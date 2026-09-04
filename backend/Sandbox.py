"""
sandbox_runner.py — EDGE-SMART tool sandbox (cross-platform)

Executes a self-generated Python tool in an isolated subprocess with:
  1. Import validation (allowlist)  -> block dangerous modules BEFORE running
  2. Subprocess isolation           -> crashes/hangs can't take down the agent
  3. Resource limits (CPU, memory)  -> runaway code gets killed  [Linux]
  4. Hard timeout                   -> infinite loops get killed  [all OSes]
  5. Clean environment              -> no inherited secrets/proxies

CROSS-PLATFORM: Windows, macOS, and Linux. No WSL needed.
  - Layers 1, 2, 4, 5 work on every OS.
  - Layer 3 (resource.setrlimit) is Linux-focused:
      * Linux: CPU + RLIMIT_AS + NPROC where supported
      * macOS: skip RLIMIT_AS/CPU (they raise or fire SIGXCPU early);
        parent monitors RSS and enforces wall-clock timeout instead
      * Windows: no `resource` module; wall-clock timeout is the backstop

This is a HACKATHON-GRADE sandbox: reasonable, layered, explainable.
It is NOT a bulletproof security boundary.
"""
area = length * breadth
import ast
import subprocess
import sys
import tempfile
import os
import signal
import platform
import textwrap
import time

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
TMP_DIR = tempfile.gettempdir()

# ---------------------------------------------------------------------------
# LAYER 1: Static import validation  (identical on every OS)
# ---------------------------------------------------------------------------
ALLOWED_IMPORTS = {
    "math", "datetime", "json", "re", "random",
    "statistics", "decimal", "fractions", "itertools",
    "collections", "string", "textwrap",
}


def validate_imports(code: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Rejects any import not in the allowlist."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"Disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"Disallowed import: {node.module}"
    return True, "ok"


# ---------------------------------------------------------------------------
# LAYER 3: Resource-limit preamble
# ---------------------------------------------------------------------------
# Child runs this FIRST. On Linux it caps CPU/memory. On macOS we skip
# RLIMIT_AS (raises ValueError) and RLIMIT_CPU (SIGXCPU races the parent
# timeout). On Windows `resource` does not exist.

def _child_preamble(cpu_seconds: int, mem_mb: int) -> str:
    if IS_WINDOWS:
        return (
            "# (Windows: no resource module; "
            "parent wall-clock timeout is the backstop)\n"
        )

    # Child decides by its own platform so the same parent code works
    # if you copy the generated script around.
    return textwrap.dedent(f"""
        import resource
        import platform

        _sys = platform.system()

        # --- CPU limit (Linux). Skip on macOS to avoid early SIGXCPU. ---
        if _sys != "Darwin":
            try:
                _soft, _hard = resource.getrlimit(resource.RLIMIT_CPU)
                _want = {cpu_seconds}
                if _hard == resource.RLIM_INFINITY:
                    _safe = _want
                else:
                    _safe = min(_want, _hard)
                if _safe > 0:
                    resource.setrlimit(resource.RLIMIT_CPU, (_safe, _safe))
            except (ValueError, OSError):
                pass

        # --- Memory / address-space cap (Linux). ---
        # macOS rejects RLIMIT_AS with:
        #   ValueError: current limit exceeds maximum limit
        # so we never call it on Darwin. Parent RSS monitoring covers macOS.
        if _sys != "Darwin":
            try:
                _mem = {mem_mb} * 1024 * 1024
                _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
                if _hard != resource.RLIM_INFINITY:
                    _mem = min(_mem, _hard)
                resource.setrlimit(resource.RLIMIT_AS, (_mem, _mem))
            except (ValueError, OSError):
                pass

        # --- Block spawning more processes where supported ---
        if _sys != "Darwin":
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            except (ValueError, OSError):
                pass
    """)


def _clean_env() -> dict:
    """Minimal environment: no inherited tokens/proxies. OS-appropriate."""
    if IS_WINDOWS:
        return {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        }
    return {"PATH": "/usr/bin:/bin"}


def _kill(process: subprocess.Popen) -> None:
    """Kill a runaway process (and its group on Unix)."""
    if process.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError, AttributeError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass


def _process_rss_kb(pid: int) -> int | None:
    """Best-effort RSS for parent-side memory monitoring (macOS/Linux)."""
    if IS_WINDOWS:
        return None
    try:
        ps_result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1,
        )
        rss_text = ps_result.stdout.strip()
        return int(rss_text) if rss_text else None
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# LAYERS 2 + 4: isolated subprocess + wall-clock timeout (+ macOS RSS)
# ---------------------------------------------------------------------------
def run_tool(
    code: str,
    entry_call: str,
    cpu_seconds: int = 2,
    mem_mb: int = 128,
    wall_timeout: int = 5,
) -> dict:
    """
    Execute `code` in an isolated subprocess, then run `entry_call`
    (e.g. "print(add(2, 3))") and capture the output.

    Returns: {ok, stdout, stderr, reason}
    """
    safe, reason = validate_imports(code)
    if not safe:
        return {"ok": False, "stdout": "", "stderr": "", "reason": reason}

    full_script = f"{_child_preamble(cpu_seconds, mem_mb)}\n{code}\n{entry_call}\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=TMP_DIR
    ) as f:
        f.write(full_script)
        script_path = f.name

    # New session/process group so we can kill the whole tree on Unix.
    # On Windows, start_new_session maps to CREATE_NEW_PROCESS_GROUP.
    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": _clean_env(),
        "cwd": TMP_DIR,
        "start_new_session": True,
    }

    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", script_path],
            **popen_kwargs,
        )

        # On macOS (and as a backup elsewhere), poll for timeout + RSS.
        # On Linux, RLIMIT_AS usually stops memory bombs; parent RSS is extra.
        start_time = time.monotonic()
        memory_limit_kb = mem_mb * 1024
        monitor_memory = IS_MAC or IS_LINUX

        while process.poll() is None:
            elapsed = time.monotonic() - start_time

            if elapsed >= wall_timeout:
                _kill(process)
                stdout, stderr = process.communicate()
                return {
                    "ok": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "reason": "timeout",
                }

            if monitor_memory:
                rss_kb = _process_rss_kb(process.pid)
                if rss_kb is not None and rss_kb > memory_limit_kb:
                    _kill(process)
                    stdout, stderr = process.communicate()
                    return {
                        "ok": False,
                        "stdout": stdout,
                        "stderr": stderr,
                        "reason": (
                            f"memory limit exceeded "
                            f"({rss_kb // 1024} MB > {mem_mb} MB)"
                        ),
                    }

            time.sleep(0.05)

        stdout, stderr = process.communicate()
        return {
            "ok": process.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "reason": "ok" if process.returncode == 0 else "nonzero exit",
        }
    finally:
        if process is not None and process.poll() is None:
            _kill(process)
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running on: {platform.system()}\n")

    print("=== Case 1: a good tool (agent wrote a working function) ===")
    print(run_tool("def add(a, b):\n    return a + b", "print(add(2, 3))"))

    print("\n=== Case 2: dangerous import (blocked before running) ===")
    print(
        run_tool(
            "import os\ndef wipe():\n    os.system('echo bad')",
            "wipe()",
        )
    )

    print("\n=== Case 3: infinite loop (killed by timeout) ===")
    print(run_tool("def loop():\n    while True:\n        pass", "loop()"))

    print(
        "\n=== Case 4: memory bomb "
        "(Linux: rlimit/RSS; macOS: RSS; Windows: timeout) ==="
    )
    print(run_tool("def bomb():\n    x = [0] * (10**9)", "bomb()"))
