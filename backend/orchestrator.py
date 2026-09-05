"""Agent orchestrator — decision loop + toolbox (mocked factory)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from Sandbox import run_tool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
TOOLBOX_DIR = BACKEND_DIR / "toolbox"
MANIFEST_PATH = TOOLBOX_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# MOCK — Sunday: replace with `from tool_factory import create_tool`
# ---------------------------------------------------------------------------
def create_tool(task_spec: dict) -> dict:
    """Stub matching Soka's tool_factory.create_tool contract."""
    return {
        "success": True,
        "tool_name": task_spec["name"],
        "code": (
            "def speed(distance_km, time_hr): "
            "return distance_km / time_hr"
        ),
        "attempts": 1,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Toolbox helpers
# ---------------------------------------------------------------------------
def _ensure_toolbox() -> None:
    TOOLBOX_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text("[]\n", encoding="utf-8")


def _load_manifest() -> list[dict]:
    _ensure_toolbox()
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(entries: list[dict]) -> None:
    _ensure_toolbox()
    MANIFEST_PATH.write_text(
        json.dumps(entries, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_tool(name: str, code: str, description: str, inputs: list[str]) -> None:
    """Write <name>.py and upsert a manifest entry."""
    _ensure_toolbox()
    (TOOLBOX_DIR / f"{name}.py").write_text(code.rstrip() + "\n", encoding="utf-8")

    signature = f"{name}({', '.join(inputs)})"
    entry = {
        "name": name,
        "signature": signature,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest = _load_manifest()
    manifest = [e for e in manifest if e.get("name") != name]
    manifest.append(entry)
    _save_manifest(manifest)


def _load_tool_code(name: str) -> str | None:
    path = TOOLBOX_DIR / f"{name}.py"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _find_matching_tool(task: str, manifest: list[dict]) -> dict | None:
    """Simple match: tool name or description keywords appear in the task."""
    lower = task.lower()
    for entry in manifest:
        name = (entry.get("name") or "").lower()
        if name and name in lower:
            return entry
        desc = (entry.get("description") or "").lower()
        # overlap on content words (skip tiny words)
        desc_words = {w for w in re.findall(r"[a-z]+", desc) if len(w) > 3}
        task_words = {w for w in re.findall(r"[a-z]+", lower) if len(w) > 3}
        if desc_words and len(desc_words & task_words) >= 2:
            return entry
    return None


# ---------------------------------------------------------------------------
# Trivial vs needs-a-tool heuristic (simple + predictable)
# ---------------------------------------------------------------------------
_ARITH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)")

# Described computations that deserve a reusable tool (demo: train-speed).
_TOOL_KEYWORDS = (
    "speed",
    "distance",
    "compute",
    "calculate",
    "convert",
    "average",
    "velocity",
)


def _is_arithmetic(task: str) -> bool:
    return bool(_ARITH_RE.search(task))


def _looks_like_tool_task(task: str) -> bool:
    """True for a described computation (params / domain words), not chat."""
    lower = task.lower()
    return any(k in lower for k in _TOOL_KEYWORDS)


def _is_trivial(task: str) -> bool:
    """Arithmetic or a direct question that is not a parameterized computation."""
    if _is_arithmetic(task):
        return True
    return not _looks_like_tool_task(task)


def _eval_arithmetic(task: str) -> str:
    match = _ARITH_RE.search(task)
    if not match:
        return "0"
    a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
    ops = {
        "+": a + b,
        "-": a - b,
        "*": a * b,
        "/": (a / b) if b != 0 else float("nan"),
    }
    result = ops[op]
    # Prefer clean ints when exact (judge: "2+5" -> "7")
    if result == int(result):
        return str(int(result))
    return str(result)


def _answer_trivial(task: str) -> str:
    if _is_arithmetic(task):
        return _eval_arithmetic(task)
    return f"Answered directly (no tool needed): {task.strip()}"


# ---------------------------------------------------------------------------
# Task -> factory spec / run args
# ---------------------------------------------------------------------------
def _build_task_spec(task: str) -> dict:
    """Build the create_tool contract. Demo focuses on speed."""
    lower = task.lower()
    if any(k in lower for k in ("speed", "distance", "velocity", "time")):
        return {
            "name": "speed",
            "description": task.strip(),
            "inputs": ["distance_km", "time_hr"],
            "tests": [{"args": [120, 2], "expected": 60}],
        }
    # Generic fallback for other tool-worthy tasks
    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")[:32] or "custom_tool"
    return {
        "name": slug,
        "description": task.strip(),
        "inputs": ["x"],
        "tests": [{"args": [1], "expected": 1}],
    }


def _param_count(signature: str) -> int:
    """Parse `name(a, b)` -> 2. Empty () -> 0."""
    match = re.search(r"\((.*)\)", signature or "")
    if not match:
        return 0
    inner = match.group(1).strip()
    if not inner:
        return 0
    return len([p for p in inner.split(",") if p.strip()])


def _extract_numbers(task: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", task)]


def _args_for_tool(task: str, signature: str) -> list[float]:
    """Numbers from the task, padded with demo defaults (120, 2)."""
    n = _param_count(signature)
    nums = _extract_numbers(task)
    defaults = [120.0, 2.0, 1.0, 1.0]
    while len(nums) < n:
        nums.append(defaults[len(nums) % len(defaults)])
    return nums[:n] if n else nums


def _format_arg(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _run_existing_tool(entry: dict, task: str) -> tuple[str | None, dict]:
    """Load toolbox/<name>.py and execute via Sandbox.run_tool."""
    name = entry["name"]
    code = _load_tool_code(name)
    if code is None:
        return None, {"ok": False, "reason": f"missing file {name}.py"}

    args = _args_for_tool(task, entry.get("signature", ""))
    call = f"print({name}({', '.join(_format_arg(a) for a in args)}))"
    result = run_tool(code, call)
    if result.get("ok"):
        return (result.get("stdout") or "").strip(), result
    return None, result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def handle_task(task: str) -> dict:
    """
    Decide:
      1. trivial  -> answer directly
      2. have tool -> load + sandbox run
      3. need tool -> mocked create_tool, then save to toolbox
    """
    trace: list[str] = [f"received task: {task!r}"]
    _ensure_toolbox()

    # --- Path 1: trivial ---
    if _is_trivial(task):
        answer = _answer_trivial(task)
        trace.append("classified as trivial — answering directly, no tool")
        if _is_arithmetic(task):
            trace.append(f"evaluated arithmetic -> {answer}")
        return {"path": "trivial", "answer": answer, "trace": trace}

    # --- Path 2: reuse from toolbox ---
    trace.append("not trivial — checking toolbox manifest")
    manifest = _load_manifest()
    match = _find_matching_tool(task, manifest)
    if match is not None:
        trace.append(f"found matching tool: {match['name']} ({match.get('signature')})")
        answer, sandbox_result = _run_existing_tool(match, task)
        if answer is not None:
            trace.append(f"ran toolbox tool via sandbox -> {answer}")
            return {
                "path": "reuse",
                "answer": answer,
                "tool_name": match["name"],
                "trace": trace,
            }
        trace.append(
            f"sandbox run failed ({sandbox_result.get('reason')}); "
            "falling through to factory"
        )

    # --- Path 3: create via mocked factory ---
    trace.append("no usable tool in toolbox — calling create_tool (mock)")
    task_spec = _build_task_spec(task)
    trace.append(f"task_spec name={task_spec['name']!r}")
    result = create_tool(task_spec)

    if not result.get("success"):
        trace.append(f"factory failed: {result.get('error')}")
        return {
            "path": "factory",
            "answer": None,
            "factory": result,
            "trace": trace,
            "error": result.get("error"),
        }

    name = result["tool_name"]
    code = result["code"]
    _save_tool(name, code, task_spec["description"], task_spec["inputs"])
    trace.append(f"saved tool to toolbox/{name}.py and updated manifest")

    return {
        "path": "factory",
        "answer": result,
        "tool_name": name,
        "trace": trace,
    }
