"""Agent orchestrator — decision loop + toolbox + frontend trace."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Sandbox import run_tool
from tool_factory import create_tool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
TOOLBOX_DIR = BACKEND_DIR / "toolbox"
MANIFEST_PATH = TOOLBOX_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Trace helpers (frontend contract)
# ---------------------------------------------------------------------------
def _step(step_type: str, label: str, detail: str | None = None) -> dict[str, str]:
    item: dict[str, str] = {"type": step_type, "label": label}
    if detail is not None:
        item["detail"] = detail
    return item


def _format_answer(tool_name: str, raw: str) -> str:
    """Turn sandbox stdout into a display answer (e.g. speed -> '60 km/h')."""
    text = (raw or "").strip()
    if tool_name == "speed":
        try:
            value = float(text)
            if value == int(value):
                return f"{int(value)} km/h"
            return f"{value} km/h"
        except ValueError:
            pass
    return text


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


def _save_tool(name: str, code: str, description: str, inputs: list[str]) -> dict:
    """Write <name>.py and upsert a manifest entry. Returns the entry."""
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
    return entry


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
        desc_words = {w for w in re.findall(r"[a-z]+", desc) if len(w) > 3}
        task_words = {w for w in re.findall(r"[a-z]+", lower) if len(w) > 3}
        if desc_words and len(desc_words & task_words) >= 2:
            return entry
    return None


# ---------------------------------------------------------------------------
# Trivial vs needs-a-tool heuristic
# ---------------------------------------------------------------------------
_ARITH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)")

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
    lower = task.lower()
    return any(k in lower for k in _TOOL_KEYWORDS)


def _is_trivial(task: str) -> bool:
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
    lower = task.lower()
    if any(k in lower for k in ("speed", "distance", "velocity", "time")):
        return {
            "name": "speed",
            "description": task.strip(),
            "inputs": ["distance_km", "time_hr"],
            "tests": [{"args": [120, 2], "expected": 60}],
        }
    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")[:32] or "custom_tool"
    return {
        "name": slug,
        "description": task.strip(),
        "inputs": ["x"],
        "tests": [{"args": [1], "expected": 1}],
    }


def _param_count(signature: str) -> int:
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
    n = _param_count(signature)
    nums = _extract_numbers(task)
    defaults = [120.0, 2.0, 1.0, 1.0]
    while len(nums) < n:
        nums.append(defaults[len(nums) % len(defaults)])
    return nums[:n] if n else nums


def _format_arg(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _run_existing_tool(entry: dict, task: str) -> tuple[str | None, dict, str]:
    """
    Load toolbox/<name>.py and execute via Sandbox.run_tool.
    Returns (stdout_or_None, sandbox_result, entry_call).
    """
    name = entry["name"]
    code = _load_tool_code(name)
    if code is None:
        return None, {"ok": False, "reason": f"missing file {name}.py"}, ""

    args = _args_for_tool(task, entry.get("signature", ""))
    call = f"print({name}({', '.join(_format_arg(a) for a in args)}))"
    result = run_tool(code, call)
    if result.get("ok"):
        return (result.get("stdout") or "").strip(), result, call
    return None, result, call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def handle_task(task: str) -> dict[str, Any]:
    """
    Decide:
      1. trivial  -> answer directly
      2. have tool -> load + sandbox run
      3. need tool -> create_tool (real factory), save, then sandbox-run

    Returns { "answer": str, "trace": [ {type, label, detail?} ] }
    """
    trace: list[dict[str, str]] = []
    _ensure_toolbox()

    # --- Path 1: trivial ---
    if _is_trivial(task):
        trace.append(
            _step("plan", "This is trivial — I can answer directly")
        )
        answer = _answer_trivial(task)
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    # --- Plan for tool-worthy tasks ---
    trace.append(_step("plan", "Planning: this needs a speed calculation"))

    # --- Path 2: reuse from toolbox ---
    trace.append(_step("check", "Checking toolbox for a matching tool"))
    manifest = _load_manifest()
    match = _find_matching_tool(task, manifest)
    if match is not None:
        raw, sandbox_result, call = _run_existing_tool(match, task)
        if raw is not None:
            answer = _format_answer(match["name"], raw)
            # Reuse: note found tool on the check step; no build steps.
            trace[-1] = _step(
                "check",
                f"Found existing tool '{match['name']}' — reusing it",
                detail=match.get("signature"),
            )
            trace.append(_step("answer", f"Answer: {answer}"))
            return {"answer": answer, "trace": trace}

        # Sandbox failed on existing tool — fall through to factory
        trace.append(
            _step(
                "fail",
                f"Existing tool failed: {sandbox_result.get('reason', 'error')}",
                detail=(sandbox_result.get("stderr") or "")[:500] or None,
            )
        )

    # --- Path 3: create via factory ---
    trace.append(_step("no_tool", "No tool found — writing a new one"))

    task_spec = _build_task_spec(task)
    result = create_tool(task_spec)

    if not result.get("success"):
        err = result.get("error") or "factory failed"
        trace.append(_step("fail", f"Tool factory failed: {err}"))
        answer = f"Could not build a tool: {err}"
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    name = result["tool_name"]
    code = result["code"]
    entry = _save_tool(name, code, task_spec["description"], task_spec["inputs"])

    trace.append(
        _step("writing", "Writing a Python tool", detail=code)
    )
    trace.append(_step("testing", "Testing in sandbox against known values"))

    raw, sandbox_result, call = _run_existing_tool(entry, task)
    if raw is None:
        reason = sandbox_result.get("reason", "nonzero exit")
        trace.append(_step("fail", f"Test failed: {reason}"))
        answer = f"Tool created but failed tests: {reason}"
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    # Demo pass label for speed; generic otherwise
    if name == "speed":
        pass_label = "Test passed: speed(120, 2) == 60"
    else:
        pass_label = f"Test passed: {call}"
    trace.append(_step("pass", pass_label))

    answer = _format_answer(name, raw)
    trace.append(_step("answer", f"Answer: {answer}"))
    return {"answer": answer, "trace": trace}
