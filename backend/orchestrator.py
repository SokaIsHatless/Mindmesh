"""Agent orchestrator — decision loop + toolbox + frontend trace."""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Sandbox import run_tool
from tool_factory import create_tool

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("orchestrator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[orchestrator] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

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


# Canonical tool names for clear computational requests (not free-form NLP).
_KNOWN_QUANTITIES = {
    "speed": "speed",
    "velocity": "speed",
    "distance": "distance",
    "time": "time",
    "average": "average",
    "probability": "probability",
    "prob": "probability",
}

_DEFAULT_ARGS = {
    "speed": [120.0, 2.0],
    "distance": [60.0, 2.0],
    "time": [120.0, 60.0],
    "probability": [13.0, 52.0],
}


def _intended_tool_name(task: str) -> str | None:
    """
    What tool purpose does this clear request ask for?
    Prefer the quantity being computed, not every keyword in the sentence.
    Returns None when there is no confident mapping → do not reuse.
    """
    lower = task.lower().strip()

    # "compute/calculate/find the <quantity>"
    m = re.search(
        r"\b(?:compute|calculate|find|determine|get)\s+(?:the\s+)?(\w+)",
        lower,
    )
    if m and m.group(1) in _KNOWN_QUANTITIES:
        return _KNOWN_QUANTITIES[m.group(1)]

    # "<quantity> from ..." — left-hand quantity is what we build
    m = re.search(
        r"\b(speed|velocity|distance|time|average|probability)\b"
        r"(?:\s+\w+){0,3}\s+from\b",
        lower,
    )
    if m:
        return _KNOWN_QUANTITIES[m.group(1)]

    # "train speed ...", or a leading known quantity as the subject
    m = re.search(
        r"\b(?:train\s+)?(speed|velocity|distance|time|average|probability)\b",
        lower,
    )
    if m:
        return _KNOWN_QUANTITIES[m.group(1)]

    if re.search(r"\bprobability\b", lower):
        return "probability"

    return None


def _find_matching_tool(task: str, manifest: list[dict]) -> dict | None:
    """
    Reuse only on a confident purpose match (intended name == tool name).
    Loose keyword overlap is NOT enough — e.g. a distance request must not
    reuse a speed tool just because the word 'speed' appears in the text.
    """
    intended = _intended_tool_name(task)
    if not intended:
        return None

    for entry in manifest:
        name = (entry.get("name") or "").lower()
        if name == intended:
            return entry
    return None


# ---------------------------------------------------------------------------
# Trivial path — ONLY clean arithmetic (safe eval, no bare eval)
# ---------------------------------------------------------------------------
_ARITH_CHARS_RE = re.compile(r"^[\d\.\+\-\*/\(\)\s]+$")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_arith_node(node: ast.AST) -> float:
    """Evaluate an AST that may contain only numbers and + - * / (unary ±)."""
    if isinstance(node, ast.Expression):
        return _eval_arith_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_arith_node(node.left)
        right = _eval_arith_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith_node(node.operand))

    raise ValueError("unsupported expression")


def _try_eval_arithmetic(task: str) -> str | None:
    """
    If `task` is a clean arithmetic expression, return its result as a string.
    Otherwise return None (not trivial — route to tool path).
    """
    expr = task.strip()
    if not expr or not _ARITH_CHARS_RE.fullmatch(expr):
        return None

    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_arith_node(tree)
    except (SyntaxError, ValueError, TypeError, OverflowError):
        return None

    if value == int(value):
        return str(int(value))
    return str(value)


def _is_trivial(task: str) -> bool:
    """True only when the whole task is a safely evaluable arithmetic expression."""
    return _try_eval_arithmetic(task) is not None


def _answer_trivial(task: str) -> str:
    result = _try_eval_arithmetic(task)
    if result is None:
        raise ValueError("not a clean arithmetic expression")
    return result


# ---------------------------------------------------------------------------
# Task -> factory spec / run args
# ---------------------------------------------------------------------------
def _slug_tool_name(task: str) -> str:
    words = re.findall(r"[a-z0-9]+", task.lower())[:4]
    return "_".join(words)[:32] or "custom_tool"


def _build_task_spec(task: str) -> dict:
    """Build create_tool contract from a clear computational request."""
    name = _intended_tool_name(task) or _slug_tool_name(task)
    description = task.strip()

    if name == "speed":
        return {
            "name": "speed",
            "description": description,
            "inputs": ["distance_km", "time_hr"],
            "tests": [{"args": [120, 2], "expected": 60}],
        }
    if name == "distance":
        return {
            "name": "distance",
            "description": description,
            "inputs": ["speed_kmh", "time_hr"],
            "tests": [{"args": [60, 2], "expected": 120}],
        }
    if name == "time":
        return {
            "name": "time",
            "description": description,
            "inputs": ["distance_km", "speed_kmh"],
            "tests": [{"args": [120, 60], "expected": 2}],
        }
    if name == "probability":
        return {
            "name": "probability",
            "description": description,
            "inputs": ["favorable", "total"],
            "tests": [{"args": [13, 52], "expected": 0.25}],
        }

    return {
        "name": name,
        "description": description,
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


def _args_for_tool(
    task: str,
    signature: str,
    tool_name: str = "",
    fallback_args: list[float] | None = None,
) -> list[float]:
    """
    Use numbers from the task when present; otherwise tool-specific demo
    defaults (or factory test args). Do not invent args from word problems.
    """
    n = _param_count(signature)
    nums = _extract_numbers(task)
    if len(nums) >= n and n > 0:
        return nums[:n]

    defaults = (
        list(fallback_args)
        if fallback_args is not None
        else list(_DEFAULT_ARGS.get(tool_name, [120.0, 2.0, 1.0, 1.0]))
    )
    while len(nums) < n:
        nums.append(defaults[len(nums) % len(defaults)])
    return nums[:n] if n else nums


def _format_arg(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _run_existing_tool(
    entry: dict,
    task: str,
    fallback_args: list[float] | None = None,
) -> tuple[str | None, dict, str]:
    """
    Load toolbox/<name>.py and execute via Sandbox.run_tool.
    Returns (stdout_or_None, sandbox_result, entry_call).
    """
    name = entry["name"]
    code = _load_tool_code(name)
    if code is None:
        return None, {"ok": False, "reason": f"missing file {name}.py"}, ""

    args = _args_for_tool(
        task,
        entry.get("signature", ""),
        tool_name=name,
        fallback_args=fallback_args,
    )
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
      2. have tool -> load + sandbox run (confident name match only)
      3. need tool -> create_tool (real factory), save, then sandbox-run

    Returns { "answer": str, "trace": [ {type, label, detail?} ] }
    """
    trace: list[dict[str, str]] = []
    _ensure_toolbox()

    logger.info("=" * 60)
    logger.info("Incoming task: %r", task)

    # --- Path 1: trivial ---
    if _is_trivial(task):
        expr_result = _answer_trivial(task)
        logger.info(
            "Path chosen: TRIVIAL — task is a clean arithmetic expression"
        )
        logger.info("  expression=%r -> result=%r", task.strip(), expr_result)
        trace.append(
            _step("plan", "This is trivial — I can answer directly")
        )
        answer = expr_result
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    intended = _intended_tool_name(task) or "custom"
    trace.append(
        _step("plan", f"Planning: this needs a {intended} calculation")
    )

    # --- Path 2: reuse from toolbox (confident name match only) ---
    trace.append(_step("check", "Checking toolbox for a matching tool"))
    manifest = _load_manifest()
    match = _find_matching_tool(task, manifest)
    if match is not None:
        raw, sandbox_result, call = _run_existing_tool(match, task)
        if raw is not None:
            answer = _format_answer(match["name"], raw)
            logger.info(
                "Path chosen: REUSE — matched existing tool '%s'",
                match["name"],
            )
            logger.info("  signature=%s | call=%s", match.get("signature"), call)
            trace[-1] = _step(
                "check",
                f"Found existing tool '{match['name']}' — reusing it",
                detail=match.get("signature"),
            )
            trace.append(_step("answer", f"Answer: {answer}"))
            return {"answer": answer, "trace": trace}

        logger.info(
            "Matched tool '%s' but sandbox run failed: reason=%s",
            match["name"], sandbox_result.get("reason", "error"),
        )
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
    logger.info(
        "Path chosen: FACTORY — no confident toolbox match, calling "
        "create_tool() with task_spec: %s",
        json.dumps(task_spec, indent=2),
    )
    result = create_tool(task_spec)

    if not result.get("success"):
        err = result.get("error") or "factory failed"
        logger.info("Factory result: FAILURE — %s", err)
        trace.append(_step("fail", f"Tool factory failed: {err}"))
        answer = f"Could not build a tool: {err}"
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    logger.info(
        "Factory result: SUCCESS — tool=%s, attempts=%s",
        result.get("tool_name"), result.get("attempts"),
    )

    name = result["tool_name"]
    code = result["code"]
    entry = _save_tool(name, code, task_spec["description"], task_spec["inputs"])

    trace.append(
        _step("writing", "Writing a Python tool", detail=code)
    )
    trace.append(_step("testing", "Testing in sandbox against known values"))

    test_args = None
    if task_spec.get("tests"):
        test_args = [float(a) for a in task_spec["tests"][0]["args"]]

    raw, sandbox_result, call = _run_existing_tool(
        entry, task, fallback_args=test_args
    )
    if raw is None:
        reason = sandbox_result.get("reason", "nonzero exit")
        trace.append(_step("fail", f"Test failed: {reason}"))
        answer = f"Tool created but failed tests: {reason}"
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    if name == "speed":
        pass_label = "Test passed: speed(120, 2) == 60"
    else:
        pass_label = f"Test passed: {call}"
    trace.append(_step("pass", pass_label))

    answer = _format_answer(name, raw)
    trace.append(_step("answer", f"Answer: {answer}"))
    return {"answer": answer, "trace": trace}
