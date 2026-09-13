"""Agent orchestrator — decision loop + toolbox + frontend trace.

Non-trivial flow:
  task → interpret_task → CalculationRequest → reuse | factory → sandbox
  → {answer, trace}
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from Sandbox import run_tool
from interpreter import interpret_task
from models import CalculationRequest
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

UNSUPPORTED_MESSAGE = (
    "I'm designed for well-defined calculations. Try a clear arithmetic "
    "expression like '2+5', or a computation like 'train speed from distance "
    "and time'."
)

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


def _format_arg(value: float | int) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


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


def _find_tool_by_operation(operation: str, manifest: list[dict]) -> dict | None:
    """Reuse only when toolbox tool name matches request.operation exactly."""
    wanted = operation.lower().strip()
    if not wanted:
        return None
    for entry in manifest:
        name = (entry.get("name") or "").lower()
        if name == wanted:
            return entry
    return None


# ---------------------------------------------------------------------------
# Trivial path — ONLY clean arithmetic (safe eval, no bare eval)
# ---------------------------------------------------------------------------
_ARITH_CHARS_RE = re.compile(r"^[\d\.\+\-\*/\(\)\s]+$")
_X_AS_MUL_RE = re.compile(r"(?<=[\d\)])\s*[xX]\s*(?=[\d\(])")

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


def _normalize_multiply(expr: str) -> str:
    """Turn × and operator-x into * before the restricted arithmetic check."""
    s = expr.replace("×", "*")
    s = _X_AS_MUL_RE.sub(" * ", s)
    return s


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
    """If task is clean arithmetic, return result string; else None."""
    expr = _normalize_multiply(task.strip())
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
    return _try_eval_arithmetic(task) is not None


def _answer_trivial(task: str) -> str:
    result = _try_eval_arithmetic(task)
    if result is None:
        raise ValueError("not a clean arithmetic expression")
    return result


# ---------------------------------------------------------------------------
# Interpreter → CalculationRequest → tool args / factory spec
# ---------------------------------------------------------------------------
def _parse_calculation_request(raw: dict[str, Any]) -> CalculationRequest | dict[str, Any]:
    """
    Validate interpreter output as CalculationRequest.
    On validation failure return an error-shaped dict for the caller.
    """
    try:
        return CalculationRequest(**raw)
    except (ValidationError, TypeError, ValueError) as exc:
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "error": f"invalid interpreter result: {exc}",
        }


def _param_names(signature: str) -> list[str]:
    match = re.search(r"\((.*)\)", signature or "")
    if not match:
        return []
    inner = match.group(1).strip()
    if not inner:
        return []
    return [p.strip() for p in inner.split(",") if p.strip()]


def _args_from_request(
    signature: str,
    inputs: dict[str, float | int],
) -> tuple[list[float | int] | None, list[str]]:
    """
    Build positional args from request.inputs in signature order.
    Never invent defaults. Returns (args, missing_names).
    """
    params = _param_names(signature)
    if not params:
        if not inputs:
            return [], []
        # No declared params — pass values in stable key order
        return [inputs[k] for k in sorted(inputs.keys())], []

    missing = [p for p in params if p not in inputs]
    if missing:
        return None, missing
    return [inputs[p] for p in params], []


def _derive_expected(operation: str, inputs: dict[str, float | int]) -> float | None:
    """
    Optional ground-truth for factory self-tests when the formula is known.
    Not used as a substitute for sandbox execution of the real answer.
    """
    op = operation.lower()
    try:
        if op == "speed" and "distance_km" in inputs and "time_hr" in inputs:
            t = float(inputs["time_hr"])
            if t == 0:
                return None
            return float(inputs["distance_km"]) / t
        if op == "distance" and "speed_kmh" in inputs and "time_hr" in inputs:
            return float(inputs["speed_kmh"]) * float(inputs["time_hr"])
        if op == "time" and "distance_km" in inputs and "speed_kmh" in inputs:
            s = float(inputs["speed_kmh"])
            if s == 0:
                return None
            return float(inputs["distance_km"]) / s
        if op == "bmi" and "height_cm" in inputs and "weight_kg" in inputs:
            h_m = float(inputs["height_cm"]) / 100.0
            if h_m == 0:
                return None
            return float(inputs["weight_kg"]) / (h_m * h_m)
        if op == "probability" and "favorable" in inputs and "total" in inputs:
            total = float(inputs["total"])
            if total == 0:
                return None
            return float(inputs["favorable"]) / total
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _build_task_spec_from_request(
    request: CalculationRequest,
    task: str,
) -> dict[str, Any]:
    """Build create_tool task_spec from the structured interpreter result."""
    name = (request.operation or "custom_tool").strip()
    # Preserve interpreter key order for factory signature
    input_names = list(request.inputs.keys())
    if not input_names:
        input_names = ["x"]

    args = [request.inputs[k] for k in input_names if k in request.inputs]
    expected = _derive_expected(name, request.inputs)

    tests: list[dict[str, Any]] = []
    if args and expected is not None:
        tests.append({"args": args, "expected": expected})
    # If we cannot derive expected (novel op), leave tests empty so the
    # factory can still generate code; we then execute with real inputs.

    return {
        "name": name,
        "description": task.strip(),
        "inputs": input_names,
        "tests": tests,
    }


def _run_tool_with_inputs(
    entry: dict,
    inputs: dict[str, float | int],
) -> tuple[str | None, dict, str, list[str]]:
    """
    Load toolbox/<name>.py and run via Sandbox with request.inputs.
    Returns (stdout_or_None, sandbox_result, entry_call, missing_params).
    """
    name = entry["name"]
    code = _load_tool_code(name)
    if code is None:
        return None, {"ok": False, "reason": f"missing file {name}.py"}, "", []

    args, missing = _args_from_request(entry.get("signature", ""), inputs)
    if missing:
        return None, {"ok": False, "reason": "missing inputs"}, "", missing
    assert args is not None

    call = f"print({name}({', '.join(_format_arg(a) for a in args)}))"
    result = run_tool(code, call)
    if result.get("ok"):
        return (result.get("stdout") or "").strip(), result, call, []
    return None, result, call, []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def handle_task(task: str) -> dict[str, Any]:
    """
    Decide:
      1. trivial arithmetic → evaluate locally
      2. interpret_task → CalculationRequest
         - needs_input / unsupported / error → short answer + trace
         - ok → reuse toolbox tool OR create via factory, then sandbox

    Returns { "answer": str, "trace": [ {type, label, detail?} ] }
    """
    trace: list[dict[str, str]] = []
    _ensure_toolbox()

    logger.info("=" * 60)
    logger.info("Incoming task: %r", task)

    # --- Path 1: trivial arithmetic (unchanged safe AST path) ---
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

    # --- Interpret natural language (source of truth) ---
    trace.append(_step("plan", "Interpreting the request with the local model"))
    logger.info("Calling interpret_task()…")
    raw = interpret_task(task)
    logger.info("Interpreter raw result: %s", json.dumps(raw, indent=2))

    parsed = _parse_calculation_request(raw)
    if isinstance(parsed, dict):
        err = parsed.get("error") or "invalid interpreter result"
        logger.info("Path chosen: ERROR — CalculationRequest validation failed")
        trace.append(_step("fail", f"Interpreter validation failed: {err}"))
        answer = f"Could not understand the request: {err}"
        trace.append(_step("answer", answer))
        return {"answer": answer, "trace": trace}

    request = parsed
    trace.append(
        _step(
            "check",
            f"Interpreter status={request.status}"
            + (f", operation={request.operation}" if request.operation else ""),
            detail=json.dumps(
                {
                    "inputs": request.inputs,
                    "missing_inputs": request.missing_inputs,
                }
            ),
        )
    )

    # --- Interpreter status gates ---
    if request.status == "needs_input":
        missing = request.missing_inputs or ["(unspecified)"]
        missing_list = ", ".join(missing)
        logger.info(
            "Path chosen: NEEDS_INPUT — missing: %s", missing_list
        )
        answer = (
            f"I need more information to compute "
            f"{request.operation or 'this'}. "
            f"Missing inputs: {missing_list}."
        )
        trace.append(
            _step("answer", answer, detail=f"missing_inputs={missing}")
        )
        return {"answer": answer, "trace": trace}

    if request.status == "unsupported":
        logger.info("Path chosen: UNSUPPORTED")
        answer = UNSUPPORTED_MESSAGE
        trace.append(_step("answer", answer))
        return {"answer": answer, "trace": trace}

    if request.status == "error":
        err = request.error or "interpreter error"
        logger.info("Path chosen: ERROR — %s", err)
        answer = f"Could not interpret the request: {err}"
        trace.append(_step("fail", answer))
        trace.append(_step("answer", answer))
        return {"answer": answer, "trace": trace}

    # status == "ok"
    if not request.operation:
        logger.info("Path chosen: ERROR — ok status but no operation")
        answer = "Could not determine which calculation to run."
        trace.append(_step("fail", answer))
        trace.append(_step("answer", answer))
        return {"answer": answer, "trace": trace}

    if not request.inputs:
        # ok without inputs should have been needs_input; fail cleanly
        logger.info("Path chosen: NEEDS_INPUT — ok but empty inputs")
        answer = (
            f"I need more information to compute {request.operation}. "
            f"No numeric inputs were provided."
        )
        trace.append(_step("answer", answer))
        return {"answer": answer, "trace": trace}

    operation = request.operation
    trace.append(
        _step("plan", f"Planning: this needs a {operation} calculation")
    )

    # --- Path 2: reuse from toolbox by operation name ---
    trace.append(_step("check", "Checking toolbox for a matching tool"))
    manifest = _load_manifest()
    match = _find_tool_by_operation(operation, manifest)
    if match is not None:
        raw_out, sandbox_result, call, missing = _run_tool_with_inputs(
            match, request.inputs
        )
        if missing:
            logger.info(
                "Tool '%s' needs params not in request.inputs: %s",
                match["name"],
                missing,
            )
            answer = (
                f"I found tool '{match['name']}' but still need: "
                f"{', '.join(missing)}."
            )
            trace.append(
                _step(
                    "fail",
                    "Toolbox tool signature does not match available inputs",
                    detail=str(missing),
                )
            )
            trace.append(_step("answer", answer))
            return {"answer": answer, "trace": trace}

        if raw_out is not None:
            answer = _format_answer(match["name"], raw_out)
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
            match["name"],
            sandbox_result.get("reason", "error"),
        )
        trace.append(
            _step(
                "fail",
                f"Existing tool failed: {sandbox_result.get('reason', 'error')}",
                detail=(sandbox_result.get("stderr") or "")[:500] or None,
            )
        )

    # --- Path 3: create via factory (new operations allowed) ---
    trace.append(_step("no_tool", "No tool found — writing a new one"))
    task_spec = _build_task_spec_from_request(request, task)
    logger.info(
        "Path chosen: FACTORY — calling create_tool() with task_spec: %s",
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
        result.get("tool_name"),
        result.get("attempts"),
    )

    name = result["tool_name"]
    code = result["code"]
    entry = _save_tool(name, code, task_spec["description"], task_spec["inputs"])

    trace.append(_step("writing", "Writing a Python tool", detail=code))
    trace.append(_step("testing", "Running the new tool in the sandbox"))

    raw_out, sandbox_result, call, missing = _run_tool_with_inputs(
        entry, request.inputs
    )
    if missing:
        answer = (
            f"Tool '{name}' was created but still needs: "
            f"{', '.join(missing)}."
        )
        trace.append(_step("fail", answer, detail=str(missing)))
        trace.append(_step("answer", answer))
        return {"answer": answer, "trace": trace}

    if raw_out is None:
        reason = sandbox_result.get("reason", "nonzero exit")
        trace.append(_step("fail", f"Sandbox run failed: {reason}"))
        answer = f"Tool created but failed when run: {reason}"
        trace.append(_step("answer", f"Answer: {answer}"))
        return {"answer": answer, "trace": trace}

    trace.append(_step("pass", f"Sandbox run succeeded: {call}"))
    answer = _format_answer(name, raw_out)
    trace.append(_step("answer", f"Answer: {answer}"))
    return {"answer": answer, "trace": trace}
