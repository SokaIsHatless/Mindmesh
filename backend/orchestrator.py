"""Agent orchestrator — pipeline decision loop + frontend trace.

Flow:
  task → interpret → normalize → resolve capability
       → reuse verified tool  OR  factory → tests → verify → register
       → execute (Sandbox) → result validate → {answer, trace}
"""

from __future__ import annotations

import ast
import logging
import operator
import re
import sys
from pathlib import Path
from typing import Any

# Support ``backend.orchestrator`` and flat ``from orchestrator import …``.
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:
    from .Sandbox import run_tool
    from .capability_lifecycle import register_verified_capability
    from .capability_resolver import resolve_capability
    from .interpreter import interpret_task
    from .models import (
        CalculationRequest,
        InputSpec,
        OutputSpec,
        TestCase,
        ToolSpec,
    )
    from .normalizer import normalize_request
    from .result_validator import validate_result
    from .test_generator import generate_tests
    from .tool_factory import create_tool
    from .verifier import verify_tool
    from .capabilities import CapabilityRegistry
except ImportError:  # pragma: no cover - flat discovery / uvicorn from backend/
    from Sandbox import run_tool
    from capability_lifecycle import register_verified_capability
    from capability_resolver import resolve_capability
    from interpreter import interpret_task
    from models import (
        CalculationRequest,
        InputSpec,
        OutputSpec,
        TestCase,
        ToolSpec,
    )
    from normalizer import normalize_request
    from result_validator import validate_result
    from test_generator import generate_tests
    from tool_factory import create_tool
    from verifier import verify_tool
    from capabilities import CapabilityRegistry

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
# Paths / registry
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
TOOLBOX_DIR = BACKEND_DIR / "toolbox"
MANIFEST_PATH = TOOLBOX_DIR / "manifest.json"
REGISTRY_DB_PATH = BACKEND_DIR / "capabilities" / "capabilities.sqlite3"

_registry: Any | None = None


def get_registry() -> Any:
    """Shared Capability Registry (lazy). Tests may replace via ``set_registry``."""
    global _registry
    if _registry is None:
        REGISTRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _registry = CapabilityRegistry(REGISTRY_DB_PATH)
    return _registry


def set_registry(registry: Any | None) -> None:
    """Replace or clear the process-wide registry (used by integration tests)."""
    global _registry
    _registry = registry


# ---------------------------------------------------------------------------
# Trace helpers (frontend contract)
# ---------------------------------------------------------------------------
def _step(step_type: str, label: str, detail: str | None = None) -> dict[str, str]:
    item: dict[str, str] = {"type": step_type, "label": label}
    if detail is not None:
        item["detail"] = detail
    return item


def _finish(answer: str, trace: list[dict[str, str]]) -> dict[str, Any]:
    trace.append(_step("answer", f"Answer: {answer}" if not answer.startswith("Answer:") else answer))
    return {"answer": answer, "trace": trace}


# ---------------------------------------------------------------------------
# Toolbox helpers (code on disk; registry holds verified metadata)
# ---------------------------------------------------------------------------
def _ensure_toolbox() -> None:
    TOOLBOX_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text("[]\n", encoding="utf-8")


def _tool_code_path(name: str) -> Path:
    return TOOLBOX_DIR / f"{name}.py"


def _write_tool_file(name: str, code: str) -> str:
    """Persist generated code under toolbox/; return absolute code_path."""
    _ensure_toolbox()
    path = _tool_code_path(name)
    path.write_text(code.rstrip() + "\n", encoding="utf-8")
    return str(path.resolve())


def _load_code(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return file_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Compatibility: known-quantity ToolSpec builders (existing demo contracts)
# ---------------------------------------------------------------------------
_KNOWN_QUANTITIES = {
    "speed": "speed",
    "velocity": "speed",
    "distance": "distance",
    "time": "time",
    "average": "average",
    "probability": "probability",
    "prob": "probability",
}


def _intended_tool_name(task: str) -> str | None:
    lower = task.lower().strip()
    m = re.search(
        r"\b(?:compute|calculate|find|determine|get)\s+(?:the\s+)?(\w+)",
        lower,
    )
    if m and m.group(1) in _KNOWN_QUANTITIES:
        return _KNOWN_QUANTITIES[m.group(1)]
    m = re.search(
        r"\b(speed|velocity|distance|time|average|probability)\b"
        r"(?:\s+\w+){0,3}\s+from\b",
        lower,
    )
    if m:
        return _KNOWN_QUANTITIES[m.group(1)]
    m = re.search(
        r"\b(?:train\s+)?(speed|velocity|distance|time|average|probability)\b",
        lower,
    )
    if m:
        return _KNOWN_QUANTITIES[m.group(1)]
    if re.search(r"\bprobability\b", lower):
        return "probability"
    return None


def _slug_tool_name(task: str) -> str:
    words = re.findall(r"[a-z0-9]+", task.lower())[:4]
    return "_".join(words)[:32] or "custom_tool"


def _build_tool_spec(task: str, operation: str | None = None) -> ToolSpec:
    """Describe a clear computational request (legacy demo contracts preserved)."""
    name: str | None = None
    if operation:
        op = operation.strip().lower()
        if op in _KNOWN_QUANTITIES:
            name = _KNOWN_QUANTITIES[op]
        elif op in set(_KNOWN_QUANTITIES.values()):
            name = op
    if name is None:
        name = _intended_tool_name(task) or _slug_tool_name(task)
    if name in _KNOWN_QUANTITIES:
        name = _KNOWN_QUANTITIES[name]
    purpose = task.strip()

    if name == "speed":
        return ToolSpec(
            name="speed",
            purpose=purpose,
            operation="speed",
            inputs=[
                InputSpec(name="distance_km", type="float", unit="km"),
                InputSpec(name="time_hr", type="float", unit="hr"),
            ],
            outputs=[OutputSpec(name="speed_kmh", type="number", unit="km/h")],
            constraints=["time_hr must not be zero"],
            examples=[
                TestCase(input={"distance_km": 120, "time_hr": 2}, expected=60)
            ],
        )
    if name == "distance":
        return ToolSpec(
            name="distance",
            purpose=purpose,
            operation="distance",
            inputs=[
                InputSpec(name="speed_kmh", type="float", unit="km/h"),
                InputSpec(name="time_hr", type="float", unit="hr"),
            ],
            outputs=[OutputSpec(name="distance_km", type="number", unit="km")],
            examples=[TestCase(input={"speed_kmh": 60, "time_hr": 2}, expected=120)],
        )
    if name == "time":
        return ToolSpec(
            name="time",
            purpose=purpose,
            operation="time",
            inputs=[
                InputSpec(name="distance_km", type="float", unit="km"),
                InputSpec(name="speed_kmh", type="float", unit="km/h"),
            ],
            outputs=[OutputSpec(name="time_hr", type="number", unit="hr")],
            constraints=["speed_kmh must not be zero"],
            examples=[
                TestCase(input={"distance_km": 120, "speed_kmh": 60}, expected=2)
            ],
        )
    if name == "probability":
        return ToolSpec(
            name="probability",
            purpose=purpose,
            operation="probability",
            inputs=[
                InputSpec(name="favorable", type="number"),
                InputSpec(name="total", type="number"),
            ],
            outputs=[OutputSpec(name="probability", type="number")],
            constraints=[
                "total must not be zero",
                "probability must be between 0 and 1",
            ],
            examples=[
                TestCase(input={"favorable": 13, "total": 52}, expected=0.25)
            ],
        )

    # Generic fallback from operation name only (no domain branches).
    return ToolSpec(
        name=name,
        purpose=purpose,
        operation=operation or name,
        inputs=[InputSpec(name="x", type="number")],
        outputs=[OutputSpec(name="result", type="number")],
        examples=[TestCase(input={"x": 1}, expected=1)],
    )


def _tool_spec_from_capability(capability: Any) -> ToolSpec:
    """Rebuild a ToolSpec from registry metadata for result validation."""
    return ToolSpec(
        name=capability.tool_id,
        purpose=capability.description,
        operation=capability.operation,
        inputs=list(capability.input_schema),
        output=capability.output_schema,
    )


def _aliases_for_tool_spec(tool_spec: ToolSpec) -> list[str]:
    aliases: list[str] = []
    op = (tool_spec.operation or "").strip().lower()
    name = tool_spec.name.strip().lower()
    if op and op != name:
        aliases.append(op)
    # Preserve legacy velocity → speed alias when registering speed.
    if name == "speed" and "velocity" not in aliases:
        aliases.append("velocity")
    return aliases


# ---------------------------------------------------------------------------
# Trivial arithmetic (unchanged behavior)
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

DECLINE_MESSAGE = (
    "I'm designed for well-defined calculations. Try a clear arithmetic "
    "expression like '2+5', or a computation like 'train speed from distance "
    "and time'."
)


def _normalize_multiply(expr: str) -> str:
    s = expr.replace("×", "*")
    s = _X_AS_MUL_RE.sub(" * ", s)
    return s


def _eval_arith_node(node: ast.AST) -> float:
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
# Sandbox execution + answer formatting
# ---------------------------------------------------------------------------
def _positional_args_from_inputs(
    input_schema: list[Any], inputs: dict[str, float]
) -> list[Any]:
    args: list[Any] = []
    for spec in input_schema:
        if spec.name in inputs:
            args.append(inputs[spec.name])
        elif not getattr(spec, "required", True):
            args.append(getattr(spec, "default", None))
        else:
            raise ValueError(f"missing required input '{spec.name}'")
    return args


def _run_code(
    code: str, function_name: str, args: list[Any]
) -> tuple[str | None, dict[str, Any], str]:
    entry_call = f"print({function_name}(*{args!r}))"
    result = run_tool(code, entry_call)
    if result.get("ok") and result.get("reason") == "ok":
        return (result.get("stdout") or "").strip(), result, entry_call
    return None, result, entry_call


def _format_answer(value: Any, unit: str | None) -> str:
    if isinstance(value, float) and value == int(value):
        text = str(int(value))
    else:
        text = str(value).strip()
    if unit:
        return f"{text} {unit}"
    return text


def _execute_and_validate(
    *,
    code: str,
    function_name: str,
    tool_spec: ToolSpec,
    inputs: dict[str, float],
    normalized_request: Any,
) -> tuple[str | None, list[dict[str, str]]]:
    """Run via Sandbox and validate. Returns (answer_or_None, extra_trace_steps)."""
    steps: list[dict[str, str]] = []
    try:
        args = _positional_args_from_inputs(tool_spec.inputs, inputs)
    except ValueError as exc:
        steps.append(_step("fail", f"Execution failed: {exc}"))
        return None, steps

    raw, sandbox_result, call = _run_code(code, function_name, args)
    if raw is None:
        reason = sandbox_result.get("reason") or "execution failed"
        steps.append(
            _step(
                "fail",
                f"Execution failed: {reason}",
                detail=(sandbox_result.get("stderr") or "")[:500] or None,
            )
        )
        return None, steps

    validation = validate_result(
        tool_spec,
        {"value": _parse_numeric_stdout(raw), "unit": tool_spec.output.unit},
        unit=tool_spec.output.unit,
        normalized_request=normalized_request,
    )
    if validation.status != "valid":
        detail = "; ".join(validation.errors) or validation.details or "invalid"
        steps.append(_step("fail", f"Result validation failed: {detail}"))
        return None, steps

    answer = _format_answer(
        validation.normalized_value
        if validation.normalized_value is not None
        else validation.value,
        validation.normalized_unit or tool_spec.output.unit,
    )
    steps.append(_step("pass", f"Result validated ({call})"))
    return answer, steps


def _parse_numeric_stdout(raw: str) -> Any:
    text = (raw or "").strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return float(text)
        except ValueError:
            return text


# ---------------------------------------------------------------------------
# Generation path: factory → tests → verify → register
# ---------------------------------------------------------------------------
def _generate_verify_register(
    *,
    task: str,
    tool_spec: ToolSpec,
    registry: Any,
    trace: list[dict[str, str]],
) -> tuple[Any | None, str | None]:
    """
    Returns (capability_or_None, error_message_or_None).
    Never registers unless verification status is verified.
    """
    trace.append(_step("no_tool", "No capability found — writing a new one"))

    logger.info("Calling tool factory for %s", tool_spec.name)
    factory_result = create_tool(tool_spec)
    if not factory_result.get("success"):
        err = factory_result.get("error") or "tool factory failed"
        trace.append(_step("fail", f"Tool factory failed: {err}"))
        return None, err

    code = factory_result.get("code") or ""
    name = factory_result.get("tool_name") or tool_spec.name
    code_path = _write_tool_file(name, code)
    trace.append(_step("writing", "Writing a Python tool", detail=code))

    trace.append(_step("testing", "Generating independent verification tests"))
    test_result = generate_tests(tool_spec)
    if not getattr(test_result, "success", False):
        err = getattr(test_result, "error", None) or "test generation failed"
        trace.append(_step("fail", f"Test generation failed: {err}"))
        return None, err

    trace.append(_step("testing", "Verifying generated tool against tests"))
    verification = verify_tool(tool_spec, code, test_result)
    if getattr(verification, "status", None) != "verified":
        err = (
            getattr(verification, "details", None)
            or "; ".join(getattr(verification, "errors", []) or [])
            or f"verification status={getattr(verification, 'status', None)!r}"
        )
        trace.append(_step("fail", f"Verification failed: {err}"))
        return None, err

    trace.append(_step("pass", "Verification passed — registering capability"))
    lifecycle = register_verified_capability(
        tool_spec,
        verification,
        code_path,
        registry=registry,
        aliases=_aliases_for_tool_spec(tool_spec),
    )
    if not getattr(lifecycle, "success", False):
        err = getattr(lifecycle, "error", None) or "registration failed"
        trace.append(_step("fail", f"Registration failed: {err}"))
        return None, err

    return lifecycle.capability, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def handle_task(task: str, *, registry: Any | None = None) -> dict[str, Any]:
    """
    Orchestrate the full offline calculation pipeline.

    Returns ``{ "answer": str, "trace": [ {type, label, detail?} ] }`` for
    ``POST /run`` (frontend-compatible).
    """
    trace: list[dict[str, str]] = []
    _ensure_toolbox()
    active_registry = registry if registry is not None else get_registry()

    logger.info("=" * 60)
    logger.info("Incoming task: %r", task)

    # --- Direct arithmetic (preserve existing behavior) ---
    if _is_trivial(task):
        expr_result = _answer_trivial(task)
        logger.info("Path: TRIVIAL arithmetic → %r", expr_result)
        trace.append(_step("plan", "This is trivial — I can answer directly"))
        return _finish(expr_result, trace)

    # --- Interpret ---
    trace.append(_step("plan", "Interpreting the request"))
    try:
        interpreted = interpret_task(task)
    except Exception as exc:  # noqa: BLE001 — surface cleanly to the client
        logger.info("Interpretation raised: %s", exc)
        trace.append(_step("fail", f"Interpretation failed: {exc}"))
        return _finish(f"Could not interpret the request: {exc}", trace)

    status = interpreted.get("status")
    if status == "unsupported":
        trace.append(
            _step("plan", "This doesn't look like a well-defined calculation")
        )
        # Match prior decline UX (answer step without "Answer:" prefix historically)
        trace.append(_step("answer", DECLINE_MESSAGE))
        return {"answer": DECLINE_MESSAGE, "trace": trace}

    if status == "needs_input":
        missing = interpreted.get("missing_inputs") or []
        missing_text = ", ".join(missing) if missing else "required inputs"
        answer = f"I need more information: {missing_text}"
        trace.append(_step("plan", f"Missing inputs: {missing_text}"))
        return _finish(answer, trace)

    if status == "error" or status != "ok":
        err = interpreted.get("error") or "interpretation failed"
        trace.append(_step("fail", f"Interpretation failed: {err}"))
        return _finish(f"Could not interpret the request: {err}", trace)

    operation = interpreted.get("operation")
    trace[-1] = _step(
        "plan",
        f"Planning: this needs a {operation or 'custom'} calculation",
    )

    # --- Normalize ---
    try:
        calc = CalculationRequest(
            status="ok",
            operation=operation,
            inputs={
                str(key): float(value)
                for key, value in (interpreted.get("inputs") or {}).items()
            },
            missing_inputs=[],
        )
        normalized = normalize_request(calc)
    except (TypeError, ValueError) as exc:
        trace.append(_step("fail", f"Normalization failed: {exc}"))
        return _finish(f"Could not normalize the request: {exc}", trace)

    # --- Resolve ---
    trace.append(_step("check", "Checking capability registry for a matching tool"))
    try:
        resolution = resolve_capability(normalized, active_registry)
    except (TypeError, ValueError) as exc:
        trace.append(_step("fail", f"Capability resolution failed: {exc}"))
        return _finish(f"Could not resolve a capability: {exc}", trace)

    capability = None
    tool_spec: ToolSpec | None = None

    if (
        resolution.status == "found"
        and resolution.capability is not None
        and getattr(resolution.capability, "verification_status", None) == "verified"
    ):
        capability = resolution.capability
        tool_spec = _tool_spec_from_capability(capability)
        trace[-1] = _step(
            "check",
            f"Found verified capability '{capability.tool_id}' v{capability.version} — reusing it",
            detail=f"{capability.operation} @ {capability.code_path}",
        )
    elif resolution.status == "ambiguous":
        trace.append(
            _step("fail", "Ambiguous capability match — cannot choose a tool")
        )
        return _finish(
            "Multiple capabilities match this request; please be more specific.",
            trace,
        )
    else:
        # Missing / unverified → generate
        tool_spec = _build_tool_spec(task, operation=normalized.operation)
        capability, gen_error = _generate_verify_register(
            task=task,
            tool_spec=tool_spec,
            registry=active_registry,
            trace=trace,
        )
        if capability is None:
            return _finish(
                f"Could not build a verified tool: {gen_error}",
                trace,
            )
        # Prefer ToolSpec used for generation (has examples/constraints).
        # Execution still uses registered code_path + tool_spec.name.

    assert tool_spec is not None
    assert capability is not None

    code = _load_code(capability.code_path)
    if code is None:
        trace.append(
            _step("fail", f"Missing tool file at {capability.code_path}")
        )
        return _finish(
            f"Registered capability is missing its code file: {capability.code_path}",
            trace,
        )

    function_name = tool_spec.name
    answer, exec_steps = _execute_and_validate(
        code=code,
        function_name=function_name,
        tool_spec=tool_spec,
        inputs=dict(normalized.inputs),
        normalized_request=normalized,
    )
    trace.extend(exec_steps)
    if answer is None:
        return _finish("Tool execution or result validation failed.", trace)

    return _finish(answer, trace)
