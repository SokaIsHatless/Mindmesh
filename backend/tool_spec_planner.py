"""
tool_spec_planner.py — Deterministic ToolSpec planning from interpreted intent.

Builds ToolSpec contracts for known inverse/forward relationships using the
supplied variables and values. Does not call an LLM and does not invent tool
names from user wording.

Used upstream of Tool Factory so generation receives real inputs/outputs and
mathematically correct examples.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models import InputSpec, OutputSpec, TestCase, ToolSpec  # noqa: E402

# Canonical quantity → (type, unit, description)
_QUANTITY_META: dict[str, tuple[str, str | None, str | None]] = {
    "bmi": ("float", "kg/m^2", "body mass index"),
    "height_cm": ("float", "cm", "height"),
    "weight_kg": ("float", "kg", "weight"),
    "distance_km": ("float", "km", "distance"),
    "time_hr": ("float", "hr", "time"),
    "speed_kmh": ("float", "km/h", "speed"),
}

SolveFn = Callable[[dict[str, float]], float]


def _bmi_from_height_weight(inputs: dict[str, float]) -> float:
    height_m = float(inputs["height_cm"]) / 100.0
    if height_m == 0:
        raise ValueError("height_cm must not be zero")
    return float(inputs["weight_kg"]) / (height_m * height_m)


def _height_cm_from_bmi_weight(inputs: dict[str, float]) -> float:
    bmi = float(inputs["bmi"])
    if bmi <= 0:
        raise ValueError("bmi must be positive")
    height_m = math.sqrt(float(inputs["weight_kg"]) / bmi)
    return height_m * 100.0


def _weight_kg_from_bmi_height(inputs: dict[str, float]) -> float:
    height_m = float(inputs["height_cm"]) / 100.0
    return float(inputs["bmi"]) * height_m * height_m


def _speed_kmh(inputs: dict[str, float]) -> float:
    time_hr = float(inputs["time_hr"])
    if time_hr == 0:
        raise ValueError("time_hr must not be zero")
    return float(inputs["distance_km"]) / time_hr


def _distance_km(inputs: dict[str, float]) -> float:
    return float(inputs["speed_kmh"]) * float(inputs["time_hr"])


def _time_hr(inputs: dict[str, float]) -> float:
    speed = float(inputs["speed_kmh"])
    if speed == 0:
        raise ValueError("speed_kmh must not be zero")
    return float(inputs["distance_km"]) / speed


def _clean_number(value: float) -> float | int:
    """Prefer ints when the value is integral; otherwise a stable float."""
    if not math.isfinite(value):
        raise ValueError("non-finite result")
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 6)


# operation → planning record
_OPERATION_PLANS: dict[str, dict[str, Any]] = {
    "bmi": {
        "name": "bmi",
        "operation": "bmi",
        "purpose": "compute BMI from height and weight",
        "inputs": ("height_cm", "weight_kg"),
        "output": "bmi",
        "constraints": [
            "height_cm must be positive",
            "weight_kg must be positive",
        ],
        "solve": _bmi_from_height_weight,
        # Fallback example when caller supplies no usable values.
        "default_inputs": {"height_cm": 150.0, "weight_kg": 50.0},
    },
    "height_from_bmi": {
        "name": "height_from_bmi",
        "operation": "height_from_bmi",
        "purpose": "compute height from BMI and weight",
        "inputs": ("bmi", "weight_kg"),
        "output": "height_cm",
        "constraints": [
            "bmi must be positive",
            "weight_kg must be positive",
        ],
        "solve": _height_cm_from_bmi_weight,
        "default_inputs": {"bmi": 20.0, "weight_kg": 80.0},
    },
    "weight_from_bmi": {
        "name": "weight_from_bmi",
        "operation": "weight_from_bmi",
        "purpose": "compute weight from BMI and height",
        "inputs": ("bmi", "height_cm"),
        "output": "weight_kg",
        "constraints": [
            "bmi must be positive",
            "height_cm must be positive",
        ],
        "solve": _weight_kg_from_bmi_height,
        "default_inputs": {"bmi": 22.0, "height_cm": 170.0},
    },
    "speed": {
        "name": "speed",
        "operation": "speed",
        "purpose": "compute speed from distance and time",
        "inputs": ("distance_km", "time_hr"),
        "output": "speed_kmh",
        "constraints": ["time_hr must not be zero"],
        "solve": _speed_kmh,
        "default_inputs": {"distance_km": 120.0, "time_hr": 2.0},
    },
    "distance": {
        "name": "distance",
        "operation": "distance",
        "purpose": "compute distance from speed and time",
        "inputs": ("speed_kmh", "time_hr"),
        "output": "distance_km",
        "constraints": [],
        "solve": _distance_km,
        "default_inputs": {"speed_kmh": 60.0, "time_hr": 2.0},
    },
    "time": {
        "name": "time",
        "operation": "time",
        "purpose": "compute time from distance and speed",
        "inputs": ("distance_km", "speed_kmh"),
        "output": "time_hr",
        "constraints": ["speed_kmh must not be zero"],
        "solve": _time_hr,
        "default_inputs": {"distance_km": 120.0, "speed_kmh": 60.0},
    },
}

# provided frozenset → operation (same families as the interpreter)
_PROVIDED_TO_OPERATION: dict[frozenset[str], str] = {
    frozenset({"height_cm", "weight_kg"}): "bmi",
    frozenset({"bmi", "weight_kg"}): "height_from_bmi",
    frozenset({"bmi", "height_cm"}): "weight_from_bmi",
    frozenset({"distance_km", "time_hr"}): "speed",
    frozenset({"speed_kmh", "time_hr"}): "distance",
    frozenset({"distance_km", "speed_kmh"}): "time",
}


def _input_spec(name: str) -> InputSpec:
    type_name, unit, description = _QUANTITY_META.get(
        name, ("number", None, None)
    )
    return InputSpec(
        name=name,
        type=type_name,  # type: ignore[arg-type]
        unit=unit,
        description=description,
    )


def _output_spec(name: str) -> OutputSpec:
    type_name, unit, _description = _QUANTITY_META.get(
        name, ("number", None, None)
    )
    return OutputSpec(
        name=name,
        type=type_name,  # type: ignore[arg-type]
        unit=unit,
    )


def _resolve_operation(
    operation: str | None,
    inputs: dict[str, float],
    requested_output: str | None,
) -> str | None:
    """Choose a stable operation name from intent — never from user prose."""
    if operation:
        key = operation.strip().lower()
        if key in _OPERATION_PLANS:
            return key
        if key in {"velocity"}:
            return "speed"
        if key in {"body_mass_index", "calculate_bmi"}:
            return "bmi"

    provided = frozenset(inputs)

    if requested_output:
        for op_name, plan in _OPERATION_PLANS.items():
            if plan["output"] != requested_output:
                continue
            needed = set(plan["inputs"])
            if needed <= provided:
                return op_name

    for pair, op_name in _PROVIDED_TO_OPERATION.items():
        if pair <= provided:
            return op_name

    return None


def _example_inputs_for_plan(
    plan: dict[str, Any], supplied: dict[str, float]
) -> dict[str, float]:
    """Use real supplied values when complete; otherwise plan defaults."""
    required = plan["inputs"]
    if all(name in supplied for name in required):
        return {name: float(supplied[name]) for name in required}
    defaults: dict[str, float] = dict(plan["default_inputs"])
    # Overlay any partial supplied values that match required names.
    for name in required:
        if name in supplied:
            defaults[name] = float(supplied[name])
    return defaults


def plan_tool_spec(
    *,
    task: str,
    operation: str | None,
    inputs: dict[str, float] | None = None,
    requested_output: str | None = None,
) -> ToolSpec | None:
    """
    Build a ToolSpec for a known forward/inverse relationship.

    Returns None when the intent is outside the planning catalog (caller may
    fall back to legacy builders). Never embeds user-specific literals into
    the tool name.
    """
    supplied = {
        str(name): float(value)
        for name, value in (inputs or {}).items()
        if isinstance(name, str) and name.strip()
    }

    op_name = _resolve_operation(operation, supplied, requested_output)
    if op_name is None:
        return None

    plan = _OPERATION_PLANS[op_name]
    example_inputs = _example_inputs_for_plan(plan, supplied)
    try:
        expected = _clean_number(plan["solve"](example_inputs))
    except (ValueError, ZeroDivisionError, KeyError):
        return None

    purpose = (task or "").strip() or str(plan["purpose"])
    # Keep purpose descriptive but never let it become the tool identity.
    if len(purpose) > 200:
        purpose = str(plan["purpose"])

    return ToolSpec(
        name=str(plan["name"]),
        purpose=purpose,
        operation=str(plan["operation"]),
        inputs=[_input_spec(name) for name in plan["inputs"]],
        output=_output_spec(str(plan["output"])),
        constraints=list(plan["constraints"]),
        examples=[
            TestCase(inputs=dict(example_inputs), expected=expected),
        ],
    )
