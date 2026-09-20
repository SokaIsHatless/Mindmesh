"""
result_validator.py — Validate an executed tool result against a ToolSpec.

Deterministic only: no Ollama, no tool execution, no registry writes.
Compares the actual result to the ToolSpec output contract (and optional
normalized-request / expected context).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models import ToolSpec  # noqa: E402

FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-9

ValidationStatus = Literal["valid", "invalid", "error"]

# Generic unit synonyms → one canonical spelling (not operation-specific).
_UNIT_SYNONYMS: dict[str, str] = {
    "kmh": "km/h",
    "kph": "km/h",
    "kmph": "km/h",
    "km/hr": "km/h",
    "kmperh": "km/h",
    "kmperhour": "km/h",
    "m/sec": "m/s",
    "mps": "m/s",
    "metre/s": "m/s",
    "meter/s": "m/s",
    "kg/m²": "kg/m^2",
    "kg/m2": "kg/m^2",
    "kgm^-2": "kg/m^2",
    "kgm-2": "kg/m^2",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "metre": "m",
    "meter": "m",
    "metres": "m",
    "meters": "m",
    "kilometre": "km",
    "kilometer": "km",
    "kilometres": "km",
    "kilometers": "km",
    "kilogram": "kg",
    "kilograms": "kg",
}


class ResultValidationResult(BaseModel):
    """Structured outcome of validating one executed tool result."""

    status: ValidationStatus
    value: Any | None = None
    normalized_value: Any | None = None
    unit: str | None = None
    normalized_unit: str | None = None
    errors: list[str] = Field(default_factory=list)
    details: str | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


def _looks_like_tool_spec(value: Any) -> bool:
    if isinstance(value, ToolSpec):
        return True
    return (
        type(value).__name__ == "ToolSpec"
        and hasattr(value, "output")
        and hasattr(value, "name")
    )


def _canonicalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    if not isinstance(unit, str):
        raise TypeError("unit must be a string or None")
    cleaned = (
        unit.strip()
        .lower()
        .replace(" ", "")
        .replace("·", "/")
        .replace("∕", "/")
    )
    if not cleaned:
        return None
    return _UNIT_SYNONYMS.get(cleaned, cleaned)


def _units_compatible(expected_unit: str | None, actual_unit: str | None) -> bool:
    """True when units are absent-on-one-side, equal, or synonym-equivalent."""
    left = _canonicalize_unit(expected_unit)
    right = _canonicalize_unit(actual_unit)
    if left is None or right is None:
        return True
    return left == right


def _is_execution_error(result: Any) -> str | None:
    """Return an error message if ``result`` represents a failed execution."""
    if isinstance(result, BaseException):
        return f"execution error: {result}"
    if isinstance(result, dict):
        if result.get("ok") is False:
            detail = result.get("error") or result.get("reason") or "execution failed"
            return f"execution error: {detail}"
        if result.get("status") in {"error", "failed"}:
            detail = result.get("error") or result.get("detail") or result["status"]
            return f"execution error: {detail}"
        if result.get("error") and result.get("value") is None and "ok" not in result:
            return f"execution error: {result['error']}"
    if hasattr(result, "ok") and getattr(result, "ok") is False:
        detail = getattr(result, "error", None) or getattr(result, "reason", None)
        return f"execution error: {detail or 'execution failed'}"
    return None


def _extract_payload(
    result: Any, output_name: str
) -> tuple[Any | None, str | None, list[str]]:
    """Pull (value, unit, errors) out of a bare or structured execution payload."""
    errors: list[str] = []

    if result is None:
        errors.append("missing output: result is None")
        return None, None, errors

    if isinstance(result, dict):
        unit = result.get("unit")
        if not isinstance(unit, str):
            unit = None

        if "value" in result:
            return result.get("value"), unit, errors

        if output_name in result:
            nested = result[output_name]
            if isinstance(nested, dict) and "value" in nested:
                nested_unit = nested.get("unit")
                if not isinstance(nested_unit, str):
                    nested_unit = unit
                return nested.get("value"), nested_unit, errors
            return nested, unit, errors

        # Only metadata keys — treat as missing the required output value.
        meta_keys = {"ok", "status", "error", "reason", "detail", "unit", "stderr"}
        if set(result) <= meta_keys:
            errors.append(f"missing output: expected value for '{output_name}'")
            return None, unit, errors

    if hasattr(result, "value"):
        unit = getattr(result, "unit", None)
        if not isinstance(unit, str):
            unit = None
        return getattr(result, "value"), unit, errors

    return result, None, errors


def _type_matches(value: Any, type_name: str) -> bool:
    if type_name in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return False


def _normalize_numeric(value: Any, type_name: str) -> Any:
    if type_name == "int" and isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if type_name in {"float", "number"} and isinstance(value, (int, float)):
        return float(value)
    return value


def values_close(expected: Any, actual: Any) -> bool:
    """Float-tolerant comparison for optional expected-value checks."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual if isinstance(expected, bool) else expected == actual
    try:
        expected_f = float(expected)
        actual_f = float(actual)
        if math.isfinite(expected_f) and math.isfinite(actual_f):
            return math.isclose(
                expected_f,
                actual_f,
                rel_tol=FLOAT_REL_TOL,
                abs_tol=FLOAT_ABS_TOL,
            )
    except (TypeError, ValueError):
        pass
    return expected == actual


def validate_result(
    tool_spec: ToolSpec,
    result: Any,
    *,
    expected: Any | None = None,
    unit: str | None = None,
    normalized_request: Any | None = None,
) -> ResultValidationResult:
    """Validate an executed result against ``tool_spec.output``.

    ``normalized_request`` is accepted for pipeline context; validation itself
    is driven by the ToolSpec output contract. ``unit`` overrides a unit
    embedded in a structured result when provided.
    """
    if not _looks_like_tool_spec(tool_spec):
        return ResultValidationResult(
            status="error",
            errors=["tool_spec must be a ToolSpec"],
            details="tool_spec must be a ToolSpec",
        )

    output = tool_spec.output
    output_name = output.name
    output_type = output.type
    expected_unit = output.unit

    exec_error = _is_execution_error(result)
    if exec_error is not None:
        return ResultValidationResult(
            status="error",
            errors=[exec_error],
            details=exec_error,
        )

    value, embedded_unit, extract_errors = _extract_payload(result, output_name)
    actual_unit = unit if unit is not None else embedded_unit

    if extract_errors:
        return ResultValidationResult(
            status="invalid",
            value=value,
            unit=actual_unit,
            normalized_unit=_canonicalize_unit(actual_unit),
            errors=extract_errors,
            details=extract_errors[0],
        )

    if value is None:
        message = f"missing output: required result '{output_name}' is absent"
        return ResultValidationResult(
            status="invalid",
            unit=actual_unit,
            normalized_unit=_canonicalize_unit(actual_unit),
            errors=[message],
            details=message,
        )

    errors: list[str] = []

    if not _type_matches(value, output_type):
        errors.append(
            f"wrong output type: expected {output_type}, "
            f"got {type(value).__name__}"
        )

    normalized_value: Any = value
    if not errors and output_type in {"float", "number", "int"}:
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            as_float = None
        if as_float is None or not math.isfinite(as_float):
            errors.append("non-finite numeric result")
        else:
            normalized_value = _normalize_numeric(value, output_type)

    if not _units_compatible(expected_unit, actual_unit):
        errors.append(
            f"unit mismatch: expected {expected_unit!r}, got {actual_unit!r}"
        )

    # Optional expected-value check (float-tolerant).
    if expected is not None and not errors:
        if not values_close(expected, normalized_value):
            errors.append(
                f"value mismatch: expected {expected!r}, got {normalized_value!r}"
            )

    # Optional normalized-request sanity: operation label when both present.
    if normalized_request is not None and not errors:
        req_op = getattr(normalized_request, "operation", None)
        spec_op = getattr(tool_spec, "operation", None) or tool_spec.name
        if isinstance(req_op, str) and req_op.strip():
            if req_op.strip().lower() != str(spec_op).strip().lower():
                errors.append(
                    f"operation mismatch: request {req_op!r} vs tool {spec_op!r}"
                )

    normalized_unit = _canonicalize_unit(actual_unit) or _canonicalize_unit(
        expected_unit
    )

    if errors:
        return ResultValidationResult(
            status="invalid",
            value=value,
            normalized_value=normalized_value,
            unit=actual_unit,
            normalized_unit=normalized_unit,
            errors=errors,
            details=errors[0],
        )

    return ResultValidationResult(
        status="valid",
        value=value,
        normalized_value=normalized_value,
        unit=actual_unit if actual_unit is not None else expected_unit,
        normalized_unit=normalized_unit,
        errors=[],
        details="ok",
    )
