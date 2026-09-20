"""
normalizer.py — CalculationRequest → NormalizedRequest (deterministic).

No LLM. The interpreter owns language understanding; this module only
standardizes operations, input names, and units for later capability resolution.
"""

from __future__ import annotations

import sys
from math import isfinite
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:  # Prefer package path so CalculationRequest matches ``backend.orchestrator``.
    from .models import CalculationRequest
except ImportError:
    from models import CalculationRequest

# --- Operation aliases (source → canonical) ---
OPERATION_ALIASES: dict[str, str] = {
    "velocity": "speed",
    "prob": "probability",
}

# --- Bare input-name aliases (assume value already in canonical units) ---
INPUT_ALIASES: dict[str, str] = {
    "height": "height_cm",
    "weight": "weight_kg",
    "distance": "distance_km",
    "time": "time_hr",
}

# Quantity family → (canonical key, factors mapping unit token → multiply-to-canonical)
# Canonical units: height=cm, weight=kg, distance=km, time=hours, speed=km/h
_HEIGHT_TO_CM: dict[str, float] = {
    "m": 100.0,
    "metre": 100.0,
    "meter": 100.0,
    "metres": 100.0,
    "meters": 100.0,
    "cm": 1.0,
    "centimetre": 1.0,
    "centimeter": 1.0,
    "centimetres": 1.0,
    "centimeters": 1.0,
    "km": 100_000.0,
    "kilometre": 100_000.0,
    "kilometer": 100_000.0,
    "kilometres": 100_000.0,
    "kilometers": 100_000.0,
}

_WEIGHT_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
}

_DISTANCE_TO_KM: dict[str, float] = {
    "m": 0.001,
    "metre": 0.001,
    "meter": 0.001,
    "metres": 0.001,
    "meters": 0.001,
    "cm": 0.00001,
    "centimetre": 0.00001,
    "centimeter": 0.00001,
    "centimetres": 0.00001,
    "centimeters": 0.00001,
    "km": 1.0,
    "kilometre": 1.0,
    "kilometer": 1.0,
    "kilometres": 1.0,
    "kilometers": 1.0,
}

_TIME_TO_HR: dict[str, float] = {
    "h": 1.0,
    "hr": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "min": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
}

_SPEED_TO_KMH: dict[str, float] = {
    "kmh": 1.0,
    "kph": 1.0,
    "kmph": 1.0,
}

# base quantity name → (canonical input key, unit→factor)
_QUANTITIES: dict[str, tuple[str, dict[str, float]]] = {
    "height": ("height_cm", _HEIGHT_TO_CM),
    "weight": ("weight_kg", _WEIGHT_TO_KG),
    "distance": ("distance_km", _DISTANCE_TO_KM),
    "time": ("time_hr", _TIME_TO_HR),
    "speed": ("speed_kmh", _SPEED_TO_KMH),
}

# Canonical keys that already encode the target unit
_CANONICAL_KEYS: frozenset[str] = frozenset(
    canonical for canonical, _ in _QUANTITIES.values()
)

# Longest-first unit tokens so "centimetre" wins over "m", "minutes" over "min", etc.
_ALL_UNIT_TOKENS: tuple[str, ...] = tuple(
    sorted(
        {
            *_HEIGHT_TO_CM,
            *_WEIGHT_TO_KG,
            *_DISTANCE_TO_KM,
            *_TIME_TO_HR,
            *_SPEED_TO_KMH,
        },
        key=len,
        reverse=True,
    )
)


class NormalizedRequest(BaseModel):
    """Canonical calculation request after deterministic name/unit normalization."""

    operation: str | None = None
    inputs: dict[str, float] = Field(default_factory=dict)
    original_inputs: dict[str, float] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("operation", mode="before")
    @classmethod
    def validate_operation(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("operation must be a string or None")
        value = value.strip()
        if not value:
            raise ValueError("operation must not be blank")
        return value

    @field_validator("inputs", "original_inputs", mode="before")
    @classmethod
    def validate_float_maps(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("inputs maps must be dictionaries")
        validated: dict[str, float] = {}
        for name, numeric_value in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("input names must be nonblank strings")
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise ValueError(
                    f"input '{name}' must be an integer or float, not "
                    f"{type(numeric_value).__name__}"
                )
            as_float = float(numeric_value)
            if not isfinite(as_float):
                raise ValueError(f"input '{name}' must be finite")
            validated[name.strip()] = as_float
        return validated

    @field_validator("missing_inputs", mode="before")
    @classmethod
    def validate_missing_inputs(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("missing_inputs must be a list")
        validated: list[str] = []
        for name in value:
            if not isinstance(name, str):
                raise ValueError("missing input names must be strings")
            name = name.strip()
            if not name:
                raise ValueError("missing input names must not be blank")
            validated.append(name)
        return validated


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"input '{name}' must be numeric, not {type(value).__name__}"
        )
    as_float = float(value)
    if not isfinite(as_float):
        raise ValueError(f"input '{name}' must be finite")
    return as_float


def _split_quantity_unit(key: str) -> tuple[str, str | None]:
    """Split ``height_m`` → (``height``, ``m``); unknown keys keep unit None."""
    lowered = key.strip().lower()
    for unit in _ALL_UNIT_TOKENS:
        suffix = f"_{unit}"
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return lowered[: -len(suffix)], unit
    return lowered, None


def _resolve_input_key(raw_key: str) -> tuple[str, str | None, dict[str, float] | None]:
    """
    Map a raw input key to (canonical_name, unit_token_or_None, factor_table_or_None).

    When ``unit`` is None the value is already assumed to be in canonical units
    (or the key is not a known quantity and is passed through unchanged).
    """
    lowered = raw_key.strip().lower()
    base, unit = _split_quantity_unit(raw_key)

    # Canonical key written with its own unit suffix, e.g. height_cm / time_hr
    if base in _QUANTITIES:
        canonical, factors = _QUANTITIES[base]
        return canonical, unit, factors

    # height_furlong / time_fortnight: quantity prefix with an unknown unit token
    for quantity in _QUANTITIES:
        prefix = f"{quantity}_"
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            raise ValueError(
                f"impossible or unsupported unit on input '{raw_key}'"
            )

    # Already the full canonical name with no further unit (defensive)
    if base in _CANONICAL_KEYS and unit is None:
        return base, None, None

    # Bare alias without unit: height → height_cm
    if base in INPUT_ALIASES and unit is None:
        return INPUT_ALIASES[base], None, None

    if unit is not None:
        # Unknown quantity with a recognized unit suffix — ambiguous
        raise ValueError(
            f"ambiguous or unsupported input key '{raw_key}': "
            f"unknown quantity '{base}' with unit '{unit}'"
        )

    return base, None, None


def _convert_value(
    raw_key: str,
    value: float,
    unit: str | None,
    factors: dict[str, float] | None,
) -> float:
    """Apply unit conversion when a unit token is present; reject impossible cases."""
    numeric = _require_finite(raw_key, value)

    if unit is None:
        return numeric

    if factors is None:
        raise ValueError(f"cannot convert '{raw_key}': no unit table for quantity")

    if unit not in factors:
        raise ValueError(
            f"impossible or unsupported unit '{unit}' on input '{raw_key}'"
        )

    converted = numeric * factors[unit]
    if not isfinite(converted):
        raise ValueError(f"conversion of '{raw_key}' produced a non-finite value")
    return converted


def _normalize_operation(operation: str | None) -> str | None:
    if operation is None:
        return None
    key = operation.strip().lower()
    if not key:
        raise ValueError("operation must not be blank")
    return OPERATION_ALIASES.get(key, key)


def _normalize_missing_name(raw_name: str) -> str:
    """Alias missing-input labels to canonical names; never invent values."""
    canonical, unit, factors = _resolve_input_key(raw_name)
    if unit is not None and factors is not None and unit not in factors:
        raise ValueError(
            f"impossible or unsupported unit on missing input '{raw_name}'"
        )
    return canonical


def _looks_like_calculation_request(value: Any) -> bool:
    """Accept CalculationRequest across flat ``models`` / ``backend.models``."""
    if isinstance(value, CalculationRequest):
        return True
    return (
        type(value).__name__ == "CalculationRequest"
        and hasattr(value, "operation")
        and hasattr(value, "inputs")
        and hasattr(value, "missing_inputs")
        and hasattr(value, "status")
    )


class RequestNormalizer:
    """Deterministic CalculationRequest → NormalizedRequest transformer."""

    def normalize(self, request: CalculationRequest) -> NormalizedRequest:
        if not _looks_like_calculation_request(request):
            raise TypeError("request must be a CalculationRequest")

        # Rebuild onto this module's CalculationRequest when dual-imported.
        if not isinstance(request, CalculationRequest):
            request = CalculationRequest.model_validate(
                request.model_dump(mode="python")
            )

        original_inputs = {
            str(name): _require_finite(str(name), value)
            for name, value in request.inputs.items()
        }

        normalized_inputs: dict[str, float] = {}
        # Track which raw keys claimed each canonical name (duplicate detection)
        claimed_by: dict[str, str] = {}

        for raw_key, raw_value in request.inputs.items():
            canonical, unit, factors = _resolve_input_key(raw_key)
            converted = _convert_value(str(raw_key), raw_value, unit, factors)

            if canonical in claimed_by:
                raise ValueError(
                    f"duplicate canonical input '{canonical}' from "
                    f"'{claimed_by[canonical]}' and '{raw_key}'"
                )
            claimed_by[canonical] = str(raw_key)
            normalized_inputs[canonical] = converted

        missing: list[str] = []
        seen_missing: set[str] = set()
        for raw_missing in request.missing_inputs:
            canonical_missing = _normalize_missing_name(raw_missing)
            if canonical_missing in seen_missing:
                raise ValueError(
                    f"duplicate canonical missing input '{canonical_missing}'"
                )
            # A supplied value and a missing flag for the same canonical name
            # would silently disagree — reject rather than discard.
            if canonical_missing in normalized_inputs:
                raise ValueError(
                    f"input '{canonical_missing}' cannot be both present and missing"
                )
            seen_missing.add(canonical_missing)
            missing.append(canonical_missing)

        return NormalizedRequest(
            operation=_normalize_operation(request.operation),
            inputs=normalized_inputs,
            original_inputs=original_inputs,
            missing_inputs=missing,
        )


def normalize_request(request: CalculationRequest) -> NormalizedRequest:
    """Public API: standardize a validated CalculationRequest."""
    return RequestNormalizer().normalize(request)
