"""Pydantic contracts for interpreted calculation requests and run responses."""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, Field, StrictStr, validator


class CalculationRequest(BaseModel):
    """Validated calculation intent emitted by the natural-language interpreter."""

    status: Literal["ok", "needs_input", "unsupported", "error"]
    operation: str | None = None
    inputs: dict[str, float] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    error: str | None = None

    class Config:
        """Reject unknown fields so untrusted interpreter output is explicit."""

        extra = "forbid"

    @validator("operation", pre=True)
    def validate_operation(cls, value: Any) -> str | None:
        """Require a nonblank operation name when one is supplied."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("operation must be a string or None")
        value = value.strip()
        if not value:
            raise ValueError("operation must not be blank")
        return value

    @validator("inputs", pre=True)
    def validate_inputs(cls, value: Any) -> dict[str, float | int]:
        """Accept only finite integer or float values, never coerced strings/bools."""
        if not isinstance(value, dict):
            raise TypeError("inputs must be a dictionary")

        validated: dict[str, float | int] = {}
        for name, numeric_value in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("input names must be nonblank strings")
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(
                    f"input '{name}' must be an integer or float, not "
                    f"{type(numeric_value).__name__}"
                )
            try:
                numeric_as_float = float(numeric_value)
            except OverflowError as exc:
                raise ValueError(f"input '{name}' must fit in a float") from exc
            if not isfinite(numeric_as_float):
                raise ValueError(f"input '{name}' must be finite")
            validated[name.strip()] = numeric_value
        return validated

    @validator("missing_inputs", pre=True)
    def validate_missing_inputs(cls, value: Any) -> list[str]:
        """Require a list of nonblank missing-input names."""
        if not isinstance(value, list):
            raise TypeError("missing_inputs must be a list")

        validated: list[str] = []
        for name in value:
            if not isinstance(name, str):
                raise TypeError("missing input names must be strings")
            name = name.strip()
            if not name:
                raise ValueError("missing input names must not be blank")
            validated.append(name)
        return validated

    @validator("error", pre=True)
    def validate_error(cls, value: Any) -> str | None:
        """Keep optional interpreter error detail as text without coercion."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("error must be a string or None")
        return value


class TraceStep(BaseModel):
    """One human-readable step in the backend reasoning trace."""

    type: StrictStr
    label: StrictStr
    detail: StrictStr | None = None

    class Config:
        """Reject undeclared trace fields."""

        extra = "forbid"


class RunResponse(BaseModel):
    """Structured form of the current backend ``/run`` response."""

    answer: StrictStr
    trace: list[TraceStep]

    class Config:
        """Reject undeclared response fields."""

        extra = "forbid"
