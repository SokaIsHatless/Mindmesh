"""Pydantic contracts for interpreted calculation requests and run responses."""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)


ToolValueType = Literal[
    "float",
    "int",
    "string",
    "boolean",
    "number",
    "object",
    "array",
]


def _nonblank_text(value: Any, field_name: str) -> str:
    """Return normalized text or reject values that are not meaningful labels."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    return value


class _ToolSpecModel(BaseModel):
    """Base configuration for the pure capability-description models."""

    model_config = ConfigDict(extra="forbid")


class InputSpec(_ToolSpecModel):
    """One named value accepted by a computational capability."""

    name: str
    type: ToolValueType
    description: str | None = None
    unit: str | None = None
    required: StrictBool = True
    default: Any = None

    @field_validator("name", "type", mode="before")
    @classmethod
    def validate_required_text(cls, value: Any, info: ValidationInfo):
        return _nonblank_text(value, info.field_name)

    @field_validator("description", "unit", mode="before")
    @classmethod
    def validate_optional_text(cls, value: Any, info: ValidationInfo):
        if value is None:
            return None
        return _nonblank_text(value, info.field_name)

    @field_validator("default")
    @classmethod
    def validate_default_for_required_input(
        cls, value: Any, info: ValidationInfo
    ):
        if info.data.get("required", True) and value is not None:
            raise ValueError("required inputs cannot declare a default")
        return value


class OutputSpec(_ToolSpecModel):
    """The named value type and optional unit of a capability result."""

    name: str
    type: ToolValueType
    unit: str | None = None

    @field_validator("name", "type", mode="before")
    @classmethod
    def validate_required_text(cls, value: Any, info: ValidationInfo):
        return _nonblank_text(value, info.field_name)

    @field_validator("unit", mode="before")
    @classmethod
    def validate_optional_text(cls, value: Any, info: ValidationInfo):
        if value is None:
            return None
        return _nonblank_text(value, info.field_name)


class TestCase(_ToolSpecModel):
    """An example input mapping and the expected result, without execution logic."""

    inputs: dict[str, Any]
    expected: Any = Field(...)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_input(cls, value: Any) -> Any:
        """Accept the orchestrator's old key while exposing only ``inputs``."""
        if not isinstance(value, dict) or "input" not in value or "inputs" in value:
            return value
        normalized = dict(value)
        normalized["inputs"] = normalized.pop("input")
        return normalized

    @field_validator("inputs", mode="before")
    @classmethod
    def validate_input_mapping(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("input must be a dictionary")

        normalized: dict[str, Any] = {}
        for name, input_value in value.items():
            name = _nonblank_text(name, "test input names")
            if name in normalized:
                raise ValueError(f"duplicate test input name: {name}")
            normalized[name] = input_value
        return normalized

    @property
    def input(self) -> dict[str, Any]:
        """Read-only bridge for the unchanged orchestrator's legacy access."""
        return self.inputs


class ToolSpec(_ToolSpecModel):
    """Declarative contract for what a computational capability does."""

    name: str
    purpose: str
    operation: str | None = None
    inputs: list[InputSpec]
    output: OutputSpec
    constraints: list[str] = Field(default_factory=list)
    examples: list[TestCase] = Field(default_factory=list)
    allowed_dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_output(cls, value: Any) -> Any:
        """Bridge the unchanged orchestrator's one-item ``outputs`` list."""
        if not isinstance(value, dict) or "outputs" not in value or "output" in value:
            return value

        legacy_outputs = value["outputs"]
        if not isinstance(legacy_outputs, list) or len(legacy_outputs) != 1:
            raise ValueError("legacy outputs must contain exactly one output")
        normalized = dict(value)
        normalized["output"] = normalized.pop("outputs")[0]
        return normalized

    @field_validator("name", "purpose", mode="before")
    @classmethod
    def validate_required_text(cls, value: Any, info: ValidationInfo):
        return _nonblank_text(value, info.field_name)

    @field_validator("operation", mode="before")
    @classmethod
    def validate_optional_operation(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _nonblank_text(value, "operation")

    @field_validator("constraints", "allowed_dependencies", mode="before")
    @classmethod
    def validate_text_lists(cls, value: Any, info: ValidationInfo) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{info.field_name} must be a list")
        normalized: list[str] = []
        for item in value:
            item = _nonblank_text(item, info.field_name)
            if item in normalized:
                raise ValueError(f"duplicate {info.field_name} value: {item}")
            normalized.append(item)
        return normalized

    @field_validator("inputs")
    @classmethod
    def validate_unique_input_names(cls, value: list[InputSpec]) -> list[InputSpec]:
        names = [input_spec.name for input_spec in value]
        if len(names) != len(set(names)):
            raise ValueError("duplicate input names are not allowed")
        return value

    @model_validator(mode="after")
    def validate_example_inputs(self) -> "ToolSpec":
        if self.operation is None:
            self.operation = self.name

        inputs = self.inputs
        input_names = {input_spec.name for input_spec in inputs}
        required_names = {
            input_spec.name for input_spec in inputs if input_spec.required
        }

        for index, example in enumerate(self.examples):
            example_names = set(example.inputs)
            unknown_names = example_names - input_names
            missing_names = required_names - example_names
            if unknown_names:
                raise ValueError(
                    f"example {index} contains undeclared inputs: "
                    f"{', '.join(sorted(unknown_names))}"
                )
            if missing_names:
                raise ValueError(
                    f"example {index} is missing required inputs: "
                    f"{', '.join(sorted(missing_names))}"
                )
        return self


class CalculationRequest(BaseModel):
    """Validated calculation intent emitted by the natural-language interpreter."""

    status: Literal["ok", "needs_input", "unsupported", "error"]
    operation: str | None = None
    inputs: dict[str, float] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    error: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("operation", mode="before")
    @classmethod
    def validate_operation(cls, value: Any) -> str | None:
        """Require a nonblank operation name when one is supplied."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("operation must be a string or None")
        value = value.strip()
        if not value:
            raise ValueError("operation must not be blank")
        return value

    @field_validator("inputs", mode="before")
    @classmethod
    def validate_inputs(cls, value: Any) -> dict[str, float | int]:
        """Accept only finite integer or float values, never coerced strings/bools."""
        if not isinstance(value, dict):
            raise ValueError("inputs must be a dictionary")

        validated: dict[str, float | int] = {}
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
            try:
                numeric_as_float = float(numeric_value)
            except OverflowError as exc:
                raise ValueError(f"input '{name}' must fit in a float") from exc
            if not isfinite(numeric_as_float):
                raise ValueError(f"input '{name}' must be finite")
            validated[name.strip()] = numeric_value
        return validated

    @field_validator("missing_inputs", mode="before")
    @classmethod
    def validate_missing_inputs(cls, value: Any) -> list[str]:
        """Require a list of nonblank missing-input names."""
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

    @field_validator("error", mode="before")
    @classmethod
    def validate_error(cls, value: Any) -> str | None:
        """Keep optional interpreter error detail as text without coercion."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("error must be a string or None")
        return value


class TraceStep(BaseModel):
    """One human-readable step in the backend reasoning trace."""

    type: StrictStr
    label: StrictStr
    detail: StrictStr | None = None

    model_config = ConfigDict(extra="forbid")


class RunResponse(BaseModel):
    """Structured form of the current backend ``/run`` response."""

    answer: StrictStr
    trace: list[TraceStep]

    model_config = ConfigDict(extra="forbid")
