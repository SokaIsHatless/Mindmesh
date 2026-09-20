"""Pydantic models for declarative capability metadata.

These models describe tool contracts and locations only.  They never import,
read, generate, or execute the Python files referenced by ``code_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

try:  # Supports ``backend.capabilities`` package and backend test discovery.
    from ..models import InputSpec, OutputSpec, ToolSpec
except ImportError:  # pragma: no cover - exercised by discovery import layout
    from models import InputSpec, OutputSpec, ToolSpec


VerificationStatus = Literal["pending", "verified", "failed"]


def normalize_identifier(value: Any, field_name: str) -> str:
    """Return a canonical registry identifier or reject unusable text."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _nonblank_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


class Capability(BaseModel):
    """Metadata for one versioned computational capability."""

    tool_id: str
    operation: str
    version: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    input_schema: list[InputSpec]
    output_schema: OutputSpec
    code_path: str
    verification_status: VerificationStatus = "pending"
    enabled: StrictBool = True

    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other: Any) -> bool:
        """Value equality when nested ToolSpec models come from dual imports.

        Alias lists are compared as sets because the repository reads aliases
        with ``ORDER BY alias`` while in-memory registration preserves
        insertion order.
        """
        if other is self:
            return True
        if not isinstance(other, BaseModel):
            return NotImplemented
        if type(other).__name__ != "Capability":
            return NotImplemented
        left = self.model_dump(mode="python")
        right = other.model_dump(mode="python")
        left_aliases = left.pop("aliases", [])
        right_aliases = right.pop("aliases", [])
        return left == right and set(left_aliases) == set(right_aliases)

    def __hash__(self) -> int:
        return id(self)

    @field_validator("tool_id", "operation", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any, info: Any) -> str:
        return normalize_identifier(value, info.field_name)

    @field_validator("version", "description", mode="before")
    @classmethod
    def validate_text(cls, value: Any, info: Any) -> str:
        return _nonblank_text(value, info.field_name)

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("aliases must be a list")
        aliases: list[str] = []
        for alias in value:
            normalized = normalize_identifier(alias, "aliases")
            if normalized in aliases:
                raise ValueError(f"duplicate alias: {normalized}")
            aliases.append(normalized)
        return aliases

    @field_validator("code_path", mode="before")
    @classmethod
    def validate_code_path(cls, value: Any) -> str:
        path_text = _nonblank_text(value, "code_path")
        path = Path(path_text)
        if path.suffix.lower() != ".py":
            raise ValueError("code_path must point to a .py file")
        if not path.is_file():
            raise ValueError("code_path must point to an existing Python file")
        return str(path)

    @classmethod
    def from_tool_spec(
        cls,
        tool_spec: ToolSpec,
        *,
        version: str,
        code_path: str,
        tool_id: str | None = None,
        aliases: list[str] | None = None,
        verification_status: VerificationStatus = "pending",
        enabled: bool = True,
    ) -> "Capability":
        """Build registry metadata from the existing declarative tool contract."""
        if not isinstance(tool_spec, ToolSpec):
            # Survive flat ``models`` vs ``backend.models`` dual imports.
            if type(tool_spec).__name__ != "ToolSpec" or not hasattr(
                tool_spec, "model_dump"
            ):
                raise TypeError("tool_spec must be a ToolSpec")
            tool_spec = ToolSpec.model_validate(tool_spec.model_dump(mode="python"))
        return cls(
            tool_id=tool_id if tool_id is not None else tool_spec.name,
            operation=tool_spec.operation if tool_spec.operation is not None else tool_spec.name,
            version=version,
            description=tool_spec.purpose,
            aliases=aliases or [],
            input_schema=tool_spec.inputs,
            output_schema=tool_spec.output,
            code_path=code_path,
            verification_status=verification_status,
            enabled=enabled,
        )
