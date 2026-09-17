"""Public, non-executing API for the capability metadata registry."""

from __future__ import annotations

from pathlib import Path

try:  # Supports ``backend.capabilities`` package and backend test discovery.
    from .models import Capability, normalize_identifier
    from .repository import CapabilityRepository
except ImportError:  # pragma: no cover - exercised by discovery import layout
    from capabilities.models import Capability, normalize_identifier
    from capabilities.repository import CapabilityRepository


class CapabilityRegistry:
    """Answer exact metadata queries without resolving or executing tools."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self._repository = CapabilityRepository(database_path)

    def register(self, capability: Capability) -> Capability:
        if not isinstance(capability, Capability):
            raise TypeError("capability must be a Capability")
        return self._repository.register(capability)

    def get_by_operation(self, operation: str) -> list[Capability]:
        return self._repository.get_by_operation(
            normalize_identifier(operation, "operation")
        )

    def get_by_alias(self, alias: str) -> Capability | None:
        return self._repository.get_by_alias(normalize_identifier(alias, "alias"))

    def get_version(self, tool_id: str, version: str) -> Capability | None:
        normalized_tool_id = normalize_identifier(tool_id, "tool_id")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version must be a nonblank string")
        return self._repository.get_version(normalized_tool_id, version.strip())

    def list_capabilities(self, enabled: bool | None = None) -> list[Capability]:
        if enabled is not None and not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool or None")
        return self._repository.list_capabilities(enabled)
