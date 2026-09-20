"""
capability_lifecycle.py — Register verified tools into the Capability Registry.

Assigns deterministic integer versions (1, 2, 3, …) and never overwrites an
existing version. Only VerificationResult.status == "verified" may register.

Class-identity note: tests may import CapabilityRegistry via ``backend.capabilities``
while ToolSpec arrives from flat ``models``. This module duck-types those objects and
builds Capability instances using the registry's own class objects so registration
never fails solely because of duplicate module paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:
    from .capabilities import Capability, CapabilityRegistry
    from .capabilities.models import normalize_identifier
    from .models import ToolSpec
except ImportError:  # Flat ``unittest discover -s backend``
    from capabilities import Capability, CapabilityRegistry  # noqa: E402
    from capabilities.models import normalize_identifier  # noqa: E402
    from models import ToolSpec  # noqa: E402

try:
    from .verifier import VerificationResult
except ImportError:
    from verifier import VerificationResult  # noqa: E402


class LifecycleResult(BaseModel):
    """Outcome of attempting to register a verified capability version."""

    success: bool
    tool_id: str = ""
    version: str | None = None
    # Any: preserve the registry's Capability class across dual imports.
    capability: Any | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


def _looks_like_verification_result(value: Any) -> bool:
    if isinstance(value, VerificationResult):
        return True
    return (
        type(value).__name__ == "VerificationResult"
        and hasattr(value, "status")
        and hasattr(value, "tool_name")
    )


def _looks_like_tool_spec(value: Any) -> bool:
    if isinstance(value, ToolSpec):
        return True
    return (
        type(value).__name__ == "ToolSpec"
        and hasattr(value, "name")
        and hasattr(value, "purpose")
        and hasattr(value, "inputs")
        and hasattr(value, "output")
    )


def _looks_like_registry(value: Any) -> bool:
    if isinstance(value, CapabilityRegistry):
        return True
    return (
        type(value).__name__ == "CapabilityRegistry"
        and callable(getattr(value, "register", None))
        and callable(getattr(value, "list_capabilities", None))
        and callable(getattr(value, "get_version", None))
    )


def _capability_cls_for_registry(registry: Any) -> type:
    """Capability class object the given registry's ``register`` will accept."""
    register_fn = type(registry).register
    globals_dict = (
        register_fn.__func__.__globals__
        if hasattr(register_fn, "__func__")
        else register_fn.__globals__
    )
    return globals_dict.get("Capability", Capability)


def _normalize_aliases(aliases: list[str]) -> list[str]:
    normalized: list[str] = []
    for alias in aliases:
        item = normalize_identifier(alias, "aliases")
        if item in normalized:
            raise ValueError(f"duplicate alias: {item}")
        normalized.append(item)
    return normalized


def _validated_code_path(code_path: str) -> str:
    path_text = code_path.strip()
    path = Path(path_text)
    if path.suffix.lower() != ".py":
        raise ValueError("code_path must point to a .py file")
    if not path.is_file():
        raise ValueError("code_path must point to an existing Python file")
    return str(path)


def _build_capability(
    capability_cls: type,
    tool_spec: Any,
    *,
    version: str,
    code_path: str,
    tool_id: str,
    aliases: list[str],
) -> Any:
    """Build a Capability using the registry's class, preserving schema objects.

    ``model_construct`` avoids re-wrapping InputSpec/OutputSpec through a
    different ``models`` module path, so schema equality with the caller's
    ToolSpec remains stable under dual imports.
    """
    operation = (
        tool_spec.operation if tool_spec.operation is not None else tool_spec.name
    )
    return capability_cls.model_construct(
        tool_id=normalize_identifier(tool_id, "tool_id"),
        operation=normalize_identifier(operation, "operation"),
        version=version.strip(),
        description=str(tool_spec.purpose).strip(),
        aliases=_normalize_aliases(aliases),
        input_schema=list(tool_spec.inputs),
        output_schema=tool_spec.output,
        code_path=_validated_code_path(code_path),
        verification_status="verified",
        enabled=True,
    )


def _tool_id_for(tool_spec: Any) -> str:
    return normalize_identifier(tool_spec.name, "tool_id")


def _aliases_for(tool_spec: Any, aliases: list[str] | None) -> list[str]:
    """Prefer explicit aliases; otherwise preserve any ToolSpec-provided ones."""
    if aliases is not None:
        return list(aliases)
    extra = getattr(tool_spec, "aliases", None)
    if isinstance(extra, list):
        return list(extra)
    return []


def _existing_versions(registry: Any, tool_id: str) -> list[Any]:
    """All stored versions for tool_id, including disabled ones."""
    normalized = normalize_identifier(tool_id, "tool_id")
    return [
        capability
        for capability in registry.list_capabilities()
        if capability.tool_id == normalized
    ]


def _parse_integer_version(version: str) -> int | None:
    text = version.strip()
    if not text.isdigit():
        return None
    return int(text)


def next_version_for(registry: Any, tool_id: str) -> str:
    """Deterministic next integer version string; never reuses an existing one."""
    used = {
        parsed
        for capability in _existing_versions(registry, tool_id)
        if (parsed := _parse_integer_version(capability.version)) is not None
    }
    candidate = 1
    while candidate in used:
        candidate += 1
    return str(candidate)


def register_verified_capability(
    tool_spec: ToolSpec,
    verification: VerificationResult,
    code_path: str,
    *,
    registry: CapabilityRegistry,
    aliases: list[str] | None = None,
) -> LifecycleResult:
    """Register a verified tool as a new immutable capability version.

    Only ``verification.status == "verified"`` is accepted. Versions are
    assigned as ``"1"``, ``"2"``, ``"3"``, … without overwriting prior rows
    (including disabled ones). Executable code stays on disk at ``code_path``.
    """
    if not _looks_like_tool_spec(tool_spec):
        return LifecycleResult(
            success=False,
            error="tool_spec must be a ToolSpec",
        )
    if not _looks_like_registry(registry):
        return LifecycleResult(
            success=False,
            error="registry must be a CapabilityRegistry",
        )
    if not _looks_like_verification_result(verification):
        return LifecycleResult(
            success=False,
            tool_id=_tool_id_for(tool_spec),
            error="verification must be a VerificationResult",
        )
    if verification.status != "verified":
        return LifecycleResult(
            success=False,
            tool_id=_tool_id_for(tool_spec),
            error=(
                "only verified tools may be registered; "
                f"got status={verification.status!r}"
            ),
        )
    if not isinstance(code_path, str) or not code_path.strip():
        return LifecycleResult(
            success=False,
            tool_id=_tool_id_for(tool_spec),
            error="code_path must be a nonblank string",
        )

    tool_id = _tool_id_for(tool_spec)
    version = next_version_for(registry, tool_id)

    if registry.get_version(tool_id, version) is not None:
        return LifecycleResult(
            success=False,
            tool_id=tool_id,
            version=version,
            error=f"version {version!r} already exists for tool_id={tool_id!r}",
        )

    capability_cls = _capability_cls_for_registry(registry)
    try:
        capability = _build_capability(
            capability_cls,
            tool_spec,
            version=version,
            code_path=code_path,
            tool_id=tool_id,
            aliases=_aliases_for(tool_spec, aliases),
        )
        stored = registry.register(capability)
    except (TypeError, ValueError, OSError) as exc:
        return LifecycleResult(
            success=False,
            tool_id=tool_id,
            version=version,
            error=str(exc),
        )

    return LifecycleResult(
        success=True,
        tool_id=stored.tool_id,
        version=stored.version,
        capability=stored,
        error=None,
    )
