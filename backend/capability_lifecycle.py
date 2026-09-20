"""
capability_lifecycle.py — Register verified tools into the Capability Registry.

Assigns deterministic integer versions (1, 2, 3, …) and never overwrites an
existing version. Only VerificationResult.status == "verified" may register.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

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
    capability: Capability | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


def _looks_like_verification_result(value: Any) -> bool:
    """Accept VerificationResult across package/flat dual imports."""
    if isinstance(value, VerificationResult):
        return True
    return (
        type(value).__name__ == "VerificationResult"
        and hasattr(value, "status")
        and hasattr(value, "tool_name")
    )


def _tool_id_for(tool_spec: ToolSpec) -> str:
    return normalize_identifier(tool_spec.name, "tool_id")


def _aliases_for(
    tool_spec: ToolSpec, aliases: list[str] | None
) -> list[str]:
    """Prefer explicit aliases; otherwise preserve any ToolSpec-provided ones."""
    if aliases is not None:
        return list(aliases)
    # ToolSpec has no first-class aliases field today; keep a generic hook
    # for future metadata without hard-coding domain names.
    extra = getattr(tool_spec, "aliases", None)
    if isinstance(extra, list):
        return list(extra)
    return []


def _existing_versions(
    registry: CapabilityRegistry, tool_id: str
) -> list[Capability]:
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


def next_version_for(
    registry: CapabilityRegistry, tool_id: str
) -> str:
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
    if not isinstance(tool_spec, ToolSpec):
        return LifecycleResult(
            success=False,
            error="tool_spec must be a ToolSpec",
        )
    if not isinstance(registry, CapabilityRegistry):
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

    # Guard against accidental overwrite if a concurrent/non-integer gap exists.
    if registry.get_version(tool_id, version) is not None:
        return LifecycleResult(
            success=False,
            tool_id=tool_id,
            version=version,
            error=f"version {version!r} already exists for tool_id={tool_id!r}",
        )

    try:
        capability = Capability.from_tool_spec(
            tool_spec,
            version=version,
            code_path=code_path,
            tool_id=tool_id,
            aliases=_aliases_for(tool_spec, aliases),
            verification_status="verified",
            enabled=True,
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
