"""
capability_resolver.py — NormalizedRequest → ResolutionResult.

Resolves against the Capability Registry only. Does not calculate, execute,
generate tools, or mutate registry state. Semantic/embedding matching is
reserved in the result model but not implemented yet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Support both ``backend.capability_resolver`` and ``unittest discover -s backend``.
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from capabilities import Capability, CapabilityRegistry  # noqa: E402
from normalizer import NormalizedRequest  # noqa: E402


ResolutionStatus = Literal["found", "missing", "ambiguous"]
MatchType = Literal["exact", "alias", "schema", "semantic", "none"]


class ResolutionResult(BaseModel):
    """Outcome of resolving a normalized request to registry metadata."""

    status: ResolutionStatus
    capability: Capability | None = None
    match_type: MatchType

    model_config = ConfigDict(extra="forbid")


def _version_sort_key(version: str) -> tuple[object, ...]:
    """Deterministic version ordering without semver dependency."""
    parts: list[object] = []
    for part in version.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return tuple(parts)


def _inputs_compatible(request: NormalizedRequest, capability: Capability) -> bool:
    """True when provided inputs satisfy the capability schema.

    - Every required schema input must appear in ``request.inputs``.
    - Every provided input must be declared in the schema (no unsupported keys).
    - ``request.missing_inputs`` never invents values; those names stay absent.
    """
    schema_names = {spec.name for spec in capability.input_schema}
    provided = set(request.inputs)

    for spec in capability.input_schema:
        if spec.required and spec.name not in provided:
            return False

    if not provided.issubset(schema_names):
        return False

    return True


def _finalize(
    matches: list[Capability], match_type: MatchType
) -> ResolutionResult:
    """Pick one capability or report ambiguity.

    Multiple versions of the same ``tool_id`` resolve to the highest version.
    Distinct ``tool_id`` values at the same match stage are ambiguous.
    """
    if not matches:
        return ResolutionResult(status="missing", capability=None, match_type="none")

    tool_ids = {capability.tool_id for capability in matches}
    if len(tool_ids) > 1:
        return ResolutionResult(
            status="ambiguous", capability=None, match_type=match_type
        )

    chosen = max(matches, key=lambda capability: _version_sort_key(capability.version))
    return ResolutionResult(
        status="found", capability=chosen, match_type=match_type
    )


class CapabilityResolver:
    """Match a NormalizedRequest to registry metadata, without execution."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        # Duck-type: avoid brittle isinstance failures when the same class is
        # imported once as ``backend.capabilities`` and once as ``capabilities``.
        required = ("get_by_operation", "get_by_alias", "list_capabilities")
        if not all(callable(getattr(registry, name, None)) for name in required):
            raise TypeError("registry must be a CapabilityRegistry")
        self._registry = registry

    def resolve(self, request: NormalizedRequest) -> ResolutionResult:
        if not hasattr(request, "operation") or not hasattr(request, "inputs"):
            raise TypeError("request must be a NormalizedRequest")

        # 1. Exact operation match
        if request.operation is not None:
            exact = [
                capability
                for capability in self._registry.get_by_operation(request.operation)
                if capability.enabled and _inputs_compatible(request, capability)
            ]
            if exact:
                return _finalize(exact, "exact")

            # 2. Alias match (operation text may be a registered alias)
            aliased = self._registry.get_by_alias(request.operation)
            if (
                aliased is not None
                and aliased.enabled
                and _inputs_compatible(request, aliased)
            ):
                return ResolutionResult(
                    status="found", capability=aliased, match_type="alias"
                )

        # 3. Compatible input-schema match across enabled capabilities
        #    (semantic / embedding matching intentionally not implemented)
        schema_matches = [
            capability
            for capability in self._registry.list_capabilities(enabled=True)
            if _inputs_compatible(request, capability)
        ]
        if schema_matches:
            return _finalize(schema_matches, "schema")

        # 4. Nothing matched
        return ResolutionResult(status="missing", capability=None, match_type="none")


def resolve_capability(
    request: NormalizedRequest, registry: CapabilityRegistry
) -> ResolutionResult:
    """Public function API for capability resolution."""
    return CapabilityResolver(registry).resolve(request)
