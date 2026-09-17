"""Persistent metadata registry for computational capabilities."""

from .models import Capability, VerificationStatus
from .registry import CapabilityRegistry

__all__ = ["Capability", "CapabilityRegistry", "VerificationStatus"]
