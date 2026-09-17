"""Unit tests for the Capability Resolver (registry lookup only)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    # Import resolver first so it places ``backend/`` on sys.path; then use the
    # same flat imports the resolver uses (avoids duplicate class objects).
    from .capability_resolver import CapabilityResolver, resolve_capability
    from capabilities import Capability, CapabilityRegistry
    from models import InputSpec, OutputSpec, ToolSpec
    from normalizer import NormalizedRequest
except ImportError:  # Supports ``unittest discover -s backend``.
    from capability_resolver import CapabilityResolver, resolve_capability
    from capabilities import Capability, CapabilityRegistry
    from models import InputSpec, OutputSpec, ToolSpec
    from normalizer import NormalizedRequest


class CapabilityResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.database_path = self.root / "capabilities.sqlite3"
        self.speed_path = self._tool_file("speed.py")
        self.bmi_path = self._tool_file("bmi.py")
        self.other_path = self._tool_file("other_speed.py")
        self.registry = CapabilityRegistry(self.database_path)
        self.resolver = CapabilityResolver(self.registry)

    def tearDown(self) -> None:
        self.registry.close()
        self._temporary_directory.cleanup()

    def _tool_file(self, name: str) -> str:
        path = self.root / name
        path.write_text("def placeholder():\n    return None\n", encoding="utf-8")
        return str(path)

    def _speed_spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculate_speed",
            operation="speed",
            purpose="Calculate speed from distance and time.",
            inputs=[
                InputSpec(name="distance_km", type="float", unit="km"),
                InputSpec(name="time_hr", type="float", unit="hr"),
            ],
            output=OutputSpec(name="speed_kmh", type="number", unit="km/h"),
        )

    def _bmi_spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculate_bmi",
            operation="bmi",
            purpose="Calculate body mass index from height and weight.",
            inputs=[
                InputSpec(name="height_cm", type="float", unit="cm"),
                InputSpec(name="weight_kg", type="float", unit="kg"),
            ],
            output=OutputSpec(name="bmi", type="number", unit="kg/m^2"),
        )

    def _speed_capability(self, **overrides: object) -> Capability:
        values: dict[str, object] = {
            "version": "1.0.0",
            "code_path": self.speed_path,
            "aliases": ["velocity"],
        }
        values.update(overrides)
        capability = Capability.from_tool_spec(
            self._speed_spec(),
            version=values.pop("version"),  # type: ignore[arg-type]
            code_path=values.pop("code_path"),  # type: ignore[arg-type]
            tool_id=values.pop("tool_id", None),  # type: ignore[arg-type]
            aliases=values.pop("aliases", None),  # type: ignore[arg-type]
            verification_status=values.pop("verification_status", "pending"),  # type: ignore[arg-type]
            enabled=values.pop("enabled", True),  # type: ignore[arg-type]
        )
        data = capability.model_dump()
        data.update(values)
        return Capability.model_validate(data)

    def _bmi_capability(self, **overrides: object) -> Capability:
        values: dict[str, object] = {
            "version": "1.0.0",
            "code_path": self.bmi_path,
            "aliases": ["body_mass_index"],
        }
        values.update(overrides)
        capability = Capability.from_tool_spec(
            self._bmi_spec(),
            version=values.pop("version"),  # type: ignore[arg-type]
            code_path=values.pop("code_path"),  # type: ignore[arg-type]
            tool_id=values.pop("tool_id", None),  # type: ignore[arg-type]
            aliases=values.pop("aliases", None),  # type: ignore[arg-type]
            verification_status=values.pop("verification_status", "pending"),  # type: ignore[arg-type]
            enabled=values.pop("enabled", True),  # type: ignore[arg-type]
        )
        data = capability.model_dump()
        data.update(values)
        return Capability.model_validate(data)

    def _request(
        self,
        *,
        operation: str | None = None,
        inputs: dict[str, float] | None = None,
        missing_inputs: list[str] | None = None,
    ) -> NormalizedRequest:
        provided = inputs or {}
        return NormalizedRequest(
            operation=operation,
            inputs=provided,
            original_inputs=dict(provided),
            missing_inputs=missing_inputs or [],
        )

    def test_exact_operation_match(self) -> None:
        capability = self.registry.register(self._speed_capability())

        result = self.resolver.resolve(
            self._request(
                operation="speed",
                inputs={"distance_km": 120, "time_hr": 2},
            )
        )

        self.assertEqual(result.status, "found")
        self.assertEqual(result.match_type, "exact")
        self.assertEqual(result.capability, capability)

    def test_alias_match_velocity_to_speed(self) -> None:
        capability = self.registry.register(self._speed_capability())

        result = self.resolver.resolve(
            self._request(
                operation="velocity",
                inputs={"distance_km": 120, "time_hr": 2},
            )
        )

        self.assertEqual(result.status, "found")
        self.assertEqual(result.match_type, "alias")
        self.assertEqual(result.capability, capability)

    def test_missing_capability(self) -> None:
        result = self.resolver.resolve(
            self._request(
                operation="orbit",
                inputs={"semi_major_axis_km": 7000, "eccentricity": 0.01},
            )
        )

        self.assertEqual(result.status, "missing")
        self.assertIsNone(result.capability)
        self.assertEqual(result.match_type, "none")

    def test_missing_required_input(self) -> None:
        self.registry.register(self._speed_capability())

        result = self.resolver.resolve(
            self._request(
                operation="speed",
                inputs={"distance_km": 120},
                missing_inputs=["time_hr"],
            )
        )

        self.assertEqual(result.status, "missing")
        self.assertIsNone(result.capability)
        self.assertEqual(result.match_type, "none")

    def test_incompatible_input_schema(self) -> None:
        self.registry.register(self._speed_capability())

        result = self.resolver.resolve(
            self._request(
                operation="speed",
                inputs={"height_cm": 180, "weight_kg": 70},
            )
        )

        self.assertEqual(result.status, "missing")
        self.assertIsNone(result.capability)
        self.assertEqual(result.match_type, "none")

    def test_compatible_input_schema_match(self) -> None:
        capability = self.registry.register(self._bmi_capability())

        result = self.resolver.resolve(
            self._request(
                operation=None,
                inputs={"height_cm": 180, "weight_kg": 70},
            )
        )

        self.assertEqual(result.status, "found")
        self.assertEqual(result.match_type, "schema")
        self.assertEqual(result.capability, capability)

    def test_ambiguous_match(self) -> None:
        self.registry.register(self._speed_capability())
        self.registry.register(
            self._speed_capability(
                tool_id="alternate_speed",
                code_path=self.other_path,
                aliases=["pace"],
            )
        )

        result = self.resolver.resolve(
            self._request(
                operation="speed",
                inputs={"distance_km": 100, "time_hr": 1},
            )
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.capability)
        self.assertEqual(result.match_type, "exact")

    def test_multiple_versions_handled_deterministically(self) -> None:
        self.registry.register(self._speed_capability(version="1.0.0", aliases=[]))
        newer = self.registry.register(
            self._speed_capability(version="2.0.0", aliases=["velocity"])
        )

        result = self.resolver.resolve(
            self._request(
                operation="speed",
                inputs={"distance_km": 50, "time_hr": 0.5},
            )
        )

        self.assertEqual(result.status, "found")
        self.assertEqual(result.match_type, "exact")
        self.assertEqual(result.capability, newer)
        self.assertEqual(result.capability.version, "2.0.0")

    def test_disabled_capability_ignored(self) -> None:
        self.registry.register(self._speed_capability(enabled=False))

        result = self.resolver.resolve(
            self._request(
                operation="speed",
                inputs={"distance_km": 120, "time_hr": 2},
            )
        )

        self.assertEqual(result.status, "missing")
        self.assertIsNone(result.capability)
        self.assertEqual(result.match_type, "none")

    def test_generic_bmi_and_speed_capabilities(self) -> None:
        speed = self.registry.register(self._speed_capability())
        bmi = self.registry.register(self._bmi_capability())

        speed_result = resolve_capability(
            self._request(
                operation="speed",
                inputs={"distance_km": 120, "time_hr": 2},
            ),
            self.registry,
        )
        bmi_result = resolve_capability(
            self._request(
                operation="bmi",
                inputs={"height_cm": 180, "weight_kg": 70},
            ),
            self.registry,
        )

        self.assertEqual(speed_result.status, "found")
        self.assertEqual(speed_result.capability, speed)
        self.assertEqual(bmi_result.status, "found")
        self.assertEqual(bmi_result.capability, bmi)


if __name__ == "__main__":
    unittest.main()
