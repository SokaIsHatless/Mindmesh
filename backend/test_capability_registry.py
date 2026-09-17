"""Unit tests for the isolated SQLite capability registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .capabilities import Capability, CapabilityRegistry
    from .models import InputSpec, OutputSpec, ToolSpec
except ImportError:  # Supports ``unittest discover -s backend``.
    from capabilities import Capability, CapabilityRegistry
    from models import InputSpec, OutputSpec, ToolSpec


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.database_path = self.root / "capabilities.sqlite3"
        self.speed_path = self._tool_file("speed.py")
        self.bmi_path = self._tool_file("bmi.py")
        self.registry = CapabilityRegistry(self.database_path)

    def tearDown(self) -> None:
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

    def test_register_and_retrieve_by_operation(self) -> None:
        registered = self.registry.register(
            self._speed_capability(tool_id=" Calculate_Speed ")
        )

        matches = self.registry.get_by_operation(" SPEED ")

        self.assertEqual(registered.tool_id, "calculate_speed")
        self.assertEqual(matches, [registered])

    def test_register_and_retrieve_by_alias(self) -> None:
        capability = self.registry.register(self._speed_capability())

        self.assertEqual(self.registry.get_by_alias(" VELOCITY "), capability)

    def test_retrieve_specific_version(self) -> None:
        capability = self.registry.register(self._speed_capability(version="2.0.0"))

        self.assertEqual(
            self.registry.get_version("CALCULATE_SPEED", "2.0.0"), capability
        )

    def test_missing_tool_or_version_returns_none(self) -> None:
        self.registry.register(self._speed_capability())

        self.assertIsNone(self.registry.get_version("calculate_speed", "9.9.9"))
        self.assertIsNone(self.registry.get_version("unknown", "1.0.0"))

    def test_duplicate_tool_version_replaces_metadata_and_aliases(self) -> None:
        self.registry.register(self._speed_capability(aliases=["velocity"]))
        replacement = self._speed_capability(
            description="Updated speed calculation.",
            aliases=["pace"],
            verification_status="verified",
        )

        self.assertEqual(self.registry.register(replacement), replacement)
        self.assertEqual(
            self.registry.get_version("calculate_speed", "1.0.0"), replacement
        )
        self.assertIsNone(self.registry.get_by_alias("velocity"))
        self.assertEqual(self.registry.get_by_alias("pace"), replacement)

    def test_verification_status_is_preserved(self) -> None:
        capability = self.registry.register(
            self._speed_capability(verification_status="failed")
        )

        stored = self.registry.get_version(capability.tool_id, capability.version)

        self.assertIsNotNone(stored)
        self.assertEqual(stored.verification_status, "failed")

    def test_disabled_capability_is_preserved_and_filterable(self) -> None:
        disabled = self.registry.register(self._speed_capability(enabled=False))

        self.assertEqual(
            self.registry.get_by_operation("speed"),
            [disabled],
        )
        self.assertEqual(self.registry.list_capabilities(), [disabled])
        self.assertEqual(self.registry.list_capabilities(enabled=True), [])
        self.assertEqual(self.registry.list_capabilities(enabled=False), [disabled])

    def test_input_and_output_schema_metadata_is_preserved(self) -> None:
        capability = self.registry.register(self._speed_capability())

        stored = self.registry.get_version(capability.tool_id, capability.version)

        self.assertIsNotNone(stored)
        self.assertEqual(stored.input_schema, capability.input_schema)
        self.assertEqual(stored.output_schema, capability.output_schema)

    def test_data_persists_across_registry_instances(self) -> None:
        capability = self.registry.register(self._speed_capability())
        reopened = CapabilityRegistry(self.database_path)

        self.assertEqual(
            reopened.get_version(capability.tool_id, capability.version), capability
        )

    def test_bmi_and_speed_use_the_same_generic_metadata_structure(self) -> None:
        speed = self.registry.register(self._speed_capability())
        bmi_spec = ToolSpec(
            name="calculate_bmi",
            operation="bmi",
            purpose="Calculate body mass index from height and weight.",
            inputs=[
                InputSpec(name="height_cm", type="float", unit="cm"),
                InputSpec(name="weight_kg", type="float", unit="kg"),
            ],
            output=OutputSpec(name="bmi", type="number", unit="kg/m^2"),
        )
        bmi = self.registry.register(
            Capability.from_tool_spec(
                bmi_spec,
                version="1.0.0",
                code_path=self.bmi_path,
                aliases=["body_mass_index"],
                verification_status="verified",
            )
        )

        self.assertEqual(self.registry.get_by_operation("speed"), [speed])
        self.assertEqual(self.registry.get_by_operation("bmi"), [bmi])
        self.assertEqual(self.registry.get_by_alias("body_mass_index"), bmi)


if __name__ == "__main__":
    unittest.main()
