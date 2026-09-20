"""Unit tests for verified capability lifecycle + deterministic versioning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .capabilities import Capability, CapabilityRegistry
    from .capability_lifecycle import (
        next_version_for,
        register_verified_capability,
    )
    from .verifier import VerificationResult
    from models import InputSpec, OutputSpec, ToolSpec
except ImportError:  # ``unittest discover -s backend``
    from capabilities import Capability, CapabilityRegistry
    from capability_lifecycle import (
        next_version_for,
        register_verified_capability,
    )
    from verifier import VerificationResult
    from models import InputSpec, OutputSpec, ToolSpec


class CapabilityLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.database_path = self.root / "capabilities.sqlite3"
        self.registry = CapabilityRegistry(self.database_path)

    def tearDown(self) -> None:
        self.registry.close()
        self._temporary_directory.cleanup()

    def _tool_file(self, name: str, body: str | None = None) -> str:
        path = self.root / name
        path.write_text(
            body or "def placeholder():\n    return None\n",
            encoding="utf-8",
        )
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
                InputSpec(name="height_m", type="float", unit="m"),
                InputSpec(name="weight_kg", type="float", unit="kg"),
            ],
            output=OutputSpec(name="bmi", type="number", unit="kg/m^2"),
        )

    def _verified(self, tool_name: str = "calculate_speed") -> VerificationResult:
        return VerificationResult(
            status="verified",
            tool_name=tool_name,
            details="all verifiable tests passed",
        )

    def _failed(self, tool_name: str = "calculate_speed") -> VerificationResult:
        return VerificationResult(
            status="failed",
            tool_name=tool_name,
            errors=["one or more tests failed"],
            details="one or more tests failed",
        )

    def test_first_registration_creates_v1(self) -> None:
        code_path = self._tool_file("speed_v1.py")
        result = register_verified_capability(
            self._speed_spec(),
            self._verified(),
            code_path,
            registry=self.registry,
            aliases=["velocity"],
        )

        self.assertTrue(result.success)
        self.assertEqual(result.version, "1")
        self.assertIsNotNone(result.capability)
        self.assertEqual(result.capability.version, "1")
        self.assertEqual(
            self.registry.get_version("calculate_speed", "1"),
            result.capability,
        )

    def test_second_registration_creates_v2(self) -> None:
        register_verified_capability(
            self._speed_spec(),
            self._verified(),
            self._tool_file("speed_v1.py"),
            registry=self.registry,
        )
        result = register_verified_capability(
            self._speed_spec(),
            self._verified(),
            self._tool_file("speed_v2.py"),
            registry=self.registry,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.version, "2")
        self.assertEqual(
            self.registry.get_version("calculate_speed", "2").version,
            "2",
        )

    def test_v1_remains_unchanged_after_v2(self) -> None:
        path_v1 = self._tool_file("speed_v1.py", "def speed(a, b):\n    return a / b\n")
        first = register_verified_capability(
            self._speed_spec(),
            self._verified(),
            path_v1,
            registry=self.registry,
            aliases=["velocity"],
        )
        path_v2 = self._tool_file(
            "speed_v2.py", "def speed(a, b):\n    return (a / b) * 1.0\n"
        )
        register_verified_capability(
            self._speed_spec(),
            self._verified(),
            path_v2,
            registry=self.registry,
            aliases=["pace"],
        )

        stored_v1 = self.registry.get_version("calculate_speed", "1")
        self.assertIsNotNone(stored_v1)
        self.assertEqual(stored_v1, first.capability)
        self.assertEqual(stored_v1.code_path, path_v1)
        self.assertEqual(stored_v1.aliases, ["velocity"])
        self.assertEqual(stored_v1.verification_status, "verified")

    def test_failed_verification_is_rejected(self) -> None:
        code_path = self._tool_file("speed_bad.py")
        result = register_verified_capability(
            self._speed_spec(),
            self._failed(),
            code_path,
            registry=self.registry,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.capability)
        self.assertIsNone(result.version)
        self.assertIn("only verified", result.error or "")
        self.assertEqual(self.registry.list_capabilities(), [])

    def test_not_verifiable_status_is_rejected(self) -> None:
        result = register_verified_capability(
            self._speed_spec(),
            VerificationResult(
                status="not_verifiable",
                tool_name="calculate_speed",
            ),
            self._tool_file("speed_nv.py"),
            registry=self.registry,
        )
        self.assertFalse(result.success)
        self.assertEqual(self.registry.list_capabilities(), [])

    def test_successful_verification_registers_as_verified(self) -> None:
        result = register_verified_capability(
            self._speed_spec(),
            self._verified(),
            self._tool_file("speed_ok.py"),
            registry=self.registry,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.capability.verification_status, "verified")
        stored = self.registry.get_version("calculate_speed", "1")
        self.assertEqual(stored.verification_status, "verified")

    def test_code_path_preserved(self) -> None:
        code_path = self._tool_file("speed_code.py")
        result = register_verified_capability(
            self._speed_spec(),
            self._verified(),
            code_path,
            registry=self.registry,
        )
        self.assertEqual(result.capability.code_path, code_path)
        self.assertEqual(
            self.registry.get_version("calculate_speed", "1").code_path,
            code_path,
        )

    def test_aliases_and_schema_preserved(self) -> None:
        spec = self._speed_spec()
        result = register_verified_capability(
            spec,
            self._verified(),
            self._tool_file("speed_alias.py"),
            registry=self.registry,
            aliases=["velocity", "pace"],
        )
        stored = result.capability
        self.assertEqual(stored.aliases, ["velocity", "pace"])
        self.assertEqual(stored.input_schema, spec.inputs)
        self.assertEqual(stored.output_schema, spec.output)
        self.assertEqual(stored.description, spec.purpose)
        self.assertEqual(self.registry.get_by_alias("velocity"), stored)

    def test_different_tools_each_start_at_v1(self) -> None:
        speed = register_verified_capability(
            self._speed_spec(),
            self._verified("calculate_speed"),
            self._tool_file("speed.py"),
            registry=self.registry,
        )
        bmi = register_verified_capability(
            self._bmi_spec(),
            self._verified("calculate_bmi"),
            self._tool_file("bmi.py"),
            registry=self.registry,
        )

        self.assertEqual(speed.version, "1")
        self.assertEqual(bmi.version, "1")
        self.assertEqual(speed.tool_id, "calculate_speed")
        self.assertEqual(bmi.tool_id, "calculate_bmi")
        self.assertIsNotNone(self.registry.get_version("calculate_speed", "1"))
        self.assertIsNotNone(self.registry.get_version("calculate_bmi", "1"))

    def test_disabled_existing_versions_do_not_get_overwritten(self) -> None:
        disabled_path = self._tool_file("speed_disabled.py")
        disabled = Capability.from_tool_spec(
            self._speed_spec(),
            version="1",
            code_path=disabled_path,
            aliases=["old_velocity"],
            verification_status="verified",
            enabled=False,
        )
        self.registry.register(disabled)

        new_path = self._tool_file("speed_new.py")
        result = register_verified_capability(
            self._speed_spec(),
            self._verified(),
            new_path,
            registry=self.registry,
            aliases=["velocity"],
        )

        self.assertTrue(result.success)
        self.assertEqual(result.version, "2")

        still_v1 = self.registry.get_version("calculate_speed", "1")
        self.assertIsNotNone(still_v1)
        self.assertEqual(still_v1.code_path, disabled_path)
        self.assertFalse(still_v1.enabled)
        self.assertEqual(still_v1.aliases, ["old_velocity"])

        stored_v2 = self.registry.get_version("calculate_speed", "2")
        self.assertEqual(stored_v2.code_path, new_path)
        self.assertTrue(stored_v2.enabled)

    def test_version_assignment_is_deterministic(self) -> None:
        self.assertEqual(next_version_for(self.registry, "calculate_speed"), "1")
        register_verified_capability(
            self._speed_spec(),
            self._verified(),
            self._tool_file("speed_d1.py"),
            registry=self.registry,
        )
        self.assertEqual(next_version_for(self.registry, "calculate_speed"), "2")
        register_verified_capability(
            self._speed_spec(),
            self._verified(),
            self._tool_file("speed_d2.py"),
            registry=self.registry,
        )
        self.assertEqual(next_version_for(self.registry, "calculate_speed"), "3")

        # Repeated calls with no new registration stay stable.
        self.assertEqual(next_version_for(self.registry, "calculate_speed"), "3")
        self.assertEqual(next_version_for(self.registry, "calculate_speed"), "3")


if __name__ == "__main__":
    unittest.main()
