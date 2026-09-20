"""Integration tests for the orchestrator architecture pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from . import orchestrator
    from .capabilities import Capability, CapabilityRegistry
    from .models import InputSpec, OutputSpec, TestCase, ToolSpec
    from .orchestrator import handle_task, set_registry
    from .result_validator import ResultValidationResult
    from .test_generator import GeneratedTestCase, TestGenerationResult
    from .verifier import VerificationResult
except ImportError:  # ``unittest discover -s backend``
    import orchestrator
    from capabilities import Capability, CapabilityRegistry
    from models import InputSpec, OutputSpec, TestCase, ToolSpec
    from orchestrator import handle_task, set_registry
    from result_validator import ResultValidationResult
    from test_generator import GeneratedTestCase, TestGenerationResult
    from verifier import VerificationResult


class OrchestratorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.database_path = self.root / "capabilities.sqlite3"
        self.toolbox_dir = self.root / "toolbox"
        self.toolbox_dir.mkdir(parents=True, exist_ok=True)
        self.registry = CapabilityRegistry(self.database_path)
        set_registry(self.registry)

        self._toolbox_patch = patch.object(orchestrator, "TOOLBOX_DIR", self.toolbox_dir)
        self._manifest_patch = patch.object(
            orchestrator,
            "MANIFEST_PATH",
            self.toolbox_dir / "manifest.json",
        )
        self._toolbox_patch.start()
        self._manifest_patch.start()

    def tearDown(self) -> None:
        self._manifest_patch.stop()
        self._toolbox_patch.stop()
        set_registry(None)
        self.registry.close()
        self._temporary_directory.cleanup()

    def _speed_code(self) -> str:
        return "def speed(distance_km, time_hr):\n    return distance_km / time_hr\n"

    def _speed_spec(self) -> ToolSpec:
        return ToolSpec(
            name="speed",
            purpose="compute speed from distance and time",
            operation="speed",
            inputs=[
                InputSpec(name="distance_km", type="float", unit="km"),
                InputSpec(name="time_hr", type="float", unit="hr"),
            ],
            output=OutputSpec(name="speed_kmh", type="number", unit="km/h"),
            constraints=["time_hr must not be zero"],
            examples=[
                TestCase(inputs={"distance_km": 120, "time_hr": 2}, expected=60),
            ],
        )

    def _ok_interpretation(
        self, *, operation: str = "speed", inputs: dict | None = None
    ) -> dict:
        return {
            "status": "ok",
            "operation": operation,
            "inputs": inputs or {"distance_km": 120, "time_hr": 2},
            "missing_inputs": [],
        }

    def _register_speed_capability(self) -> Capability:
        code_path = self.toolbox_dir / "speed.py"
        code_path.write_text(self._speed_code(), encoding="utf-8")
        capability = Capability.from_tool_spec(
            self._speed_spec(),
            version="1",
            code_path=str(code_path),
            aliases=["velocity"],
            verification_status="verified",
        )
        return self.registry.register(capability)

    def _trace_types(self, result: dict) -> list[str]:
        return [step["type"] for step in result["trace"]]

    def test_existing_capability_is_reused(self) -> None:
        self._register_speed_capability()
        with (
            patch.object(
                orchestrator,
                "interpret_task",
                return_value=self._ok_interpretation(),
            ),
            patch.object(orchestrator, "create_tool") as create_tool_mock,
        ):
            result = handle_task("train speed 120 km in 2 hours")

        create_tool_mock.assert_not_called()
        self.assertIn("60", result["answer"])
        self.assertIn("km/h", result["answer"])
        labels = " ".join(step["label"] for step in result["trace"])
        self.assertIn("reusing", labels.lower())
        self.assertIn("check", self._trace_types(result))
        self.assertNotIn("no_tool", self._trace_types(result))
        self.assertNotIn("writing", self._trace_types(result))

    def test_missing_capability_generates_and_verifies(self) -> None:
        factory_payload = {
            "success": True,
            "tool_name": "speed",
            "code": self._speed_code(),
            "attempts": 1,
            "error": None,
        }
        tests = TestGenerationResult(
            success=True,
            tool_name="speed",
            tests=[
                GeneratedTestCase(
                    inputs={"distance_km": 120, "time_hr": 2},
                    category="valid",
                    expectation_status="known",
                    expected=60,
                )
            ],
        )
        verification = VerificationResult(
            status="verified",
            tool_name="speed",
            details="ok",
        )

        with (
            patch.object(
                orchestrator,
                "interpret_task",
                return_value=self._ok_interpretation(),
            ),
            patch.object(orchestrator, "create_tool", return_value=factory_payload),
            patch.object(orchestrator, "generate_tests", return_value=tests),
            patch.object(orchestrator, "verify_tool", return_value=verification),
        ):
            result = handle_task("compute speed from distance and time")

        self.assertIn("60", result["answer"])
        types = self._trace_types(result)
        self.assertIn("no_tool", types)
        self.assertIn("writing", types)
        self.assertIn("testing", types)
        self.assertIn("pass", types)
        stored = self.registry.get_version("speed", "1")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.verification_status, "verified")

    def test_failed_verification_prevents_registration(self) -> None:
        factory_payload = {
            "success": True,
            "tool_name": "speed",
            "code": self._speed_code(),
            "attempts": 1,
            "error": None,
        }
        tests = TestGenerationResult(success=True, tool_name="speed", tests=[])
        verification = VerificationResult(
            status="failed",
            tool_name="speed",
            errors=["sandbox mismatch"],
            details="sandbox mismatch",
        )

        with (
            patch.object(
                orchestrator,
                "interpret_task",
                return_value=self._ok_interpretation(),
            ),
            patch.object(orchestrator, "create_tool", return_value=factory_payload),
            patch.object(orchestrator, "generate_tests", return_value=tests),
            patch.object(orchestrator, "verify_tool", return_value=verification),
            patch.object(
                orchestrator, "register_verified_capability"
            ) as register_mock,
        ):
            result = handle_task("compute speed from distance and time")

        register_mock.assert_not_called()
        self.assertEqual(self.registry.list_capabilities(), [])
        self.assertIn("Verification failed", " ".join(s["label"] for s in result["trace"]))
        self.assertIn("fail", self._trace_types(result))

    def test_successful_new_capability_is_registered_and_executed(self) -> None:
        factory_payload = {
            "success": True,
            "tool_name": "speed",
            "code": self._speed_code(),
            "attempts": 1,
            "error": None,
        }
        tests = TestGenerationResult(
            success=True,
            tool_name="speed",
            tests=[
                GeneratedTestCase(
                    inputs={"distance_km": 90, "time_hr": 3},
                    category="valid",
                    expectation_status="known",
                    expected=30,
                )
            ],
        )
        verification = VerificationResult(status="verified", tool_name="speed")

        with (
            patch.object(
                orchestrator,
                "interpret_task",
                return_value=self._ok_interpretation(
                    inputs={"distance_km": 90, "time_hr": 3}
                ),
            ),
            patch.object(orchestrator, "create_tool", return_value=factory_payload),
            patch.object(orchestrator, "generate_tests", return_value=tests),
            patch.object(orchestrator, "verify_tool", return_value=verification),
        ):
            result = handle_task("speed for 90 km in 3 hours")

        self.assertIn("30", result["answer"])
        self.assertIsNotNone(self.registry.get_version("speed", "1"))
        self.assertTrue((self.toolbox_dir / "speed.py").is_file())

    def test_result_validation_failure_is_handled(self) -> None:
        self._register_speed_capability()
        invalid = ResultValidationResult(
            status="invalid",
            errors=["unit mismatch: expected 'km/h', got 'm/s'"],
            details="unit mismatch",
        )
        with (
            patch.object(
                orchestrator,
                "interpret_task",
                return_value=self._ok_interpretation(),
            ),
            patch.object(orchestrator, "validate_result", return_value=invalid),
        ):
            result = handle_task("train speed 120 km in 2 hours")

        self.assertIn("validation", result["answer"].lower())
        self.assertIn("fail", self._trace_types(result))
        self.assertTrue(
            any("Result validation failed" in step["label"] for step in result["trace"])
        )

    def test_unsupported_request_is_handled(self) -> None:
        with patch.object(
            orchestrator,
            "interpret_task",
            return_value={
                "status": "unsupported",
                "operation": None,
                "inputs": {},
                "missing_inputs": [],
            },
        ):
            result = handle_task("hello how are you?")

        self.assertEqual(result["answer"], orchestrator.DECLINE_MESSAGE)
        self.assertIn("plan", self._trace_types(result))
        self.assertIn("answer", self._trace_types(result))

    def test_missing_inputs_are_handled(self) -> None:
        with patch.object(
            orchestrator,
            "interpret_task",
            return_value={
                "status": "needs_input",
                "operation": "speed",
                "inputs": {},
                "missing_inputs": ["distance_km", "time_hr"],
            },
        ):
            result = handle_task("calculate speed")

        self.assertIn("distance_km", result["answer"])
        self.assertIn("time_hr", result["answer"])
        self.assertIn("need more information", result["answer"].lower())

    def test_trace_contains_major_pipeline_stages(self) -> None:
        factory_payload = {
            "success": True,
            "tool_name": "speed",
            "code": self._speed_code(),
            "attempts": 1,
            "error": None,
        }
        tests = TestGenerationResult(
            success=True,
            tool_name="speed",
            tests=[
                GeneratedTestCase(
                    inputs={"distance_km": 120, "time_hr": 2},
                    category="valid",
                    expectation_status="known",
                    expected=60,
                )
            ],
        )
        verification = VerificationResult(status="verified", tool_name="speed")

        with (
            patch.object(
                orchestrator,
                "interpret_task",
                return_value=self._ok_interpretation(),
            ),
            patch.object(orchestrator, "create_tool", return_value=factory_payload),
            patch.object(orchestrator, "generate_tests", return_value=tests),
            patch.object(orchestrator, "verify_tool", return_value=verification),
        ):
            result = handle_task("compute speed from distance and time")

        types = set(self._trace_types(result))
        for required in ("plan", "check", "no_tool", "writing", "testing", "pass", "answer"):
            self.assertIn(required, types)

    def test_direct_arithmetic_still_works(self) -> None:
        with patch.object(orchestrator, "interpret_task") as interpret_mock:
            result = handle_task("2+5")

        interpret_mock.assert_not_called()
        self.assertEqual(result["answer"], "7")
        self.assertEqual(result["trace"][0]["type"], "plan")
        self.assertIn("trivial", result["trace"][0]["label"].lower())
        self.assertEqual(result["trace"][-1]["type"], "answer")


if __name__ == "__main__":
    unittest.main()
