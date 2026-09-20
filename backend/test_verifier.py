"""Unit tests for the independent tool verification pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from . import verifier
    from .test_generator import GeneratedTestCase, TestGenerationResult
    from .verifier import verify_tool
    from models import InputSpec, OutputSpec, TestCase, ToolSpec
except ImportError:  # ``unittest discover -s backend``
    import verifier
    from test_generator import GeneratedTestCase, TestGenerationResult
    from verifier import verify_tool
    from models import InputSpec, OutputSpec, TestCase, ToolSpec


class VerifierTests(unittest.TestCase):
    def _speed_spec(self) -> ToolSpec:
        return ToolSpec(
            name="speed",
            purpose="compute speed from distance and time",
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

    def _bmi_spec(self) -> ToolSpec:
        return ToolSpec(
            name="bmi",
            purpose="compute body mass index from weight and height",
            inputs=[
                InputSpec(name="weight_kg", type="float", unit="kg"),
                InputSpec(name="height_m", type="float", unit="m"),
            ],
            output=OutputSpec(name="bmi", type="number"),
            constraints=["height_m must not be zero"],
            examples=[
                TestCase(inputs={"weight_kg": 70, "height_m": 1.75}, expected=22.857142857),
            ],
        )

    def _known(
        self, inputs: dict, expected, category: str = "valid"
    ) -> GeneratedTestCase:
        return GeneratedTestCase(
            inputs=inputs,
            category=category,  # type: ignore[arg-type]
            expectation_status="known",
            expected=expected,
        )

    def _unknown(
        self, inputs: dict, category: str = "valid", constraint: str | None = None
    ) -> GeneratedTestCase:
        return GeneratedTestCase(
            inputs=inputs,
            category=category,  # type: ignore[arg-type]
            expectation_status="unknown",
            expected=None,
            constraint=constraint,
        )

    def test_correct_generated_code_is_verified(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = [
            self._known({"distance_km": 120, "time_hr": 2}, 60),
            self._known({"distance_km": 90, "time_hr": 3}, 30),
        ]
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "verified")
        self.assertEqual(len(result.passed_tests), 2)
        self.assertEqual(result.failed_tests, [])
        self.assertEqual(result.errors, [])

    def test_incorrect_generated_code_fails(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km * time_hr\n"
        )
        tests = [self._known({"distance_km": 120, "time_hr": 2}, 60)]
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.failed_tests), 1)
        self.assertEqual(result.passed_tests, [])

    def test_malformed_unsafe_code_fails_static_validation(self) -> None:
        unsafe = (
            "import os\n"
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = [self._known({"distance_km": 120, "time_hr": 2}, 60)]
        result = verify_tool(self._speed_spec(), unsafe, tests)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.errors)
        self.assertIn("static validation", result.errors[0])
        self.assertIn("Disallowed import", result.errors[0])

        malformed = "def speed(distance_km, time_hr)\n    return 1\n"
        result2 = verify_tool(self._speed_spec(), malformed, tests)
        self.assertEqual(result2.status, "failed")
        self.assertTrue(
            any("Syntax error" in err for err in result2.errors)
        )

        missing_fn = "def other(a, b):\n    return a / b\n"
        result3 = verify_tool(self._speed_spec(), missing_fn, tests)
        self.assertEqual(result3.status, "failed")
        self.assertTrue(
            any("missing required function" in err for err in result3.errors)
        )

    def test_sandbox_execution_failure_fails_verification(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = [self._known({"distance_km": 120, "time_hr": 2}, 60)]
        with patch.object(
            verifier,
            "run_tool",
            return_value={
                "ok": False,
                "stdout": "",
                "stderr": "boom",
                "reason": "nonzero exit",
            },
        ):
            result = verify_tool(self._speed_spec(), code, tests)

        self.assertEqual(result.status, "failed")
        self.assertTrue(result.errors)
        self.assertIn("sandbox execution failed", result.errors[0])

    def test_unknown_expected_values_are_not_verifiable(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = [
            self._unknown({"distance_km": 50.0, "time_hr": 2.0}),
            self._unknown(
                {"distance_km": 10.0, "time_hr": 0.0},
                category="constraint_invalid",
                constraint="time_hr must not be zero",
            ),
        ]
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "not_verifiable")
        self.assertEqual(result.passed_tests, [])
        self.assertEqual(result.failed_tests, [])
        self.assertEqual(len(result.skipped_tests), 2)
        self.assertTrue(
            all(item.status == "not_verifiable" for item in result.skipped_tests)
        )

    def test_mixed_pass_and_fail_results(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = [
            self._known({"distance_km": 120, "time_hr": 2}, 60),
            self._known({"distance_km": 100, "time_hr": 2}, 999),  # wrong expected
            self._unknown({"distance_km": 10.0, "time_hr": 1.0}),
        ]
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.passed_tests), 1)
        self.assertEqual(len(result.failed_tests), 1)
        self.assertEqual(len(result.skipped_tests), 1)
        self.assertEqual(result.skipped_tests[0].status, "not_verifiable")

    def test_floating_point_comparison_tolerance(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        # 10/3 ≈ 3.333... — exact float equality would be brittle.
        tests = [self._known({"distance_km": 10.0, "time_hr": 3.0}, 3.333333333)]
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "verified")
        self.assertEqual(len(result.passed_tests), 1)

        # Far outside tolerance must still fail.
        far = [self._known({"distance_km": 10.0, "time_hr": 3.0}, 3.5)]
        result_fail = verify_tool(self._speed_spec(), code, far)
        self.assertEqual(result_fail.status, "failed")

    def test_generic_toolspecs_speed_and_bmi(self) -> None:
        speed_code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        speed_tests = TestGenerationResult(
            success=True,
            tool_name="speed",
            tests=[self._known({"distance_km": 120, "time_hr": 2}, 60)],
        )
        speed_result = verify_tool(self._speed_spec(), speed_code, speed_tests)
        self.assertEqual(speed_result.status, "verified")
        self.assertEqual(speed_result.tool_name, "speed")

        bmi_code = (
            "def bmi(weight_kg, height_m):\n"
            "    return weight_kg / (height_m * height_m)\n"
        )
        bmi_tests = [
            self._known(
                {"weight_kg": 70, "height_m": 1.75},
                70 / (1.75 * 1.75),
            )
        ]
        bmi_result = verify_tool(self._bmi_spec(), bmi_code, bmi_tests)
        self.assertEqual(bmi_result.status, "verified")
        self.assertEqual(bmi_result.tool_name, "bmi")
        self.assertEqual(len(bmi_result.passed_tests), 1)

    def test_failed_test_generation_result_fails(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = TestGenerationResult(
            success=False,
            tool_name="speed",
            tests=[],
            error="model returned invalid JSON",
        )
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "failed")
        self.assertIn("model returned invalid JSON", result.errors[0])

    def test_known_plus_unknown_can_still_verify(self) -> None:
        code = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        tests = [
            self._known({"distance_km": 120, "time_hr": 2}, 60),
            self._unknown({"distance_km": 40.0, "time_hr": 2.0}),
        ]
        result = verify_tool(self._speed_spec(), code, tests)
        self.assertEqual(result.status, "verified")
        self.assertEqual(len(result.passed_tests), 1)
        self.assertEqual(len(result.skipped_tests), 1)


if __name__ == "__main__":
    unittest.main()
