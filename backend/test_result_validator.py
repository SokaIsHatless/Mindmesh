"""Unit tests for deterministic ToolSpec result validation."""

from __future__ import annotations

import math
import unittest

try:
    from .result_validator import ResultValidationResult, validate_result
    from models import InputSpec, OutputSpec, ToolSpec
except ImportError:  # ``unittest discover -s backend``
    from result_validator import ResultValidationResult, validate_result
    from models import InputSpec, OutputSpec, ToolSpec


class ResultValidatorTests(unittest.TestCase):
    def _speed_spec(self) -> ToolSpec:
        return ToolSpec(
            name="speed",
            purpose="compute speed from distance and time",
            inputs=[
                InputSpec(name="distance_km", type="float", unit="km"),
                InputSpec(name="time_hr", type="float", unit="hr"),
            ],
            output=OutputSpec(name="speed_kmh", type="number", unit="km/h"),
        )

    def _bmi_spec(self) -> ToolSpec:
        return ToolSpec(
            name="bmi",
            purpose="compute body mass index",
            inputs=[
                InputSpec(name="weight_kg", type="float", unit="kg"),
                InputSpec(name="height_m", type="float", unit="m"),
            ],
            output=OutputSpec(name="bmi", type="number", unit="kg/m^2"),
        )

    def test_valid_numeric_result(self) -> None:
        result = validate_result(self._speed_spec(), 60.0, unit="km/h")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.value, 60.0)
        self.assertEqual(result.normalized_value, 60.0)
        self.assertEqual(result.errors, [])
        self.assertIsInstance(result, ResultValidationResult)

    def test_wrong_output_type(self) -> None:
        result = validate_result(self._speed_spec(), "sixty", unit="km/h")
        self.assertEqual(result.status, "invalid")
        self.assertTrue(any("wrong output type" in err for err in result.errors))

    def test_missing_output(self) -> None:
        result = validate_result(self._speed_spec(), None)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(any("missing output" in err for err in result.errors))

        empty = validate_result(self._speed_spec(), {"ok": True})
        self.assertEqual(empty.status, "invalid")
        self.assertTrue(any("missing output" in err for err in empty.errors))

    def test_non_finite_numeric_result(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                result = validate_result(self._speed_spec(), bad, unit="km/h")
                self.assertEqual(result.status, "invalid")
                self.assertTrue(
                    any("non-finite" in err for err in result.errors)
                )

    def test_unit_mismatch(self) -> None:
        result = validate_result(self._speed_spec(), 60.0, unit="m/s")
        self.assertEqual(result.status, "invalid")
        self.assertTrue(any("unit mismatch" in err for err in result.errors))

    def test_valid_compatible_units(self) -> None:
        for unit in ("km/h", "kmh", "kph", "km/hr"):
            with self.subTest(unit=unit):
                result = validate_result(self._speed_spec(), 60.0, unit=unit)
                self.assertEqual(result.status, "valid")
                self.assertEqual(result.normalized_unit, "km/h")

        structured = validate_result(
            self._speed_spec(),
            {"value": 60.0, "unit": "kmh"},
        )
        self.assertEqual(structured.status, "valid")
        self.assertEqual(structured.normalized_unit, "km/h")

    def test_floating_point_tolerance(self) -> None:
        # 10/3 with expected nearby — within abs_tol 1e-6.
        result = validate_result(
            self._speed_spec(),
            10.0 / 3.0,
            expected=3.333333333,
            unit="km/h",
        )
        self.assertEqual(result.status, "valid")

        far = validate_result(
            self._speed_spec(),
            10.0 / 3.0,
            expected=3.5,
            unit="km/h",
        )
        self.assertEqual(far.status, "invalid")
        self.assertTrue(any("value mismatch" in err for err in far.errors))

    def test_execution_error(self) -> None:
        result = validate_result(
            self._speed_spec(),
            {"ok": False, "error": "division by zero"},
        )
        self.assertEqual(result.status, "error")
        self.assertTrue(any("execution error" in err for err in result.errors))
        self.assertIn("division by zero", result.errors[0])

        result2 = validate_result(self._speed_spec(), RuntimeError("boom"))
        self.assertEqual(result2.status, "error")
        self.assertIn("boom", result2.errors[0])

    def test_generic_speed_and_bmi_outputs(self) -> None:
        speed = validate_result(
            self._speed_spec(),
            {"value": 60.0, "unit": "km/h"},
        )
        self.assertEqual(speed.status, "valid")
        self.assertEqual(speed.normalized_value, 60.0)
        self.assertEqual(speed.normalized_unit, "km/h")

        bmi_value = 70 / (1.75 * 1.75)
        bmi = validate_result(
            self._bmi_spec(),
            {"bmi": bmi_value, "unit": "kg/m2"},
        )
        self.assertEqual(bmi.status, "valid")
        self.assertEqual(bmi.normalized_unit, "kg/m^2")
        self.assertTrue(
            abs(float(bmi.normalized_value) - bmi_value) < 1e-9
        )

        bmi_mismatch = validate_result(
            self._bmi_spec(),
            {"value": bmi_value, "unit": "km/h"},
        )
        self.assertEqual(bmi_mismatch.status, "invalid")
        self.assertTrue(
            any("unit mismatch" in err for err in bmi_mismatch.errors)
        )


if __name__ == "__main__":
    unittest.main()
