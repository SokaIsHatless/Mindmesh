"""Unit tests for deterministic CalculationRequest normalization."""

from __future__ import annotations

import unittest

from models import CalculationRequest
from normalizer import NormalizedRequest, normalize_request


class NormalizerTests(unittest.TestCase):
    def _req(
        self,
        *,
        operation: str | None = None,
        inputs: dict[str, float] | None = None,
        missing_inputs: list[str] | None = None,
        status: str = "ok",
    ) -> CalculationRequest:
        return CalculationRequest(
            status=status,  # type: ignore[arg-type]
            operation=operation,
            inputs=inputs or {},
            missing_inputs=missing_inputs or [],
        )

    def test_velocity_aliases_to_speed(self) -> None:
        result = normalize_request(
            self._req(operation="velocity", inputs={"distance_km": 120, "time_hr": 2})
        )
        self.assertEqual(result.operation, "speed")

    def test_prob_aliases_to_probability(self) -> None:
        result = normalize_request(
            self._req(operation="prob", inputs={"favorable": 1, "total": 4})
        )
        self.assertEqual(result.operation, "probability")
        self.assertEqual(result.inputs, {"favorable": 1.0, "total": 4.0})

    def test_height_meters_to_centimeters(self) -> None:
        result = normalize_request(self._req(operation="bmi", inputs={"height_m": 1.8}))
        self.assertEqual(result.inputs["height_cm"], 180.0)

    def test_minutes_to_hours(self) -> None:
        result = normalize_request(
            self._req(operation="speed", inputs={"time_minutes": 30})
        )
        self.assertAlmostEqual(result.inputs["time_hr"], 0.5)

    def test_bmi_normalization(self) -> None:
        result = normalize_request(
            self._req(
                operation="bmi",
                inputs={"height_m": 1.8, "weight_kg": 70},
            )
        )
        self.assertEqual(result.operation, "bmi")
        self.assertEqual(result.inputs["height_cm"], 180.0)
        self.assertEqual(result.inputs["weight_kg"], 70.0)
        self.assertEqual(result.original_inputs, {"height_m": 1.8, "weight_kg": 70.0})

    def test_speed_normalization(self) -> None:
        result = normalize_request(
            self._req(
                operation="velocity",
                inputs={"distance": 120, "time_min": 30},
            )
        )
        self.assertEqual(result.operation, "speed")
        self.assertEqual(result.inputs["distance_km"], 120.0)
        self.assertAlmostEqual(result.inputs["time_hr"], 0.5)

    def test_missing_values_remain_missing(self) -> None:
        result = normalize_request(
            self._req(
                status="needs_input",
                operation="bmi",
                inputs={"weight": 70},
                missing_inputs=["height"],
            )
        )
        self.assertEqual(result.inputs, {"weight_kg": 70.0})
        self.assertEqual(result.missing_inputs, ["height_cm"])
        self.assertNotIn("height_cm", result.inputs)

    def test_original_inputs_are_preserved(self) -> None:
        raw = {"height_m": 1.8, "weight": 68.5}
        result = normalize_request(self._req(operation="bmi", inputs=raw))
        self.assertEqual(result.original_inputs, {"height_m": 1.8, "weight": 68.5})
        self.assertEqual(result.inputs["height_cm"], 180.0)
        self.assertEqual(result.inputs["weight_kg"], 68.5)

    def test_duplicate_canonical_names_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_request(
                self._req(
                    operation="bmi",
                    inputs={"height": 180, "height_cm": 170},
                )
            )
        with self.assertRaises(ValueError):
            normalize_request(
                self._req(
                    operation="bmi",
                    inputs={"height_m": 1.8, "height_cm": 180},
                )
            )

    def test_invalid_and_nonfinite_values_are_rejected(self) -> None:
        # Bypass CalculationRequest validators to exercise normalizer checks.
        bad = CalculationRequest.model_construct(
            status="ok",
            operation="bmi",
            inputs={"height_cm": float("nan")},
            missing_inputs=[],
            error=None,
        )
        with self.assertRaises(ValueError):
            normalize_request(bad)

        overflow = CalculationRequest.model_construct(
            status="ok",
            operation="speed",
            inputs={"distance_km": float("inf")},
            missing_inputs=[],
            error=None,
        )
        with self.assertRaises(ValueError):
            normalize_request(overflow)

        with self.assertRaises(ValueError):
            normalize_request(
                self._req(operation="bmi", inputs={"height_furlong": 2.0})
            )

    def test_common_unit_spellings(self) -> None:
        metre = normalize_request(self._req(inputs={"height_metre": 1.8}))
        self.assertEqual(metre.inputs["height_cm"], 180.0)

        centimeter = normalize_request(self._req(inputs={"height_centimeter": 180}))
        self.assertEqual(centimeter.inputs["height_cm"], 180.0)

        kilometre = normalize_request(self._req(inputs={"distance_kilometre": 12}))
        self.assertEqual(kilometre.inputs["distance_km"], 12.0)

        kilogram = normalize_request(self._req(inputs={"weight_kilogram": 50}))
        self.assertEqual(kilogram.inputs["weight_kg"], 50.0)

        hour = normalize_request(self._req(inputs={"time_hour": 2}))
        self.assertEqual(hour.inputs["time_hr"], 2.0)

    def test_returns_normalized_request_model(self) -> None:
        result = normalize_request(self._req(operation="speed", inputs={"distance": 10}))
        self.assertIsInstance(result, NormalizedRequest)


if __name__ == "__main__":
    unittest.main()
