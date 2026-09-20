"""Unit tests for the interpreter (inverse relationships + unknown detection)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

try:
    from . import interpreter
    from .interpreter import _validate_result, interpret_task
    from .models import CalculationRequest
except ImportError:  # ``unittest discover -s backend``
    import interpreter
    from interpreter import _validate_result, interpret_task
    from models import CalculationRequest


class InterpreterRelationshipTests(unittest.TestCase):
    def test_bmi_from_weight_and_height(self) -> None:
        result = _validate_result(
            {
                "status": "ok",
                "operation": "bmi",
                "inputs": {"height_cm": 150, "weight_kg": 50},
                "missing_inputs": [],
                "requested_output": "bmi",
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["operation"], "bmi")
        self.assertEqual(result["inputs"]["height_cm"], 150)
        self.assertEqual(result["inputs"]["weight_kg"], 50)
        self.assertEqual(result["missing_inputs"], [])
        self.assertEqual(result["requested_output"], "bmi")
        CalculationRequest.model_validate(result)

    def test_height_from_bmi_and_weight(self) -> None:
        """Regression: BMI + weight asks for height — height is unknown, not missing."""
        result = _validate_result(
            {
                "status": "needs_input",
                "operation": "bmi",
                "inputs": {"bmi": 20, "weight_kg": 80},
                # Model wrongly lists the unknown (and provided bmi) as missing.
                "missing_inputs": ["bmi", "height_cm"],
                "requested_output": None,
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["operation"], "height_from_bmi")
        self.assertEqual(result["inputs"], {"bmi": 20, "weight_kg": 80})
        self.assertEqual(result["missing_inputs"], [])
        self.assertEqual(result["requested_output"], "height_cm")
        self.assertNotIn("height_cm", result["inputs"])
        self.assertNotIn("height_cm", result["missing_inputs"])
        CalculationRequest.model_validate(result)

    def test_weight_from_bmi_and_height(self) -> None:
        result = _validate_result(
            {
                "status": "ok",
                "operation": "bmi",
                "inputs": {"bmi": 22, "height_cm": 170},
                "missing_inputs": [],
                "requested_output": "weight_kg",
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["operation"], "weight_from_bmi")
        self.assertEqual(result["inputs"], {"bmi": 22, "height_cm": 170})
        self.assertEqual(result["missing_inputs"], [])
        self.assertEqual(result["requested_output"], "weight_kg")
        CalculationRequest.model_validate(result)

    def test_missing_information_for_height_from_bmi(self) -> None:
        result = _validate_result(
            {
                "status": "needs_input",
                "operation": "height_from_bmi",
                "inputs": {"bmi": 20},
                "missing_inputs": ["height_cm", "weight_kg"],
                "requested_output": "height_cm",
            }
        )
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["operation"], "height_from_bmi")
        self.assertEqual(result["inputs"], {"bmi": 20})
        self.assertEqual(result["requested_output"], "height_cm")
        # Unknown must not be listed as missing; weight is genuinely required.
        self.assertEqual(result["missing_inputs"], ["weight_kg"])
        self.assertNotIn("height_cm", result["missing_inputs"])
        CalculationRequest.model_validate(result)

    def test_unsupported_request(self) -> None:
        result = _validate_result(
            {
                "status": "unsupported",
                "operation": "bmi",
                "inputs": {"bmi": 20},
                "missing_inputs": ["height_cm"],
                "requested_output": "height_cm",
            }
        )
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["operation"])
        self.assertEqual(result["inputs"], {})
        self.assertEqual(result["missing_inputs"], [])
        self.assertIsNone(result["requested_output"])

    def test_forward_speed_still_works(self) -> None:
        result = _validate_result(
            {
                "status": "ok",
                "operation": "speed",
                "inputs": {"distance_km": 120, "time_hr": 2},
                "missing_inputs": [],
                "requested_output": "speed_kmh",
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["operation"], "speed")
        self.assertEqual(result["inputs"], {"distance_km": 120, "time_hr": 2})
        self.assertEqual(result["requested_output"], "speed_kmh")

    def test_interpret_task_height_from_bmi_with_mocked_model(self) -> None:
        payload = {
            "status": "needs_input",
            "operation": "bmi",
            "inputs": {"bmi": 20, "weight_kg": 80},
            "missing_inputs": ["height_cm"],
            "requested_output": "height_cm",
        }
        with patch.object(
            interpreter, "_call_ollama", return_value=json.dumps(payload)
        ):
            result = interpret_task(
                "If BMI is 20 and weight is 80 kg, what is the height?"
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["operation"], "height_from_bmi")
        self.assertEqual(result["inputs"]["bmi"], 20)
        self.assertEqual(result["inputs"]["weight_kg"], 80)
        self.assertEqual(result["missing_inputs"], [])
        self.assertEqual(result["requested_output"], "height_cm")

    def test_interpret_task_unsupported_with_mocked_model(self) -> None:
        payload = {
            "status": "unsupported",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "requested_output": None,
        }
        with patch.object(
            interpreter, "_call_ollama", return_value=json.dumps(payload)
        ):
            result = interpret_task("hello how are you?")

        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["operation"])
        self.assertEqual(result["inputs"], {})

    def test_calculation_request_rejects_unknown_listed_as_missing(self) -> None:
        with self.assertRaises(Exception):
            CalculationRequest(
                status="needs_input",
                operation="height_from_bmi",
                inputs={"bmi": 20},
                missing_inputs=["height_cm", "weight_kg"],
                requested_output="height_cm",
            )


if __name__ == "__main__":
    unittest.main()
