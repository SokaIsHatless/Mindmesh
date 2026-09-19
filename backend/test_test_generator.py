"""Unit tests for the independent ToolSpec test generator."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

try:
    from . import test_generator
    from .test_generator import (
        GeneratedTestCase,
        build_generation_prompt,
        generate_tests,
        validate_and_build_tests,
    )
    from models import InputSpec, OutputSpec, TestCase, ToolSpec
except ImportError:  # ``unittest discover -s backend``
    import test_generator
    from test_generator import (
        GeneratedTestCase,
        build_generation_prompt,
        generate_tests,
        validate_and_build_tests,
    )
    from models import InputSpec, OutputSpec, TestCase, ToolSpec


class TestGeneratorTests(unittest.TestCase):
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

    def test_prompt_uses_toolspec_metadata_not_code(self) -> None:
        prompt = build_generation_prompt(self._speed_spec())
        self.assertIn("compute speed from distance and time", prompt)
        self.assertIn("distance_km", prompt)
        self.assertIn("speed_kmh", prompt)
        self.assertIn("time_hr must not be zero", prompt)
        self.assertIn("Do NOT invent or include expected outputs", prompt)
        self.assertNotIn("def speed", prompt)

    def test_examples_become_known_cases(self) -> None:
        tests = validate_and_build_tests([], self._speed_spec())
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0].expectation_status, "known")
        self.assertEqual(tests[0].expected, 60)
        self.assertIsInstance(tests[0].test_case, TestCase)

    def test_model_expected_values_are_discarded(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 90.0, "time_hr": 3.0},
                "category": "valid",
                "expected": 999,  # invented — must be ignored
            }
        ]
        tests = validate_and_build_tests(raw, self._speed_spec())
        added = [item for item in tests if item.inputs["distance_km"] == 90.0][0]
        self.assertEqual(added.expectation_status, "unknown")
        self.assertIsNone(added.expected)

    def test_matching_example_inputs_keep_known_expected(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 120, "time_hr": 2},
                "category": "valid",
            }
        ]
        with self.assertRaises(ValueError):
            # Exact duplicate of the seeded example must be rejected.
            validate_and_build_tests(raw, self._speed_spec())

    def test_reject_unknown_inputs(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 10.0, "time_hr": 1.0, "extra": 1.0},
                "category": "valid",
            }
        ]
        with self.assertRaises(ValueError):
            validate_and_build_tests(raw, self._speed_spec())

    def test_reject_invalid_types(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": "far", "time_hr": 1.0},
                "category": "valid",
            }
        ]
        with self.assertRaises(ValueError):
            validate_and_build_tests(raw, self._speed_spec())

    def test_reject_duplicate_candidates(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 10.0, "time_hr": 1.0},
                "category": "valid",
            },
            {
                "inputs": {"distance_km": 10.0, "time_hr": 1.0},
                "category": "boundary",
            },
        ]
        with self.assertRaises(ValueError):
            validate_and_build_tests(raw, self._speed_spec())

    def test_constraint_invalid_requires_known_constraint(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 10.0, "time_hr": 0.0},
                "category": "constraint_invalid",
                "constraint": "time_hr must not be zero",
            }
        ]
        tests = validate_and_build_tests(raw, self._speed_spec())
        invalid = [item for item in tests if item.category == "constraint_invalid"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].expectation_status, "unknown")

        bad = [
            {
                "inputs": {"distance_km": 10.0, "time_hr": 0.0},
                "category": "constraint_invalid",
                "constraint": "not a real constraint",
            }
        ]
        with self.assertRaises(ValueError):
            validate_and_build_tests(bad, self._speed_spec())

    def test_paraphrased_constraint_time_hr_must_be_positive_is_rejected(
        self,
    ) -> None:
        """Regression: LLM must not rewrite ToolSpec constraint text."""
        prompt = build_generation_prompt(self._speed_spec())
        self.assertIn("copy these strings VERBATIM", prompt)
        self.assertIn("Never paraphrase", prompt)
        self.assertIn('"time_hr must not be zero"', prompt)

        paraphrased = [
            {
                "inputs": {"distance_km": 10.0, "time_hr": 0.0},
                "category": "constraint_invalid",
                "constraint": "time_hr must be positive",
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_and_build_tests(paraphrased, self._speed_spec())
        self.assertIn("time_hr must be positive", str(ctx.exception))
        self.assertIn("exact entry", str(ctx.exception))

    def test_boundary_and_valid_categories(self) -> None:
        raw = [
            {
                # Zero is a boundary implied by "time_hr must not be zero".
                "inputs": {"distance_km": 10.0, "time_hr": 0.0},
                "category": "boundary",
            },
            {
                "inputs": {"distance_km": 50.0, "time_hr": 2.0},
                "category": "valid",
            },
        ]
        tests = validate_and_build_tests(raw, self._speed_spec())
        categories = {item.category for item in tests}
        self.assertIn("boundary", categories)
        self.assertIn("valid", categories)

    def test_time_hr_zero_is_valid_constraint_violation(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 10.0, "time_hr": 0.0},
                "category": "constraint_invalid",
                "constraint": "time_hr must not be zero",
            }
        ]
        tests = validate_and_build_tests(raw, self._speed_spec())
        invalid = [item for item in tests if item.category == "constraint_invalid"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].inputs["time_hr"], 0.0)
        self.assertEqual(invalid[0].expectation_status, "unknown")
        self.assertIsNone(invalid[0].expected)

    def test_time_hr_one_must_not_be_constraint_invalid(self) -> None:
        raw = [
            {
                "inputs": {"distance_km": 0.0, "time_hr": 1.0},
                "category": "constraint_invalid",
                "constraint": "time_hr must not be zero",
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_and_build_tests(raw, self._speed_spec())
        self.assertIn("do not violate", str(ctx.exception))

    def test_unspecified_boundaries_must_not_be_invented(self) -> None:
        # distance_km=0 is not implied by any ToolSpec constraint.
        invented = [
            {
                "inputs": {"distance_km": 0.0, "time_hr": 1.0},
                "category": "boundary",
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_and_build_tests(invented, self._speed_spec())
        self.assertIn("invents a boundary", str(ctx.exception))

        # No constraints → no defined boundaries at all.
        unconstrained = ToolSpec(
            name="add",
            purpose="add two numbers",
            inputs=[
                InputSpec(name="a", type="float"),
                InputSpec(name="b", type="float"),
            ],
            output=OutputSpec(name="sum", type="number"),
            constraints=[],
        )
        with self.assertRaises(ValueError) as ctx2:
            validate_and_build_tests(
                [{"inputs": {"a": 0.0, "b": 1.0}, "category": "boundary"}],
                unconstrained,
            )
        self.assertIn("no constraint-derived boundaries", str(ctx2.exception))

    def test_generate_tests_success_with_mocked_model(self) -> None:
        payload = {
            "tests": [
                {
                    "inputs": {"distance_km": 10.0, "time_hr": 2.0},
                    "category": "valid",
                },
                {
                    "inputs": {"distance_km": 10.0, "time_hr": 0.0},
                    "category": "constraint_invalid",
                    "constraint": "time_hr must not be zero",
                },
            ]
        }
        with patch.object(
            test_generator, "call_model", return_value=json.dumps(payload)
        ):
            result = generate_tests(self._speed_spec())

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(len(result.tests), 3)
        self.assertTrue(
            any(item.expectation_status == "known" for item in result.tests)
        )
        self.assertTrue(
            any(item.expectation_status == "unknown" for item in result.tests)
        )

    def test_generate_tests_handles_model_failure(self) -> None:
        with patch.object(
            test_generator,
            "call_model",
            side_effect=RuntimeError("Could not reach Ollama"),
        ):
            result = generate_tests(self._speed_spec())

        self.assertFalse(result.success)
        self.assertEqual(result.tests, [])
        self.assertIn("Could not reach Ollama", result.error or "")

    def test_generate_tests_rejects_malformed_output(self) -> None:
        with patch.object(test_generator, "call_model", return_value="not-json"):
            result = generate_tests(self._speed_spec())

        self.assertFalse(result.success)
        self.assertTrue(result.error)

    def test_invalid_tool_spec_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ToolSpec(name="incomplete")

        result = generate_tests("nope")  # type: ignore[arg-type]
        self.assertFalse(result.success)

    def test_generated_case_model_forbids_invented_expected_when_unknown(self) -> None:
        case = GeneratedTestCase(
            inputs={"distance_km": 1.0, "time_hr": 1.0},
            category="valid",
            expectation_status="unknown",
            expected=123,
        )
        self.assertIsNone(case.expected)
        self.assertIsNone(case.test_case)


if __name__ == "__main__":
    unittest.main()
