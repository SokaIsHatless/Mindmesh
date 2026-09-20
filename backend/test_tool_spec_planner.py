"""Regression tests for relationship-aware ToolSpec planning."""

from __future__ import annotations

import math
import unittest

try:
    from .tool_spec_planner import plan_tool_spec
except ImportError:  # ``unittest discover -s backend``
    from tool_spec_planner import plan_tool_spec


class ToolSpecPlannerTests(unittest.TestCase):
    def test_bmi_from_weight_and_height_toolspec(self) -> None:
        task = "height = 150cm, weight = 50kg calculate bmi"
        spec = plan_tool_spec(
            task=task,
            operation="bmi",
            inputs={"height_cm": 150, "weight_kg": 50},
            requested_output="bmi",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "bmi")
        self.assertEqual(spec.operation, "bmi")
        self.assertEqual([i.name for i in spec.inputs], ["height_cm", "weight_kg"])
        self.assertEqual(spec.output.name, "bmi")
        self.assertNotIn("20", spec.name)
        self.assertNotIn("150", spec.name)
        self.assertEqual(len(spec.examples), 1)
        self.assertEqual(
            spec.examples[0].inputs, {"height_cm": 150.0, "weight_kg": 50.0}
        )
        expected = 50.0 / ((150.0 / 100.0) ** 2)
        self.assertAlmostEqual(float(spec.examples[0].expected), expected, places=5)

    def test_height_from_bmi_and_weight_toolspec(self) -> None:
        task = "If BMI is 20 and weight is 80 kg, what is the height?"
        spec = plan_tool_spec(
            task=task,
            operation="height_from_bmi",
            inputs={"bmi": 20, "weight_kg": 80},
            requested_output="height_cm",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "height_from_bmi")
        self.assertEqual(spec.operation, "height_from_bmi")
        self.assertEqual([i.name for i in spec.inputs], ["bmi", "weight_kg"])
        self.assertEqual(spec.output.name, "height_cm")
        self.assertEqual(spec.output.unit, "cm")
        # Must not slug user prose / embed literal values into the tool name.
        self.assertNotEqual(spec.name, "if_bmi_is_20")
        self.assertFalse(any(ch.isdigit() for ch in spec.name))
        self.assertNotIn("x", [i.name for i in spec.inputs])
        self.assertEqual(
            spec.examples[0].inputs, {"bmi": 20.0, "weight_kg": 80.0}
        )
        # height_m = sqrt(80/20) = 2 → 200 cm
        self.assertEqual(spec.examples[0].expected, 200)
        self.assertAlmostEqual(
            math.sqrt(80.0 / 20.0) * 100.0, float(spec.examples[0].expected)
        )

    def test_weight_from_bmi_and_height_toolspec(self) -> None:
        task = "BMI is 22 and height is 170 cm; find the weight"
        spec = plan_tool_spec(
            task=task,
            operation="weight_from_bmi",
            inputs={"bmi": 22, "height_cm": 170},
            requested_output="weight_kg",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "weight_from_bmi")
        self.assertEqual([i.name for i in spec.inputs], ["bmi", "height_cm"])
        self.assertEqual(spec.output.name, "weight_kg")
        self.assertFalse(any(ch.isdigit() for ch in spec.name))
        expected = 22.0 * ((170.0 / 100.0) ** 2)
        self.assertAlmostEqual(float(spec.examples[0].expected), expected, places=5)
        self.assertEqual(
            spec.examples[0].inputs, {"bmi": 22.0, "height_cm": 170.0}
        )

    def test_resolves_operation_from_inputs_when_omitted(self) -> None:
        spec = plan_tool_spec(
            task="find height",
            operation=None,
            inputs={"bmi": 20, "weight_kg": 80},
            requested_output="height_cm",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "height_from_bmi")
        self.assertEqual(spec.examples[0].expected, 200)

    def test_unknown_operation_returns_none(self) -> None:
        spec = plan_tool_spec(
            task="do something weird",
            operation="not_a_real_op",
            inputs={"foo": 1.0},
            requested_output=None,
        )
        self.assertIsNone(spec)

    def test_forward_bmi_name_stable_across_different_values(self) -> None:
        a = plan_tool_spec(
            task="bmi for 160cm 60kg",
            operation="bmi",
            inputs={"height_cm": 160, "weight_kg": 60},
            requested_output="bmi",
        )
        b = plan_tool_spec(
            task="bmi for 180cm 90kg",
            operation="bmi",
            inputs={"height_cm": 180, "weight_kg": 90},
            requested_output="bmi",
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        assert a is not None and b is not None
        self.assertEqual(a.name, b.name)
        self.assertEqual(a.name, "bmi")
        self.assertNotEqual(a.examples[0].expected, b.examples[0].expected)


if __name__ == "__main__":
    unittest.main()
