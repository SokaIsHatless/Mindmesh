"""Unit tests for the pure ToolSpec capability-description models."""

import unittest

from pydantic import ValidationError

from models import InputSpec, OutputSpec, TestCase, ToolSpec


class ToolSpecTests(unittest.TestCase):
    def _speed_spec(self) -> ToolSpec:
        return ToolSpec(
            operation="speed",
            name="calculate_speed",
            purpose="Calculate speed from distance and time.",
            inputs=[
                InputSpec(
                    name="distance_km",
                    type="float",
                    description="Distance travelled.",
                    unit="km",
                ),
                InputSpec(name="time_hr", type="float", unit="hr"),
            ],
            output=OutputSpec(name="speed_kmh", type="number", unit="km/h"),
            constraints=["time_hr must not be zero"],
            examples=[
                TestCase(
                    inputs={"distance_km": 120, "time_hr": 2},
                    expected=60,
                )
            ],
            allowed_dependencies=["math"],
        )

    def test_valid_tool_spec_creation(self) -> None:
        spec = self._speed_spec()

        self.assertEqual(spec.name, "calculate_speed")
        self.assertEqual(spec.operation, "speed")
        self.assertEqual(spec.inputs[0].name, "distance_km")
        self.assertEqual(spec.output.name, "speed_kmh")

    def test_missing_required_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ToolSpec(name="incomplete")

    def test_blank_name_and_purpose_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ToolSpec(
                name="   ",
                purpose="valid purpose",
                inputs=[],
                output=OutputSpec(name="result", type="number"),
            )
        with self.assertRaises(ValidationError):
            ToolSpec(
                name="valid_name",
                purpose="   ",
                inputs=[],
                output=OutputSpec(name="result", type="number"),
            )

    def test_duplicate_input_names_are_rejected_after_normalization(self) -> None:
        with self.assertRaises(ValidationError):
            ToolSpec(
                name="duplicate_inputs",
                purpose="Reject duplicate inputs.",
                inputs=[
                    InputSpec(name="value", type="number"),
                    InputSpec(name=" value ", type="number"),
                ],
                output=OutputSpec(name="result", type="number"),
            )

    def test_duplicate_test_input_names_are_rejected_after_normalization(self) -> None:
        with self.assertRaises(ValidationError):
            TestCase(
                inputs={"value": 1, " value ": 2},
                expected=1,
            )

    def test_optional_default_fields_are_valid_and_required_defaults_are_not(self) -> None:
        optional = InputSpec(
            name="precision",
            type="int",
            required=False,
            default=2,
        )
        spec = ToolSpec(
            name="round_number",
            purpose="Round a number to an optional precision.",
            inputs=[InputSpec(name="value", type="number"), optional],
            output=OutputSpec(name="rounded", type="number"),
            examples=[TestCase(inputs={"value": 1.234}, expected=1.23)],
        )

        self.assertEqual(spec.inputs[1].default, 2)
        with self.assertRaises(ValidationError):
            InputSpec(name="value", type="number", default=0)

    def test_constraints_and_test_cases_are_declarative(self) -> None:
        spec = self._speed_spec()

        self.assertEqual(spec.constraints, ["time_hr must not be zero"])
        self.assertEqual(spec.examples[0].inputs["time_hr"], 2)
        self.assertEqual(spec.examples[0].expected, 60)

    def test_invalid_test_cases_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TestCase(inputs=[("value", 1)], expected=1)
        with self.assertRaises(ValidationError):
            TestCase(inputs={"value": 1})
        with self.assertRaises(ValidationError):
            ToolSpec(
                name="bad_example",
                purpose="Reject undeclared example inputs.",
                inputs=[InputSpec(name="value", type="number")],
                output=OutputSpec(name="result", type="number"),
                examples=[TestCase(inputs={"other": 1}, expected=1)],
            )
        with self.assertRaises(ValidationError):
            ToolSpec(
                name="missing_example_input",
                purpose="Reject missing required example inputs.",
                inputs=[InputSpec(name="value", type="number")],
                output=OutputSpec(name="result", type="number"),
                examples=[TestCase(inputs={}, expected=1)],
            )

    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TestCase(inputs={"value": 1}, expected=1, unexpected=True)
        with self.assertRaises(ValidationError):
            ToolSpec(
                name="unknown_field",
                purpose="Reject undeclared fields.",
                inputs=[],
                output=OutputSpec(name="result", type="number"),
                unexpected=True,
            )

    def test_calculate_bmi_is_representable_without_a_special_model(self) -> None:
        bmi = ToolSpec(
            operation="bmi",
            name="calculate_bmi",
            purpose="Calculate body mass index from height and weight.",
            inputs=[
                InputSpec(name="height_cm", type="float", unit="cm"),
                InputSpec(name="weight_kg", type="float", unit="kg"),
            ],
            output=OutputSpec(name="bmi", type="number", unit="kg/m^2"),
            constraints=["height_cm must be greater than zero"],
            examples=[
                TestCase(
                    inputs={"height_cm": 150, "weight_kg": 50},
                    expected=22.22,
                )
            ],
        )

        self.assertEqual(bmi.name, "calculate_bmi")
        self.assertEqual(bmi.examples[0].expected, 22.22)


if __name__ == "__main__":
    unittest.main()
