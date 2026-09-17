"""Focused unit tests for ToolSpec-native tool_factory generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

try:
    # Package form: ``py -m unittest backend.test_tool_factory``
    # Import tool_factory first so it places ``backend/`` on sys.path; then use
    # the same flat ``models`` import the factory uses (one ToolSpec class).
    from . import tool_factory
    from .tool_factory import (
        adapt_legacy_task_spec,
        build_initial_prompt,
        coerce_tool_spec,
        create_tool,
    )
    from models import InputSpec, OutputSpec, TestCase, ToolSpec
except ImportError:  # Supports ``unittest discover -s backend``.
    import tool_factory
    from tool_factory import (
        adapt_legacy_task_spec,
        build_initial_prompt,
        coerce_tool_spec,
        create_tool,
    )
    from models import InputSpec, OutputSpec, TestCase, ToolSpec


class ToolFactoryToolSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.toolbox_dir = self.root / "toolbox"
        self.manifest_path = self.toolbox_dir / "manifest.json"
        self._toolbox_patch = patch.object(tool_factory, "TOOLBOX_DIR", self.toolbox_dir)
        self._manifest_patch = patch.object(
            tool_factory, "MANIFEST_PATH", self.manifest_path
        )
        self._toolbox_patch.start()
        self._manifest_patch.start()

    def tearDown(self) -> None:
        self._manifest_patch.stop()
        self._toolbox_patch.stop()
        self._temporary_directory.cleanup()

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
                TestCase(inputs={"distance_km": 90, "time_hr": 3}, expected=30),
            ],
            allowed_dependencies=["math"],
        )

    def test_generate_tool_from_valid_tool_spec(self) -> None:
        generated = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )

        with patch.object(tool_factory, "call_model", return_value=generated):
            result = create_tool(self._speed_spec())

        self.assertTrue(result["success"])
        self.assertEqual(result["tool_name"], "speed")
        self.assertEqual(result["attempts"], 1)
        self.assertIsNone(result["error"])
        self.assertIn("def speed", result["code"])
        self.assertTrue((self.toolbox_dir / "speed.py").is_file())

    def test_required_input_and_output_reflected_in_prompt_and_code(self) -> None:
        spec = self._speed_spec()
        prompt = build_initial_prompt(spec)

        self.assertIn("distance_km", prompt)
        self.assertIn("time_hr", prompt)
        self.assertIn("speed_kmh", prompt)
        self.assertIn("float", prompt)
        self.assertIn("km/h", prompt)
        self.assertIn("time_hr must not be zero", prompt)
        self.assertIn(spec.purpose, prompt)

        generated = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        with patch.object(tool_factory, "call_model", return_value=generated) as mocked:
            result = create_tool(spec)

        sent_prompt = mocked.call_args.args[0]
        self.assertIn("distance_km", sent_prompt)
        self.assertIn("speed_kmh", sent_prompt)
        self.assertIn("def speed(distance_km, time_hr)", result["code"])
        self.assertTrue(result["success"])

    def test_invalid_tool_spec_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ToolSpec(name="incomplete")

        result = create_tool({"name": "broken"})
        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], 0)
        self.assertIn("invalid tool spec", result["error"])

        result_none = create_tool(None)  # type: ignore[arg-type]
        self.assertFalse(result_none["success"])
        self.assertEqual(result_none["attempts"], 0)

    def test_generation_failure_is_handled_cleanly(self) -> None:
        with patch.object(
            tool_factory,
            "call_model",
            side_effect=RuntimeError("Could not reach Ollama"),
        ):
            result = create_tool(self._speed_spec())

        self.assertFalse(result["success"])
        self.assertEqual(result["tool_name"], "speed")
        self.assertEqual(result["attempts"], 1)
        self.assertIn("Could not reach Ollama", result["error"])
        self.assertEqual(result["code"], "")

    def test_sandbox_failure_exhausts_retries_cleanly(self) -> None:
        bad_code = "def speed(distance_km, time_hr):\n    return distance_km * time_hr\n"

        with patch.object(tool_factory, "call_model", return_value=bad_code):
            result = create_tool(self._speed_spec())

        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], tool_factory.MAX_RETRIES)
        self.assertIn("failed after", result["error"])
        self.assertIn("def speed", result["code"])

    def test_legacy_task_spec_adapter_still_works(self) -> None:
        legacy = {
            "name": "speed",
            "description": "compute speed from distance and time",
            "inputs": ["distance_km", "time_hr"],
            "tests": [
                {"args": [120, 2], "expected": 60},
                {"args": [90, 3], "expected": 30},
            ],
        }
        adapted = adapt_legacy_task_spec(legacy)
        self.assertIsInstance(adapted, ToolSpec)
        self.assertEqual(adapted.name, "speed")
        self.assertEqual(adapted.purpose, legacy["description"])
        self.assertEqual(
            [item.name for item in adapted.inputs], ["distance_km", "time_hr"]
        )

        generated = (
            "def speed(distance_km, time_hr):\n"
            "    return distance_km / time_hr\n"
        )
        with patch.object(tool_factory, "call_model", return_value=generated):
            result = create_tool(legacy)

        self.assertTrue(result["success"])
        self.assertEqual(result["tool_name"], "speed")

    def test_coerce_accepts_tool_spec_dict_payload(self) -> None:
        payload = self._speed_spec().model_dump(mode="json")
        coerced = coerce_tool_spec(payload)
        self.assertEqual(coerced.name, "speed")
        self.assertEqual(coerced.output.name, "speed_kmh")


if __name__ == "__main__":
    unittest.main()
