"""
test_generator.py — Independent test-case generation from a ToolSpec.

Uses Ollama only to propose candidate input scenarios. Expected values are
never invented here: they come only from ToolSpec.examples, or are marked
explicitly unknown. Never inspects or executes generated tool code.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models import InputSpec, TestCase, ToolSpec  # noqa: E402

MODEL = "phi4-mini"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_SEC = 60

TestCategory = Literal["valid", "boundary", "constraint_invalid"]
ExpectationStatus = Literal["known", "unknown"]


class GeneratedTestCase(BaseModel):
    """One structured test case produced for later verification."""

    inputs: dict[str, Any]
    category: TestCategory
    expectation_status: ExpectationStatus
    expected: Any | None = None
    constraint: str | None = None
    # Populated when expectation_status == "known" (from ToolSpec.examples).
    test_case: TestCase | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_expectation_consistency(self) -> "GeneratedTestCase":
        if self.expectation_status == "unknown":
            # Never keep an invented expected value.
            self.expected = None
            self.test_case = None
        elif self.test_case is None:
            self.test_case = TestCase(inputs=self.inputs, expected=self.expected)
        if self.category == "constraint_invalid" and not self.constraint:
            raise ValueError(
                "constraint_invalid cases must name the violated constraint"
            )
        return self


class TestGenerationResult(BaseModel):
    """Outcome of independent test generation for a ToolSpec."""

    success: bool
    tool_name: str
    tests: list[GeneratedTestCase] = Field(default_factory=list)
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


def call_model(prompt: str) -> str:
    """Call local Ollama for JSON candidate tests only."""
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt, "stream": False, "format": "json"}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SEC) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Is `ollama serve` running and is `{MODEL}` pulled? ({exc})"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama timed out after {OLLAMA_TIMEOUT_SEC}s ({exc})"
        ) from exc

    text = body.get("response")
    if not isinstance(text, str):
        raise RuntimeError("Ollama response missing 'response' string field")
    return text


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model response")

    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"tests": data}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def _inputs_fingerprint(inputs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Stable identity for duplicate detection (JSON-normalized values)."""
    items: list[tuple[str, str]] = []
    for key in sorted(inputs):
        items.append((key, json.dumps(inputs[key], sort_keys=True, default=str)))
    return tuple(items)


def _value_matches_type(value: Any, type_name: str) -> bool:
    if type_name in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return False


def _validate_inputs_against_spec(
    inputs: dict[str, Any], tool_spec: ToolSpec
) -> dict[str, Any]:
    """Reject unknown names, missing required fields, and wrong types."""
    if not isinstance(inputs, dict):
        raise ValueError("test inputs must be a dictionary")

    schema = {spec.name: spec for spec in tool_spec.inputs}
    unknown = set(inputs) - set(schema)
    if unknown:
        raise ValueError(
            f"unknown input name(s): {', '.join(sorted(unknown))}"
        )

    required = {spec.name for spec in tool_spec.inputs if spec.required}
    missing = required - set(inputs)
    if missing:
        raise ValueError(
            f"missing required input(s): {', '.join(sorted(missing))}"
        )

    validated: dict[str, Any] = {}
    for name, value in inputs.items():
        input_spec = schema[name]
        if not _value_matches_type(value, input_spec.type):
            raise ValueError(
                f"input '{name}' has invalid type for {input_spec.type}: "
                f"{type(value).__name__}"
            )
        validated[name] = value
    return validated


def _example_expected_lookup(tool_spec: ToolSpec) -> dict[tuple, Any]:
    return {
        _inputs_fingerprint(example.inputs): example.expected
        for example in tool_spec.examples
    }


def _tests_from_examples(tool_spec: ToolSpec) -> list[GeneratedTestCase]:
    """Deterministic seed cases — expected values come only from ToolSpec."""
    generated: list[GeneratedTestCase] = []
    seen: set[tuple] = set()
    for example in tool_spec.examples:
        fingerprint = _inputs_fingerprint(example.inputs)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        generated.append(
            GeneratedTestCase(
                inputs=dict(example.inputs),
                category="valid",
                expectation_status="known",
                expected=example.expected,
                test_case=TestCase(
                    inputs=dict(example.inputs), expected=example.expected
                ),
            )
        )
    return generated


def build_generation_prompt(tool_spec: ToolSpec) -> str:
    """Ask the model for input scenarios only — not derived from tool code."""
    input_lines = []
    for spec in tool_spec.inputs:
        line = f"- {spec.name}: type={spec.type}, required={spec.required}"
        if spec.unit:
            line += f", unit={spec.unit}"
        if spec.description:
            line += f", description={spec.description}"
        input_lines.append(line)

    constraints = list(tool_spec.constraints)
    examples_blob = [
        {"inputs": example.inputs, "expected": example.expected}
        for example in tool_spec.examples
    ]
    constraints_blob = json.dumps(constraints, indent=2)

    return (
        "You generate independent verification scenarios for a computational "
        "capability described ONLY by the metadata below. "
        "Do NOT write or assume any Python implementation.\n\n"
        f"Name: {tool_spec.name}\n"
        f"Purpose: {tool_spec.purpose}\n"
        f"Operation: {tool_spec.operation}\n"
        f"Output: {tool_spec.output.name} ({tool_spec.output.type}"
        f"{f', unit={tool_spec.output.unit}' if tool_spec.output.unit else ''})\n"
        "Inputs:\n"
        + "\n".join(input_lines)
        + "\nConstraints (copy these strings VERBATIM when referencing them; "
        "character-for-character exact match required):\n"
        + constraints_blob
        + "\nKnown examples (do not contradict these inputs):\n"
        + json.dumps(examples_blob, indent=2)
        + "\n\nReturn ONLY a JSON object of the form:\n"
        '{\n  "tests": [\n'
        "    {\n"
        '      "inputs": { ... },\n'
        '      "category": "valid" | "boundary" | "constraint_invalid",\n'
        '      "constraint": null or an EXACT string from the Constraints '
        "JSON array above\n"
        "    }\n"
        "  ]\n}\n"
        "Rules:\n"
        "- Propose additional valid, boundary, and constraint_invalid cases "
        "when the ToolSpec supports them.\n"
        "- Use ONLY declared input names.\n"
        "- Include every required input.\n"
        "- Do NOT invent or include expected outputs.\n"
        "- For constraint_invalid, the constraint field MUST be copied "
        "verbatim from the Constraints JSON array above. Never paraphrase, "
        "rewrite, shorten, expand, or invent constraint text "
        '(e.g. do not change "must not be zero" into "must be positive").\n'
        "- constraint_invalid inputs MUST demonstrably violate that exact "
        "constraint (e.g. for \"time_hr must not be zero\", time_hr must be 0). "
        "Do not attach a constraint label to inputs that satisfy it.\n"
        "- Do NOT invent or infer new constraints that are not listed.\n"
        "- If Constraints is an empty array [], do not emit "
        "constraint_invalid cases.\n"
        "- Emit boundary cases ONLY for input values that are boundaries "
        "implied by ToolSpec constraints (e.g. zero when a constraint says "
        "an input must not be zero). Do not invent other boundary values.\n"
        "- If no constraint-defined boundaries exist, do not emit boundary "
        "cases.\n"
        "- Do NOT invent or include expected outputs; leave expectations "
        "unknown unless they already appear in Known examples.\n"
    )


class _CandidateModel(BaseModel):
    """Loose candidate shape before deterministic enrichment."""

    inputs: dict[str, Any]
    category: TestCategory
    constraint: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("constraint", mode="before")
    @classmethod
    def blank_constraint_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


def _normalize_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "tests" not in payload:
        raise ValueError("model output missing 'tests' array")
    tests = payload["tests"]
    if not isinstance(tests, list):
        raise ValueError("'tests' must be a list")
    return tests


def _resolve_expectation(
    inputs: dict[str, Any],
    example_lookup: dict[tuple, Any],
) -> tuple[ExpectationStatus, Any | None, TestCase | None]:
    fingerprint = _inputs_fingerprint(inputs)
    if fingerprint in example_lookup:
        expected = example_lookup[fingerprint]
        return (
            "known",
            expected,
            TestCase(inputs=dict(inputs), expected=expected),
        )
    return "unknown", None, None


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _constraint_violation_checker(
    constraint: str, input_names: set[str]
) -> Any | None:
    """Return a predicate ``inputs -> bool`` if the constraint is interpretable.

    Only exact, generic templates tied to declared input names are supported.
    Unparseable constraints cannot be shown to be violated, so callers reject.
    """
    for name in input_names:
        if constraint == f"{name} must not be zero":

            def _violated_zero(
                inputs: dict[str, Any], *, _name: str = name
            ) -> bool:
                value = inputs.get(_name)
                return _is_numeric(value) and value == 0

            return _violated_zero

        if constraint in {
            f"{name} must be positive",
            f"{name} must be greater than zero",
        }:

            def _violated_positive(
                inputs: dict[str, Any], *, _name: str = name
            ) -> bool:
                value = inputs.get(_name)
                return _is_numeric(value) and value <= 0

            return _violated_positive

        if constraint in {
            f"{name} must not be negative",
            f"{name} must be non-negative",
        }:

            def _violated_non_negative(
                inputs: dict[str, Any], *, _name: str = name
            ) -> bool:
                value = inputs.get(_name)
                return _is_numeric(value) and value < 0

            return _violated_non_negative

    return None


def _inputs_violate_constraint(
    constraint: str, inputs: dict[str, Any], tool_spec: ToolSpec
) -> bool:
    """True only when inputs demonstrably violate the exact constraint text."""
    checker = _constraint_violation_checker(
        constraint, {spec.name for spec in tool_spec.inputs}
    )
    if checker is None:
        raise ValueError(
            f"constraint {constraint!r} cannot be checked deterministically; "
            "constraint_invalid requires a demonstrable violation"
        )
    return bool(checker(inputs))


def _boundary_values_from_tool_spec(
    tool_spec: ToolSpec,
) -> dict[str, set[float]]:
    """Boundary values implied only by ToolSpec constraints — never invented."""
    boundaries: dict[str, set[float]] = {}
    input_names = {spec.name for spec in tool_spec.inputs}
    for constraint in tool_spec.constraints:
        for name in input_names:
            if constraint == f"{name} must not be zero":
                boundaries.setdefault(name, set()).add(0.0)
            elif constraint in {
                f"{name} must be positive",
                f"{name} must be greater than zero",
                f"{name} must not be negative",
                f"{name} must be non-negative",
            }:
                boundaries.setdefault(name, set()).add(0.0)
    return boundaries


def _is_constraint_defined_boundary(
    inputs: dict[str, Any], tool_spec: ToolSpec
) -> bool:
    """True when at least one input sits on a ToolSpec-defined boundary."""
    boundaries = _boundary_values_from_tool_spec(tool_spec)
    if not boundaries:
        return False
    for name, allowed in boundaries.items():
        if name not in inputs:
            continue
        value = inputs[name]
        if _is_numeric(value) and float(value) in allowed:
            return True
    return False


def validate_and_build_tests(
    raw_candidates: list[dict[str, Any]], tool_spec: ToolSpec
) -> list[GeneratedTestCase]:
    """Deterministically validate model candidates and merge ToolSpec examples."""
    example_lookup = _example_expected_lookup(tool_spec)
    constraint_set = set(tool_spec.constraints)
    merged = _tests_from_examples(tool_spec)
    seen = {_inputs_fingerprint(item.inputs) for item in merged}

    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError(f"tests[{index}] must be an object")

        # Strip any invented expected fields from the model payload.
        cleaned = {
            key: value
            for key, value in raw.items()
            if key not in {"expected", "expectation_status", "test_case"}
        }
        try:
            candidate = _CandidateModel.model_validate(cleaned)
        except Exception as exc:
            raise ValueError(f"tests[{index}] malformed: {exc}") from exc

        inputs = _validate_inputs_against_spec(candidate.inputs, tool_spec)
        fingerprint = _inputs_fingerprint(inputs)
        if fingerprint in seen:
            raise ValueError(f"tests[{index}] duplicates an existing test case")

        if candidate.category == "constraint_invalid":
            if not tool_spec.constraints:
                raise ValueError(
                    f"tests[{index}] is constraint_invalid but ToolSpec "
                    "declares no constraints"
                )
            if candidate.constraint not in constraint_set:
                raise ValueError(
                    f"tests[{index}] references constraint "
                    f"{candidate.constraint!r} which is not an exact "
                    f"entry in ToolSpec.constraints; paraphrases and "
                    f"invented constraints are rejected"
                )
            if not _inputs_violate_constraint(
                candidate.constraint, inputs, tool_spec
            ):
                raise ValueError(
                    f"tests[{index}] is constraint_invalid for "
                    f"{candidate.constraint!r} but inputs {inputs!r} "
                    f"do not violate that constraint"
                )
        elif candidate.constraint is not None:
            raise ValueError(
                f"tests[{index}] may only set constraint for "
                "constraint_invalid cases"
            )

        if candidate.category == "boundary":
            if not _boundary_values_from_tool_spec(tool_spec):
                raise ValueError(
                    f"tests[{index}] uses category 'boundary' but ToolSpec "
                    "defines no constraint-derived boundaries"
                )
            if not _is_constraint_defined_boundary(inputs, tool_spec):
                raise ValueError(
                    f"tests[{index}] invents a boundary not defined by "
                    f"ToolSpec constraints: inputs={inputs!r}"
                )

        status, expected, test_case = _resolve_expectation(inputs, example_lookup)
        merged.append(
            GeneratedTestCase(
                inputs=inputs,
                category=candidate.category,
                expectation_status=status,
                expected=expected,
                constraint=candidate.constraint,
                test_case=test_case,
            )
        )
        seen.add(fingerprint)

    return merged


def generate_tests(tool_spec: ToolSpec) -> TestGenerationResult:
    """Public API: ToolSpec → independently generated structured tests."""
    if not isinstance(tool_spec, ToolSpec):
        return TestGenerationResult(
            success=False,
            tool_name="",
            error="tool_spec must be a ToolSpec",
        )

    prompt = build_generation_prompt(tool_spec)
    try:
        raw = call_model(prompt)
        payload = _extract_json_object(raw)
        candidates = _normalize_candidates(payload)
        tests = validate_and_build_tests(candidates, tool_spec)
    except (RuntimeError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return TestGenerationResult(
            success=False,
            tool_name=tool_spec.name,
            error=str(exc),
        )

    return TestGenerationResult(
        success=True,
        tool_name=tool_spec.name,
        tests=tests,
        error=None,
    )
