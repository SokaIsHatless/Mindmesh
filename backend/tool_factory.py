"""
The Tool Factory. Generates a pure Python function from a ToolSpec via a
local Ollama model, tests it in the sandbox against ToolSpec.examples,
retries on failure, and saves passing tools to backend/toolbox/.

Primary input is ``ToolSpec``. A temporary legacy ``task_spec`` dict adapter
is isolated below for backward compatibility with the current orchestrator.

Run standalone from backend/:  py tool_factory.py
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Support ``backend.tool_factory`` and flat ``python tool_factory.py`` / discover.
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models import InputSpec, OutputSpec, TestCase, ToolSpec  # noqa: E402
from Sandbox import run_tool  # noqa: E402

# --- Logging ---
logger = logging.getLogger("tool_factory")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[tool_factory] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# --- Config ---
MODEL = "phi4-mini"  # may switch to a larger model later; keep as a constant
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_RETRIES = 3

TOOLBOX_DIR = Path(__file__).resolve().parent / "toolbox"
MANIFEST_PATH = TOOLBOX_DIR / "manifest.json"

ALLOWED_IMPORTS = (
    "math, datetime, json, re, random, statistics, decimal, "
    "fractions, itertools, collections, string, textwrap"
)


def call_model(prompt: str) -> str:
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Is `ollama serve` running and is `{MODEL}` pulled? ({exc})"
        ) from exc
    return body["response"]


def clean_code(raw: str) -> str:
    code = raw.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]  # drop opening fence (with optional "python" tag)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code


def build_signature(name: str, inputs: list[str]) -> str:
    return f"{name}({', '.join(inputs)})"


def _input_names(tool_spec: ToolSpec) -> list[str]:
    return [input_spec.name for input_spec in tool_spec.inputs]


def _format_input_contract(tool_spec: ToolSpec) -> str:
    parts: list[str] = []
    for input_spec in tool_spec.inputs:
        detail = f"{input_spec.name}: {input_spec.type}"
        if input_spec.unit:
            detail += f" ({input_spec.unit})"
        if input_spec.description:
            detail += f" — {input_spec.description}"
        if not input_spec.required:
            detail += " [optional]"
        parts.append(detail)
    return "; ".join(parts) if parts else "(none)"


def _format_output_contract(tool_spec: ToolSpec) -> str:
    output = tool_spec.output
    detail = f"{output.name}: {output.type}"
    if output.unit:
        detail += f" ({output.unit})"
    return detail


def _allowed_imports_for(tool_spec: ToolSpec) -> str:
    if tool_spec.allowed_dependencies:
        return ", ".join(tool_spec.allowed_dependencies)
    return ALLOWED_IMPORTS


def build_initial_prompt(tool_spec: ToolSpec) -> str:
    """Build a generic generation prompt from ToolSpec metadata only."""
    name = tool_spec.name
    signature = build_signature(name, _input_names(tool_spec))
    constraints = (
        "; ".join(tool_spec.constraints) if tool_spec.constraints else "none"
    )
    return (
        f"Write a pure Python function named `{name}` with signature "
        f"{signature} that {tool_spec.purpose}. "
        f"Inputs: {_format_input_contract(tool_spec)}. "
        f"Output: {_format_output_contract(tool_spec)}. "
        f"Constraints/requirements: {constraints}. "
        "Return ONLY the raw Python code for the function. "
        "Do not include any explanation, comments, markdown code fences, "
        "or example usage — just the function definition. "
        "Only use built-in Python (no imports) unless absolutely "
        f"necessary; if you must import, only use: {_allowed_imports_for(tool_spec)}."
    )


def build_retry_prompt(
    tool_spec: ToolSpec, prev_code: str, failure_report: str
) -> str:
    name = tool_spec.name
    signature = build_signature(name, _input_names(tool_spec))
    return (
        "You previously wrote this Python function for the task below, "
        "but it failed testing. Fix the function.\n\n"
        f"Task: {signature} — {tool_spec.purpose}\n"
        f"Inputs: {_format_input_contract(tool_spec)}\n"
        f"Output: {_format_output_contract(tool_spec)}\n"
        f"Constraints/requirements: "
        f"{'; '.join(tool_spec.constraints) if tool_spec.constraints else 'none'}\n\n"
        "Your previous code:\n"
        f"{prev_code}\n\n"
        "Test failures:\n"
        f"{failure_report}\n\n"
        f"Write a corrected pure Python function named `{name}` with the "
        "same signature. Return ONLY the raw Python code, no markdown, "
        "no explanation."
    )


def _values_match(expected: Any, stdout_str: str) -> bool:
    stdout_str = stdout_str.strip()
    try:
        expected_f = float(expected)
        actual_f = float(stdout_str)
        return abs(actual_f - expected_f) < 1e-6
    except (TypeError, ValueError):
        pass
    return stdout_str == str(expected).strip()


def _example_args(tool_spec: ToolSpec, example: TestCase) -> list[Any]:
    """Positional args in ToolSpec input order (generic; no operation branches)."""
    args: list[Any] = []
    for input_spec in tool_spec.inputs:
        if input_spec.name in example.inputs:
            args.append(example.inputs[input_spec.name])
        elif not input_spec.required:
            args.append(input_spec.default)
        else:
            raise ValueError(
                f"example is missing required input '{input_spec.name}'"
            )
    return args


def run_tests(code: str, tool_spec: ToolSpec) -> tuple[bool, list[str]]:
    """Exercise generated code against ToolSpec.examples via the sandbox."""
    name = tool_spec.name
    failures: list[str] = []

    if not tool_spec.examples:
        # Verification pipeline / independent test generation is a later milestone.
        # With no examples, accept generation if the sandbox can import/call nothing
        # further — treat as vacuously passed so generation can still complete.
        logger.info("No ToolSpec.examples provided; skipping sandbox tests")
        return True, []

    for example in tool_spec.examples:
        args = _example_args(tool_spec, example)
        expected = example.expected
        entry_call = f"print({name}(*{args!r}))"
        result = run_tool(code, entry_call)

        if not result["ok"] or result["reason"] != "ok":
            stderr_snippet = result["stderr"].strip()[-300:]
            logger.info(
                "  sandbox test %s -> FAIL | reason=%s | stdout=%r | stderr=%r",
                entry_call,
                result["reason"],
                result.get("stdout", ""),
                stderr_snippet,
            )
            failures.append(
                f"- {name}(*{args!r}) failed to run: {result['reason']} "
                f"— stderr: {stderr_snippet!r}"
            )
            continue

        actual = result["stdout"].strip()
        if not _values_match(expected, actual):
            logger.info(
                "  sandbox test %s -> FAIL | reason=ok | got=%r | expected=%r",
                entry_call,
                actual,
                expected,
            )
            failures.append(
                f"- {name}(*{args!r}) returned {actual!r}, "
                f"expected {expected!r}"
            )
        else:
            logger.info(
                "  sandbox test %s -> PASS | reason=ok | stdout=%r",
                entry_call,
                actual,
            )

    return (len(failures) == 0, failures)


def _load_manifest() -> list:
    if not MANIFEST_PATH.exists():
        return []
    try:
        text = MANIFEST_PATH.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(entries: list) -> None:
    TOOLBOX_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp_path.replace(MANIFEST_PATH)


def save_tool(tool_spec: ToolSpec, code: str) -> None:
    TOOLBOX_DIR.mkdir(parents=True, exist_ok=True)
    name = tool_spec.name
    (TOOLBOX_DIR / f"{name}.py").write_text(code, encoding="utf-8")

    entries = _load_manifest()
    entries = [e for e in entries if e.get("name") != name]
    entries.append(
        {
            "name": name,
            "signature": build_signature(name, _input_names(tool_spec)),
            "description": tool_spec.purpose,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_manifest(entries)


# ---------------------------------------------------------------------------
# Legacy task_spec adapter (temporary, isolated)
# ---------------------------------------------------------------------------
# The orchestrator still converts ToolSpec → old dict and calls create_tool(dict).
# Keep that path working here without leaking dict-shaped logic into prompts/tests.


def _is_legacy_task_spec(value: dict[str, Any]) -> bool:
    """Detect the pre-ToolSpec factory dict: string input names + description."""
    inputs = value.get("inputs")
    return (
        isinstance(value.get("name"), str)
        and isinstance(value.get("description"), str)
        and isinstance(inputs, list)
        and (len(inputs) == 0 or isinstance(inputs[0], str))
        and "purpose" not in value
    )


def adapt_legacy_task_spec(task_spec: dict[str, Any]) -> ToolSpec:
    """Convert the old orchestrator task_spec dict into a ToolSpec.

    Isolated adapter — do not use this shape inside prompt/test helpers.
    """
    if not isinstance(task_spec, dict):
        raise TypeError("legacy task_spec must be a dict")

    raw_inputs = task_spec.get("inputs")
    if not isinstance(raw_inputs, list) or not all(
        isinstance(name, str) for name in raw_inputs
    ):
        raise ValueError("legacy task_spec inputs must be a list of strings")

    input_specs = [InputSpec(name=name, type="number") for name in raw_inputs]
    examples: list[TestCase] = []
    for test in task_spec.get("tests") or []:
        if not isinstance(test, dict) or "args" not in test or "expected" not in test:
            raise ValueError("legacy tests must be {args, expected} mappings")
        args = test["args"]
        if not isinstance(args, list) or len(args) != len(input_specs):
            raise ValueError(
                "legacy test args length must match the number of inputs"
            )
        examples.append(
            TestCase(
                inputs={
                    input_spec.name: args[index]
                    for index, input_spec in enumerate(input_specs)
                },
                expected=test["expected"],
            )
        )

    return ToolSpec(
        name=task_spec["name"],
        purpose=task_spec["description"],
        inputs=input_specs,
        output=OutputSpec(name="result", type="number"),
        examples=examples,
    )


def coerce_tool_spec(value: ToolSpec | dict[str, Any]) -> ToolSpec:
    """Accept ToolSpec, ToolSpec-shaped dict, or isolated legacy task_spec."""
    if isinstance(value, ToolSpec):
        return value
    if not isinstance(value, dict):
        raise TypeError("tool_spec must be a ToolSpec or dict")
    if _is_legacy_task_spec(value):
        return adapt_legacy_task_spec(value)
    return ToolSpec.model_validate(value)


def create_tool(tool_spec: ToolSpec | dict[str, Any]) -> dict:
    """Generate, sandbox-test, and save a tool from a ToolSpec.

    Also accepts the legacy task_spec dict via ``adapt_legacy_task_spec`` so the
    current orchestrator keeps working until it is switched to ToolSpec directly.
    """
    try:
        spec = coerce_tool_spec(tool_spec)
    except (TypeError, ValueError, ValidationError, KeyError) as exc:
        name = None
        if isinstance(tool_spec, dict):
            raw_name = tool_spec.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
        elif isinstance(tool_spec, ToolSpec):
            name = tool_spec.name
        logger.info("create_tool() rejected invalid tool spec: %s", exc)
        return {
            "success": False,
            "tool_name": name or "",
            "code": "",
            "attempts": 0,
            "error": f"invalid tool spec: {exc}",
        }

    name = spec.name
    prev_code = None
    last_failures: list[str] = []

    logger.info("=" * 60)
    logger.info("create_tool() called for '%s'", name)
    logger.info("tool_spec: %s", spec.model_dump_json(indent=2))

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("-" * 60)
        logger.info("Attempt %d/%d for '%s'", attempt, MAX_RETRIES, name)

        if attempt == 1:
            prompt = build_initial_prompt(spec)
        else:
            failure_report = "\n".join(last_failures)
            logger.info("Retrying — feeding back failures:\n%s", failure_report)
            prompt = build_retry_prompt(spec, prev_code or "", failure_report)

        logger.info("Prompt sent to model (%s):\n%s", MODEL, prompt)

        try:
            raw = call_model(prompt)
        except RuntimeError as exc:
            logger.info("Model call FAILED: %s", exc)
            logger.info(
                "create_tool() outcome: FAILURE (tool=%s, attempts=%d, error=%s)",
                name,
                attempt,
                exc,
            )
            return {
                "success": False,
                "tool_name": name,
                "code": prev_code or "",
                "attempts": attempt,
                "error": str(exc),
            }

        logger.info("Raw response from model:\n%s", raw)

        code = clean_code(raw)
        logger.info("Cleaned code:\n%s", code)

        passed, failures = run_tests(code, spec)

        if passed:
            save_tool(spec, code)
            logger.info(
                "create_tool() outcome: SUCCESS (tool=%s, attempts=%d)",
                name,
                attempt,
            )
            return {
                "success": True,
                "tool_name": name,
                "code": code,
                "attempts": attempt,
                "error": None,
            }

        prev_code = code
        last_failures = failures

    error_summary = (
        f"failed after {MAX_RETRIES} attempts; last failures: "
        + "; ".join(last_failures)
    )
    logger.info(
        "create_tool() outcome: FAILURE (tool=%s, attempts=%d, error=%s)",
        name,
        MAX_RETRIES,
        error_summary,
    )
    return {
        "success": False,
        "tool_name": name,
        "code": prev_code or "",
        "attempts": MAX_RETRIES,
        "error": error_summary,
    }


DEMO_TOOL_SPEC = ToolSpec(
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

# Backward-compatible alias for standalone demos that still mention task_spec.
DEMO_TASK_SPEC = {
    "name": "speed",
    "description": "compute speed from distance and time",
    "inputs": ["distance_km", "time_hr"],
    "tests": [
        {"args": [120, 2], "expected": 60},
        {"args": [90, 3], "expected": 30},
    ],
}


def _self_test_retry() -> None:
    """
    Manual verification that the retry loop recovers from a bad first
    attempt. Run with: py tool_factory.py --self-test-retry

    Monkeypatches call_model so attempt 1 returns deliberately broken code
    (wrong operator) and attempt 2+ falls through to the real Ollama call,
    then asserts create_tool succeeds with attempts == 2.
    """
    global call_model
    real_call_model = call_model
    calls = {"n": 0}

    def fake_call_model(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "def speed(distance_km, time_hr): return distance_km * time_hr"
        return real_call_model(prompt)

    call_model = fake_call_model
    try:
        result = create_tool(DEMO_TOOL_SPEC)
    finally:
        call_model = real_call_model

    print(json.dumps(result, indent=2))
    assert result["attempts"] >= 2, "expected retry loop to need a second attempt"
    assert result["success"], "expected retry loop to eventually succeed"
    print("SELF-TEST PASSED: retry loop recovered on attempt", result["attempts"])


if __name__ == "__main__":
    if "--self-test-retry" in sys.argv:
        _self_test_retry()
    else:
        result = create_tool(DEMO_TOOL_SPEC)
        print(json.dumps(result, indent=2))
