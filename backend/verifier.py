"""
verifier.py — Independent verification of generated tool code.

Checks candidate Python against an independently generated test set before a
tool can be considered verified. Never generates code or tests. Never imports
or executes generated code in-process; all execution goes through Sandbox.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Flat models/Sandbox match how test_verifier constructs ToolSpec.
from models import ToolSpec  # noqa: E402
from Sandbox import run_tool, validate_imports  # noqa: E402

try:  # Prefer package path so GeneratedTestCase identity matches test imports.
    from .test_generator import GeneratedTestCase, TestGenerationResult
except ImportError:
    from test_generator import GeneratedTestCase, TestGenerationResult  # noqa: E402

FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-9

VerificationStatus = Literal["verified", "failed", "not_verifiable"]
TestOutcomeStatus = Literal["passed", "failed", "skipped", "not_verifiable"]


class TestVerificationDetail(BaseModel):
    """Outcome for one generated test case."""

    inputs: dict[str, Any]
    category: str | None = None
    status: TestOutcomeStatus
    expected: Any | None = None
    actual: Any | None = None
    detail: str | None = None

    model_config = ConfigDict(extra="forbid")


class VerificationResult(BaseModel):
    """Structured result of the verification pipeline."""

    status: VerificationStatus
    tool_name: str
    passed_tests: list[TestVerificationDetail] = Field(default_factory=list)
    failed_tests: list[TestVerificationDetail] = Field(default_factory=list)
    skipped_tests: list[TestVerificationDetail] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    details: str | None = None

    model_config = ConfigDict(extra="forbid")


def _looks_like_generation_result(value: Any) -> bool:
    """True for TestGenerationResult across package/flat dual imports."""
    if isinstance(value, TestGenerationResult):
        return True
    # Duck-type when the same Pydantic model was loaded under two module paths.
    return (
        type(value).__name__ == "TestGenerationResult"
        and hasattr(value, "success")
        and hasattr(value, "tests")
        and hasattr(value, "error")
    )


def _looks_like_generated_case(value: Any) -> bool:
    """True for GeneratedTestCase across package/flat dual imports."""
    if isinstance(value, GeneratedTestCase):
        return True
    return (
        type(value).__name__ == "GeneratedTestCase"
        and hasattr(value, "inputs")
        and hasattr(value, "expectation_status")
        and hasattr(value, "expected")
        and hasattr(value, "category")
    )


def _normalize_tests(
    tests: TestGenerationResult | list[GeneratedTestCase],
) -> tuple[list[Any], str | None]:
    """Accept TestGenerationResult or a bare list; surface generation errors."""
    if _looks_like_generation_result(tests):
        if not tests.success:
            return [], tests.error or "test generation was unsuccessful"
        return list(tests.tests), None
    if isinstance(tests, list):
        return list(tests), None
    raise TypeError(
        "tests must be a TestGenerationResult or list of GeneratedTestCase"
    )


def _static_validate_code(code: str, tool_name: str) -> list[str]:
    """Stage (a): syntax, import allowlist, and required function presence."""
    errors: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return ["generated code must be a non-empty string"]

    safe, reason = validate_imports(code)
    if not safe:
        errors.append(f"static validation failed: {reason}")
        return errors

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"static validation failed: Syntax error: {exc}"]

    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if tool_name not in function_names:
        errors.append(
            f"static validation failed: missing required function "
            f"'{tool_name}'"
        )
    return errors


def _positional_args(tool_spec: ToolSpec, inputs: dict[str, Any]) -> list[Any]:
    """Build call args in ToolSpec input order (generic; no domain branches)."""
    args: list[Any] = []
    for input_spec in tool_spec.inputs:
        if input_spec.name in inputs:
            args.append(inputs[input_spec.name])
        elif not input_spec.required:
            args.append(input_spec.default)
        else:
            raise ValueError(
                f"test is missing required input '{input_spec.name}'"
            )
    return args


def _parse_stdout(stdout: str) -> Any:
    """Best-effort parse of sandbox stdout into a Python value."""
    text = (stdout or "").strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return text


def values_match(expected: Any, actual_stdout: str) -> bool:
    """Compare expected vs sandbox stdout with float-tolerant numeric checks."""
    actual = _parse_stdout(actual_stdout)

    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual

    try:
        expected_f = float(expected)
        actual_f = float(actual)
        if math.isfinite(expected_f) and math.isfinite(actual_f):
            return math.isclose(
                expected_f,
                actual_f,
                rel_tol=FLOAT_REL_TOL,
                abs_tol=FLOAT_ABS_TOL,
            )
    except (TypeError, ValueError):
        pass

    if expected == actual:
        return True
    return str(actual).strip() == str(expected).strip()


def _is_verifiable(case: Any) -> bool:
    """Known expected values are required for pass/fail comparison."""
    return case.expectation_status == "known" and case.expected is not None


def _smoke_execute(code: str, tool_name: str) -> dict[str, Any]:
    """Stage (b): exercise the code in the sandbox without importing it here."""
    entry_call = f"print({tool_name})"
    return run_tool(code, entry_call)


def _run_one_test(
    code: str, tool_spec: ToolSpec, case: Any
) -> TestVerificationDetail:
    """Execute one verifiable case via the sandbox and compare outputs."""
    try:
        args = _positional_args(tool_spec, case.inputs)
    except ValueError as exc:
        return TestVerificationDetail(
            inputs=dict(case.inputs),
            category=case.category,
            status="failed",
            expected=case.expected,
            detail=str(exc),
        )

    entry_call = f"print({tool_spec.name}(*{args!r}))"
    result = run_tool(code, entry_call)

    if not result.get("ok") or result.get("reason") != "ok":
        stderr = (result.get("stderr") or "").strip()[-300:]
        reason = result.get("reason") or "sandbox error"
        return TestVerificationDetail(
            inputs=dict(case.inputs),
            category=case.category,
            status="failed",
            expected=case.expected,
            actual=(result.get("stdout") or "").strip() or None,
            detail=f"sandbox execution failed: {reason}"
            + (f" — stderr: {stderr!r}" if stderr else ""),
        )

    stdout = result.get("stdout") or ""
    actual = _parse_stdout(stdout)
    if values_match(case.expected, stdout):
        return TestVerificationDetail(
            inputs=dict(case.inputs),
            category=case.category,
            status="passed",
            expected=case.expected,
            actual=actual,
            detail="ok",
        )

    return TestVerificationDetail(
        inputs=dict(case.inputs),
        category=case.category,
        status="failed",
        expected=case.expected,
        actual=actual,
        detail=(
            f"expected {case.expected!r}, got {actual!r} "
            f"(stdout={stdout.strip()!r})"
        ),
    )


def _finalize(
    tool_name: str,
    passed: list[TestVerificationDetail],
    failed: list[TestVerificationDetail],
    skipped: list[TestVerificationDetail],
    errors: list[str],
) -> VerificationResult:
    """Derive overall status from stage outcomes and per-test buckets."""
    if errors or failed:
        status: VerificationStatus = "failed"
        details = "; ".join(errors) if errors else "one or more tests failed"
    elif passed:
        status = "verified"
        details = "all verifiable tests passed"
    else:
        status = "not_verifiable"
        details = "no verifiable tests with known expected values"

    return VerificationResult(
        status=status,
        tool_name=tool_name,
        passed_tests=passed,
        failed_tests=failed,
        skipped_tests=skipped,
        errors=errors,
        details=details,
    )


def verify_tool(
    tool_spec: ToolSpec,
    code: str,
    tests: TestGenerationResult | list[GeneratedTestCase],
) -> VerificationResult:
    """Run the verification pipeline against generated code and tests.

    Stages:
      (a) static validation of the generated Python
      (b) smoke execution through Sandbox.run_tool
      (c) run applicable known/valid (verifiable) cases via the sandbox
      (d) compare actual vs expected with float-tolerant matching
      (e) return a structured VerificationResult
    """
    if not isinstance(tool_spec, ToolSpec):
        return VerificationResult(
            status="failed",
            tool_name="",
            errors=["tool_spec must be a ToolSpec"],
            details="invalid tool_spec",
        )

    tool_name = tool_spec.name
    try:
        cases, generation_error = _normalize_tests(tests)
    except TypeError as exc:
        return VerificationResult(
            status="failed",
            tool_name=tool_name,
            errors=[str(exc)],
            details=str(exc),
        )

    if generation_error:
        return VerificationResult(
            status="failed",
            tool_name=tool_name,
            errors=[generation_error],
            details=generation_error,
        )

    # --- (a) Static validation ------------------------------------------------
    static_errors = _static_validate_code(code, tool_name)
    if static_errors:
        return VerificationResult(
            status="failed",
            tool_name=tool_name,
            errors=static_errors,
            details=static_errors[0],
        )

    # --- (b) Sandbox smoke execution ------------------------------------------
    smoke = _smoke_execute(code, tool_name)
    if not smoke.get("ok") or smoke.get("reason") != "ok":
        reason = smoke.get("reason") or "sandbox error"
        stderr = (smoke.get("stderr") or "").strip()[-300:]
        message = f"sandbox execution failed: {reason}"
        if stderr:
            message = f"{message} — stderr: {stderr!r}"
        return VerificationResult(
            status="failed",
            tool_name=tool_name,
            errors=[message],
            details=message,
        )

    passed: list[TestVerificationDetail] = []
    failed: list[TestVerificationDetail] = []
    skipped: list[TestVerificationDetail] = []

    # --- (c)+(d) Run verifiable cases and compare -----------------------------
    for case in cases:
        if not _looks_like_generated_case(case):
            failed.append(
                TestVerificationDetail(
                    inputs={},
                    status="failed",
                    detail="test case must be a GeneratedTestCase",
                )
            )
            continue

        if not _is_verifiable(case):
            skipped.append(
                TestVerificationDetail(
                    inputs=dict(case.inputs),
                    category=case.category,
                    status="not_verifiable",
                    expected=None,
                    detail=(
                        "skipped: expectation_status is unknown "
                        "(no expected value to verify against)"
                    ),
                )
            )
            continue

        outcome = _run_one_test(code, tool_spec, case)
        if outcome.status == "passed":
            passed.append(outcome)
        else:
            failed.append(outcome)

    # --- (e) Structured result ------------------------------------------------
    return _finalize(tool_name, passed, failed, skipped, errors=[])
