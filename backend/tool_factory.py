"""
The real Tool Factory. Generates a pure Python function for a task via a
local Ollama model, tests it in the sandbox, retries on failure, and saves
passing tools to backend/toolbox/.

Run standalone from backend/:  py tool_factory.py
"""

import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Sandbox import run_tool

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


def build_signature(name: str, inputs: list) -> str:
    return f"{name}({', '.join(inputs)})"


def build_initial_prompt(task_spec: dict) -> str:
    name = task_spec["name"]
    signature = build_signature(name, task_spec["inputs"])
    return (
        f"Write a pure Python function named `{name}` with signature "
        f"{signature} that {task_spec['description']}. "
        "Return ONLY the raw Python code for the function. "
        "Do not include any explanation, comments, markdown code fences, "
        "or example usage — just the function definition. "
        "Only use built-in Python (no imports) unless absolutely "
        f"necessary; if you must import, only use: {ALLOWED_IMPORTS}."
    )


def build_retry_prompt(task_spec: dict, prev_code: str, failure_report: str) -> str:
    name = task_spec["name"]
    signature = build_signature(name, task_spec["inputs"])
    return (
        "You previously wrote this Python function for the task below, "
        "but it failed testing. Fix the function.\n\n"
        f"Task: {signature} — {task_spec['description']}\n\n"
        "Your previous code:\n"
        f"{prev_code}\n\n"
        "Test failures:\n"
        f"{failure_report}\n\n"
        f"Write a corrected pure Python function named `{name}` with the "
        "same signature. Return ONLY the raw Python code, no markdown, "
        "no explanation."
    )


def _values_match(expected, stdout_str: str) -> bool:
    stdout_str = stdout_str.strip()
    try:
        expected_f = float(expected)
        actual_f = float(stdout_str)
        return abs(actual_f - expected_f) < 1e-6
    except (TypeError, ValueError):
        pass
    return stdout_str == str(expected).strip()


def run_tests(code: str, task_spec: dict) -> tuple:
    name = task_spec["name"]
    failures = []

    for test in task_spec["tests"]:
        args = test["args"]
        expected = test["expected"]
        entry_call = f"print({name}(*{args!r}))"
        result = run_tool(code, entry_call)

        if not result["ok"] or result["reason"] != "ok":
            stderr_snippet = result["stderr"].strip()[-300:]
            failures.append(
                f"- {name}(*{args!r}) failed to run: {result['reason']} "
                f"— stderr: {stderr_snippet!r}"
            )
            continue

        actual = result["stdout"].strip()
        if not _values_match(expected, actual):
            failures.append(
                f"- {name}(*{args!r}) returned {actual!r}, "
                f"expected {expected!r}"
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


def save_tool(task_spec: dict, code: str) -> None:
    TOOLBOX_DIR.mkdir(parents=True, exist_ok=True)
    name = task_spec["name"]
    (TOOLBOX_DIR / f"{name}.py").write_text(code, encoding="utf-8")

    entries = _load_manifest()
    entries = [e for e in entries if e.get("name") != name]
    entries.append(
        {
            "name": name,
            "signature": build_signature(name, task_spec["inputs"]),
            "description": task_spec["description"],
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_manifest(entries)


def create_tool(task_spec: dict) -> dict:
    name = task_spec["name"]
    prev_code = None
    last_failures = []

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            prompt = build_initial_prompt(task_spec)
        else:
            prompt = build_retry_prompt(task_spec, prev_code, "\n".join(last_failures))

        try:
            raw = call_model(prompt)
        except RuntimeError as exc:
            return {
                "success": False,
                "tool_name": name,
                "code": prev_code or "",
                "attempts": attempt,
                "error": str(exc),
            }

        code = clean_code(raw)
        passed, failures = run_tests(code, task_spec)

        if passed:
            save_tool(task_spec, code)
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
    return {
        "success": False,
        "tool_name": name,
        "code": prev_code or "",
        "attempts": MAX_RETRIES,
        "error": error_summary,
    }


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
        result = create_tool(DEMO_TASK_SPEC)
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
        result = create_tool(DEMO_TASK_SPEC)
        print(json.dumps(result, indent=2))
