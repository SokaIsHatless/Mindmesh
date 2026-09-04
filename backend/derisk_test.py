"""
THROWAWAY de-risk experiment — NOT the real tool factory.

Question this answers: can Phi-4-mini (via Ollama) write a correct pure
Python function that passes the sandbox's tests, and how reliably?

Run from the backend/ folder (or repo root) with Ollama already running and
`phi4-mini` already pulled:

    py derisk_test.py
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Sandbox import run_tool

MODEL = "phi4-mini"  # adjust if you pulled a differently-named tag
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = (
    "Write a pure Python function named `speed` with signature "
    "speed(distance_km, time_hr) that returns distance_km divided by "
    "time_hr. Return ONLY the raw Python code for the function. "
    "Do not include any explanation, comments, markdown code fences, "
    "or example usage — just the function definition."
)

TEST_CASES = [
    ((120, 2), 60),
    ((90, 3), 30),
    ((100, 4), 25),
]

NUM_RUNS = 3


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


def main() -> None:
    total_passed = 0
    total_tests = NUM_RUNS * len(TEST_CASES)

    for attempt in range(1, NUM_RUNS + 1):
        print(f"\n{'=' * 60}")
        print(f"ATTEMPT {attempt}/{NUM_RUNS}")
        print(f"{'=' * 60}")

        raw_code = call_model(PROMPT)
        code = clean_code(raw_code)
        print("--- generated code ---")
        print(code)
        print("--- tests ---")

        attempt_passed = 0
        for args, expected in TEST_CASES:
            entry_call = f"print(speed({args[0]}, {args[1]}))"
            result = run_tool(code, entry_call)

            if not result["ok"]:
                print(
                    f"speed{args} -> FAIL "
                    f"(reason={result['reason']!r}, stderr={result['stderr'].strip()!r})"
                )
                continue

            output = result["stdout"].strip()
            try:
                actual = float(output)
                passed = abs(actual - expected) < 1e-6
            except ValueError:
                passed = False

            status = "PASS" if passed else "FAIL"
            if passed:
                attempt_passed += 1
            print(f"speed{args} -> {status} (expected={expected}, got={output!r})")

        total_passed += attempt_passed
        print(f"--- attempt {attempt} result: {attempt_passed}/{len(TEST_CASES)} tests passed ---")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed across {NUM_RUNS} attempts")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
