"""
interpreter.py — NL → structured calculation request (Ollama / phi4-mini).

First step of the backend refactor. Standalone: does NOT talk to the
orchestrator, sandbox, toolbox, or tool factory.

Usage (from backend/):
    py interpreter.py
    from interpreter import interpret_task
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

# --- Config (aligned with tool_factory) ---
MODEL = "phi4-mini"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_SEC = 60

VALID_STATUSES = frozenset({"ok", "needs_input", "unsupported", "error"})

# Soft hints for the model only — not used as a rigid regex classifier.
_KNOWN_OPS_HINT = """
Common operations and their usual SI-ish inputs (extend freely for new tools):
  - speed: distance_km, time_hr
  - distance: speed_kmh, time_hr
  - time: distance_km, speed_kmh
  - bmi: height_cm, weight_kg
  - average: values (list) OR sum + count
  - probability: favorable, total
Prefer normalized numeric SI units in inputs:
  height_cm, weight_kg, distance_km, speed_kmh, time_hr
Convert when the user gives other units (e.g. 1.5 m → height_cm 150;
  30 minutes → time_hr 0.5; 5000 m → distance_km 5).
"""


def _call_ollama(prompt: str) -> str:
    """Call local Ollama; raise RuntimeError on connection/timeout failures."""
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
    """Parse a JSON object from model output; tolerate markdown fences."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model response")

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: first {...} block in the text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def _build_prompt(task: str) -> str:
    return f"""You are a request interpreter for an offline calculation agent.
Your ONLY job is to understand the user's natural-language request and return
ONE JSON object describing what calculation they want. Do NOT compute the
answer. Do NOT write Python code. Do NOT invent numbers the user did not give.

{_KNOWN_OPS_HINT}

Return ONLY a JSON object with exactly these keys:
  "status": one of "ok" | "needs_input" | "unsupported" | "error"
  "operation": short snake_case name of the calculation (or null if unknown)
  "inputs": object of named numeric inputs you extracted (empty object if none)
  "missing_inputs": array of input names still required (empty if none)

Status rules:
  - "ok": operation is clear AND all required numeric inputs are present
    (use values the user stated; convert units into the preferred keys above).
  - "needs_input": operation is clear but one or more required numbers are
    missing. List those names in missing_inputs. NEVER invent values.
  - "unsupported": greeting, chat, or not a well-defined calculation
    (e.g. "hello how are you?"). Use operation null, empty inputs/missing.
  - "error": only if the request is a calculation but too broken to interpret.

Examples of intent (illustrative — follow the same shape):
  User: "height = 150cm, weight = 50kg calculate bmi"
  → status ok, operation bmi, inputs {{"height_cm": 150, "weight_kg": 50}},
    missing_inputs []

  User: "calculate speed"
  → status needs_input, operation speed, inputs {{}},
    missing_inputs ["distance_km", "time_hr"]

  User: "hello how are you?"
  → status unsupported, operation null, inputs {{}}, missing_inputs []

User request:
{task.strip()}

JSON:"""


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else value
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            num = float(s)
            return int(num) if num == int(num) else num
        except ValueError:
            return None
    return None


def _normalize_inputs(raw_inputs: Any) -> dict[str, float | int]:
    if not isinstance(raw_inputs, dict):
        return {}
    out: dict[str, float | int] = {}
    for key, value in raw_inputs.items():
        if not isinstance(key, str) or not key.strip():
            continue
        num = _coerce_number(value)
        if num is None:
            # Do not invent; skip non-numeric silently
            continue
        out[key.strip()] = num
    return out


def _normalize_missing(raw_missing: Any) -> list[str]:
    if not isinstance(raw_missing, list):
        return []
    names: list[str] = []
    for item in raw_missing:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
    return names


def _validate_result(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate model JSON into the public contract."""
    status = data.get("status")
    if not isinstance(status, str) or status not in VALID_STATUSES:
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "error": f"invalid status from model: {status!r}",
        }

    operation = data.get("operation")
    if operation is not None and not isinstance(operation, str):
        operation = None
    if isinstance(operation, str):
        operation = operation.strip() or None

    inputs = _normalize_inputs(data.get("inputs"))
    missing = _normalize_missing(data.get("missing_inputs"))

    # Consistency: ok should not list missing required inputs
    if status == "ok" and missing:
        status = "needs_input"

    # needs_input with no operation and no missing names → unsupported
    if status == "needs_input" and operation is None and not missing:
        status = "unsupported"

    if status == "unsupported":
        return {
            "status": "unsupported",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
        }

    result: dict[str, Any] = {
        "status": status,
        "operation": operation,
        "inputs": inputs,
        "missing_inputs": missing,
    }
    if status == "ok":
        result["missing_inputs"] = []
    if status == "error" and isinstance(data.get("error"), str):
        result["error"] = data["error"]
    return result


def interpret_task(task: str) -> dict[str, Any]:
    """
    Interpret a natural-language calculation request via local Ollama.

    Returns a dict:
      {
        "status": "ok" | "needs_input" | "unsupported" | "error",
        "operation": str | None,
        "inputs": { name: number, ... },
        "missing_inputs": [str, ...],
      }

    Does not calculate, generate code, run tools, or write to disk.
    """
    if not isinstance(task, str) or not task.strip():
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "error": "empty task",
        }

    try:
        raw = _call_ollama(_build_prompt(task))
    except RuntimeError as exc:
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "error": str(exc),
        }

    try:
        data = _extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "error": f"malformed model JSON: {exc}",
        }

    return _validate_result(data)


# ---------------------------------------------------------------------------
# Manual smoke tests (not wired to orchestrator)
# ---------------------------------------------------------------------------
_DEMO_TASKS = [
    "height = 150cm, weight = 50kg calculate bmi",
    "A train travels 120 km in 2 hours. What is its speed?",
    "calculate speed",
    "hello how are you?",
]


if __name__ == "__main__":
    for demo in _DEMO_TASKS:
        print("=" * 60)
        print("task:", demo)
        print(json.dumps(interpret_task(demo), indent=2))
