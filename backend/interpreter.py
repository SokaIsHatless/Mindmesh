"""
interpreter.py — NL → structured calculation request (Ollama / phi4-mini).

First step of the backend refactor. Standalone: does NOT talk to the
orchestrator, sandbox, toolbox, or tool factory.

Understands provided inputs vs the requested unknown, and resolves inverse
relationships (e.g. BMI + weight → height) without computing numeric answers.

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
  - speed: distance_km, time_hr → speed_kmh
  - distance: speed_kmh, time_hr → distance_km
  - time: distance_km, speed_kmh → time_hr
  - bmi: height_cm, weight_kg → bmi
  - height_from_bmi: bmi, weight_kg → height_cm
  - weight_from_bmi: bmi, height_cm → weight_kg
  - average: values (list) OR sum + count
  - probability: favorable, total
Prefer normalized numeric SI units in inputs:
  height_cm, weight_kg, distance_km, speed_kmh, time_hr, bmi
Convert when the user gives other units (e.g. 1.5 m → height_cm 150;
  30 minutes → time_hr 0.5; 5000 m → distance_km 5).

Inverse / unknown-output rules (critical):
  - Identify which quantity the user is ASKING FOR (the unknown). Put that
    name in requested_output. Do NOT put it in missing_inputs.
  - Put only PROVIDED numeric values in inputs.
  - missing_inputs = other values still needed to solve — never the unknown.
  - BMI family (any two determine the third):
      height_cm + weight_kg → bmi (operation bmi)
      bmi + weight_kg → height_cm (operation height_from_bmi)
      bmi + height_cm → weight_kg (operation weight_from_bmi)
  - Speed family (any two determine the third):
      distance_km + time_hr → speed_kmh (operation speed)
      speed_kmh + time_hr → distance_km (operation distance)
      distance_km + speed_kmh → time_hr (operation time)
"""

# ---------------------------------------------------------------------------
# Deterministic relationship catalog (planning layer — no numeric solving)
# ---------------------------------------------------------------------------
# Alias → canonical quantity name used in relationships / ToolSpec planning.
_INPUT_ALIASES: dict[str, str] = {
    "height": "height_cm",
    "height_cm": "height_cm",
    "height_m": "height_cm",  # value conversion left to normalizer when key kept;
    "weight": "weight_kg",
    "weight_kg": "weight_kg",
    "bmi": "bmi",
    "body_mass_index": "bmi",
    "distance": "distance_km",
    "distance_km": "distance_km",
    "time": "time_hr",
    "time_hr": "time_hr",
    "speed": "speed_kmh",
    "velocity": "speed_kmh",
    "speed_kmh": "speed_kmh",
}

# Keys whose values are already converted by the model into canonical units
# when the alias form is used; height_m is special — keep as height_m for
# normalizer unit conversion when present, but treat as height_cm family.
_ALIAS_PASS_THROUGH_RAW: frozenset[str] = frozenset(
    {
        "height_m",
        "height_cm",
        "weight_kg",
        "distance_km",
        "time_hr",
        "speed_kmh",
        "bmi",
    }
)

# Relationship families: any two variables determine the third.
# solutions: frozenset(provided) → (operation, requested_output)
_RELATIONSHIP_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "bmi",
        "variables": frozenset({"height_cm", "weight_kg", "bmi"}),
        "solutions": {
            frozenset({"height_cm", "weight_kg"}): ("bmi", "bmi"),
            frozenset({"bmi", "weight_kg"}): ("height_from_bmi", "height_cm"),
            frozenset({"bmi", "height_cm"}): ("weight_from_bmi", "weight_kg"),
        },
    },
    {
        "name": "motion",
        "variables": frozenset({"distance_km", "time_hr", "speed_kmh"}),
        "solutions": {
            frozenset({"distance_km", "time_hr"}): ("speed", "speed_kmh"),
            frozenset({"speed_kmh", "time_hr"}): ("distance", "distance_km"),
            frozenset({"distance_km", "speed_kmh"}): ("time", "time_hr"),
        },
    },
)


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
  "inputs": object of named numeric inputs the user PROVIDED (empty if none)
  "missing_inputs": array of input names still required to solve (empty if none)
  "requested_output": name of the quantity to solve for (the unknown), or null

Status rules:
  - "ok": operation is clear AND enough provided numbers exist to solve for
    requested_output (use values the user stated; convert units into preferred
    keys above). requested_output must NOT appear in inputs or missing_inputs.
  - "needs_input": operation/unknown is clear but one or more OTHER numbers
    needed to solve are absent. List only those in missing_inputs — never the
    requested_output itself.
  - "unsupported": greeting, chat, or not a well-defined calculation
    (e.g. "hello how are you?"). Use operation null, empty inputs/missing,
    requested_output null.
  - "error": only if the request is a calculation but too broken to interpret.

Examples of intent (illustrative — follow the same shape):
  User: "height = 150cm, weight = 50kg calculate bmi"
  → status ok, operation bmi,
    inputs {{"height_cm": 150, "weight_kg": 50}},
    missing_inputs [], requested_output "bmi"

  User: "If BMI is 20 and weight is 80 kg, what is the height?"
  → status ok, operation height_from_bmi,
    inputs {{"bmi": 20, "weight_kg": 80}},
    missing_inputs [], requested_output "height_cm"

  User: "BMI is 22 and height is 170 cm; find the weight"
  → status ok, operation weight_from_bmi,
    inputs {{"bmi": 22, "height_cm": 170}},
    missing_inputs [], requested_output "weight_kg"

  User: "calculate height from BMI 20"
  → status needs_input, operation height_from_bmi,
    inputs {{"bmi": 20}},
    missing_inputs ["weight_kg"], requested_output "height_cm"

  User: "calculate speed"
  → status needs_input, operation speed, inputs {{}},
    missing_inputs ["distance_km", "time_hr"], requested_output "speed_kmh"

  User: "hello how are you?"
  → status unsupported, operation null, inputs {{}}, missing_inputs [],
    requested_output null

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


def _normalize_requested_output(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None


def _canonical_quantity(name: str) -> str:
    """Map aliases to relationship-canonical names without inventing values."""
    key = name.strip().lower()
    return _INPUT_ALIASES.get(key, key)


def _canonicalize_input_map(
    inputs: dict[str, float | int],
) -> dict[str, float | int]:
    """
    Soft-alias input keys for relationship matching.

    Preserves unit-bearing keys the normalizer understands (e.g. height_m)
    when they already encode a unit; otherwise maps bare aliases to canonical
    relationship names (height → height_cm).
    """
    out: dict[str, float | int] = {}
    for raw_key, value in inputs.items():
        lowered = raw_key.strip().lower()
        if lowered in _ALIAS_PASS_THROUGH_RAW:
            # height_m stays height_m for normalizer conversion; for family
            # matching we also expose the canonical quantity separately below.
            out[lowered] = value
            continue
        canonical = _canonical_quantity(lowered)
        if canonical in out and out[canonical] != value:
            # Conflicting duplicates — keep the first; relationship logic
            # still sees one value for the quantity.
            continue
        out[canonical] = value
    return out


def _quantities_present(inputs: dict[str, float | int]) -> set[str]:
    """Canonical quantity names implied by the provided input keys."""
    present: set[str] = set()
    for key in inputs:
        lowered = key.strip().lower()
        if lowered == "height_m":
            present.add("height_cm")
        else:
            present.add(_canonical_quantity(lowered))
    return present


def _rewrite_inputs_for_family(
    inputs: dict[str, float | int], family_vars: frozenset[str]
) -> dict[str, float | int]:
    """
    Keep provided values keyed for the pipeline.

    Maps relationship-relevant aliases onto canonical family names while
    leaving unrelated keys intact. ``height_m`` is rewritten to ``height_cm``
    only after converting metres→cm here when the model left a metre key
    (deterministic unit fix for relationship planning; normalizer also
    converts height_m if left unchanged — we convert once to height_cm).
    """
    rewritten: dict[str, float | int] = {}
    for key, value in inputs.items():
        lowered = key.strip().lower()
        if lowered == "height_m":
            # metres → centimetres so family matching + downstream agree
            cm = value * 100 if isinstance(value, (int, float)) else value
            if "height_cm" in family_vars:
                rewritten["height_cm"] = (
                    int(cm) if isinstance(cm, float) and cm == int(cm) else cm
                )
            else:
                rewritten[key] = value
            continue
        canonical = _canonical_quantity(lowered)
        if canonical in family_vars:
            rewritten[canonical] = value
        else:
            rewritten[key] = value
    return rewritten


def _apply_relationships(
    *,
    status: str,
    operation: str | None,
    inputs: dict[str, float | int],
    missing: list[str],
    requested_output: str | None,
) -> tuple[str, str | None, dict[str, float | int], list[str], str | None]:
    """
    Deterministically resolve inverse relationships.

    Does not compute numeric answers — only chooses operation, unknown, and
    which names are genuinely missing.
    """
    if status in {"unsupported", "error"}:
        return status, operation, inputs, missing, requested_output

    present = _quantities_present(inputs)
    requested = (
        _canonical_quantity(requested_output) if requested_output else None
    )

    for family in _RELATIONSHIP_FAMILIES:
        variables: frozenset[str] = family["variables"]
        overlap = present & variables
        if len(overlap) < 1:
            continue

        # Infer requested_output when exactly one family variable is absent
        # and two are provided — classic inverse / forward case.
        solutions: dict[frozenset[str], tuple[str, str]] = family["solutions"]

        if len(overlap) == 2:
            provided_pair = frozenset(overlap)
            if provided_pair in solutions:
                op_name, output_name = solutions[provided_pair]
                # If the model named a different unknown outside this family,
                # do not override unrelated requests.
                if requested is not None and requested not in variables:
                    continue
                # If model claimed a provided quantity as the unknown, prefer
                # the solvable unknown from the relationship.
                if requested in provided_pair:
                    requested = output_name
                if requested is None:
                    requested = output_name
                if requested != output_name and requested in variables:
                    # User asked for a provided value — still solvable as
                    # identity-ish; keep relationship operation for the
                    # natural unknown of this pair.
                    requested = output_name

                cleaned_inputs = _rewrite_inputs_for_family(inputs, variables)
                # Drop any value for the unknown if the model wrongly included it
                cleaned_inputs = {
                    k: v
                    for k, v in cleaned_inputs.items()
                    if _canonical_quantity(k) != requested
                    and not (k == "height_m" and requested == "height_cm")
                }
                # Genuinely missing: none for a complete pair
                return "ok", op_name, cleaned_inputs, [], requested

        if len(overlap) == 1 or len(overlap) == 2:
            # Incomplete: need one more family variable (not the unknown).
            if requested is None:
                # Prefer inferring unknown from operation hints / remaining vars
                if operation:
                    op_l = operation.strip().lower()
                    for provided_pair, (op_name, output_name) in solutions.items():
                        if op_l == op_name or op_l == output_name:
                            requested = output_name
                            break
                if requested is None and len(overlap) == 1:
                    # Cannot uniquely choose unknown among two remaining vars
                    remaining = sorted(variables - overlap)
                    cleaned_inputs = _rewrite_inputs_for_family(inputs, variables)
                    return (
                        "needs_input",
                        operation,
                        cleaned_inputs,
                        remaining,
                        None,
                    )

            if requested in variables:
                needed = variables - overlap - {requested}
                # Also if overlap somehow includes requested (model put unknown
                # in inputs), treat it as not provided.
                if requested in overlap:
                    overlap = overlap - {requested}
                    needed = variables - overlap - {requested}

                cleaned_inputs = _rewrite_inputs_for_family(inputs, variables)
                cleaned_inputs = {
                    k: v
                    for k, v in cleaned_inputs.items()
                    if _canonical_quantity(k) != requested
                }

                # Choose operation for the intended unknown if we can
                op_name = operation
                for provided_pair, (cand_op, output_name) in solutions.items():
                    if output_name == requested:
                        # Ideal provided set is variables - {requested}
                        if overlap <= (variables - {requested}):
                            op_name = cand_op
                            break

                if not needed:
                    # Should have been handled by len==2 branch; treat as ok
                    for provided_pair, (cand_op, output_name) in solutions.items():
                        if provided_pair == frozenset(overlap) and output_name == requested:
                            return "ok", cand_op, cleaned_inputs, [], requested
                    return "ok", op_name, cleaned_inputs, [], requested

                missing_names = sorted(needed)
                return (
                    "needs_input",
                    op_name,
                    cleaned_inputs,
                    missing_names,
                    requested,
                )

    # No family matched strongly — still strip unknown from missing/inputs
    if requested is not None:
        missing = [
            m
            for m in missing
            if _canonical_quantity(m) != requested
        ]
        inputs = {
            k: v
            for k, v in inputs.items()
            if _canonical_quantity(k) != requested
        }
        if status == "ok" and missing:
            status = "needs_input"
        if status == "needs_input" and not missing and inputs:
            # Model listed only the unknown as missing — promote to ok if
            # inputs look complete enough; otherwise keep needs_input empty
            # only when operation is clear and inputs non-empty without gaps.
            status = "ok"

    return status, operation, inputs, missing, requested


def _validate_result(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate model JSON into the public contract."""
    status = data.get("status")
    if not isinstance(status, str) or status not in VALID_STATUSES:
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "requested_output": None,
            "error": f"invalid status from model: {status!r}",
        }

    operation = data.get("operation")
    if operation is not None and not isinstance(operation, str):
        operation = None
    if isinstance(operation, str):
        operation = operation.strip() or None

    inputs = _normalize_inputs(data.get("inputs"))
    missing = _normalize_missing(data.get("missing_inputs"))
    requested_output = _normalize_requested_output(data.get("requested_output"))

    (
        status,
        operation,
        inputs,
        missing,
        requested_output,
    ) = _apply_relationships(
        status=status,
        operation=operation,
        inputs=inputs,
        missing=missing,
        requested_output=requested_output,
    )

    # Consistency: ok should not list missing required inputs
    if status == "ok" and missing:
        status = "needs_input"

    # needs_input with no operation and no missing names → unsupported
    if status == "needs_input" and operation is None and not missing:
        status = "unsupported"

    # Unknown must never be listed as missing
    if requested_output is not None:
        req_canon = _canonical_quantity(requested_output)
        missing = [m for m in missing if _canonical_quantity(m) != req_canon]
        inputs = {
            k: v
            for k, v in inputs.items()
            if _canonical_quantity(k) != req_canon
        }

    if status == "unsupported":
        return {
            "status": "unsupported",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "requested_output": None,
        }

    result: dict[str, Any] = {
        "status": status,
        "operation": operation,
        "inputs": inputs,
        "missing_inputs": missing,
        "requested_output": requested_output,
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
        "inputs": { name: number, ... },       # provided values only
        "missing_inputs": [str, ...],          # genuinely missing only
        "requested_output": str | None,        # unknown being solved for
      }

    Does not calculate, generate code, run tools, or write to disk.
    """
    if not isinstance(task, str) or not task.strip():
        return {
            "status": "error",
            "operation": None,
            "inputs": {},
            "missing_inputs": [],
            "requested_output": None,
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
            "requested_output": None,
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
            "requested_output": None,
            "error": f"malformed model JSON: {exc}",
        }

    return _validate_result(data)


# ---------------------------------------------------------------------------
# Manual smoke tests (not wired to orchestrator)
# ---------------------------------------------------------------------------
_DEMO_TASKS = [
    "height = 150cm, weight = 50kg calculate bmi",
    "If BMI is 20 and weight is 80 kg, what is the height?",
    "BMI is 22 and height is 170 cm; find the weight",
    "A train travels 120 km in 2 hours. What is its speed?",
    "calculate speed",
    "hello how are you?",
]


if __name__ == "__main__":
    for demo in _DEMO_TASKS:
        print("=" * 60)
        print("task:", demo)
        print(json.dumps(interpret_task(demo), indent=2))
