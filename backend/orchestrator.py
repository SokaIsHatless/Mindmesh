"""Agent orchestrator — skeleton with mocked tool factory."""

import re


# ---------------------------------------------------------------------------
# MOCK — replace with: from tool_factory import create_tool
# ---------------------------------------------------------------------------
def create_tool(task_spec: dict) -> dict:
    """Stub matching Soka's tool_factory.create_tool contract."""
    return {
        "success": True,
        "tool_name": task_spec["name"],
        "code": "def speed(d,t): return d/t",
        "attempts": 1,
        "error": None,
    }


def _is_trivial_arithmetic(task: str) -> bool:
    """Placeholder: treat simple digit+op+digit expressions as trivial."""
    return bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", task))


def _eval_trivial(task: str) -> str:
    """Placeholder direct answer for trivial arithmetic."""
    match = re.search(r"(\d+)\s*([\+\-\*/])\s*(\d+)", task)
    if not match:
        return "0"
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    ops = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}
    return str(ops[op])


def handle_task(task: str) -> dict:
    """
    Decision stub (real logic Friday):
      1. trivial arithmetic -> answer directly, skip factory
      2. otherwise          -> call mocked create_tool()
    """
    if _is_trivial_arithmetic(task):
        answer = _eval_trivial(task)
        return {
            "path": "trivial",
            "answer": answer,
            "trace": ["detected trivial arithmetic", f"answered directly: {answer}"],
        }

    # Placeholder task_spec — real parsing comes later
    task_spec = {
        "name": "speed",
        "description": task,
        "inputs": ["distance_km", "time_hr"],
        "tests": [{"args": [120, 2], "expected": 60}],
    }
    result = create_tool(task_spec)
    return {
        "path": "factory",
        "answer": result,
        "trace": ["not trivial — calling tool factory (mock)", f"factory result: {result}"],
    }
