# AGENTS.md — Guidance for Cursor on this project

## Project: EDGE-SMART (team MindMesh)
An offline AI agent that writes its own Python tools. When it hits a task it
has no tool for, it generates a small pure-function tool with a local model
(Phi-4-mini via Ollama), tests that tool in a sandbox, and saves it for reuse —
all fully offline. See README.md for the full picture.

## ⚠️ MY LANE — read this first
I (Cursor, used by Miss O) work ONLY on the **Orchestrator + backend entry**.

**Files I own and may edit:**
- `backend/orchestrator.py`   ← the agent loop / decision logic
- `backend/main.py`           ← FastAPI entry point
- `backend/toolbox/manifest.json` (read/write as part of toolbox logic)

**Files I may READ but must NOT edit** (owned by teammates):
- `backend/tool_factory.py`   ← owned by Soka (Claude Code). I CALL it; I do not change it.
- `backend/Sandbox.py`        ← owned by Miss S.

**Directories that are OFF-LIMITS — never create, edit, or delete files here:**
- `frontend/`                 ← Miss S's Next.js app. Not my territory at all.
- any config at the repo root (package.json, next.config.ts, tsconfig.json, etc.)

If a task seems to need changing a file I don't own, STOP and tell Miss O to
coordinate with that file's owner instead of editing it directly.

## What the Orchestrator does (my job)
`orchestrator.py` runs the agent loop. For each incoming task it decides:

1. **Trivial** (e.g. "what's 2+5") -> answer directly, do NOT build a tool.
2. **Have a tool already** -> load it from the toolbox and run it.
3. **Need a tool I don't have** -> call the tool factory (Soka's code).

Then it observes the result and returns the answer + a reasoning trace.

## How to call the tool factory (owned by Soka — I only call it)
This is the CONTRACT. Do not change these shapes without telling Soka.

```python
from tool_factory import create_tool

task_spec = {
    "name": "speed",
    "description": "compute speed from distance and time",
    "inputs": ["distance_km", "time_hr"],
    "tests": [ {"args": [120, 2], "expected": 60}, ... ],
}
result = create_tool(task_spec)
# result = { "success": bool, "tool_name": str, "code": str, "attempts": int, "error": str|None }
```

**While the real factory isn't ready:** mock `create_tool()` to return a
hardcoded success dict, so I can build my loop without waiting on Soka.

## Toolbox storage
- Tools live as `backend/toolbox/<name>.py`
- Indexed in `backend/toolbox/manifest.json`:
  `[{ "name", "signature", "description", "created_at" }, ...]`
- No database. The filesystem is the toolbox.
- To check "do I have a tool for this?", read manifest.json.

## Tech + conventions
- Python + FastAPI. Model served locally via Ollama.
- Keep imports simple (flat files in backend/): `from tool_factory import create_tool`.
- The frontend talks to this backend — expose the reasoning trace so the
  Next.js UI can stream it (SSE or websocket). Coordinate the endpoint shape
  with Miss S, but build the backend side here.

## Scope discipline
This is a hackathon. Get the decision loop working for the ONE demo task
(train-speed) end to end first. The "don't build a tool for trivial tasks"
guard matters — a judge will test "2+5". Don't over-engineer.
