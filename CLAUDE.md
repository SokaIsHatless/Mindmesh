# CLAUDE.md — Guidance for Claude Code on this project

## Project: EDGE-SMART (team MindMesh)
An offline AI agent that writes its own Python tools. When it hits a task it
has no tool for, it generates a small pure-function tool with a local model
(Phi-4-mini via Ollama), tests that tool in a sandbox, and saves it for reuse —
all fully offline. See README.md for the full picture.

## ⚠️ MY LANE — read this first
I (Claude Code, used by Soka) work ONLY on the **Tool Factory**.

**Files I own and may edit:**
- `backend/tool_factory.py`   ← my main file

**Files I may READ but must NOT edit** (owned by teammates):
- `backend/Sandbox.py`        ← owned by Miss S. I import `run_tool` from it; I do not change it.
- `backend/orchestrator.py`   ← owned by Miss O (Cursor)
- `backend/main.py`           ← owned by Miss O (Cursor)

**Directories that are OFF-LIMITS — never create, edit, or delete files here:**
- `frontend/`                 ← Miss S's Next.js app. Not my territory at all.
- any config at the repo root (package.json, next.config.ts, tsconfig.json, etc.)

If a task seems to need changing a file I don't own, STOP and tell Soka to
coordinate with that file's owner instead of editing it myself.

## What the Tool Factory does (my job)
`tool_factory.py` exposes one main function:

```python
def create_tool(task_spec: dict) -> dict:
    # 1. Prompt the local model to write a pure Python function for the task
    # 2. Call Sandbox.run_tool() to test the generated code against task_spec["tests"]
    # 3. On pass  -> save the .py to backend/toolbox/ + update manifest.json, return success
    # 4. On fail  -> feed the error back, regenerate, retry up to N times, then give up gracefully
```

**Input (`task_spec`) and output shape** — this is the CONTRACT with the
orchestrator (Miss O). Do not change it without telling her:

```python
task_spec = {
    "name": "speed",
    "description": "compute speed from distance and time",
    "inputs": ["distance_km", "time_hr"],
    "tests": [ {"args": [120, 2], "expected": 60}, ... ],
}
returns = {
    "success": True/False,
    "tool_name": "speed",
    "code": "def speed(...): ...",
    "attempts": 2,
    "error": None,
}
```

## How to call the sandbox (owned by Miss S — I only import it)
```python
from Sandbox import run_tool   # note: capital S, matches the filename
result = run_tool(code, entry_call)   # returns {ok, stdout, stderr, reason}
```
Treat `run_tool` as a black box that safely runs code and reports pass/fail.
I do not modify Sandbox.py.

## Toolbox storage (I write here)
- Generated tools are saved as `backend/toolbox/<name>.py`
- Metadata indexed in `backend/toolbox/manifest.json`:
  `[{ "name", "signature", "description", "created_at" }, ...]`
- No database. The filesystem is the toolbox.

## Tech + conventions
- Python + FastAPI backend, model served locally via Ollama (Phi-4-mini).
- Tools must be PURE functions: input -> output, no files/network/OS.
- Allowed imports in generated tools: math, datetime, json, re, random,
  statistics (must match Sandbox.py's allowlist).
- Keep it simple — flat files in backend/, simple imports, no over-engineering.

## Scope discipline
This is a hackathon. Build the ONE demo task (train-speed) working cleanly
end to end before anything else. Do not add features, extra tools, or
abstractions that aren't needed for the core loop.
