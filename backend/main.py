"""FastAPI entry point for EDGE-SMART orchestrator."""

from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator import handle_task

app = FastAPI(title="EDGE-SMART")


class RunRequest(BaseModel):
    task: str


@app.post("/run")
def run(req: RunRequest):
    """Accept {"task": "..."} and return answer + reasoning trace."""
    return handle_task(req.task)
