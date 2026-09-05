"""FastAPI entry point for EDGE-SMART orchestrator."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orchestrator import handle_task

app = FastAPI(title="EDGE-SMART")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    task: str


@app.post("/run")
def run(req: RunRequest):
    """Accept {"task": "..."} and return {"answer", "trace"} for the frontend."""
    return handle_task(req.task)
