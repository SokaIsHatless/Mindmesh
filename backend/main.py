"""FastAPI entry point — minimal skeleton."""

from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator import handle_task

app = FastAPI(title="EDGE-SMART")


class RunRequest(BaseModel):
    task: str


@app.post("/run")
def run(req: RunRequest):
    return handle_task(req.task)
