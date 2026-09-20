"""
execution_ledger.py — Persistent execution history / telemetry for Mindmesh runs.

Metadata and pipeline history only. Never stores executable Python tool code.
Independent of Ollama; uses SQLite under backend/runtime/.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_RUNTIME_DIR = _BACKEND_DIR / "runtime"
DEFAULT_DATABASE_PATH = DEFAULT_RUNTIME_DIR / "execution_ledger.sqlite3"

RunStatus = str  # in_progress | completed | failed


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_or_none(value: Any) -> str | None:
    """Serialize structured values for storage; never invent content."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _parse_json_maybe(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class ExecutionLedger:
    """Small SQLite ledger for structured Mindmesh run history."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = (
            Path(database_path)
            if database_path is not None
            else DEFAULT_DATABASE_PATH
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("ExecutionLedger is closed")
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit/rollback, then always close.

        ``sqlite3.Connection`` as a context manager only commits or rolls back;
        it does **not** close the connection. Leaving it open keeps the database
        file locked on Windows and breaks temporary-directory cleanup.
        """
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def close(self) -> None:
        """Mark the ledger closed. Safe to call more than once."""
        self._closed = True

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    original_task TEXT NOT NULL,
                    final_status TEXT NOT NULL
                        CHECK (final_status IN (
                            'in_progress', 'completed', 'failed'
                        )),
                    tool_name TEXT,
                    tool_version TEXT,
                    final_answer TEXT,
                    error TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS execution_stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    recorded_at TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    FOREIGN KEY (execution_id)
                        REFERENCES executions (execution_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_executions_timestamp
                    ON executions (timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_stages_execution
                    ON execution_stages (execution_id, sequence);
                """
            )

    def create_run(
        self,
        original_task: str,
        *,
        execution_id: str | None = None,
        tool_name: str | None = None,
        tool_version: str | None = None,
    ) -> str:
        """Create a new in-progress execution and return its id."""
        if not isinstance(original_task, str) or not original_task.strip():
            raise ValueError("original_task must be a non-blank string")

        run_id = execution_id or str(uuid.uuid4())
        timestamp = _utc_now_iso()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO executions (
                    execution_id,
                    timestamp,
                    original_task,
                    final_status,
                    tool_name,
                    tool_version,
                    final_answer,
                    error,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    run_id,
                    timestamp,
                    original_task.strip(),
                    "in_progress",
                    tool_name,
                    tool_version,
                ),
            )
        return run_id

    def append_stage(
        self,
        execution_id: str,
        stage_name: str,
        *,
        status: str = "recorded",
        detail: Any | None = None,
    ) -> int:
        """Append a pipeline stage record. Returns the stage row id."""
        if not stage_name or not str(stage_name).strip():
            raise ValueError("stage_name must be a non-blank string")

        recorded_at = _utc_now_iso()
        detail_text = _json_or_none(detail)

        with self._connection() as connection:
            self._require_execution(connection, execution_id)
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS max_seq
                FROM execution_stages
                WHERE execution_id = ?
                """,
                (execution_id,),
            ).fetchone()
            sequence = int(row["max_seq"]) + 1
            cursor = connection.execute(
                """
                INSERT INTO execution_stages (
                    execution_id,
                    stage_name,
                    status,
                    detail,
                    recorded_at,
                    sequence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    stage_name.strip(),
                    status,
                    detail_text,
                    recorded_at,
                    sequence,
                ),
            )
            return int(cursor.lastrowid)

    def update_stage(
        self,
        execution_id: str,
        stage_name: str,
        *,
        status: str | None = None,
        detail: Any | None = None,
    ) -> None:
        """Update the most recent stage with the given name for this run."""
        if status is None and detail is None:
            raise ValueError("update_stage requires status and/or detail")

        with self._connection() as connection:
            self._require_execution(connection, execution_id)
            row = connection.execute(
                """
                SELECT id FROM execution_stages
                WHERE execution_id = ? AND stage_name = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (execution_id, stage_name.strip()),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"no stage named {stage_name!r} for execution "
                    f"{execution_id!r}"
                )

            fields: list[str] = []
            values: list[Any] = []
            if status is not None:
                fields.append("status = ?")
                values.append(status)
            if detail is not None:
                fields.append("detail = ?")
                values.append(_json_or_none(detail))
            values.append(row["id"])
            connection.execute(
                f"UPDATE execution_stages SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )

    def complete_run(
        self,
        execution_id: str,
        *,
        final_answer: Any | None = None,
        tool_name: str | None = None,
        tool_version: str | None = None,
    ) -> None:
        """Mark a run completed and store the final answer/result metadata."""
        completed_at = _utc_now_iso()
        with self._connection() as connection:
            self._require_execution(connection, execution_id)
            connection.execute(
                """
                UPDATE executions
                SET final_status = 'completed',
                    final_answer = ?,
                    error = NULL,
                    completed_at = ?,
                    tool_name = COALESCE(?, tool_name),
                    tool_version = COALESCE(?, tool_version)
                WHERE execution_id = ?
                """,
                (
                    _json_or_none(final_answer),
                    completed_at,
                    tool_name,
                    tool_version,
                    execution_id,
                ),
            )

    def record_failure(
        self,
        execution_id: str,
        error: str,
        *,
        tool_name: str | None = None,
        tool_version: str | None = None,
        final_answer: Any | None = None,
    ) -> None:
        """Mark a run failed and record the error message."""
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a non-blank string")

        completed_at = _utc_now_iso()
        with self._connection() as connection:
            self._require_execution(connection, execution_id)
            connection.execute(
                """
                UPDATE executions
                SET final_status = 'failed',
                    error = ?,
                    final_answer = COALESCE(?, final_answer),
                    completed_at = ?,
                    tool_name = COALESCE(?, tool_name),
                    tool_version = COALESCE(?, tool_version)
                WHERE execution_id = ?
                """,
                (
                    error.strip(),
                    _json_or_none(final_answer),
                    completed_at,
                    tool_name,
                    tool_version,
                    execution_id,
                ),
            )

    def get_run(self, execution_id: str) -> dict[str, Any] | None:
        """Retrieve one execution including ordered stage history."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT execution_id, timestamp, original_task, final_status,
                       tool_name, tool_version, final_answer, error,
                       completed_at
                FROM executions
                WHERE execution_id = ?
                """,
                (execution_id,),
            ).fetchone()
            if row is None:
                return None

            stage_rows = connection.execute(
                """
                SELECT id, stage_name, status, detail, recorded_at, sequence
                FROM execution_stages
                WHERE execution_id = ?
                ORDER BY sequence ASC
                """,
                (execution_id,),
            ).fetchall()

        return self._execution_dict(row, stage_rows)

    def list_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent executions (newest first), each with stages."""
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT execution_id, timestamp, original_task, final_status,
                       tool_name, tool_version, final_answer, error,
                       completed_at
                FROM executions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                stage_rows = connection.execute(
                    """
                    SELECT id, stage_name, status, detail, recorded_at,
                           sequence
                    FROM execution_stages
                    WHERE execution_id = ?
                    ORDER BY sequence ASC
                    """,
                    (row["execution_id"],),
                ).fetchall()
                results.append(self._execution_dict(row, stage_rows))
            return results

    @staticmethod
    def _require_execution(
        connection: sqlite3.Connection, execution_id: str
    ) -> None:
        row = connection.execute(
            "SELECT 1 FROM executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown execution_id: {execution_id!r}")

    @staticmethod
    def _execution_dict(
        row: sqlite3.Row, stage_rows: list[sqlite3.Row]
    ) -> dict[str, Any]:
        return {
            "execution_id": row["execution_id"],
            "timestamp": row["timestamp"],
            "original_task": row["original_task"],
            "final_status": row["final_status"],
            "tool_name": row["tool_name"],
            "tool_version": row["tool_version"],
            "final_answer": _parse_json_maybe(row["final_answer"]),
            "error": row["error"],
            "completed_at": row["completed_at"],
            "stages": [
                {
                    "id": stage["id"],
                    "stage_name": stage["stage_name"],
                    "status": stage["status"],
                    "detail": _parse_json_maybe(stage["detail"]),
                    "recorded_at": stage["recorded_at"],
                    "sequence": stage["sequence"],
                }
                for stage in stage_rows
            ],
        }
