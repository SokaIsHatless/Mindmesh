"""SQLite persistence for capability metadata only."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # Supports ``backend.capabilities`` package and backend test discovery.
    from .models import Capability
    from ..models import InputSpec, OutputSpec
except ImportError:  # pragma: no cover - exercised by discovery import layout
    from capabilities.models import Capability
    from models import InputSpec, OutputSpec


class CapabilityRepository:
    """Store and retrieve declarative capability metadata in SQLite."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = (
            Path(database_path)
            if database_path is not None
            else Path(__file__).resolve().parent / "capabilities.sqlite3"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("CapabilityRepository is closed")
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit/rollback the transaction, then always close.

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
        """Mark the repository closed. Safe to call more than once.

        Connections are opened per operation and closed before each method
        returns; ``close`` exists so callers (and tests) can end the lifecycle
        explicitly before deleting the database file on Windows.
        """
        self._closed = True

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS capabilities (
                    tool_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    description TEXT NOT NULL,
                    input_schema_json TEXT NOT NULL,
                    output_schema_json TEXT NOT NULL,
                    code_path TEXT NOT NULL,
                    verification_status TEXT NOT NULL
                        CHECK (verification_status IN ('pending', 'verified', 'failed')),
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    PRIMARY KEY (tool_id, version)
                );

                CREATE TABLE IF NOT EXISTS capability_aliases (
                    alias TEXT PRIMARY KEY,
                    tool_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    FOREIGN KEY (tool_id, version)
                        REFERENCES capabilities (tool_id, version)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_capabilities_operation
                    ON capabilities (operation, tool_id, version);
                """
            )

    @staticmethod
    def _to_record(capability: Capability) -> tuple[object, ...]:
        return (
            capability.tool_id,
            capability.version,
            capability.operation,
            capability.description,
            json.dumps(
                [schema.model_dump(mode="json") for schema in capability.input_schema],
                sort_keys=True,
            ),
            json.dumps(capability.output_schema.model_dump(mode="json"), sort_keys=True),
            capability.code_path,
            capability.verification_status,
            int(capability.enabled),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row, aliases: list[str]) -> Capability:
        return Capability(
            tool_id=row["tool_id"],
            operation=row["operation"],
            version=row["version"],
            description=row["description"],
            aliases=aliases,
            input_schema=[
                InputSpec.model_validate(value)
                for value in json.loads(row["input_schema_json"])
            ],
            output_schema=OutputSpec.model_validate(
                json.loads(row["output_schema_json"])
            ),
            code_path=row["code_path"],
            verification_status=row["verification_status"],
            enabled=bool(row["enabled"]),
        )

    def register(self, capability: Capability) -> Capability:
        """Atomically upsert one capability and replace its aliases."""
        record = self._to_record(capability)
        with self._connection() as connection:
            for alias in capability.aliases:
                owner = connection.execute(
                    """
                    SELECT tool_id, version FROM capability_aliases
                    WHERE alias = ?
                    """,
                    (alias,),
                ).fetchone()
                if owner is not None and (
                    owner["tool_id"] != capability.tool_id
                    or owner["version"] != capability.version
                ):
                    raise ValueError(
                        f"alias '{alias}' is already registered to "
                        f"{owner['tool_id']} version {owner['version']}"
                    )

            connection.execute(
                """
                INSERT INTO capabilities (
                    tool_id, version, operation, description, input_schema_json,
                    output_schema_json, code_path, verification_status, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id, version) DO UPDATE SET
                    operation = excluded.operation,
                    description = excluded.description,
                    input_schema_json = excluded.input_schema_json,
                    output_schema_json = excluded.output_schema_json,
                    code_path = excluded.code_path,
                    verification_status = excluded.verification_status,
                    enabled = excluded.enabled
                """,
                record,
            )
            connection.execute(
                "DELETE FROM capability_aliases WHERE tool_id = ? AND version = ?",
                (capability.tool_id, capability.version),
            )
            connection.executemany(
                """
                INSERT INTO capability_aliases (alias, tool_id, version)
                VALUES (?, ?, ?)
                """,
                [
                    (alias, capability.tool_id, capability.version)
                    for alias in capability.aliases
                ],
            )
        return capability

    @staticmethod
    def _aliases_for(
        connection: sqlite3.Connection, tool_id: str, version: str
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT alias FROM capability_aliases
            WHERE tool_id = ? AND version = ?
            ORDER BY alias
            """,
            (tool_id, version),
        ).fetchall()
        return [row["alias"] for row in rows]

    def _capabilities_from_rows(
        self, connection: sqlite3.Connection, rows: list[sqlite3.Row]
    ) -> list[Capability]:
        return [
            self._from_row(
                row,
                self._aliases_for(connection, row["tool_id"], row["version"]),
            )
            for row in rows
        ]

    def get_by_operation(self, operation: str) -> list[Capability]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM capabilities WHERE operation = ?
                ORDER BY tool_id, version
                """,
                (operation,),
            ).fetchall()
            return self._capabilities_from_rows(connection, rows)

    def get_by_alias(self, alias: str) -> Capability | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT capabilities.* FROM capabilities
                INNER JOIN capability_aliases
                    ON capabilities.tool_id = capability_aliases.tool_id
                    AND capabilities.version = capability_aliases.version
                WHERE capability_aliases.alias = ?
                """,
                (alias,),
            ).fetchone()
            if row is None:
                return None
            return self._from_row(
                row, self._aliases_for(connection, row["tool_id"], row["version"])
            )

    def get_version(self, tool_id: str, version: str) -> Capability | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM capabilities WHERE tool_id = ? AND version = ?",
                (tool_id, version),
            ).fetchone()
            if row is None:
                return None
            return self._from_row(
                row, self._aliases_for(connection, row["tool_id"], row["version"])
            )

    def list_capabilities(self, enabled: bool | None = None) -> list[Capability]:
        with self._connection() as connection:
            if enabled is None:
                rows = connection.execute(
                    "SELECT * FROM capabilities ORDER BY tool_id, version"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM capabilities WHERE enabled = ?
                    ORDER BY tool_id, version
                    """,
                    (int(enabled),),
                ).fetchall()
            return self._capabilities_from_rows(connection, rows)
