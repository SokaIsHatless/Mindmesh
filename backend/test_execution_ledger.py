"""Unit tests for the standalone execution ledger / telemetry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .execution_ledger import ExecutionLedger
except ImportError:  # ``unittest discover -s backend``
    from execution_ledger import ExecutionLedger


class ExecutionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "execution_ledger.sqlite3"
        self.ledger = ExecutionLedger(self.db_path)
        self.addCleanup(self.ledger.close)

    def test_create_and_retrieve_execution(self) -> None:
        run_id = self.ledger.create_run("compute train speed")
        record = self.ledger.get_run(run_id)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["execution_id"], run_id)
        self.assertEqual(record["original_task"], "compute train speed")
        self.assertEqual(record["final_status"], "in_progress")
        self.assertIsNone(record["tool_name"])
        self.assertIsNone(record["tool_version"])
        self.assertIsNone(record["final_answer"])
        self.assertIsNone(record["error"])
        self.assertIsNone(record["completed_at"])
        self.assertEqual(record["stages"], [])
        self.assertTrue(record["timestamp"])

    def test_stage_recording(self) -> None:
        run_id = self.ledger.create_run("compute train speed")
        stage_id = self.ledger.append_stage(
            run_id,
            "interpret",
            status="completed",
            detail={"inputs": ["distance_km", "time_hr"]},
        )
        self.ledger.append_stage(run_id, "resolve", status="started")
        self.ledger.update_stage(run_id, "resolve", status="completed")

        record = self.ledger.get_run(run_id)
        assert record is not None
        self.assertEqual(len(record["stages"]), 2)
        self.assertEqual(record["stages"][0]["id"], stage_id)
        self.assertEqual(record["stages"][0]["stage_name"], "interpret")
        self.assertEqual(record["stages"][0]["status"], "completed")
        self.assertEqual(
            record["stages"][0]["detail"],
            {"inputs": ["distance_km", "time_hr"]},
        )
        self.assertEqual(record["stages"][0]["sequence"], 1)
        self.assertEqual(record["stages"][1]["stage_name"], "resolve")
        self.assertEqual(record["stages"][1]["status"], "completed")
        self.assertEqual(record["stages"][1]["sequence"], 2)

    def test_successful_completion(self) -> None:
        run_id = self.ledger.create_run("compute train speed")
        self.ledger.append_stage(run_id, "execute", status="completed")
        self.ledger.complete_run(
            run_id,
            final_answer={"speed_kmh": 60},
            tool_name="speed",
            tool_version="1.0.0",
        )

        record = self.ledger.get_run(run_id)
        assert record is not None
        self.assertEqual(record["final_status"], "completed")
        self.assertEqual(record["final_answer"], {"speed_kmh": 60})
        self.assertEqual(record["tool_name"], "speed")
        self.assertEqual(record["tool_version"], "1.0.0")
        self.assertIsNone(record["error"])
        self.assertTrue(record["completed_at"])

    def test_failed_execution(self) -> None:
        run_id = self.ledger.create_run("compute train speed")
        self.ledger.append_stage(
            run_id, "verify", status="failed", detail={"reason": "tests failed"}
        )
        self.ledger.record_failure(run_id, "verification failed")

        record = self.ledger.get_run(run_id)
        assert record is not None
        self.assertEqual(record["final_status"], "failed")
        self.assertEqual(record["error"], "verification failed")
        self.assertTrue(record["completed_at"])
        self.assertEqual(len(record["stages"]), 1)
        self.assertEqual(record["stages"][0]["status"], "failed")

    def test_tool_version_metadata(self) -> None:
        run_id = self.ledger.create_run(
            "compute train speed",
            tool_name="speed",
            tool_version="1.0.0",
        )
        self.ledger.complete_run(
            run_id,
            final_answer=60,
            tool_name="speed",
            tool_version="1.2.0",
        )

        record = self.ledger.get_run(run_id)
        assert record is not None
        self.assertEqual(record["tool_name"], "speed")
        self.assertEqual(record["tool_version"], "1.2.0")
        self.assertEqual(record["final_answer"], 60)

        # Failure path can also retain tool metadata.
        failed_id = self.ledger.create_run("another task")
        self.ledger.record_failure(
            failed_id,
            "factory failed",
            tool_name="speed",
            tool_version="1.0.0",
        )
        failed = self.ledger.get_run(failed_id)
        assert failed is not None
        self.assertEqual(failed["tool_name"], "speed")
        self.assertEqual(failed["tool_version"], "1.0.0")

    def test_multiple_executions_remain_separate(self) -> None:
        first = self.ledger.create_run("task one")
        second = self.ledger.create_run("task two")
        self.ledger.append_stage(first, "interpret", status="completed")
        self.ledger.append_stage(second, "resolve", status="started")
        self.ledger.complete_run(first, final_answer="a")
        self.ledger.record_failure(second, "boom")

        left = self.ledger.get_run(first)
        right = self.ledger.get_run(second)
        assert left is not None and right is not None

        self.assertNotEqual(left["execution_id"], right["execution_id"])
        self.assertEqual(left["original_task"], "task one")
        self.assertEqual(right["original_task"], "task two")
        self.assertEqual(left["final_status"], "completed")
        self.assertEqual(right["final_status"], "failed")
        self.assertEqual([s["stage_name"] for s in left["stages"]], ["interpret"])
        self.assertEqual([s["stage_name"] for s in right["stages"]], ["resolve"])

    def test_recent_run_listing(self) -> None:
        ids = [
            self.ledger.create_run(f"task {index}")
            for index in range(3)
        ]
        recent = self.ledger.list_recent_runs(limit=2)

        self.assertEqual(len(recent), 2)
        # Newest first.
        self.assertEqual(recent[0]["execution_id"], ids[-1])
        self.assertEqual(recent[1]["execution_id"], ids[-2])
        self.assertEqual(recent[0]["original_task"], "task 2")

        all_runs = self.ledger.list_recent_runs(limit=10)
        self.assertEqual(len(all_runs), 3)

    def test_persistence_across_ledger_instances(self) -> None:
        run_id = self.ledger.create_run("persisted task")
        self.ledger.append_stage(run_id, "execute", status="completed")
        self.ledger.complete_run(
            run_id,
            final_answer={"value": 42},
            tool_name="speed",
            tool_version="1.0.0",
        )
        self.ledger.close()

        reopened = ExecutionLedger(self.db_path)
        try:
            record = reopened.get_run(run_id)
            assert record is not None
            self.assertEqual(record["original_task"], "persisted task")
            self.assertEqual(record["final_status"], "completed")
            self.assertEqual(record["final_answer"], {"value": 42})
            self.assertEqual(record["tool_name"], "speed")
            self.assertEqual(record["tool_version"], "1.0.0")
            self.assertEqual(len(record["stages"]), 1)
            self.assertEqual(record["stages"][0]["stage_name"], "execute")
        finally:
            reopened.close()

    def test_sqlite_resources_released_cleanly(self) -> None:
        """Windows must be able to delete the DB after the ledger is closed."""
        isolated = tempfile.TemporaryDirectory()
        try:
            root = Path(isolated.name)
            database_path = root / "execution_ledger.sqlite3"
            ledger = ExecutionLedger(database_path)
            run_id = ledger.create_run("cleanup task")
            ledger.append_stage(run_id, "done", status="completed")
            self.assertTrue(database_path.is_file())
            ledger.close()
            database_path.unlink()
            self.assertFalse(database_path.exists())
        finally:
            isolated.cleanup()

    def test_ledger_does_not_store_code_fields(self) -> None:
        """Schema is metadata-only — no executable code column."""
        run_id = self.ledger.create_run("compute train speed")
        self.ledger.complete_run(run_id, final_answer=60)
        record = self.ledger.get_run(run_id)
        assert record is not None
        self.assertNotIn("code", record)
        self.assertNotIn("source", record)
        for stage in record["stages"]:
            self.assertNotIn("code", stage)


if __name__ == "__main__":
    unittest.main()
