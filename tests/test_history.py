"""
Tests for grdl_rt.execution.history — ExecutionHistoryDB and execution
record lifecycle.

Verifies that history records are created for success and failure, that
queries filter correctly, and that the integration with WorkflowExecutor
writes accurate timing and status.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-11
"""

import uuid
from unittest.mock import patch

import numpy as np
import pytest

from grdl_rt.execution.history import ExecutionHistoryDB

# ======================================================================
# ExecutionHistoryDB — basic operations
# ======================================================================


class TestExecutionHistoryDB:
    def test_creates_database(self, tmp_path):
        db_path = tmp_path / "history.db"
        db = ExecutionHistoryDB(db_path=db_path)
        assert db_path.exists()
        db.close()

    def test_record_start(self, tmp_path):
        db_path = tmp_path / "history.db"
        db = ExecutionHistoryDB(db_path=db_path)
        run_id = str(uuid.uuid4())
        row_id = db.record_start(
            workflow_id="Test:1.0",
            run_id=run_id,
            workflow_hash="abc123",
            parameters_json='{"name": "Test"}',
        )
        assert isinstance(row_id, int)
        assert row_id > 0

        record = db.get_execution(run_id)
        assert record is not None
        assert record.status == "running"
        assert record.workflow_id == "Test:1.0"
        assert record.workflow_hash == "abc123"
        db.close()

    def test_record_completion_success(self, tmp_path):
        db_path = tmp_path / "history.db"
        db = ExecutionHistoryDB(db_path=db_path)
        run_id = str(uuid.uuid4())
        db.record_start("Test:1.0", run_id, "abc123")
        db.record_completion(
            run_id=run_id,
            status="success",
            step_count=5,
            metrics_json='{"total_wall_time_s": 2.5}',
        )
        record = db.get_execution(run_id)
        assert record.status == "success"
        assert record.step_count == 5
        assert record.end_time is not None
        assert record.metrics_json is not None
        db.close()

    def test_record_completion_failed(self, tmp_path):
        db_path = tmp_path / "history.db"
        db = ExecutionHistoryDB(db_path=db_path)
        run_id = str(uuid.uuid4())
        db.record_start("Test:1.0", run_id, "abc123")
        db.record_completion(
            run_id=run_id,
            status="failed",
            step_count=2,
            error_message="Step 'ProcessorX' failed: division by zero",
        )
        record = db.get_execution(run_id)
        assert record.status == "failed"
        assert record.step_count == 2
        assert "division by zero" in record.error_message
        db.close()

    def test_record_completion_cancelled_with_checkpoint(self, tmp_path):
        db_path = tmp_path / "history.db"
        db = ExecutionHistoryDB(db_path=db_path)
        run_id = str(uuid.uuid4())
        db.record_start("Test:1.0", run_id, "abc123")
        db.record_completion(
            run_id=run_id,
            status="cancelled",
            step_count=3,
            checkpoint_path="/tmp/grdl_checkpoint_run1",
        )
        record = db.get_execution(run_id)
        assert record.status == "cancelled"
        assert record.checkpoint_path == "/tmp/grdl_checkpoint_run1"
        db.close()


# ======================================================================
# Query operations
# ======================================================================


class TestHistoryQueries:
    def _populate_db(self, db_path):
        db = ExecutionHistoryDB(db_path=db_path)
        # Create several runs with different statuses
        for i in range(5):
            run_id = f"run-success-{i}"
            db.record_start(f"Workflow{i}:1.0", run_id, f"hash{i}")
            db.record_completion(run_id, "success", step_count=i + 1)

        for i in range(3):
            run_id = f"run-failed-{i}"
            db.record_start(f"FailWf{i}:1.0", run_id, f"fhash{i}")
            db.record_completion(
                run_id,
                "failed",
                step_count=i,
                error_message=f"Error {i}",
            )

        run_id = "run-running-0"
        db.record_start("Running:1.0", run_id, "rhash")
        return db

    def test_list_all(self, tmp_path):
        db = self._populate_db(tmp_path / "h.db")
        records = db.list_executions(limit=50)
        assert len(records) == 9  # 5 success + 3 failed + 1 running
        db.close()

    def test_list_filter_by_status(self, tmp_path):
        db = self._populate_db(tmp_path / "h.db")
        successes = db.list_executions(status="success")
        assert len(successes) == 5
        for r in successes:
            assert r.status == "success"

        failures = db.list_executions(status="failed")
        assert len(failures) == 3

        running = db.list_executions(status="running")
        assert len(running) == 1
        db.close()

    def test_list_respects_limit(self, tmp_path):
        db = self._populate_db(tmp_path / "h.db")
        records = db.list_executions(limit=3)
        assert len(records) == 3
        db.close()

    def test_list_ordered_by_start_time_desc(self, tmp_path):
        db = self._populate_db(tmp_path / "h.db")
        records = db.list_executions(limit=50)
        for i in range(len(records) - 1):
            assert records[i].start_time >= records[i + 1].start_time
        db.close()

    def test_get_nonexistent_returns_none(self, tmp_path):
        db = ExecutionHistoryDB(db_path=tmp_path / "h.db")
        assert db.get_execution("no-such-run") is None
        db.close()

    def test_count_by_status(self, tmp_path):
        db = self._populate_db(tmp_path / "h.db")
        counts = db.count_by_status()
        assert counts["success"] == 5
        assert counts["failed"] == 3
        assert counts["running"] == 1
        db.close()


# ======================================================================
# Context manager
# ======================================================================


class TestHistoryContextManager:
    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "h.db"
        with ExecutionHistoryDB(db_path=db_path) as db:
            run_id = "ctx-test"
            db.record_start("Test:1.0", run_id, "hash")
            db.record_completion(run_id, "success", step_count=1)

        # DB is closed, reopen to verify data persisted
        db2 = ExecutionHistoryDB(db_path=db_path)
        record = db2.get_execution("ctx-test")
        assert record is not None
        assert record.status == "success"
        db2.close()


# ======================================================================
# Integration: WorkflowExecutor writes history
# ======================================================================


class _SimpleTransform:
    def apply(self, source, **kwargs):
        kwargs.pop("progress_callback", None)
        return source * 2.0


class _FailTransform:
    def apply(self, source, **kwargs):
        raise RuntimeError("intentional failure")


class TestHistoryWithExecutor:
    def _make_workflow(self, n_steps=3):
        from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

        wf = WorkflowDefinition(name="HistTest", version="1.0.0")
        for _ in range(n_steps):
            wf.add_step(ProcessingStep("Transform", "1.0.0"))
        return wf

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_success_recorded(self, mock_resolve, tmp_path):
        """Successful execution creates a history record with status=success."""
        mock_resolve.return_value = _SimpleTransform

        db = ExecutionHistoryDB(db_path=tmp_path / "h.db")
        wf = self._make_workflow(n_steps=3)

        from grdl_rt.execution.executor import WorkflowExecutor

        executor = WorkflowExecutor(wf, history_db=db)

        source = np.ones((2, 2), dtype=np.float64)
        wr = executor.execute(
            source,
            enable_memory_check=False,
            enable_shutdown_handler=False,
        )
        assert wr.metrics.status == "success"

        records = db.list_executions()
        assert len(records) == 1
        assert records[0].status == "success"
        assert records[0].step_count == 3
        assert records[0].workflow_id == "HistTest:1.0.0"
        assert records[0].end_time is not None
        assert records[0].metrics_json is not None
        db.close()

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_failure_recorded(self, mock_resolve, tmp_path):
        """Failed execution creates a history record with status=failed."""
        mock_resolve.return_value = _FailTransform

        db = ExecutionHistoryDB(db_path=tmp_path / "h.db")
        wf = self._make_workflow(n_steps=3)

        from grdl_rt.execution.executor import WorkflowExecutor

        executor = WorkflowExecutor(wf, history_db=db)

        source = np.ones((2, 2), dtype=np.float64)
        with pytest.raises(RuntimeError, match="intentional failure"):
            executor.execute(
                source,
                enable_memory_check=False,
                enable_shutdown_handler=False,
            )

        records = db.list_executions()
        assert len(records) == 1
        assert records[0].status == "failed"
        assert "intentional failure" in records[0].error_message
        db.close()

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_accurate_timing(self, mock_resolve, tmp_path):
        """History record has accurate start_time and end_time."""
        mock_resolve.return_value = _SimpleTransform

        db = ExecutionHistoryDB(db_path=tmp_path / "h.db")
        wf = self._make_workflow(n_steps=1)

        from grdl_rt.execution.executor import WorkflowExecutor

        executor = WorkflowExecutor(wf, history_db=db)

        source = np.ones((2, 2))
        executor.execute(
            source,
            enable_memory_check=False,
            enable_shutdown_handler=False,
        )

        records = db.list_executions()
        r = records[0]
        assert r.start_time is not None
        assert r.end_time is not None
        # end_time should be >= start_time
        assert r.end_time >= r.start_time
        db.close()

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_multiple_executions_tracked(self, mock_resolve, tmp_path):
        """Each execution gets its own history record."""
        mock_resolve.return_value = _SimpleTransform

        db = ExecutionHistoryDB(db_path=tmp_path / "h.db")
        wf = self._make_workflow(n_steps=2)

        from grdl_rt.execution.executor import WorkflowExecutor

        executor = WorkflowExecutor(wf, history_db=db)

        source = np.ones((2, 2))
        for _ in range(3):
            executor.execute(
                source,
                enable_memory_check=False,
                enable_shutdown_handler=False,
            )

        records = db.list_executions()
        assert len(records) == 3
        # Each should have a unique run_id
        run_ids = {r.run_id for r in records}
        assert len(run_ids) == 3
        db.close()
