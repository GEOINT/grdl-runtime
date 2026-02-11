# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.checkpoint — CheckpointState, CheckpointManager,
workflow hashing, atomic writes, and resume validation.

Also tests the full checkpoint/resume lifecycle through WorkflowExecutor:
run a multi-step workflow, simulate failure at step 3, resume, and verify
steps 1–3 are skipped.

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

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from grdl_rt.execution.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    CheckpointState,
    compute_workflow_hash,
)
from grdl_rt.execution.errors import CheckpointError, ResumeError


# ======================================================================
# compute_workflow_hash
# ======================================================================


class TestComputeWorkflowHash:
    def test_deterministic(self):
        wf = {"name": "Test", "version": "1.0", "steps": []}
        h1 = compute_workflow_hash(wf)
        h2 = compute_workflow_hash(wf)
        assert h1 == h2

    def test_different_steps_different_hash(self):
        wf1 = {
            "name": "Test", "version": "1.0",
            "steps": [{"processor": "A", "version": "1.0"}],
        }
        wf2 = {
            "name": "Test", "version": "1.0",
            "steps": [{"processor": "B", "version": "1.0"}],
        }
        assert compute_workflow_hash(wf1) != compute_workflow_hash(wf2)

    def test_description_change_does_not_change_hash(self):
        """Documentation-only changes must not invalidate checkpoints."""
        wf1 = {
            "name": "Test", "version": "1.0",
            "description": "Original",
            "steps": [{"processor": "A"}],
        }
        wf2 = {
            "name": "Test", "version": "1.0",
            "description": "Updated description",
            "steps": [{"processor": "A"}],
        }
        assert compute_workflow_hash(wf1) == compute_workflow_hash(wf2)

    def test_name_change_changes_hash(self):
        wf1 = {"name": "Alpha", "version": "1.0", "steps": []}
        wf2 = {"name": "Beta", "version": "1.0", "steps": []}
        assert compute_workflow_hash(wf1) != compute_workflow_hash(wf2)

    def test_returns_hex_string(self):
        h = compute_workflow_hash({"name": "X", "version": "1", "steps": []})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex


# ======================================================================
# CheckpointState
# ======================================================================


class TestCheckpointState:
    def test_defaults(self):
        s = CheckpointState()
        assert s.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert s.step_index == -1
        assert s.intermediate_files == []
        assert s.metrics_so_far == []

    def test_roundtrip(self):
        s = CheckpointState(
            schema_version="2.0",
            workflow_definition_hash="abc123",
            step_index=4,
            intermediate_files=["/tmp/a.npy", "/tmp/b.npy"],
            metrics_so_far=[{"step_index": 0, "wall_time_s": 1.5}],
            execution_context={"run_id": "test-run"},
            timestamp="2026-02-11T12:00:00Z",
            workflow_dict={"name": "Test", "steps": []},
        )
        d = s.to_dict()
        s2 = CheckpointState.from_dict(d)
        assert s2.schema_version == "2.0"
        assert s2.workflow_definition_hash == "abc123"
        assert s2.step_index == 4
        assert len(s2.intermediate_files) == 2
        assert s2.execution_context["run_id"] == "test-run"

    def test_to_dict_is_json_serializable(self):
        s = CheckpointState(
            workflow_definition_hash="hash",
            step_index=2,
            workflow_dict={"name": "Test"},
        )
        # Must not raise
        json.dumps(s.to_dict())


# ======================================================================
# CheckpointManager — writing
# ======================================================================


class TestCheckpointManagerWrite:
    def test_write_step_intermediate(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = mgr.write_step_intermediate("run-1", 0, arr)
        assert Path(path).exists()
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, arr)

    def test_write_checkpoint_creates_json(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        state = CheckpointState(
            workflow_definition_hash="abc",
            step_index=2,
            timestamp="2026-02-11T12:00:00Z",
        )
        ckpt_path = mgr.write_checkpoint(state, "run-1")
        assert ckpt_path.exists()
        data = json.loads(ckpt_path.read_text())
        assert data["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert data["step_index"] == 2

    def test_write_is_atomic(self, tmp_path):
        """After write, no temp files remain in the checkpoint directory."""
        mgr = CheckpointManager(tmp_path)
        arr = np.zeros(10, dtype=np.float64)
        mgr.write_step_intermediate("run-1", 0, arr)
        state = CheckpointState(step_index=0)
        mgr.write_checkpoint(state, "run-1")

        run_dir = mgr.run_dir("run-1")
        tmp_files = list(run_dir.glob(".tmp_*"))
        assert tmp_files == []

    def test_run_dir(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        rd = mgr.run_dir("abc-123")
        assert rd == tmp_path / "grdl_checkpoint_abc-123"


# ======================================================================
# CheckpointManager — loading and validation
# ======================================================================


class TestCheckpointManagerLoad:
    def _write_checkpoint(self, tmp_path, wf_dict, step_index=2):
        mgr = CheckpointManager(tmp_path)
        run_id = "test-run"

        # Write intermediates
        files = []
        for i in range(step_index + 1):
            arr = np.ones(5) * (i + 1)
            p = mgr.write_step_intermediate(run_id, i, arr)
            files.append(p)

        wf_hash = compute_workflow_hash(wf_dict)
        state = CheckpointState(
            workflow_definition_hash=wf_hash,
            step_index=step_index,
            intermediate_files=files,
            metrics_so_far=[
                {"step_index": i, "processor_name": "Proc",
                 "wall_time_s": 0.1, "cpu_time_s": 0.1,
                 "peak_rss_bytes": 100, "gpu_used": False,
                 "status": "success"}
                for i in range(step_index + 1)
            ],
            execution_context={"run_id": run_id},
            timestamp="2026-02-11T12:00:00Z",
            workflow_dict=wf_dict,
        )
        ckpt_path = mgr.write_checkpoint(state, run_id)
        return ckpt_path

    def test_load_checkpoint_from_file(self, tmp_path):
        wf = {"name": "Test", "version": "1.0", "steps": []}
        ckpt_path = self._write_checkpoint(tmp_path, wf)
        state = CheckpointManager.load_checkpoint(ckpt_path)
        assert state.step_index == 2
        assert len(state.intermediate_files) == 3

    def test_load_checkpoint_from_directory(self, tmp_path):
        wf = {"name": "Test", "version": "1.0", "steps": []}
        ckpt_path = self._write_checkpoint(tmp_path, wf)
        # Pass the directory instead of the file
        state = CheckpointManager.load_checkpoint(ckpt_path.parent)
        assert state.step_index == 2

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(CheckpointError, match="not found"):
            CheckpointManager.load_checkpoint(tmp_path / "nope.json")

    def test_load_corrupt_json_raises(self, tmp_path):
        bad_file = tmp_path / "checkpoint.json"
        bad_file.write_text("not json!!!", encoding="utf-8")
        with pytest.raises(CheckpointError, match="Failed to read"):
            CheckpointManager.load_checkpoint(bad_file)

    def test_validate_matching_hash_passes(self, tmp_path):
        wf = {"name": "Test", "version": "1.0", "steps": [{"processor": "A"}]}
        ckpt_path = self._write_checkpoint(tmp_path, wf)
        state = CheckpointManager.load_checkpoint(ckpt_path)
        # Should not raise
        CheckpointManager.validate_for_resume(state, wf)

    def test_validate_changed_workflow_rejects(self, tmp_path):
        wf_original = {"name": "Test", "version": "1.0", "steps": [{"processor": "A"}]}
        ckpt_path = self._write_checkpoint(tmp_path, wf_original)
        state = CheckpointManager.load_checkpoint(ckpt_path)

        wf_modified = {"name": "Test", "version": "1.0", "steps": [{"processor": "B"}]}
        with pytest.raises(ResumeError, match="Workflow definition has changed"):
            CheckpointManager.validate_for_resume(state, wf_modified)

    def test_validate_missing_intermediate_rejects(self, tmp_path):
        wf = {"name": "Test", "version": "1.0", "steps": []}
        ckpt_path = self._write_checkpoint(tmp_path, wf)
        state = CheckpointManager.load_checkpoint(ckpt_path)
        # Delete an intermediate file
        Path(state.intermediate_files[0]).unlink()

        with pytest.raises(ResumeError, match="intermediate file.*missing"):
            CheckpointManager.validate_for_resume(state, wf)

    def test_validate_unsupported_schema_rejects(self, tmp_path):
        state = CheckpointState(
            schema_version="99.0",
            workflow_definition_hash="abc",
        )
        with pytest.raises(ResumeError, match="Unsupported checkpoint schema"):
            CheckpointManager.validate_for_resume(
                state, {"name": "Test", "version": "1.0", "steps": []}
            )

    def test_load_last_intermediate(self, tmp_path):
        wf = {"name": "Test", "version": "1.0", "steps": []}
        ckpt_path = self._write_checkpoint(tmp_path, wf, step_index=2)
        state = CheckpointManager.load_checkpoint(ckpt_path)
        arr = CheckpointManager.load_last_intermediate(state)
        # Step 2 writes np.ones(5) * 3
        np.testing.assert_array_equal(arr, np.ones(5) * 3)

    def test_load_last_intermediate_empty_raises(self):
        state = CheckpointState(intermediate_files=[])
        with pytest.raises(CheckpointError, match="no intermediate"):
            CheckpointManager.load_last_intermediate(state)


# ======================================================================
# Full lifecycle: run → kill at step 3 → resume → complete
# ======================================================================


class _CountingTransform:
    """Processor that tracks which steps have been called."""
    call_log = []

    def apply(self, source, **kwargs):
        kwargs.pop("progress_callback", None)
        step_idx = kwargs.get("_step_idx", -1)
        _CountingTransform.call_log.append(step_idx)
        return source * 2.0


class _FailAtStepTransform:
    """Processor that fails at a specific call count."""
    call_count = 0
    fail_at = 3  # 0-indexed: fail on the 4th call

    def apply(self, source, **kwargs):
        kwargs.pop("progress_callback", None)
        _FailAtStepTransform.call_count += 1
        if _FailAtStepTransform.call_count > _FailAtStepTransform.fail_at:
            raise RuntimeError("Simulated failure at step")
        return source * 2.0


class TestCheckpointResumeLifecycle:
    """Integration: 10-step workflow killed at step 5 resumes correctly."""

    def _make_workflow(self, n_steps=10):
        from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition
        wf = WorkflowDefinition(name="MultiStep", version="1.0.0")
        for i in range(n_steps):
            wf.add_step(ProcessingStep(f"Transform", "1.0.0"))
        return wf

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_run_with_checkpointing(self, mock_resolve, tmp_path):
        """Each step produces a checkpoint file."""
        mock_resolve.return_value = _CountingTransform
        _CountingTransform.call_log = []

        wf = self._make_workflow(n_steps=5)
        mgr = CheckpointManager(tmp_path)

        from grdl_rt.execution.executor import WorkflowExecutor
        executor = WorkflowExecutor(wf, checkpoint_manager=mgr)

        source = np.ones((4, 4), dtype=np.float64)
        wr = executor.execute(
            source,
            enable_checkpointing=True,
            enable_memory_check=False,
            enable_shutdown_handler=False,
        )

        # All 5 steps executed
        assert len(_CountingTransform.call_log) == 5
        # Result: source * 2^5 = 32
        np.testing.assert_array_almost_equal(
            wr.result, np.ones((4, 4)) * 32.0,
        )
        # Checkpoint file exists
        run_dirs = list(tmp_path.glob("grdl_checkpoint_*"))
        assert len(run_dirs) == 1
        ckpt = json.loads(
            (run_dirs[0] / "checkpoint.json").read_text()
        )
        assert ckpt["step_index"] == 4  # last step index

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_fail_at_step_then_resume(self, mock_resolve, tmp_path):
        """Run 10 steps, fail at step 3, resume and complete 4-9."""
        # Phase 1: Run and fail at step 3
        _FailAtStepTransform.call_count = 0
        _FailAtStepTransform.fail_at = 3
        mock_resolve.return_value = _FailAtStepTransform

        wf = self._make_workflow(n_steps=10)
        mgr = CheckpointManager(tmp_path)

        from grdl_rt.execution.executor import WorkflowExecutor
        executor = WorkflowExecutor(wf, checkpoint_manager=mgr)

        source = np.ones((4, 4), dtype=np.float64)
        with pytest.raises(RuntimeError, match="Simulated failure"):
            executor.execute(
                source,
                enable_checkpointing=True,
                enable_memory_check=False,
                enable_shutdown_handler=False,
            )

        # Steps 0, 1, 2 completed (3 calls), step 3 failed (4th call)
        assert _FailAtStepTransform.call_count == 4

        # Checkpoint should exist for step 2 (last completed before failure)
        run_dirs = list(tmp_path.glob("grdl_checkpoint_*"))
        assert len(run_dirs) == 1
        ckpt_dir = run_dirs[0]
        ckpt = json.loads(
            (ckpt_dir / "checkpoint.json").read_text()
        )
        assert ckpt["step_index"] == 2  # steps 0, 1, 2 completed

        # Phase 2: Resume — use a transform that always succeeds
        class _SuccessTransform:
            resume_call_count = 0

            def apply(self, source, **kwargs):
                kwargs.pop("progress_callback", None)
                _SuccessTransform.resume_call_count += 1
                return source * 2.0

        mock_resolve.return_value = _SuccessTransform

        executor2 = WorkflowExecutor(wf, checkpoint_manager=mgr)
        wr = executor2.resume(
            ckpt_dir,
            enable_memory_check=False,
            enable_shutdown_handler=False,
            enable_checkpointing=True,
        )

        # Steps 0-2 were skipped, steps 3-9 ran (7 calls)
        assert _SuccessTransform.resume_call_count == 7

        # Merged metrics: 3 from original + 7 from resume = 10
        assert len(wr.metrics.step_metrics) == 10
        assert wr.metrics.status == "success"

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_resume_rejects_modified_workflow(self, mock_resolve, tmp_path):
        """Modifying the workflow YAML and attempting resume fails."""
        mock_resolve.return_value = _CountingTransform
        _CountingTransform.call_log = []

        wf_original = self._make_workflow(n_steps=5)
        mgr = CheckpointManager(tmp_path)

        from grdl_rt.execution.executor import WorkflowExecutor
        executor = WorkflowExecutor(wf_original, checkpoint_manager=mgr)

        source = np.ones((2, 2), dtype=np.float64)
        executor.execute(
            source,
            enable_checkpointing=True,
            enable_memory_check=False,
            enable_shutdown_handler=False,
        )

        run_dirs = list(tmp_path.glob("grdl_checkpoint_*"))
        ckpt_dir = run_dirs[0]

        # Create a DIFFERENT workflow (different step count)
        wf_modified = self._make_workflow(n_steps=3)
        executor2 = WorkflowExecutor(wf_modified, checkpoint_manager=mgr)

        with pytest.raises(ResumeError, match="Workflow definition has changed"):
            executor2.resume(
                ckpt_dir,
                enable_memory_check=False,
                enable_shutdown_handler=False,
            )

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_checkpoint_files_are_valid_json(self, mock_resolve, tmp_path):
        """Checkpoint files contain all required fields as valid JSON."""
        mock_resolve.return_value = _CountingTransform
        _CountingTransform.call_log = []

        wf = self._make_workflow(n_steps=3)
        mgr = CheckpointManager(tmp_path)

        from grdl_rt.execution.executor import WorkflowExecutor
        executor = WorkflowExecutor(wf, checkpoint_manager=mgr)

        source = np.ones((2, 2), dtype=np.float64)
        executor.execute(
            source,
            enable_checkpointing=True,
            enable_memory_check=False,
            enable_shutdown_handler=False,
        )

        run_dirs = list(tmp_path.glob("grdl_checkpoint_*"))
        ckpt_path = run_dirs[0] / "checkpoint.json"
        data = json.loads(ckpt_path.read_text())

        # All required fields present
        assert "schema_version" in data
        assert "workflow_definition_hash" in data
        assert "step_index" in data
        assert "intermediate_files" in data
        assert "metrics_so_far" in data
        assert "execution_context" in data
        assert "timestamp" in data
        assert "workflow_dict" in data

        # Values are correct types
        assert isinstance(data["schema_version"], str)
        assert isinstance(data["workflow_definition_hash"], str)
        assert isinstance(data["step_index"], int)
        assert isinstance(data["intermediate_files"], list)
        assert isinstance(data["metrics_so_far"], list)
