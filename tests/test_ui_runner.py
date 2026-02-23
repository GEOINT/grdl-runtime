"""
Tests for grdl_rt.ui._runner — data models and image I/O helpers.
"""

from pathlib import Path

import numpy as np
import pytest

from grdl_rt.ui._runner import RunRequest, RunResult, _read_input, _save_output

# ── Tests: RunRequest / RunResult ────────────────────────────────────


class TestDataModels:
    def test_run_request_defaults(self):
        req = RunRequest(
            input_paths=[Path("/tmp/a.npy")],
            workflow_source=Path("/tmp/wf.yaml"),
        )
        assert req.prefer_gpu is False
        assert req.max_workers == 1
        assert req.output_dir is None
        assert req.ground_truth_path is None
        assert req.params == {}

    def test_run_result_defaults(self):
        res = RunResult()
        assert res.workflow_results == []
        assert res.accuracy is None
        assert res.error is None
        assert res.elapsed_seconds == 0.0


# ── Tests: _read_input ───────────────────────────────────────────────


class TestReadInput:
    def test_read_npy(self, tmp_path: Path):
        arr = np.random.rand(32, 32).astype(np.float32)
        p = tmp_path / "test.npy"
        np.save(str(p), arr)

        loaded = _read_input(p)
        np.testing.assert_array_equal(loaded, arr)

    def test_nonexistent_file(self, tmp_path: Path):
        with pytest.raises((FileNotFoundError, ValueError)):
            _read_input(tmp_path / "missing.npy")


# ── Tests: _save_output ─────────────────────────────────────────────


class TestSaveOutput:
    def test_save_npy(self, tmp_path: Path):
        arr = np.ones((16, 16), dtype=np.float64)
        out = tmp_path / "output.npy"
        _save_output(arr, out)

        loaded = np.load(str(out))
        np.testing.assert_array_equal(loaded, arr)
