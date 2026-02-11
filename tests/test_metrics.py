# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.metrics and grdl_rt.execution.result.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

import json

import numpy as np
import pytest

from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics
from grdl_rt.execution.result import WorkflowResult


# ---------------------------------------------------------------------------
# StepMetrics
# ---------------------------------------------------------------------------

class TestStepMetrics:
    def test_construction(self):
        sm = StepMetrics(
            step_index=0,
            processor_name="LeeFilter",
            wall_time_s=1.23,
            cpu_time_s=0.98,
            peak_rss_bytes=1024,
            gpu_used=False,
        )
        assert sm.step_index == 0
        assert sm.processor_name == "LeeFilter"
        assert sm.wall_time_s == 1.23
        assert sm.cpu_time_s == 0.98
        assert sm.peak_rss_bytes == 1024
        assert sm.gpu_used is False
        assert sm.status == "success"
        assert sm.error_message is None

    def test_failed_step(self):
        sm = StepMetrics(
            step_index=1,
            processor_name="BadFilter",
            wall_time_s=0.01,
            cpu_time_s=0.01,
            peak_rss_bytes=0,
            gpu_used=False,
            status="failed",
            error_message="division by zero",
        )
        assert sm.status == "failed"
        assert sm.error_message == "division by zero"

    def test_to_dict(self):
        sm = StepMetrics(
            step_index=0,
            processor_name="Test",
            wall_time_s=0.5,
            cpu_time_s=0.4,
            peak_rss_bytes=2048,
            gpu_used=True,
        )
        d = sm.to_dict()
        assert d["step_index"] == 0
        assert d["processor_name"] == "Test"
        assert d["wall_time_s"] == 0.5
        assert d["gpu_used"] is True
        assert d["status"] == "success"


# ---------------------------------------------------------------------------
# WorkflowMetrics
# ---------------------------------------------------------------------------

class TestWorkflowMetrics:
    def _make_metrics(self, n_steps=2, status="success"):
        steps = [
            StepMetrics(
                step_index=i,
                processor_name=f"Step{i}",
                wall_time_s=0.1 * (i + 1),
                cpu_time_s=0.05 * (i + 1),
                peak_rss_bytes=1024 * (i + 1),
                gpu_used=False,
            )
            for i in range(n_steps)
        ]
        return WorkflowMetrics(
            workflow_id="test:1.0",
            run_id="run-123",
            workflow_name="Test",
            workflow_version="1.0.0",
            total_wall_time_s=sum(s.wall_time_s for s in steps),
            total_cpu_time_s=sum(s.cpu_time_s for s in steps),
            peak_rss_bytes=max(s.peak_rss_bytes for s in steps),
            step_metrics=steps,
            started_at="2026-02-11T00:00:00Z",
            completed_at="2026-02-11T00:00:01Z",
            status=status,
        )

    def test_construction(self):
        wm = self._make_metrics()
        assert wm.workflow_id == "test:1.0"
        assert wm.run_id == "run-123"
        assert wm.workflow_name == "Test"
        assert wm.workflow_version == "1.0.0"
        assert len(wm.step_metrics) == 2
        assert wm.status == "success"

    def test_total_wall_time_positive(self):
        wm = self._make_metrics()
        assert wm.total_wall_time_s > 0

    def test_total_cpu_time_positive(self):
        wm = self._make_metrics()
        assert wm.total_cpu_time_s > 0

    def test_peak_rss_non_negative(self):
        wm = self._make_metrics()
        assert wm.peak_rss_bytes >= 0

    def test_step_count_matches(self):
        wm = self._make_metrics(n_steps=5)
        assert len(wm.step_metrics) == 5

    def test_to_dict(self):
        wm = self._make_metrics()
        d = wm.to_dict()
        assert d["workflow_id"] == "test:1.0"
        assert d["run_id"] == "run-123"
        assert isinstance(d["step_metrics"], list)
        assert len(d["step_metrics"]) == 2
        assert d["started_at"] == "2026-02-11T00:00:00Z"
        assert d["completed_at"] == "2026-02-11T00:00:01Z"

    def test_to_json(self):
        wm = self._make_metrics()
        j = wm.to_json()
        parsed = json.loads(j)
        assert parsed["workflow_name"] == "Test"
        assert len(parsed["step_metrics"]) == 2

    def test_failed_workflow(self):
        wm = self._make_metrics(status="failed")
        wm.error_message = "step 1 failed"
        d = wm.to_dict()
        assert d["status"] == "failed"
        assert d["error_message"] == "step 1 failed"


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

class TestWorkflowResult:
    def test_construction(self):
        arr = np.ones((4, 4), dtype=np.float32)
        wm = WorkflowMetrics(
            workflow_id="test:1.0",
            run_id="run-1",
            workflow_name="Test",
            workflow_version="1.0.0",
            total_wall_time_s=0.5,
            total_cpu_time_s=0.3,
            peak_rss_bytes=1024,
        )
        wr = WorkflowResult(result=arr, metrics=wm)
        np.testing.assert_array_equal(wr.result, arr)
        assert wr.metrics is wm

    def test_result_is_ndarray(self):
        arr = np.zeros((2, 2))
        wm = WorkflowMetrics(
            workflow_id="x", run_id="r", workflow_name="X",
            workflow_version="0.1", total_wall_time_s=0,
            total_cpu_time_s=0, peak_rss_bytes=0,
        )
        wr = WorkflowResult(result=arr, metrics=wm)
        assert isinstance(wr.result, np.ndarray)
