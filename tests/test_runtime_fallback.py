# -*- coding: utf-8 -*-
"""
Tests for DAGExecutor runtime fallback and as_planned/as_executed manifests.

Covers:
- Step failure triggers fallback to alternative processor
- Fallback success: output in results, substitution recorded
- Fallback failure: original exception re-raised
- No alternatives available: exception raised immediately
- Only one level of fallback (no cascading)
- as_planned.json written before first step executes
- as_executed.json written after execution with status per step
- as_executed.json reflects runtime substitution on fallback

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from grdl_rt.catalog.models import Artifact
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog
from grdl_rt.execution.dag_executor import DAGExecutor
from grdl_rt.execution.plan import (
    ParallelGroup,
    ResolvedExecutionPlan,
    ResolvedStep,
)
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

# ---------------------------------------------------------------------------
# Mock processors
# ---------------------------------------------------------------------------


class _SuccessProcessor:
    """Returns input * 2."""

    def apply(self, source, **kwargs):
        return source * 2.0


class _FailProcessor:
    """Always raises RuntimeError."""

    def apply(self, source, **kwargs):
        raise RuntimeError("primary failed")


class _FallbackProcessor:
    """Returns input * 10 (distinguishable from primary)."""

    def apply(self, source, **kwargs):
        return source * 10.0


class _FailFallbackProcessor:
    """Fallback that also fails."""

    def apply(self, source, **kwargs):
        raise RuntimeError("fallback also failed")


_PROCESSORS = {
    "Success": _SuccessProcessor,
    "Fail": _FailProcessor,
    "Fallback": _FallbackProcessor,
    "FailFallback": _FailFallbackProcessor,
}


def _mock_resolve(name):
    if name in _PROCESSORS:
        return _PROCESSORS[name]
    raise ImportError(f"Cannot resolve '{name}'")


# ---------------------------------------------------------------------------
# Mock catalog for alternatives
# ---------------------------------------------------------------------------


def _make_yaml_catalog(tmp_path, alternatives_map=None):
    """Create a YamlArtifactCatalog with processor artifacts + alternatives."""
    alternatives_map = alternatives_map or {}
    yaml_path = tmp_path / "fallback_catalog.yaml"
    cat = YamlArtifactCatalog(file_path=yaml_path)
    for proc_name, alts in alternatives_map.items():
        art = Artifact(
            name=proc_name,
            version="1.0.0",
            artifact_type="grdl_processor",
            description=f"Mock {proc_name}",
            processor_class=f"mock.{proc_name}",
            processor_version="1.0.0",
            processor_type="transform",
            alternatives=alts,
        )
        cat.add_artifact(art)
    return cat


# ---------------------------------------------------------------------------
# Helper: make a minimal ResolvedExecutionPlan
# ---------------------------------------------------------------------------


def _make_plan(steps_dict, hw_context=None):
    return ResolvedExecutionPlan(
        workflow_name="Test",
        workflow_version="1.0.0",
        resolved_at="2026-02-11T00:00:00+00:00",
        hardware_context=hw_context or {"cpu_count": 4},
        steps=steps_dict,
        parallel_groups=[
            ParallelGroup(level=0, step_ids=list(steps_dict.keys()), estimated_peak_memory_bytes=0),
        ],
        global_pass_steps=[],
        substitutions=[],
        estimated_total_memory_bytes=0,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Runtime fallback tests
# ---------------------------------------------------------------------------


class TestRuntimeFallback:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_fallback_on_failure(self, mock_resolve, tmp_path):
        """Primary processor fails → fallback used → output is fallback's."""
        cat = _make_yaml_catalog(
            tmp_path,
            {
                "Fail": [{"processor_name": "Fallback", "priority": 1}],
            },
        )
        wf = WorkflowDefinition(
            name="FallbackTest",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        executor = DAGExecutor(wf, catalog=cat)
        source = np.array([3.0])

        result = executor.execute(source, enable_memory_check=False)

        # Fallback multiplies by 10
        np.testing.assert_array_almost_equal(result.result, np.array([30.0]))
        # Runtime substitution recorded
        assert len(executor._runtime_substitutions) == 1
        sub = executor._runtime_substitutions[0]
        assert sub["original_processor"] == "Fail"
        assert sub["replacement_processor"] == "Fallback"

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_fallback_metrics_status(self, mock_resolve, tmp_path):
        """Step metrics show 'fallback' status when fallback used."""
        cat = _make_yaml_catalog(
            tmp_path,
            {
                "Fail": [{"processor_name": "Fallback", "priority": 1}],
            },
        )
        wf = WorkflowDefinition(
            name="FBStatus",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        executor = DAGExecutor(wf, catalog=cat)
        result = executor.execute(np.array([1.0]), enable_memory_check=False)

        sm = result.metrics.step_metrics[0]
        assert sm.status == "fallback"

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_fallback_also_fails_raises_original(self, mock_resolve, tmp_path):
        """If fallback also fails, original exception propagates."""
        cat = _make_yaml_catalog(
            tmp_path,
            {
                "Fail": [{"processor_name": "FailFallback", "priority": 1}],
            },
        )
        wf = WorkflowDefinition(
            name="BothFail",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        executor = DAGExecutor(wf, catalog=cat)

        with pytest.raises(RuntimeError, match="primary failed"):
            executor.execute(np.array([1.0]), enable_memory_check=False)

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_no_alternatives_raises(self, mock_resolve, tmp_path):
        """No alternatives in catalog → original exception propagates."""
        cat = _make_yaml_catalog(
            tmp_path,
            {
                "Fail": [],  # No alternatives
            },
        )
        wf = WorkflowDefinition(
            name="NoAlt",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        executor = DAGExecutor(wf, catalog=cat)

        with pytest.raises(RuntimeError, match="primary failed"):
            executor.execute(np.array([1.0]), enable_memory_check=False)

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_no_catalog_no_fallback(self, mock_resolve):
        """No catalog configured → no fallback attempted."""
        wf = WorkflowDefinition(
            name="NoCat",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        executor = DAGExecutor(wf)  # No catalog

        with pytest.raises(RuntimeError, match="primary failed"):
            executor.execute(np.array([1.0]), enable_memory_check=False)

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_success_no_fallback(self, mock_resolve, tmp_path):
        """Successful step does not trigger fallback."""
        cat = _make_yaml_catalog(
            tmp_path,
            {
                "Success": [{"processor_name": "Fallback", "priority": 1}],
            },
        )
        wf = WorkflowDefinition(
            name="OK",
            version="1.0.0",
            steps=[
                ProcessingStep("Success", "1.0", id="s1"),
            ],
        )
        executor = DAGExecutor(wf, catalog=cat)
        result = executor.execute(np.array([5.0]), enable_memory_check=False)

        np.testing.assert_array_almost_equal(result.result, np.array([10.0]))
        assert len(executor._runtime_substitutions) == 0


# ---------------------------------------------------------------------------
# As-planned / as-executed manifest tests
# ---------------------------------------------------------------------------


class TestAsPlannedManifest:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_as_planned_written(self, mock_resolve, tmp_path):
        """as_planned.json is written before execution starts."""
        wf = WorkflowDefinition(
            name="Planned",
            version="1.0.0",
            steps=[
                ProcessingStep("Success", "1.0", id="s1"),
            ],
        )
        plan = _make_plan(
            {
                "s1": ResolvedStep(
                    step_id="s1",
                    original_processor="Success",
                    resolved_processor="Success",
                    processor_class_fqn="mock.Success",
                    params={},
                    gpu_capability="cpu_only",
                    will_use_gpu=False,
                    requires_global_pass=False,
                    substitution_reason=None,
                    estimated_memory_bytes=0,
                    retry=None,
                    timeout_seconds=None,
                ),
            }
        )
        run_folder = tmp_path / "run_001"
        executor = DAGExecutor(wf)
        executor.execute(
            np.array([1.0]),
            enable_memory_check=False,
            resolved_plan=plan,
            run_folder=run_folder,
        )

        planned_path = run_folder / "as_planned.json"
        assert planned_path.exists()
        data = json.loads(planned_path.read_text())
        assert data["workflow_name"] == "Test"  # from _make_plan
        assert "s1" in data["steps"]

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_no_plan_no_file(self, mock_resolve, tmp_path):
        """No resolved_plan → no as_planned.json written."""
        wf = WorkflowDefinition(
            name="NoPlan",
            version="1.0.0",
            steps=[
                ProcessingStep("Success", "1.0", id="s1"),
            ],
        )
        run_folder = tmp_path / "run_002"
        executor = DAGExecutor(wf)
        executor.execute(
            np.array([1.0]),
            enable_memory_check=False,
            run_folder=run_folder,
        )
        # No resolved_plan → no file
        assert not (run_folder / "as_planned.json").exists()


class TestAsExecutedManifest:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_as_executed_success(self, mock_resolve, tmp_path):
        """as_executed.json written with status=success."""
        wf = WorkflowDefinition(
            name="Exec",
            version="1.0.0",
            steps=[
                ProcessingStep("Success", "1.0", id="s1"),
            ],
        )
        plan = _make_plan(
            {
                "s1": ResolvedStep(
                    step_id="s1",
                    original_processor="Success",
                    resolved_processor="Success",
                    processor_class_fqn="mock.Success",
                    params={},
                    gpu_capability="cpu_only",
                    will_use_gpu=False,
                    requires_global_pass=False,
                    substitution_reason=None,
                    estimated_memory_bytes=0,
                    retry=None,
                    timeout_seconds=None,
                ),
            }
        )
        run_folder = tmp_path / "run_003"
        executor = DAGExecutor(wf)
        executor.execute(
            np.array([1.0]),
            enable_memory_check=False,
            resolved_plan=plan,
            run_folder=run_folder,
        )

        executed_path = run_folder / "as_executed.json"
        assert executed_path.exists()
        data = json.loads(executed_path.read_text())
        assert data["status"] == "success"
        assert data["workflow_name"] == "Exec"
        assert len(data["executed_steps"]) == 1
        assert data["executed_steps"][0]["status"] == "success"
        assert data["total_wall_time_s"] > 0

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_as_executed_failure(self, mock_resolve, tmp_path):
        """as_executed.json written with status=failed on error."""
        wf = WorkflowDefinition(
            name="FailExec",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        plan = _make_plan(
            {
                "s1": ResolvedStep(
                    step_id="s1",
                    original_processor="Fail",
                    resolved_processor="Fail",
                    processor_class_fqn="mock.Fail",
                    params={},
                    gpu_capability="cpu_only",
                    will_use_gpu=False,
                    requires_global_pass=False,
                    substitution_reason=None,
                    estimated_memory_bytes=0,
                    retry=None,
                    timeout_seconds=None,
                ),
            }
        )
        run_folder = tmp_path / "run_004"
        executor = DAGExecutor(wf)

        with pytest.raises(RuntimeError):
            executor.execute(
                np.array([1.0]),
                enable_memory_check=False,
                resolved_plan=plan,
                run_folder=run_folder,
            )

        executed_path = run_folder / "as_executed.json"
        assert executed_path.exists()
        data = json.loads(executed_path.read_text())
        assert data["status"] == "failed"

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_as_executed_with_fallback(self, mock_resolve, tmp_path):
        """as_executed.json reflects runtime substitution on fallback."""
        cat = _make_yaml_catalog(
            tmp_path,
            {
                "Fail": [{"processor_name": "Fallback", "priority": 1}],
            },
        )
        wf = WorkflowDefinition(
            name="FBExec",
            version="1.0.0",
            steps=[
                ProcessingStep("Fail", "1.0", id="s1"),
            ],
        )
        plan = _make_plan(
            {
                "s1": ResolvedStep(
                    step_id="s1",
                    original_processor="Fail",
                    resolved_processor="Fail",
                    processor_class_fqn="mock.Fail",
                    params={},
                    gpu_capability="cpu_only",
                    will_use_gpu=False,
                    requires_global_pass=False,
                    substitution_reason=None,
                    estimated_memory_bytes=0,
                    retry=None,
                    timeout_seconds=None,
                ),
            }
        )
        run_folder = tmp_path / "run_005"
        executor = DAGExecutor(wf, catalog=cat)
        executor.execute(
            np.array([1.0]),
            enable_memory_check=False,
            resolved_plan=plan,
            run_folder=run_folder,
        )

        executed_path = run_folder / "as_executed.json"
        data = json.loads(executed_path.read_text())
        assert data["status"] == "success"
        assert len(data["runtime_substitutions"]) == 1
        sub = data["runtime_substitutions"][0]
        assert sub["original_processor"] == "Fail"
        assert sub["replacement_processor"] == "Fallback"

        step_record = data["executed_steps"][0]
        assert step_record["fallback_processor"] == "Fallback"
