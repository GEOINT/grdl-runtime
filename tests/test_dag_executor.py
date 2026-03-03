"""
Tests for grdl_rt.execution.dag_executor — DAGExecutor parallel execution.

Covers:
- Linear backward compatibility via DAGExecutor
- Branching workflow with numeric correctness
- Parallel execution timing (proves concurrency)
- Conditional step skip / execute
- Fan-out / fan-in patterns
- Multi-input step receives dict
- Single-input step receives array
- Metrics include step_id
- Builder DAG API (branches/merge)

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

import time
from unittest.mock import patch

import numpy as np
import pytest

from grdl_rt.execution.builder import BranchBuilder, Workflow
from grdl_rt.execution.dag_executor import DAGExecutor
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.workflow import (
    ProcessingStep,
    WorkflowDefinition,
)

# ---------------------------------------------------------------------------
# Mock processors for tests
# ---------------------------------------------------------------------------


class _Scale2:
    """Multiplies by 2."""

    def apply(self, source, **kwargs):
        return source * 2.0


class _Scale3:
    """Multiplies by 3."""

    def apply(self, source, **kwargs):
        return source * 3.0


class _Scale5:
    """Multiplies by 5."""

    def apply(self, source, **kwargs):
        return source * 5.0


class _AddOne:
    """Adds 1."""

    def apply(self, source, **kwargs):
        return source + 1.0


class _SlowProcessor:
    """Sleeps for 0.5s then passes through."""

    def apply(self, source, **kwargs):
        time.sleep(0.5)
        return source


class _MergeProcessor:
    """Merges multiple inputs by summing them."""

    def apply(self, source, **kwargs):
        if isinstance(source, dict):
            return sum(source.values())
        return source


class _Slow03:
    """Sleeps 0.3 s, passes source through."""

    def apply(self, source, **kwargs):
        time.sleep(0.3)
        return source


class _Slow06:
    """Sleeps 0.6 s, passes source through."""

    def apply(self, source, **kwargs):
        time.sleep(0.6)
        return source


# ---------------------------------------------------------------------------
# Helper: patch resolve_processor_class to return our mocks
# ---------------------------------------------------------------------------

_MOCK_PROCESSORS = {
    "Scale2": _Scale2,
    "Scale3": _Scale3,
    "Scale5": _Scale5,
    "AddOne": _AddOne,
    "SlowProcessor": _SlowProcessor,
    "MergeProcessor": _MergeProcessor,
    "Slow03": _Slow03,
    "Slow06": _Slow06,
}


def _mock_resolve(name):
    if name in _MOCK_PROCESSORS:
        return _MOCK_PROCESSORS[name]
    raise ImportError(f"Cannot resolve processor '{name}'")


# ---------------------------------------------------------------------------
# DAGExecutor — Linear backward compatibility
# ---------------------------------------------------------------------------


class TestDAGExecutorLinear:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_linear_pipeline(self, mock_resolve):
        """Old-style linear workflow (no explicit deps) executes correctly."""
        wf = WorkflowDefinition(
            name="Linear",
            version="1.0.0",
            steps=[
                ProcessingStep("Scale2", "1.0"),
                ProcessingStep("Scale3", "1.0"),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.ones((4, 4), dtype=np.float64)

        result = executor.execute(source, enable_memory_check=False)

        assert isinstance(result, WorkflowResult)
        # 1 * 2 * 3 = 6
        np.testing.assert_array_almost_equal(result.result, np.ones((4, 4)) * 6.0)

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_single_step(self, mock_resolve):
        wf = WorkflowDefinition(
            name="Single",
            steps=[
                ProcessingStep("Scale5", "1.0"),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.array([2.0, 3.0])

        result = executor.execute(source, enable_memory_check=False)

        np.testing.assert_array_almost_equal(result.result, np.array([10.0, 15.0]))

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_empty_workflow_returns_source(self, mock_resolve):
        wf = WorkflowDefinition(name="Empty")
        executor = DAGExecutor(wf)
        source = np.array([42.0])

        result = executor.execute(source, enable_memory_check=False)

        np.testing.assert_array_equal(result.result, source)


# ---------------------------------------------------------------------------
# DAGExecutor — Branching workflow
# ---------------------------------------------------------------------------


class TestDAGExecutorBranching:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_branching_workflow(self, mock_resolve):
        """root → [scale2, scale3] → merge (sum)."""
        wf = WorkflowDefinition(
            name="Branch",
            version="1.0.0",
            steps=[
                ProcessingStep("Scale2", "1.0", id="root"),
                ProcessingStep("Scale2", "1.0", id="b1", depends_on=["root"]),
                ProcessingStep("Scale3", "1.0", id="b2", depends_on=["root"]),
                ProcessingStep("MergeProcessor", "1.0", id="merge", depends_on=["b1", "b2"]),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.array([10.0])

        result = executor.execute(source, enable_memory_check=False)

        # root: 10*2=20
        # b1: 20*2=40, b2: 20*3=60
        # merge: 40+60=100
        np.testing.assert_array_almost_equal(result.result, np.array([100.0]))

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_fan_out_fan_in(self, mock_resolve):
        """One step produces output, two consume it, one merges."""
        wf = WorkflowDefinition(
            name="FanOut",
            version="1.0.0",
            steps=[
                ProcessingStep("AddOne", "1.0", id="root"),
                ProcessingStep("Scale2", "1.0", id="left", depends_on=["root"]),
                ProcessingStep("Scale5", "1.0", id="right", depends_on=["root"]),
                ProcessingStep("MergeProcessor", "1.0", id="merge", depends_on=["left", "right"]),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.array([1.0])

        result = executor.execute(source, enable_memory_check=False)

        # root: 1+1=2
        # left: 2*2=4, right: 2*5=10
        # merge: 4+10=14
        np.testing.assert_array_almost_equal(result.result, np.array([14.0]))


# ---------------------------------------------------------------------------
# DAGExecutor — Parallel execution timing
# ---------------------------------------------------------------------------


class TestDAGExecutorParallel:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_parallel_branches_are_concurrent(self, mock_resolve):
        """Two 0.5s sleep steps in parallel should take ~0.5s, not ~1.0s."""
        wf = WorkflowDefinition(
            name="Parallel",
            version="1.0.0",
            steps=[
                ProcessingStep("AddOne", "1.0", id="root"),
                ProcessingStep("SlowProcessor", "1.0", id="left", depends_on=["root"]),
                ProcessingStep("SlowProcessor", "1.0", id="right", depends_on=["root"]),
            ],
        )
        # left and right both depend on root → same level → parallel
        executor = DAGExecutor(wf, max_workers=2)
        source = np.array([1.0])

        t0 = time.perf_counter()
        executor.execute(source, enable_memory_check=False)
        elapsed = time.perf_counter() - t0

        # Should be ~0.5s if parallel, ~1.0s if sequential
        assert elapsed < 0.9, f"Expected <0.9s (parallel), got {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# DAGExecutor — Conditional steps
# ---------------------------------------------------------------------------


class TestDAGExecutorConditional:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_condition_true_executes(self, mock_resolve):
        """Step with condition=True runs normally."""
        wf = WorkflowDefinition(
            name="CondTrue",
            steps=[
                ProcessingStep("Scale2", "1.0", id="s1", condition="x > 0"),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.array([5.0])

        result = executor.execute(
            source,
            enable_memory_check=False,
            execution_context={"x": 1},
        )

        np.testing.assert_array_almost_equal(result.result, np.array([10.0]))

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_condition_false_skips(self, mock_resolve):
        """Step with condition=False is skipped, input propagated."""
        wf = WorkflowDefinition(
            name="CondFalse",
            steps=[
                ProcessingStep("Scale2", "1.0", id="s1", condition="skip == False"),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.array([5.0])

        result = executor.execute(
            source,
            enable_memory_check=False,
            execution_context={"skip": True},
        )

        # Step is skipped → source propagated unchanged
        np.testing.assert_array_equal(result.result, source)

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_skipped_step_metrics_status(self, mock_resolve):
        """Skipped step has status='skipped' in metrics."""
        wf = WorkflowDefinition(
            name="CondSkip",
            steps=[
                ProcessingStep("Scale2", "1.0", id="s1", condition="False"),
            ],
        )
        executor = DAGExecutor(wf)
        source = np.array([5.0])

        result = executor.execute(
            source,
            enable_memory_check=False,
        )

        assert result.metrics.step_metrics[0].status == "skipped"


# ---------------------------------------------------------------------------
# DAGExecutor — Multi-input vs single-input
# ---------------------------------------------------------------------------


class TestDAGExecutorInputTypes:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_single_dep_receives_array(self, mock_resolve):
        """Step with one dependency receives plain ndarray, not dict."""
        received_input = {}

        class _Inspector:
            def apply(self, source, **kwargs):
                received_input["type"] = type(source).__name__
                received_input["is_ndarray"] = isinstance(source, np.ndarray)
                return source

        _MOCK_PROCESSORS["Inspector"] = _Inspector
        try:
            wf = WorkflowDefinition(
                name="SingleDep",
                steps=[
                    ProcessingStep("Scale2", "1.0", id="root"),
                    ProcessingStep("Inspector", "1.0", id="child", depends_on=["root"]),
                ],
            )
            executor = DAGExecutor(wf)
            executor.execute(np.array([1.0]), enable_memory_check=False)
            assert received_input["is_ndarray"] is True
        finally:
            del _MOCK_PROCESSORS["Inspector"]

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_multi_dep_receives_dict(self, mock_resolve):
        """Step with multiple dependencies receives dict of arrays."""
        received_input = {}

        class _Inspector:
            def apply(self, source, **kwargs):
                received_input["type"] = type(source).__name__
                received_input["is_dict"] = isinstance(source, dict)
                if isinstance(source, dict):
                    received_input["keys"] = set(source.keys())
                return source if not isinstance(source, dict) else sum(source.values())

        _MOCK_PROCESSORS["Inspector"] = _Inspector
        try:
            wf = WorkflowDefinition(
                name="MultiDep",
                steps=[
                    ProcessingStep("Scale2", "1.0", id="a"),
                    ProcessingStep("Scale3", "1.0", id="b"),
                    ProcessingStep("Inspector", "1.0", id="merge", depends_on=["a", "b"]),
                ],
            )
            executor = DAGExecutor(wf)
            executor.execute(np.array([1.0]), enable_memory_check=False)
            assert received_input["is_dict"] is True
            assert received_input["keys"] == {"a", "b"}
        finally:
            del _MOCK_PROCESSORS["Inspector"]


# ---------------------------------------------------------------------------
# DAGExecutor — Metrics
# ---------------------------------------------------------------------------


class TestDAGExecutorMetrics:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_metrics_have_step_id(self, mock_resolve):
        wf = WorkflowDefinition(
            name="Metrics",
            steps=[
                ProcessingStep("Scale2", "1.0", id="s1"),
                ProcessingStep("Scale3", "1.0", id="s2", depends_on=["s1"]),
            ],
        )
        executor = DAGExecutor(wf)
        result = executor.execute(np.array([1.0]), enable_memory_check=False)

        step_ids = {sm.step_id for sm in result.metrics.step_metrics}
        assert "s1" in step_ids
        assert "s2" in step_ids

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_workflow_metrics_status_success(self, mock_resolve):
        wf = WorkflowDefinition(
            name="OK",
            steps=[
                ProcessingStep("Scale2", "1.0"),
            ],
        )
        executor = DAGExecutor(wf)
        result = executor.execute(np.array([1.0]), enable_memory_check=False)

        assert result.metrics.status == "success"
        assert result.metrics.total_wall_time_s > 0

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_step_results_dict(self, mock_resolve):
        """step_results contains terminal outputs only; intermediates are evicted."""
        wf = WorkflowDefinition(
            name="Steps",
            steps=[
                ProcessingStep("Scale2", "1.0", id="root"),
                ProcessingStep("Scale3", "1.0", id="child", depends_on=["root"]),
            ],
        )
        executor = DAGExecutor(wf)
        result = executor.execute(np.array([1.0]), enable_memory_check=False)

        assert result.step_results is not None
        # "root" is an intermediate consumed by "child" — evicted after child completes.
        assert "root" not in result.step_results
        # "child" is the terminal — must be present.
        assert "child" in result.step_results
        np.testing.assert_array_almost_equal(result.step_results["child"], np.array([6.0]))


# ---------------------------------------------------------------------------
# DAGExecutor — Progress callback
# ---------------------------------------------------------------------------


class TestDAGExecutorProgress:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_progress_callback_called(self, mock_resolve):
        progress = []
        wf = WorkflowDefinition(
            name="Progress",
            steps=[
                ProcessingStep("Scale2", "1.0", id="root"),
                ProcessingStep("Scale3", "1.0", id="child", depends_on=["root"]),
            ],
        )
        executor = DAGExecutor(wf)
        executor.execute(
            np.array([1.0]),
            progress_callback=progress.append,
            enable_memory_check=False,
        )
        assert len(progress) >= 1
        assert progress[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DAGExecutor — Error handling
# ---------------------------------------------------------------------------


class TestDAGExecutorErrors:

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_invalid_dag_raises(self, mock_resolve):
        """Cycle in DAG raises ValueError."""
        s1 = ProcessingStep("Scale2", "1.0", id="a", depends_on=["b"])
        s2 = ProcessingStep("Scale2", "1.0", id="b", depends_on=["a"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Cycle"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1, s2]
        wf.tags = None
        wf.state = None
        wf.schema_version = "2.0"
        executor = DAGExecutor(wf)
        with pytest.raises(ValueError, match="Invalid workflow DAG"):
            executor.execute(np.array([1.0]), enable_memory_check=False)

    def test_unresolvable_processor_raises(self):
        wf = WorkflowDefinition(
            name="Bad",
            steps=[
                ProcessingStep("NoSuchProcessor999", "1.0"),
            ],
        )
        executor = DAGExecutor(wf)
        with pytest.raises(ImportError):
            executor.execute(np.array([1.0]), enable_memory_check=False)


# ---------------------------------------------------------------------------
# Builder — branches/merge API with DAGExecutor
# ---------------------------------------------------------------------------


class TestBuilderDAGAPI:

    def test_branch_builder_step(self):
        bb = BranchBuilder("b1")
        bb.step(lambda x: x * 2, name="double")
        assert len(bb.steps) == 1
        assert bb.steps[0].name == "double"

    def test_branch_builder_chaining(self):
        bb = (
            BranchBuilder("b1")
            .step(lambda x: x * 2, name="double")
            .step(lambda x: x + 1, name="inc")
        )
        assert len(bb.steps) == 2

    def test_workflow_phase_builders(self):
        wf = (
            Workflow("Phased")
            .io_phase()
            .step(lambda x: x, name="read")
            .processing_phase()
            .step(lambda x: x * 2, name="process")
            .finalization_phase()
            .step(lambda x: x, name="finalize")
        )
        assert wf.steps[0].phase == "io"
        assert wf.steps[1].phase == "global_processing"
        assert wf.steps[2].phase == "finalization"

    def test_workflow_step_with_id_and_deps(self):
        wf = (
            Workflow("DAG Builder")
            .step(lambda x: x * 2, name="read", id="read")
            .step(lambda x: x + 1, name="process", id="proc", depends_on=["read"])
        )
        assert wf.steps[0].id == "read"
        assert wf.steps[1].id == "proc"
        assert wf.steps[1].depends_on == ["read"]

    def test_workflow_step_with_condition(self):
        wf = Workflow("Cond").step(lambda x: x, name="s1", id="s1", condition="x > 0")
        assert wf.steps[0].condition == "x > 0"


# ---------------------------------------------------------------------------
# DAGExecutor — Critical-path / readiness-based scheduler
# ---------------------------------------------------------------------------


class TestDAGExecutorCriticalPath:
    """Prove that the readiness-based dispatcher achieves critical-path timing.

    Under the old level-based scheduler the DAG below would take ~0.9 s
    because 'after_fast' had to wait for 'slow' (same topological level
    as 'fast') before it could start.

    Under the readiness-based scheduler, 'after_fast' starts as soon as
    'fast' finishes, yielding a total of ~0.6 s (the critical path).
    """

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_critical_path_asymmetric_branches(self, mock_resolve):
        """Downstream of the fast branch starts before the slow branch finishes.

        DAG::

            root (instant)
              ├── fast (0.3 s) ──► after_fast (0.3 s)
              └── slow (0.6 s)

        Level-based total ≈ 0 + 0.6 (barrier) + 0.3 = 0.9 s
        Readiness-based total ≈ 0.6 s  (critical path)
        """
        wf = WorkflowDefinition(
            name="CriticalPath",
            version="1.0.0",
            steps=[
                ProcessingStep("AddOne", "1.0", id="root"),
                ProcessingStep("Slow03", "1.0", id="fast", depends_on=["root"]),
                ProcessingStep("Slow06", "1.0", id="slow", depends_on=["root"]),
                ProcessingStep("Slow03", "1.0", id="after_fast", depends_on=["fast"]),
            ],
        )
        executor = DAGExecutor(wf, max_workers=4)
        t0 = time.perf_counter()
        result = executor.execute(np.array([1.0]), enable_memory_check=False)
        elapsed = time.perf_counter() - t0

        # Critical path is 0.6 s; level-based would be ~0.9 s.
        assert elapsed < 0.82, (
            f"Expected <0.82 s (critical path ~0.6 s), got {elapsed:.2f} s. "
            "Level-based scheduling would produce ~0.9 s."
        )
        # Numerical correctness — only terminal outputs remain in step_results.
        # "root" is an intermediate (consumed by fast and slow) and is evicted.
        assert "root" not in result.step_results
        np.testing.assert_array_almost_equal(result.step_results["after_fast"], np.array([2.0]))

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_concurrent_flag_set_for_overlapping_steps(self, mock_resolve):
        """Steps that run simultaneously must have ``concurrent=True`` in metrics."""
        wf = WorkflowDefinition(
            name="ConcurrentFlag",
            version="1.0.0",
            steps=[
                ProcessingStep("AddOne", "1.0", id="root"),
                ProcessingStep("Slow03", "1.0", id="a", depends_on=["root"]),
                ProcessingStep("Slow03", "1.0", id="b", depends_on=["root"]),
            ],
        )
        executor = DAGExecutor(wf, max_workers=2)
        result = executor.execute(np.array([1.0]), enable_memory_check=False)

        by_id = {sm.step_id: sm for sm in result.metrics.step_metrics}
        assert by_id["a"].concurrent is True, "Branch a should be concurrent"
        assert by_id["b"].concurrent is True, "Branch b should be concurrent"
        assert by_id["root"].concurrent is False, "Root runs alone, not concurrent"

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_linear_chain_correct_output(self, mock_resolve):
        """Linear A→B→C still produces correct numerical results after refactor."""
        wf = WorkflowDefinition(
            name="LinearCheck",
            version="1.0.0",
            steps=[
                ProcessingStep("Scale2", "1.0", id="a"),
                ProcessingStep("Scale3", "1.0", id="b", depends_on=["a"]),
                ProcessingStep("Scale5", "1.0", id="c", depends_on=["b"]),
            ],
        )
        result = DAGExecutor(wf).execute(np.array([1.0]), enable_memory_check=False)
        # 1 * 2 * 3 * 5 = 30
        np.testing.assert_array_almost_equal(result.result, np.array([30.0]))


# ---------------------------------------------------------------------------
# DAGExecutor — Consumer-count intermediate eviction
# ---------------------------------------------------------------------------


class TestDAGExecutorEviction:
    """Verify that the eager consumer-count eviction policy is correct.

    Intermediates (steps with at least one downstream consumer) must be
    removed from ``step_results`` once their last consumer completes.
    Terminal outputs (no downstream consumers) must always be present.
    """

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_intermediate_evicted_from_step_results(self, mock_resolve):
        """Intermediate nodes are absent from WorkflowResult.step_results."""
        # root → child  (root is intermediate, child is terminal)
        wf = WorkflowDefinition(
            name="EvictIntermediate",
            steps=[
                ProcessingStep("Scale2", "1.0", id="root"),
                ProcessingStep("Scale3", "1.0", id="child", depends_on=["root"]),
            ],
        )
        result = DAGExecutor(wf).execute(np.array([1.0]), enable_memory_check=False)

        assert result.step_results is not None
        assert (
            "root" not in result.step_results
        ), "Intermediate 'root' should be evicted after its sole consumer 'child' completes"

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_terminal_outputs_preserved(self, mock_resolve):
        """Terminal nodes are always present and numerically correct in step_results."""
        # a, b are independent roots (terminals of their own branches merged together)
        # root → [left, right] → merge
        wf = WorkflowDefinition(
            name="EvictTerminal",
            steps=[
                ProcessingStep("Scale2", "1.0", id="root"),
                ProcessingStep("Scale2", "1.0", id="left", depends_on=["root"]),
                ProcessingStep("Scale3", "1.0", id="right", depends_on=["root"]),
                ProcessingStep("MergeProcessor", "1.0", id="merge", depends_on=["left", "right"]),
            ],
        )
        result = DAGExecutor(wf).execute(np.array([1.0]), enable_memory_check=False)

        assert result.step_results is not None
        # Only terminal "merge" survives; intermediates are evicted.
        assert "merge" in result.step_results, "Terminal 'merge' must be in step_results"
        assert "root" not in result.step_results, "'root' (intermediate) must be evicted"
        assert "left" not in result.step_results, "'left' (intermediate) must be evicted"
        assert "right" not in result.step_results, "'right' (intermediate) must be evicted"
        # root:1*2=2, left:2*2=4, right:2*3=6, merge:4+6=10
        np.testing.assert_array_almost_equal(result.step_results["merge"], np.array([10.0]))

    @patch("grdl_rt.execution.dag_executor.resolve_processor_class", side_effect=_mock_resolve)
    def test_shared_intermediate_evicted_after_last_consumer(self, mock_resolve):
        """An intermediate shared by two consumers is evicted only after both complete.

        DAG::

            root → left  ─┐
                 → right ─┘→ merge

        'root' must survive until both 'left' and 'right' finish.
        """
        wf = WorkflowDefinition(
            name="SharedIntermediate",
            steps=[
                ProcessingStep("AddOne", "1.0", id="root"),
                ProcessingStep("Scale2", "1.0", id="left", depends_on=["root"]),
                ProcessingStep("Scale5", "1.0", id="right", depends_on=["root"]),
                ProcessingStep("MergeProcessor", "1.0", id="merge", depends_on=["left", "right"]),
            ],
        )
        result = DAGExecutor(wf).execute(np.array([1.0]), enable_memory_check=False)

        # root(1+1=2), left(2*2=4), right(2*5=10), merge(4+10=14)
        np.testing.assert_array_almost_equal(result.result, np.array([14.0]))
        # All intermediates evicted; only terminal present.
        assert result.step_results is not None
        assert "merge" in result.step_results
        assert "root" not in result.step_results
        assert "left" not in result.step_results
        assert "right" not in result.step_results
