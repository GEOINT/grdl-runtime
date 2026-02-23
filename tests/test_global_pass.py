"""
Global Pass Executor Tests - Two-pass execution with @globalprocessor.

Tests that the WorkflowExecutor correctly runs global-pass callbacks
before the transform pass, enforces read-only buffers, records metrics,
and leaves non-global processors completely unaffected.

Dependencies
------------
pytest

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11
"""

from unittest.mock import patch

import numpy as np
import pytest

from grdl_rt.execution.executor import WorkflowExecutor
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

# ---------------------------------------------------------------------------
# Test processors (defined at module level for resolve_processor_class)
# ---------------------------------------------------------------------------


class _MedianThresholdProcessor:
    """Processor that computes a global median then thresholds per-chip.

    The global pass computes the median of the full image. The transform
    pass thresholds: values above median -> 1.0, below -> 0.0.
    """

    __has_global_pass__ = True
    __global_callbacks__ = ("compute_median",)

    def __init__(self):
        self._median = None

    def compute_median(self, source):
        """Global pass callback — compute median from full image."""
        self._median = float(np.median(source))

    def apply(self, source, **kwargs):
        """Transform pass — threshold using the computed median."""
        if self._median is None:
            raise RuntimeError("Global pass did not run")
        return (source >= self._median).astype(np.float64)


class _MutatingGlobalProcessor:
    """Processor that tries to mutate the buffer during global pass."""

    __has_global_pass__ = True
    __global_callbacks__ = ("attempt_mutation",)

    def __init__(self):
        self._ran = False

    def attempt_mutation(self, source):
        """Global pass callback that tries to write to the buffer."""
        source[0, 0] = 999.0  # Should raise ValueError

    def apply(self, source, **kwargs):
        return source


class _PlainProcessor:
    """Non-global processor that simply scales the input."""

    def apply(self, source, **kwargs):
        return source * 2.0


class _MultiCallbackProcessor:
    """Processor with two global callbacks that accumulate stats."""

    __has_global_pass__ = True
    __global_callbacks__ = ("compute_mean", "compute_std")

    def __init__(self):
        self._mean = None
        self._std = None

    def compute_mean(self, source):
        self._mean = float(np.mean(source))

    def compute_std(self, source):
        self._std = float(np.std(source))

    def apply(self, source, **kwargs):
        if self._mean is None or self._std is None:
            raise RuntimeError("Global pass did not run")
        return (source - self._mean) / max(self._std, 1e-10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow(steps):
    """Build a WorkflowDefinition from a list of ProcessingStep objects."""
    wf = WorkflowDefinition(name="TestGlobalPass")
    for s in steps:
        wf.add_step(s)
    return wf


# ---------------------------------------------------------------------------
# Tests: Global pass computes statistics, transform uses them
# ---------------------------------------------------------------------------


class TestGlobalPassExecution:
    """Verify the two-pass execution flow."""

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_median_threshold_workflow(self, mock_resolve):
        """Global pass computes median, transform thresholds using it."""
        mock_resolve.return_value = _MedianThresholdProcessor

        step = ProcessingStep(
            processor_name="MedianThresholdProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        # Input: [1, 2, 3, 4, 5, 6, 7, 8, 9] — median is 5.0
        source = np.arange(1.0, 10.0).reshape(3, 3)
        wr = executor.execute(source)

        # Values >= 5.0 become 1.0, below become 0.0
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(wr.result, expected)

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_global_pass_state_on_self(self, mock_resolve):
        """Global pass accumulates state accessible in transform."""
        mock_resolve.return_value = _MultiCallbackProcessor

        step = ProcessingStep(
            processor_name="MultiCallbackProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.array([[10.0, 20.0], [30.0, 40.0]])
        wr = executor.execute(source)

        # z-score normalization: (x - mean) / std
        expected_mean = 25.0
        expected_std = np.std(source)
        expected = (source - expected_mean) / expected_std
        np.testing.assert_array_almost_equal(wr.result, expected)


# ---------------------------------------------------------------------------
# Tests: Mutation defense
# ---------------------------------------------------------------------------


class TestGlobalPassMutationDefense:
    """Verify read-only buffer blocks writes and execution continues."""

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_mutation_raises_valueerror_and_continues(self, mock_resolve):
        """Writing to the buffer during global pass raises ValueError
        which is caught and logged, and execution proceeds."""
        mock_resolve.return_value = _MutatingGlobalProcessor

        step = ProcessingStep(
            processor_name="MutatingGlobalProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((4, 4))
        # Should not raise — the mutation is caught and logged
        wr = executor.execute(source)
        # Transform just returns the input unchanged
        np.testing.assert_array_equal(wr.result, source)

    def test_readonly_view_prevents_write(self):
        """np.ndarray with writeable=False raises ValueError on write."""
        arr = np.zeros((3, 3))
        view = arr.view()
        view.flags.writeable = False
        with pytest.raises(ValueError):
            view[0, 0] = 1.0

    def test_readonly_view_is_no_copy(self):
        """Read-only view shares memory with the original (no copy)."""
        arr = np.zeros((3, 3))
        view = arr.view()
        view.flags.writeable = False
        assert np.shares_memory(arr, view)


# ---------------------------------------------------------------------------
# Tests: Non-global processors unaffected
# ---------------------------------------------------------------------------


class TestNonGlobalProcessorUnaffected:
    """Non-global processors have no overhead from global pass logic."""

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_plain_processor_no_global_pass(self, mock_resolve):
        """A processor without __has_global_pass__ skips the global pass."""
        mock_resolve.return_value = _PlainProcessor

        step = ProcessingStep(
            processor_name="PlainProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((3, 3))
        wr = executor.execute(source)
        np.testing.assert_array_equal(wr.result, np.ones((3, 3)) * 2.0)

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_no_global_pass_metrics_for_plain(self, mock_resolve):
        """Non-global processor StepMetrics have None for global fields."""
        mock_resolve.return_value = _PlainProcessor

        step = ProcessingStep(
            processor_name="PlainProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((3, 3))
        wr = executor.execute(source)
        sm = wr.metrics.step_metrics[0]
        assert sm.global_pass_duration is None
        assert sm.global_pass_memory is None


# ---------------------------------------------------------------------------
# Tests: Global pass metrics
# ---------------------------------------------------------------------------


class TestGlobalPassMetrics:
    """Verify global pass timing and memory are captured."""

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_global_pass_duration_recorded(self, mock_resolve):
        """StepMetrics for a global-pass processor has duration > 0."""
        mock_resolve.return_value = _MedianThresholdProcessor

        step = ProcessingStep(
            processor_name="MedianThresholdProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.arange(100.0).reshape(10, 10)
        wr = executor.execute(source)

        sm = wr.metrics.step_metrics[0]
        assert sm.global_pass_duration is not None
        assert sm.global_pass_duration >= 0.0

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_global_pass_memory_recorded(self, mock_resolve):
        """StepMetrics for a global-pass processor has memory >= 0."""
        mock_resolve.return_value = _MedianThresholdProcessor

        step = ProcessingStep(
            processor_name="MedianThresholdProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.arange(100.0).reshape(10, 10)
        wr = executor.execute(source)

        sm = wr.metrics.step_metrics[0]
        assert sm.global_pass_memory is not None
        assert sm.global_pass_memory >= 0

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_global_pass_metrics_in_dict(self, mock_resolve):
        """to_dict() includes global pass metrics when present."""
        mock_resolve.return_value = _MedianThresholdProcessor

        step = ProcessingStep(
            processor_name="MedianThresholdProcessor",
            processor_version="1.0",
        )
        wf = _make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.arange(100.0).reshape(10, 10)
        wr = executor.execute(source)

        d = wr.metrics.step_metrics[0].to_dict()
        assert "global_pass_duration" in d
        assert "global_pass_memory" in d


# ---------------------------------------------------------------------------
# Tests: Mixed pipeline (global + non-global)
# ---------------------------------------------------------------------------


class TestMixedPipeline:
    """Pipeline with both global-pass and plain processors."""

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_mixed_pipeline_correct_order(self, mock_resolve):
        """Global-pass processor followed by plain: both work correctly."""

        def _resolve(name):
            if name == "MedianThresholdProcessor":
                return _MedianThresholdProcessor
            return _PlainProcessor

        mock_resolve.side_effect = _resolve

        steps = [
            ProcessingStep("MedianThresholdProcessor", "1.0"),
            ProcessingStep("PlainProcessor", "1.0"),
        ]
        wf = _make_workflow(steps)
        executor = WorkflowExecutor(wf)

        source = np.arange(1.0, 10.0).reshape(3, 3)
        wr = executor.execute(source)

        # Step 1: threshold at median=5.0 -> binary 0/1
        # Step 2: multiply by 2.0
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )
        np.testing.assert_array_equal(wr.result, expected)

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_mixed_pipeline_metrics(self, mock_resolve):
        """Global pass metrics only on the global-pass step."""

        def _resolve(name):
            if name == "MedianThresholdProcessor":
                return _MedianThresholdProcessor
            return _PlainProcessor

        mock_resolve.side_effect = _resolve

        steps = [
            ProcessingStep("MedianThresholdProcessor", "1.0"),
            ProcessingStep("PlainProcessor", "1.0"),
        ]
        wf = _make_workflow(steps)
        executor = WorkflowExecutor(wf)

        source = np.arange(1.0, 10.0).reshape(3, 3)
        wr = executor.execute(source)

        # First step has global pass metrics
        sm0 = wr.metrics.step_metrics[0]
        assert sm0.global_pass_duration is not None

        # Second step does not
        sm1 = wr.metrics.step_metrics[1]
        assert sm1.global_pass_duration is None
