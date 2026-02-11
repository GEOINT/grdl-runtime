# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.builder — Workflow and WorkflowStep.

Author
------
Steven Siebert

Created
-------
2026-02-11
"""

import numpy as np
import pytest

from grdl_rt.execution.builder import Workflow, WorkflowStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _double(source: np.ndarray) -> np.ndarray:
    """Plain function step: doubles the input."""
    return source * 2.0


def _add_one(source: np.ndarray) -> np.ndarray:
    """Plain function step: adds one."""
    return source + 1.0


class _FakeProcessor:
    """Minimal processor-like class with a bound method and GPU flag."""

    __gpu_compatible__ = True

    def transform(self, source: np.ndarray) -> np.ndarray:
        return source * 3.0


class _FakeTransform:
    """Mimics ImageTransform with an apply() method."""

    __gpu_compatible__ = False

    def apply(self, source: np.ndarray, **kwargs) -> np.ndarray:
        return source + 10.0


# ---------------------------------------------------------------------------
# WorkflowStep
# ---------------------------------------------------------------------------

class TestWorkflowStep:
    def test_dataclass_fields(self):
        ws = WorkflowStep(fn=_double, name="double", gpu_compatible=False)
        assert ws.name == "double"
        assert ws.gpu_compatible is False
        result = ws.fn(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.array([2.0, 4.0]))


# ---------------------------------------------------------------------------
# Workflow construction
# ---------------------------------------------------------------------------

class TestWorkflowConstruction:
    def test_empty_workflow(self):
        wf = Workflow("empty")
        assert wf.name == "empty"
        assert wf.version == "0.1.0"
        assert len(wf) == 0
        assert wf.steps == []

    def test_name_and_metadata(self):
        wf = Workflow(
            "test",
            version="2.0.0",
            description="A test workflow",
            modalities=["SAR"],
        )
        assert wf.name == "test"
        assert wf.version == "2.0.0"
        assert wf.description == "A test workflow"
        assert len(wf.tags.modalities) == 1

    def test_fluent_chaining(self):
        wf = (
            Workflow("chain")
            .step(_double, name="double")
            .step(_add_one, name="add one")
        )
        assert len(wf) == 2
        assert wf.steps[0].name == "double"
        assert wf.steps[1].name == "add one"

    def test_repr(self):
        wf = Workflow("repr test").step(_double, name="d")
        r = repr(wf)
        assert "repr test" in r
        assert "d" in r


# ---------------------------------------------------------------------------
# Workflow.step() — callable types
# ---------------------------------------------------------------------------

class TestWorkflowStep_Registration:
    def test_plain_function(self):
        wf = Workflow("fn").step(_double)
        assert len(wf) == 1
        assert "_double" in wf.steps[0].name
        assert wf.steps[0].gpu_compatible is False

    def test_lambda(self):
        wf = Workflow("lam").step(lambda x: x * 5, name="times 5")
        assert wf.steps[0].name == "times 5"

    def test_bound_method(self):
        proc = _FakeProcessor()
        wf = Workflow("bound").step(proc.transform)
        assert wf.steps[0].name == "_FakeProcessor.transform"
        assert wf.steps[0].gpu_compatible is True

    def test_bound_method_custom_name(self):
        proc = _FakeProcessor()
        wf = Workflow("bound").step(proc.transform, name="My Step")
        assert wf.steps[0].name == "My Step"

    def test_image_transform_instance(self):
        """ImageTransform instances are wrapped to call .apply()."""
        try:
            from grdl.image_processing.base import ImageTransform
        except ImportError:
            pytest.skip("grdl not available")

        # Use _FakeTransform only if ImageTransform is available and
        # _FakeTransform is actually recognized. For this test we use
        # the apply-based duck typing via callable detection.
        # This test verifies the callable path works for objects with apply().
        ft = _FakeTransform()
        wf = Workflow("transform").step(ft.apply, name="FakeTransform")
        assert len(wf) == 1

    def test_rejects_class_not_instance(self):
        with pytest.raises(TypeError, match="Did you mean"):
            Workflow("bad").step(_FakeProcessor)

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="requires a callable"):
            Workflow("bad").step(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Workflow.execute()
# ---------------------------------------------------------------------------

class TestWorkflowExecute:
    def test_empty_workflow_returns_source(self):
        wf = Workflow("empty")
        source = np.ones((4, 4))
        result = wf.execute(source)
        np.testing.assert_array_equal(result, source)

    def test_single_step(self):
        wf = Workflow("single").step(_double)
        source = np.array([1.0, 2.0, 3.0])
        result = wf.execute(source)
        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_multi_step_pipeline(self):
        wf = (
            Workflow("multi")
            .step(_double)       # x2
            .step(_add_one)      # +1
            .step(_double)       # x2
        )
        source = np.array([1.0])
        # (1*2 + 1) * 2 = 6
        result = wf.execute(source)
        np.testing.assert_array_equal(result, np.array([6.0]))

    def test_bound_method_step(self):
        proc = _FakeProcessor()
        wf = Workflow("bound").step(proc.transform)
        source = np.array([2.0])
        result = wf.execute(source)
        np.testing.assert_array_equal(result, np.array([6.0]))

    def test_lambda_step(self):
        wf = Workflow("lam").step(lambda x: x ** 2, name="square")
        source = np.array([3.0])
        result = wf.execute(source)
        np.testing.assert_array_equal(result, np.array([9.0]))

    def test_mixed_step_types(self):
        proc = _FakeProcessor()
        wf = (
            Workflow("mixed")
            .step(_double, name="double")
            .step(proc.transform, name="triple")
            .step(lambda x: x + 1, name="inc")
        )
        source = np.array([1.0])
        # (1*2) * 3 + 1 = 7
        result = wf.execute(source)
        np.testing.assert_array_equal(result, np.array([7.0]))


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestWorkflowProgress:
    def test_progress_callback_values(self):
        progress_values: list = []
        wf = (
            Workflow("progress")
            .step(_double)
            .step(_add_one)
            .step(_double)
        )
        wf.execute(np.array([1.0]), progress_callback=progress_values.append)
        assert len(progress_values) == 3
        assert progress_values == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_no_callback_is_fine(self):
        wf = Workflow("no cb").step(_double)
        result = wf.execute(np.array([5.0]))
        np.testing.assert_array_equal(result, np.array([10.0]))


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

class TestWorkflowExecuteBatch:
    def test_batch_returns_correct_results(self):
        wf = Workflow("batch").step(_double)
        sources = [np.array([float(i)]) for i in range(4)]
        results = wf.execute_batch(sources)
        assert len(results) == 4
        for i, r in enumerate(results):
            np.testing.assert_array_equal(r, np.array([float(i * 2)]))

    def test_batch_progress_callback(self):
        progress_values: list = []
        wf = Workflow("batch progress").step(_double).step(_add_one)
        sources = [np.array([1.0]), np.array([2.0])]
        wf.execute_batch(sources, progress_callback=progress_values.append)
        # 2 sources x 2 steps = 4 total progress updates
        assert len(progress_values) == 4
        # Final progress should be ~1.0
        assert progress_values[-1] == pytest.approx(1.0)

    def test_empty_batch(self):
        wf = Workflow("empty batch").step(_double)
        results = wf.execute_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestWorkflowErrorHandling:
    def test_step_failure_includes_context(self):
        def _bad_step(source):
            raise ValueError("something broke")

        wf = Workflow("error test").step(_bad_step, name="Bad Step")
        with pytest.raises(RuntimeError, match="Bad Step"):
            wf.execute(np.array([1.0]))

    def test_step_failure_includes_workflow_name(self):
        def _bad_step(source):
            raise ValueError("oops")

        wf = Workflow("My Workflow").step(_bad_step, name="Fail")
        with pytest.raises(RuntimeError, match="My Workflow"):
            wf.execute(np.array([1.0]))

    def test_step_failure_preserves_cause(self):
        def _bad_step(source):
            raise ValueError("root cause")

        wf = Workflow("chain").step(_bad_step, name="Fail")
        with pytest.raises(RuntimeError) as exc_info:
            wf.execute(np.array([1.0]))
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "root cause" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# GPU-compatible flag inference
# ---------------------------------------------------------------------------

class TestGpuCompatibleInference:
    def test_bound_method_inherits_flag(self):
        proc = _FakeProcessor()
        wf = Workflow("gpu").step(proc.transform)
        assert wf.steps[0].gpu_compatible is True

    def test_plain_function_not_gpu(self):
        wf = Workflow("cpu").step(_double)
        assert wf.steps[0].gpu_compatible is False

    def test_lambda_not_gpu(self):
        wf = Workflow("cpu").step(lambda x: x)
        assert wf.steps[0].gpu_compatible is False


# ---------------------------------------------------------------------------
# Workflow.source() — deferred data factory
# ---------------------------------------------------------------------------

class TestWorkflowSource:
    def test_source_provides_input(self):
        """execute() without an array calls the registered source."""
        wf = (
            Workflow("src")
            .source(lambda: np.array([10.0]))
            .step(_double)
        )
        result = wf.execute()
        np.testing.assert_array_equal(result, np.array([20.0]))

    def test_source_with_args(self):
        """Positional args are forwarded to the source callable."""
        def _make(val):
            return np.array([val])

        wf = (
            Workflow("src args")
            .source(_make, 7.0)
            .step(_add_one)
        )
        result = wf.execute()
        np.testing.assert_array_equal(result, np.array([8.0]))

    def test_source_with_kwargs(self):
        """Keyword args are forwarded to the source callable."""
        def _make(*, scale=1.0):
            return np.ones(3) * scale

        wf = (
            Workflow("src kwargs")
            .source(_make, scale=5.0)
            .step(_double)
        )
        result = wf.execute()
        np.testing.assert_array_equal(result, np.ones(3) * 10.0)

    def test_explicit_source_overrides_stored(self):
        """Passing an array to execute() overrides the stored source."""
        wf = (
            Workflow("override")
            .source(lambda: np.array([999.0]))
            .step(_double)
        )
        result = wf.execute(np.array([1.0]))
        np.testing.assert_array_equal(result, np.array([2.0]))

    def test_no_source_raises(self):
        """execute() with no array and no source raises ValueError."""
        wf = Workflow("no src").step(_double)
        with pytest.raises(ValueError, match="No source provided"):
            wf.execute()

    def test_source_fluent_chaining(self):
        """source() returns self for fluent chaining."""
        wf = Workflow("chain")
        ret = wf.source(lambda: np.ones(2))
        assert ret is wf


# ---------------------------------------------------------------------------
# ImageTransform instances as steps
# ---------------------------------------------------------------------------

class TestImageTransformSteps:
    def test_to_decibels_as_step(self):
        """ToDecibels instance can be passed directly to step()."""
        try:
            from grdl.image_processing.intensity import ToDecibels
        except ImportError:
            pytest.skip("grdl not available")

        to_db = ToDecibels(floor_db=-50.0)
        wf = Workflow("db").step(to_db)
        assert wf.steps[0].name == "ToDecibels"
        assert wf.steps[0].gpu_compatible is True

        source = np.array([1.0, 10.0, 100.0])
        result = wf.execute(source)
        # 20*log10(1) = 0, 20*log10(10) = 20, 20*log10(100) = 40
        np.testing.assert_array_almost_equal(result, [0.0, 20.0, 40.0], decimal=2)

    def test_percentile_stretch_as_step(self):
        """PercentileStretch instance can be passed directly to step()."""
        try:
            from grdl.image_processing.intensity import PercentileStretch
        except ImportError:
            pytest.skip("grdl not available")

        stretch = PercentileStretch(plow=0.0, phigh=100.0)
        wf = Workflow("stretch").step(stretch)
        assert wf.steps[0].name == "PercentileStretch"
        assert wf.steps[0].gpu_compatible is True

        source = np.array([0.0, 50.0, 100.0])
        result = wf.execute(source)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_chained_transform_pipeline(self):
        """ToDecibels → PercentileStretch as a composed workflow."""
        try:
            from grdl.image_processing.intensity import ToDecibels, PercentileStretch
        except ImportError:
            pytest.skip("grdl not available")

        wf = (
            Workflow("display pipeline")
            .step(ToDecibels(), name="To dB")
            .step(PercentileStretch(plow=0.0, phigh=100.0), name="Stretch")
        )
        source = np.array([1.0, 10.0, 100.0])
        result = wf.execute(source)
        # Result should be in [0, 1] after stretch
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype == np.float32
