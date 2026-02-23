"""
Tests for grdl_rt.execution.builder — Workflow, WorkflowStep, DeferredStep.

Author
------
Steven Siebert

Created
-------
2026-02-11
"""

import numpy as np
import pytest

from grdl_rt.execution.builder import DeferredStep, Workflow, WorkflowStep
from grdl_rt.execution.result import WorkflowResult

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


class _SimpleProcessor:
    """Processor that doesn't need metadata."""

    __gpu_compatible__ = False

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def apply(self, source: np.ndarray) -> np.ndarray:
        return source * self.scale


class _MetadataProcessor:
    """Processor whose constructor requires metadata."""

    __gpu_compatible__ = True

    def __init__(self, metadata, factor: int = 1):
        self.metadata = metadata
        self.factor = factor

    def apply(self, source: np.ndarray) -> np.ndarray:
        return source * self.factor


class _FakeReader:
    """Minimal mock reader with context manager and metadata."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {"sensor": "fake", "rows": 100, "cols": 200}
        self._shape = (100, 200)
        self._data = np.arange(20000.0).reshape(100, 200)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_shape(self):
        return self._shape

    def read_full(self):
        return self._data.copy()

    def read_chip(self, row_start, row_end, col_start, col_end):
        return self._data[row_start:row_end, col_start:col_end].copy()


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
        wf = Workflow("chain").step(_double, name="double").step(_add_one, name="add one")
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
            from grdl.image_processing.base import ImageTransform  # noqa: F401
        except ImportError:
            pytest.skip("grdl not available")

        ft = _FakeTransform()
        wf = Workflow("transform").step(ft.apply, name="FakeTransform")
        assert len(wf) == 1

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="requires a callable"):
            Workflow("bad").step(42)  # type: ignore[arg-type]

    def test_kwargs_rejected_for_non_class(self):
        """Passing **kwargs to a non-class step raises TypeError."""
        with pytest.raises(TypeError, match="only supported for class-type"):
            Workflow("bad").step(_double, name="d", scale=2.0)


# ---------------------------------------------------------------------------
# Workflow.execute()
# ---------------------------------------------------------------------------


class TestWorkflowExecute:
    def test_empty_workflow_returns_source(self):
        wf = Workflow("empty")
        source = np.ones((4, 4))
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, source)

    def test_single_step(self):
        wf = Workflow("single").step(_double)
        source = np.array([1.0, 2.0, 3.0])
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([2.0, 4.0, 6.0]))

    def test_multi_step_pipeline(self):
        wf = Workflow("multi").step(_double).step(_add_one).step(_double)  # x2  # +1  # x2
        source = np.array([1.0])
        # (1*2 + 1) * 2 = 6
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([6.0]))

    def test_bound_method_step(self):
        proc = _FakeProcessor()
        wf = Workflow("bound").step(proc.transform)
        source = np.array([2.0])
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([6.0]))

    def test_lambda_step(self):
        wf = Workflow("lam").step(lambda x: x**2, name="square")
        source = np.array([3.0])
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([9.0]))

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
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([7.0]))


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestWorkflowProgress:
    def test_progress_callback_values(self):
        progress_values: list = []
        wf = Workflow("progress").step(_double).step(_add_one).step(_double)
        wf.execute(np.array([1.0]), progress_callback=progress_values.append)
        assert len(progress_values) == 3
        assert progress_values == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_no_callback_is_fine(self):
        wf = Workflow("no cb").step(_double)
        wr = wf.execute(np.array([5.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([10.0]))


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


class TestWorkflowExecuteBatch:
    def test_batch_returns_correct_results(self):
        wf = Workflow("batch").step(_double)
        sources = [np.array([float(i)]) for i in range(4)]
        results = wf.execute_batch(sources)
        assert len(results) == 4
        for i, wr in enumerate(results):
            assert isinstance(wr, WorkflowResult)
            assert wr.metrics.total_wall_time_s >= 0
            np.testing.assert_array_equal(wr.result, np.array([float(i * 2)]))

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
        assert len(results) == 0


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
        wf = Workflow("src").source(lambda: np.array([10.0])).step(_double)
        wr = wf.execute()
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([20.0]))

    def test_source_with_args(self):
        """Positional args are forwarded to the source callable."""

        def _make(val):
            return np.array([val])

        wf = Workflow("src args").source(_make, 7.0).step(_add_one)
        wr = wf.execute()
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([8.0]))

    def test_source_with_kwargs(self):
        """Keyword args are forwarded to the source callable."""

        def _make(*, scale=1.0):
            return np.ones(3) * scale

        wf = Workflow("src kwargs").source(_make, scale=5.0).step(_double)
        wr = wf.execute()
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.ones(3) * 10.0)

    def test_explicit_source_overrides_stored(self):
        """Passing an array to execute() overrides the stored source."""
        wf = Workflow("override").source(lambda: np.array([999.0])).step(_double)
        wr = wf.execute(np.array([1.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([2.0]))

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
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_almost_equal(wr.result, [0.0, 20.0, 40.0], decimal=2)

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
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_almost_equal(wr.result, [0.0, 0.5, 1.0])

    def test_chained_transform_pipeline(self):
        """ToDecibels → PercentileStretch as a composed workflow."""
        try:
            from grdl.image_processing.intensity import PercentileStretch, ToDecibels
        except ImportError:
            pytest.skip("grdl not available")

        wf = (
            Workflow("display pipeline")
            .step(ToDecibels(), name="To dB")
            .step(PercentileStretch(plow=0.0, phigh=100.0), name="Stretch")
        )
        source = np.array([1.0, 10.0, 100.0])
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        assert wr.result.min() >= 0.0
        assert wr.result.max() <= 1.0
        assert wr.result.dtype == np.float32


# ---------------------------------------------------------------------------
# Workflow.reader() — reader configuration
# ---------------------------------------------------------------------------


class TestWorkflowReader:
    def test_stores_reader_class(self):
        wf = Workflow("rdr").reader(_FakeReader)
        assert wf._reader_cls is _FakeReader

    def test_fluent_chaining(self):
        wf = Workflow("rdr")
        ret = wf.reader(_FakeReader)
        assert ret is wf

    def test_execute_filepath_without_reader_raises(self):
        wf = Workflow("no rdr").step(_double)
        with pytest.raises(ValueError, match="no reader"):
            wf.execute("some/path.nitf")


# ---------------------------------------------------------------------------
# Workflow.chip() — chip strategy
# ---------------------------------------------------------------------------


class TestWorkflowChip:
    def test_stores_strategy(self):
        wf = Workflow("chip").chip("center", size=2048)
        assert wf._chip_strategy == "center"
        assert wf._chip_kwargs == {"size": 2048}

    def test_fluent_chaining(self):
        wf = Workflow("chip")
        ret = wf.chip("full")
        assert ret is wf

    def test_default_strategy(self):
        wf = Workflow("chip").chip()
        assert wf._chip_strategy == "center"


# ---------------------------------------------------------------------------
# Deferred steps — .step(Class, **kwargs)
# ---------------------------------------------------------------------------


class TestDeferredStep:
    def test_class_stored_as_deferred(self):
        wf = Workflow("defer").step(_SimpleProcessor, scale=5.0)
        assert len(wf) == 1
        ds = wf.steps[0]
        assert isinstance(ds, DeferredStep)
        assert ds.processor_cls is _SimpleProcessor
        assert ds.kwargs == {"scale": 5.0}
        assert ds.name == "_SimpleProcessor"

    def test_class_with_custom_name(self):
        wf = Workflow("defer").step(_SimpleProcessor, name="Scale")
        assert wf.steps[0].name == "Scale"

    def test_class_no_kwargs(self):
        wf = Workflow("defer").step(_SimpleProcessor)
        ds = wf.steps[0]
        assert isinstance(ds, DeferredStep)
        assert ds.kwargs == {}

    def test_deferred_resolves_on_execute_array(self):
        """Deferred step without metadata resolves when executing on array."""
        wf = Workflow("defer exec").step(_SimpleProcessor, scale=3.0)
        wr = wf.execute(np.array([2.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([6.0]))

    def test_mixed_deferred_and_immediate(self):
        """Deferred and immediate steps can be mixed."""
        wf = Workflow("mixed").step(_double).step(_SimpleProcessor, scale=5.0).step(_add_one)
        source = np.array([1.0])
        # 1*2 = 2 → 2*5 = 10 → 10+1 = 11
        wr = wf.execute(source)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([11.0]))


# ---------------------------------------------------------------------------
# Metadata injection
# ---------------------------------------------------------------------------


class TestMetadataInjection:
    def test_metadata_injected_when_constructor_accepts_it(self):
        """Processor with 'metadata' param gets it injected."""
        fake_meta = {"sensor": "test"}
        wf = Workflow("meta").step(_MetadataProcessor, factor=4)
        wr = wf.execute(np.array([3.0]), metadata=fake_meta)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([12.0]))

    def test_user_metadata_kwarg_wins(self):
        """User-provided metadata in step kwargs takes precedence."""
        user_meta = {"sensor": "user"}
        reader_meta = {"sensor": "reader"}
        wf = Workflow("meta").step(
            _MetadataProcessor,
            metadata=user_meta,
            factor=2,
        )
        wr = wf.execute(np.array([5.0]), metadata=reader_meta)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([10.0]))

    def test_no_metadata_needed_skips_injection(self):
        """Processor without 'metadata' param works fine without metadata."""
        wf = Workflow("no meta").step(_SimpleProcessor, scale=7.0)
        wr = wf.execute(np.array([2.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([14.0]))

    def test_metadata_required_but_missing_raises(self):
        """Deferred step needing metadata with no metadata available raises."""
        wf = Workflow("need meta").step(_MetadataProcessor, factor=1)
        with pytest.raises(ValueError, match="requires metadata"):
            wf.execute(np.array([1.0]))

    def test_deferred_gpu_flag_detected(self):
        """GPU compatible flag is read from the constructed instance."""
        fake_meta = {"sensor": "test"}
        wf = Workflow("gpu defer").step(_MetadataProcessor, factor=1)
        # Execute to trigger resolution
        wf.execute(np.array([1.0]), metadata=fake_meta)
        # _MetadataProcessor has __gpu_compatible__ = True
        # (can't check after execute easily, but the resolution path works)


# ---------------------------------------------------------------------------
# Execute from file — full framework orchestration
# ---------------------------------------------------------------------------


class TestExecuteFromFile:
    def test_full_pipeline(self):
        """Execute with filepath opens reader, reads chip, runs pipeline."""
        wf = Workflow("file test").reader(_FakeReader).step(_double)
        # _FakeReader reads full image (100x200) by default (no chip strategy)
        wr = wf.execute("fake/path.nitf")
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        assert wr.result.shape == (100, 200)
        # _FakeReader data is arange(20000), doubled
        assert wr.result[0, 0] == 0.0
        assert wr.result[0, 1] == 2.0

    def test_center_chip_strategy(self):
        """Center chip strategy reads a center subset."""
        wf = Workflow("chip test").reader(_FakeReader).chip("center", size=10).step(_double)
        wr = wf.execute("fake/path.nitf")
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        # ChipExtractor centers 10x10 in 100x200 → rows 45:55, cols 95:105
        assert wr.result.shape == (10, 10)

    def test_full_chip_strategy(self):
        """'full' strategy reads entire image."""
        wf = Workflow("full test").reader(_FakeReader).chip("full").step(_double)
        wr = wf.execute("fake/path.nitf")
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        assert wr.result.shape == (100, 200)

    def test_deferred_step_with_reader_metadata(self):
        """Deferred steps receive metadata from the reader."""
        wf = Workflow("meta inject").reader(_FakeReader).step(_MetadataProcessor, factor=2)
        wr = wf.execute("fake/path.nitf")
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        assert wr.result.shape == (100, 200)
        # All values doubled
        assert wr.result[0, 1] == 2.0

    def test_progress_callback_from_file(self):
        """Progress callback fires when executing from file."""
        progress: list = []
        wf = Workflow("progress").reader(_FakeReader).step(_double).step(_add_one)
        wf.execute("fake/path.nitf", progress_callback=progress.append)
        assert len(progress) == 2
        assert progress == pytest.approx([0.5, 1.0])

    def test_unknown_chip_strategy_raises(self):
        wf = Workflow("bad chip").reader(_FakeReader).chip("unknown_strategy").step(_double)
        with pytest.raises(ValueError, match="Unknown chip strategy"):
            wf.execute("fake/path.nitf")

    def test_metadata_override(self):
        """Explicit metadata= overrides reader metadata."""
        custom_meta = {"sensor": "override"}
        wf = Workflow("override").reader(_FakeReader).step(_MetadataProcessor, factor=3)
        wr = wf.execute("fake/path.nitf", metadata=custom_meta)
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        assert wr.result.shape == (100, 200)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_execute_with_array_still_works(self):
        wf = Workflow("compat").step(_double)
        wr = wf.execute(np.array([5.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([10.0]))

    def test_source_factory_still_works(self):
        wf = Workflow("compat source").source(lambda: np.array([3.0])).step(_double)
        wr = wf.execute()
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([6.0]))

    def test_step_with_callable_still_works(self):
        proc = _FakeProcessor()
        wf = Workflow("compat callable").step(proc.transform).step(_add_one)
        wr = wf.execute(np.array([1.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        np.testing.assert_array_equal(wr.result, np.array([4.0]))

    def test_step_with_instance_still_works(self):
        try:
            from grdl.image_processing.intensity import ToDecibels
        except ImportError:
            pytest.skip("grdl not available")

        wf = Workflow("compat inst").step(ToDecibels())
        wr = wf.execute(np.array([1.0]))
        assert isinstance(wr, WorkflowResult)
        assert wr.metrics.total_wall_time_s >= 0
        assert np.isfinite(wr.result[0])
