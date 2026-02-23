"""
Tests for grdl_rt.execution.instrumentation.prometheus — PrometheusHook.

Validates that the Prometheus instrumentation hook correctly creates metrics,
observes step and workflow durations, increments error counters, and sets
memory and GPU utilization gauges on a dedicated collector registry.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

from unittest.mock import patch

import pytest

prometheus_client = pytest.importorskip(
    "prometheus_client", reason="prometheus_client not installed"
)

from grdl_rt.execution.context import ExecutionContext
from grdl_rt.execution.instrumentation.prometheus import PrometheusHook
from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ExecutionContext:
    """Return a minimal ExecutionContext for testing."""
    return ExecutionContext(
        workflow_id="test:1.0",
        workflow_name="TestWorkflow",
        run_id="run-abc-123",
    )


def _make_step_metrics(
    step_index: int = 0,
    processor_name: str = "LeeFilter",
    wall_time_s: float = 1.5,
    cpu_time_s: float = 1.2,
    peak_rss_bytes: int = 4096,
    gpu_used: bool = False,
) -> StepMetrics:
    """Return a StepMetrics instance with sensible defaults."""
    return StepMetrics(
        step_index=step_index,
        processor_name=processor_name,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        peak_rss_bytes=peak_rss_bytes,
        gpu_used=gpu_used,
    )


def _make_workflow_metrics(
    total_wall_time_s: float = 3.0,
    peak_rss_bytes: int = 8192,
) -> WorkflowMetrics:
    """Return a WorkflowMetrics instance with sensible defaults."""
    return WorkflowMetrics(
        workflow_id="test:1.0",
        run_id="run-abc-123",
        workflow_name="TestWorkflow",
        workflow_version="1.0.0",
        total_wall_time_s=total_wall_time_s,
        total_cpu_time_s=2.0,
        peak_rss_bytes=peak_rss_bytes,
        step_metrics=[],
        started_at="2026-02-11T00:00:00Z",
        completed_at="2026-02-11T00:00:03Z",
        status="success",
    )


def _get_sample_value(registry, metric_name, labels=None):
    """Read a sample value from the given registry.

    Uses ``prometheus_client.generate_latest`` to parse the registry
    and extract the requested metric value.

    Parameters
    ----------
    registry : prometheus_client.CollectorRegistry
        The registry to read from.
    metric_name : str
        Full metric name (including suffixes like ``_total``, ``_count``).
    labels : dict, optional
        Label key-value pairs to match.

    Returns
    -------
    float or None
        The sample value, or ``None`` if not found.
    """
    labels = labels or {}
    for metric in registry.collect():
        for sample in metric.samples:
            if sample.name == metric_name and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return sample.value
    return None


# ---------------------------------------------------------------------------
# PrometheusHook — metric creation
# ---------------------------------------------------------------------------


class TestPrometheusHookCreation:
    def test_prometheus_hook_creates_metrics(self):
        """Verify all 5 metric types exist on a fresh registry."""
        hook = PrometheusHook()
        registry = hook.registry

        assert isinstance(hook.workflow_duration, prometheus_client.Histogram)
        assert isinstance(hook.step_duration, prometheus_client.Histogram)
        assert isinstance(hook.errors_total, prometheus_client.Counter)
        assert isinstance(hook.memory_peak, prometheus_client.Gauge)
        assert isinstance(hook.gpu_used, prometheus_client.Gauge)

        # Verify registry is a CollectorRegistry (not the global one)
        assert isinstance(registry, prometheus_client.CollectorRegistry)
        assert registry is not prometheus_client.REGISTRY

    def test_prometheus_hook_custom_prefix(self):
        """Custom prefix is reflected in metric names."""
        hook = PrometheusHook(prefix="myapp")
        registry = hook.registry

        # Collect all metric names from the registry
        metric_names = set()
        for metric in registry.collect():
            for sample in metric.samples:
                metric_names.add(sample.name)

        # Trigger at least one observation so histograms appear
        ctx = _make_context()
        sm = _make_step_metrics()
        hook.on_step_end(ctx, 0, sm)

        wm = _make_workflow_metrics()
        hook.on_workflow_end(ctx, wm)

        metric_names = set()
        for metric in registry.collect():
            for sample in metric.samples:
                metric_names.add(sample.name)

        # Check that custom-prefixed names appear
        assert any(n.startswith("myapp_workflow_duration") for n in metric_names)
        assert any(n.startswith("myapp_step_duration") for n in metric_names)
        assert any(n.startswith("myapp_memory_peak") for n in metric_names)
        assert any(n.startswith("myapp_gpu_used") for n in metric_names)

        # And that the default prefix does NOT appear
        assert not any(n.startswith("grdl_rt_") for n in metric_names)

    def test_registry_property(self):
        """The .registry property returns the internal registry."""
        reg = prometheus_client.CollectorRegistry()
        hook = PrometheusHook(registry=reg)
        assert hook.registry is reg

    def test_custom_registry_used(self):
        """Metrics are registered on the provided registry, not the global one."""
        reg = prometheus_client.CollectorRegistry()
        PrometheusHook(registry=reg)

        # The custom registry should have metrics; verify by collecting.
        # Note: prometheus_client strips the "_total" suffix from Counter
        # metric names during collection, so "grdl_rt_errors_total" becomes
        # "grdl_rt_errors" at the metric level (samples still use "_total").
        names = {m.name for m in reg.collect()}
        assert "grdl_rt_workflow_duration_seconds" in names
        assert "grdl_rt_step_duration_seconds" in names
        assert "grdl_rt_errors" in names
        assert "grdl_rt_memory_peak_bytes" in names
        assert "grdl_rt_gpu_used" in names


# ---------------------------------------------------------------------------
# PrometheusHook — on_step_end
# ---------------------------------------------------------------------------


class TestStepEnd:
    def test_step_duration_observed(self):
        """Calling on_step_end observes the step histogram sample count."""
        hook = PrometheusHook()
        ctx = _make_context()
        sm = _make_step_metrics(
            processor_name="LeeFilter",
            wall_time_s=2.5,
        )

        hook.on_step_end(ctx, 0, sm)

        count = _get_sample_value(
            hook.registry,
            "grdl_rt_step_duration_seconds_count",
            labels={"processor": "LeeFilter"},
        )
        assert count == 1.0

        total = _get_sample_value(
            hook.registry,
            "grdl_rt_step_duration_seconds_sum",
            labels={"processor": "LeeFilter"},
        )
        assert total == 2.5

    def test_memory_peak_set(self):
        """on_step_end with peak_rss_bytes > 0 sets the gauge value."""
        hook = PrometheusHook()
        ctx = _make_context()
        sm = _make_step_metrics(peak_rss_bytes=1_048_576)

        hook.on_step_end(ctx, 0, sm)

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_memory_peak_bytes",
        )
        assert value == 1_048_576

    def test_memory_peak_not_set_when_zero(self):
        """on_step_end with peak_rss_bytes == 0 does not touch the gauge."""
        hook = PrometheusHook()
        ctx = _make_context()
        sm = _make_step_metrics(peak_rss_bytes=0)

        hook.on_step_end(ctx, 0, sm)

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_memory_peak_bytes",
        )
        # Gauge defaults to 0.0 and should remain untouched
        assert value == 0.0

    def test_gpu_used_set(self):
        """on_step_end with gpu_used=True sets the gauge to 1.0."""
        hook = PrometheusHook()
        ctx = _make_context()
        sm = _make_step_metrics(gpu_used=True)

        hook.on_step_end(ctx, 0, sm)

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_gpu_used",
        )
        assert value == 1.0

    def test_gpu_used_not_set_when_unused(self):
        """on_step_end with gpu_used=False sets the gauge to 0.0."""
        hook = PrometheusHook()
        ctx = _make_context()
        sm = _make_step_metrics(gpu_used=False)

        hook.on_step_end(ctx, 0, sm)

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_gpu_used",
        )
        assert value == 0.0


# ---------------------------------------------------------------------------
# PrometheusHook — on_workflow_end
# ---------------------------------------------------------------------------


class TestWorkflowEnd:
    def test_workflow_duration_observed(self):
        """Calling on_workflow_end observes the workflow histogram."""
        hook = PrometheusHook()
        ctx = _make_context()
        wm = _make_workflow_metrics(total_wall_time_s=5.75)

        hook.on_workflow_end(ctx, wm)

        count = _get_sample_value(
            hook.registry,
            "grdl_rt_workflow_duration_seconds_count",
        )
        assert count == 1.0

        total = _get_sample_value(
            hook.registry,
            "grdl_rt_workflow_duration_seconds_sum",
        )
        assert total == 5.75

    def test_workflow_end_sets_memory_peak(self):
        """on_workflow_end sets the memory peak gauge from workflow metrics."""
        hook = PrometheusHook()
        ctx = _make_context()
        wm = _make_workflow_metrics(peak_rss_bytes=2_097_152)

        hook.on_workflow_end(ctx, wm)

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_memory_peak_bytes",
        )
        assert value == 2_097_152


# ---------------------------------------------------------------------------
# PrometheusHook — on_error
# ---------------------------------------------------------------------------


class TestErrorCounter:
    def test_error_counter_increments(self):
        """on_error increments the counter by error type name."""
        hook = PrometheusHook()
        ctx = _make_context()

        hook.on_error(ctx, ValueError("bad value"))

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_errors_total",
            labels={"error_type": "ValueError"},
        )
        assert value == 1.0

    def test_error_counter_multiple_types(self):
        """Different error types get separate counter labels."""
        hook = PrometheusHook()
        ctx = _make_context()

        hook.on_error(ctx, ValueError("bad"))
        hook.on_error(ctx, TypeError("wrong type"))
        hook.on_error(ctx, ValueError("another bad"))

        val_count = _get_sample_value(
            hook.registry,
            "grdl_rt_errors_total",
            labels={"error_type": "ValueError"},
        )
        type_count = _get_sample_value(
            hook.registry,
            "grdl_rt_errors_total",
            labels={"error_type": "TypeError"},
        )
        assert val_count == 2.0
        assert type_count == 1.0

    def test_error_counter_with_step_index(self):
        """on_error works with an optional step_index argument."""
        hook = PrometheusHook()
        ctx = _make_context()

        hook.on_error(ctx, RuntimeError("crash"), step_index=3)

        value = _get_sample_value(
            hook.registry,
            "grdl_rt_errors_total",
            labels={"error_type": "RuntimeError"},
        )
        assert value == 1.0


# ---------------------------------------------------------------------------
# PrometheusHook — multiple steps
# ---------------------------------------------------------------------------


class TestMultipleSteps:
    def test_multiple_steps(self):
        """Simulate 2 steps and assert the step histogram count equals 2."""
        hook = PrometheusHook()
        ctx = _make_context()

        sm1 = _make_step_metrics(
            step_index=0,
            processor_name="FilterA",
            wall_time_s=1.0,
        )
        sm2 = _make_step_metrics(
            step_index=1,
            processor_name="FilterA",
            wall_time_s=2.0,
        )

        hook.on_step_end(ctx, 0, sm1)
        hook.on_step_end(ctx, 1, sm2)

        count = _get_sample_value(
            hook.registry,
            "grdl_rt_step_duration_seconds_count",
            labels={"processor": "FilterA"},
        )
        assert count == 2.0

        total = _get_sample_value(
            hook.registry,
            "grdl_rt_step_duration_seconds_sum",
            labels={"processor": "FilterA"},
        )
        assert total == 3.0

    def test_multiple_steps_different_processors(self):
        """Steps with different processor names get separate histogram labels."""
        hook = PrometheusHook()
        ctx = _make_context()

        sm1 = _make_step_metrics(step_index=0, processor_name="FilterA", wall_time_s=1.0)
        sm2 = _make_step_metrics(step_index=1, processor_name="FilterB", wall_time_s=2.0)

        hook.on_step_end(ctx, 0, sm1)
        hook.on_step_end(ctx, 1, sm2)

        count_a = _get_sample_value(
            hook.registry,
            "grdl_rt_step_duration_seconds_count",
            labels={"processor": "FilterA"},
        )
        count_b = _get_sample_value(
            hook.registry,
            "grdl_rt_step_duration_seconds_count",
            labels={"processor": "FilterB"},
        )
        assert count_a == 1.0
        assert count_b == 1.0


# ---------------------------------------------------------------------------
# PrometheusHook — import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_import_guard(self):
        """Mock prometheus_client as None to verify ImportError is raised."""
        with (
            patch("grdl_rt.execution.instrumentation.prometheus._prom", None),
            pytest.raises(ImportError, match="prometheus_client is required"),
        ):
            PrometheusHook()
