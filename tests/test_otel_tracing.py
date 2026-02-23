# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.instrumentation.tracing — OpenTelemetry hook.

Verifies that the ``OtelHook`` correctly creates traces with parent-child
span relationships for workflows, steps, and global passes.  Uses the
``InMemorySpanExporter`` from the OTel SDK to capture finished spans
without requiring a running collector.

Dependencies
------------
pytest, opentelemetry-api, opentelemetry-sdk

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

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry not installed")

from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from grdl_rt.execution.context import ExecutionContext  # noqa: E402
from grdl_rt.execution.instrumentation.tracing import OtelHook, _create_provider  # noqa: E402
from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter():
    """InMemorySpanExporter that captures finished spans."""
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    """TracerProvider wired to the in-memory exporter."""
    resource = Resource.create({"service.name": "test-grdl"})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    return tp


@pytest.fixture
def hook(provider):
    """OtelHook backed by the test TracerProvider."""
    return OtelHook(tracer_provider=provider)


@pytest.fixture
def ctx():
    """Minimal ExecutionContext for test runs."""
    return ExecutionContext(
        workflow_id="test-wf:1.0",
        workflow_name="TestWorkflow",
        run_id="run-001",
    )


def _make_step_metrics(
    step_index=0,
    processor_name="TestProcessor",
    wall_time_s=1.5,
    cpu_time_s=1.0,
    peak_rss_bytes=4096,
    gpu_used=False,
    status="success",
    global_pass_duration=None,
):
    """Build a StepMetrics with sensible defaults."""
    return StepMetrics(
        step_index=step_index,
        processor_name=processor_name,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        peak_rss_bytes=peak_rss_bytes,
        gpu_used=gpu_used,
        status=status,
        global_pass_duration=global_pass_duration,
    )


def _make_workflow_metrics(
    step_metrics=None,
    status="success",
    total_wall_time_s=3.0,
    total_cpu_time_s=2.0,
    peak_rss_bytes=8192,
):
    """Build a WorkflowMetrics with sensible defaults."""
    if step_metrics is None:
        step_metrics = []
    return WorkflowMetrics(
        workflow_id="test-wf:1.0",
        run_id="run-001",
        workflow_name="TestWorkflow",
        workflow_version="1.0.0",
        total_wall_time_s=total_wall_time_s,
        total_cpu_time_s=total_cpu_time_s,
        peak_rss_bytes=peak_rss_bytes,
        step_metrics=step_metrics,
        status=status,
    )


# ---------------------------------------------------------------------------
# Tests: Provider creation
# ---------------------------------------------------------------------------


class TestCreateProvider:
    def test_otel_hook_creates_provider(self, hook):
        """OtelHook exposes a .provider property that is a TracerProvider."""
        assert hook.provider is not None
        assert isinstance(hook.provider, TracerProvider)

    def test_create_provider_returns_tracer_provider(self):
        """_create_provider returns a TracerProvider with correct resource."""
        tp = _create_provider("my-service", "none")
        assert isinstance(tp, TracerProvider)

    def test_hook_with_no_provider_creates_one(self):
        """OtelHook creates its own provider when none is supplied."""
        h = OtelHook(service_name="auto-created", exporter="none")
        assert h.provider is not None
        assert isinstance(h.provider, TracerProvider)


# ---------------------------------------------------------------------------
# Tests: Workflow span lifecycle
# ---------------------------------------------------------------------------


class TestWorkflowSpan:
    def test_workflow_produces_root_span(self, hook, exporter, ctx):
        """A full workflow start/end lifecycle produces one root span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "workflow:TestWorkflow"
        assert span.parent is None  # root span

    def test_workflow_span_has_initial_attributes(self, hook, exporter, ctx):
        """Workflow span carries name, version, and run_id from start."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        span = exporter.get_finished_spans()[0]
        assert span.attributes["workflow.name"] == "TestWorkflow"
        assert span.attributes["workflow.version"] == "1.0.0"
        assert span.attributes["workflow.run_id"] == "run-001"

    def test_workflow_span_attributes(self, hook, exporter, ctx):
        """on_workflow_end sets total_wall_time, status, step_count."""
        sm = _make_step_metrics(step_index=0)
        wm = _make_workflow_metrics(
            step_metrics=[sm],
            total_wall_time_s=5.5,
            total_cpu_time_s=3.3,
            peak_rss_bytes=16384,
            status="success",
        )

        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_workflow_end(ctx, wm)

        span = exporter.get_finished_spans()[0]
        assert span.attributes["workflow.total_wall_time_s"] == 5.5
        assert span.attributes["workflow.total_cpu_time_s"] == 3.3
        assert span.attributes["workflow.peak_rss_bytes"] == 16384
        assert span.attributes["workflow.status"] == "success"
        assert span.attributes["workflow.step_count"] == 1


# ---------------------------------------------------------------------------
# Tests: Step spans
# ---------------------------------------------------------------------------


class TestStepSpan:
    def test_step_produces_child_span(self, hook, exporter, ctx):
        """A workflow with one step produces a parent span and a child span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "LeeFilter")
        hook.on_step_end(ctx, 0, _make_step_metrics())
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        span_names = {s.name for s in spans}
        assert "workflow:TestWorkflow" in span_names
        assert "step:LeeFilter" in span_names

        # Verify parent-child relationship
        workflow_span = next(s for s in spans if s.name.startswith("workflow:"))
        step_span = next(s for s in spans if s.name.startswith("step:"))

        assert step_span.parent is not None
        assert step_span.parent.span_id == workflow_span.context.span_id

    def test_multiple_steps(self, hook, exporter, ctx):
        """Two steps produce the workflow span plus two step spans."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")

        hook.on_step_start(ctx, 0, "FilterA")
        hook.on_step_end(ctx, 0, _make_step_metrics(step_index=0, processor_name="FilterA"))

        hook.on_step_start(ctx, 1, "FilterB")
        hook.on_step_end(ctx, 1, _make_step_metrics(step_index=1, processor_name="FilterB"))

        wm = _make_workflow_metrics(
            step_metrics=[
                _make_step_metrics(step_index=0, processor_name="FilterA"),
                _make_step_metrics(step_index=1, processor_name="FilterB"),
            ]
        )
        hook.on_workflow_end(ctx, wm)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        workflow_spans = [s for s in spans if s.name.startswith("workflow:")]
        step_spans = [s for s in spans if s.name.startswith("step:")]
        assert len(workflow_spans) == 1
        assert len(step_spans) == 2

        step_names = {s.name for s in step_spans}
        assert "step:FilterA" in step_names
        assert "step:FilterB" in step_names

        # Both steps are children of the workflow span
        wf_span_id = workflow_spans[0].context.span_id
        for ss in step_spans:
            assert ss.parent is not None
            assert ss.parent.span_id == wf_span_id

    def test_step_span_attributes(self, hook, exporter, ctx):
        """Step span carries processor name, timing, memory, and status."""
        sm = _make_step_metrics(
            step_index=0,
            processor_name="LeeFilter",
            wall_time_s=2.5,
            cpu_time_s=1.8,
            peak_rss_bytes=65536,
            gpu_used=True,
            status="success",
            global_pass_duration=0.3,
        )

        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "LeeFilter")
        hook.on_step_end(ctx, 0, sm)
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        step_span = next(s for s in exporter.get_finished_spans() if s.name.startswith("step:"))

        assert step_span.attributes["step.index"] == 0
        assert step_span.attributes["step.processor"] == "LeeFilter"
        assert step_span.attributes["step.wall_time_s"] == 2.5
        assert step_span.attributes["step.cpu_time_s"] == 1.8
        assert step_span.attributes["step.peak_rss_bytes"] == 65536
        assert step_span.attributes["step.gpu_used"] is True
        assert step_span.attributes["step.status"] == "success"
        assert step_span.attributes["step.global_pass_duration_s"] == 0.3

    def test_step_span_without_global_pass_duration(self, hook, exporter, ctx):
        """Step span omits global_pass_duration_s when not set."""
        sm = _make_step_metrics(global_pass_duration=None)

        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "TestProcessor")
        hook.on_step_end(ctx, 0, sm)
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        step_span = next(s for s in exporter.get_finished_spans() if s.name.startswith("step:"))
        assert "step.global_pass_duration_s" not in step_span.attributes


# ---------------------------------------------------------------------------
# Tests: Global pass spans
# ---------------------------------------------------------------------------


class TestGlobalPassSpan:
    def test_global_pass_produces_grandchild_span(self, hook, exporter, ctx):
        """A global pass creates a span under the step span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "MedianFilter")
        hook.on_global_pass_start(ctx, 0, "MedianFilter")
        hook.on_global_pass_end(ctx, 0, duration_s=0.25, peak_memory_bytes=2048)
        hook.on_step_end(ctx, 0, _make_step_metrics())
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        workflow_span = next(s for s in spans if s.name.startswith("workflow:"))
        step_span = next(s for s in spans if s.name.startswith("step:"))
        gp_span = next(s for s in spans if s.name.startswith("global_pass:"))

        assert gp_span.name == "global_pass:MedianFilter"

        # Global pass is a child of the step span
        assert gp_span.parent is not None
        assert gp_span.parent.span_id == step_span.context.span_id

        # Step span is a child of the workflow span
        assert step_span.parent is not None
        assert step_span.parent.span_id == workflow_span.context.span_id

    def test_global_pass_span_attributes(self, hook, exporter, ctx):
        """Global pass span carries duration and peak memory."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "MedianFilter")
        hook.on_global_pass_start(ctx, 0, "MedianFilter")
        hook.on_global_pass_end(ctx, 0, duration_s=0.42, peak_memory_bytes=12288)
        hook.on_step_end(ctx, 0, _make_step_metrics())
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        gp_span = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("global_pass:")
        )
        assert gp_span.attributes["global_pass.step_index"] == 0
        assert gp_span.attributes["global_pass.processor"] == "MedianFilter"
        assert gp_span.attributes["global_pass.duration_s"] == 0.42
        assert gp_span.attributes["global_pass.peak_memory_bytes"] == 12288

    def test_global_pass_without_step_context_falls_back(self, hook, exporter, ctx):
        """If no step context exists, global pass parents to workflow span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        # Start global pass without starting a step first
        hook.on_global_pass_start(ctx, 0, "OrphanPass")
        hook.on_global_pass_end(ctx, 0, duration_s=0.1, peak_memory_bytes=512)
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        spans = exporter.get_finished_spans()
        workflow_span = next(s for s in spans if s.name.startswith("workflow:"))
        gp_span = next(s for s in spans if s.name.startswith("global_pass:"))

        # Falls back to workflow context as parent
        assert gp_span.parent is not None
        assert gp_span.parent.span_id == workflow_span.context.span_id


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_error_sets_span_status_on_workflow(self, hook, exporter, ctx):
        """on_error with no step_index sets ERROR status on workflow span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")

        err = RuntimeError("workflow exploded")
        hook.on_error(ctx, err, step_index=None)
        hook.on_workflow_end(ctx, _make_workflow_metrics(status="failed"))

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "workflow exploded" in span.status.description

    def test_error_sets_span_status_on_step(self, hook, exporter, ctx):
        """on_error with a step_index sets ERROR status on the step span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "BadFilter")

        err = ValueError("bad pixel value")
        hook.on_error(ctx, err, step_index=0)

        hook.on_step_end(ctx, 0, _make_step_metrics(status="failed"))
        hook.on_workflow_end(ctx, _make_workflow_metrics(status="failed"))

        step_span = next(s for s in exporter.get_finished_spans() if s.name.startswith("step:"))
        assert step_span.status.status_code == StatusCode.ERROR
        assert "bad pixel value" in step_span.status.description

    def test_error_records_exception_event(self, hook, exporter, ctx):
        """on_error records an exception event on the span."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")

        err = RuntimeError("something went wrong")
        hook.on_error(ctx, err, step_index=None)
        hook.on_workflow_end(ctx, _make_workflow_metrics(status="failed"))

        span = exporter.get_finished_spans()[0]
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) >= 1

        event = exception_events[0]
        assert event.attributes["exception.type"] == "RuntimeError"
        assert "something went wrong" in event.attributes["exception.message"]

    def test_error_on_missing_span_is_noop(self, hook, exporter, ctx):
        """on_error with an unknown step_index does not crash."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        # step_index=99 was never started
        hook.on_error(ctx, RuntimeError("ghost"), step_index=99)
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        # Should complete without error; workflow span has no ERROR status
        span = exporter.get_finished_spans()[0]
        assert span.status.status_code != StatusCode.ERROR


# ---------------------------------------------------------------------------
# Tests: Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_import_guard_raises_when_otel_missing(self):
        """OtelHook raises ImportError when opentelemetry is not installed."""
        with patch("grdl_rt.execution.instrumentation.tracing._otel_trace", None):
            with pytest.raises(ImportError, match="opentelemetry"):
                OtelHook()


# ---------------------------------------------------------------------------
# Tests: Cleanup and idempotency
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_on_step_end_without_start_is_noop(self, hook, exporter, ctx):
        """on_step_end for an unstarted step does not crash."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_end(ctx, 99, _make_step_metrics(step_index=99))
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        # Only the workflow span should exist
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "workflow:TestWorkflow"

    def test_on_global_pass_end_without_start_is_noop(self, hook, exporter, ctx):
        """on_global_pass_end for an unstarted pass does not crash."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_global_pass_end(ctx, 0, duration_s=1.0, peak_memory_bytes=0)
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

    def test_on_workflow_end_without_start_is_noop(self, hook, exporter, ctx):
        """on_workflow_end for an unstarted workflow does not crash."""
        other_ctx = ExecutionContext(
            workflow_id="other:1.0",
            workflow_name="Other",
            run_id="run-999",
        )
        hook.on_workflow_end(other_ctx, _make_workflow_metrics())

        spans = exporter.get_finished_spans()
        assert len(spans) == 0

    def test_spans_cleaned_after_workflow_end(self, hook, exporter, ctx):
        """Internal dicts are cleaned up after a workflow completes."""
        hook.on_workflow_start(ctx, "TestWorkflow", "1.0.0")
        hook.on_step_start(ctx, 0, "Filter")
        hook.on_step_end(ctx, 0, _make_step_metrics())
        hook.on_workflow_end(ctx, _make_workflow_metrics())

        assert ctx.run_id not in hook._workflow_spans
        assert ctx.run_id not in hook._workflow_contexts
        assert f"{ctx.run_id}:0" not in hook._step_spans
        assert f"{ctx.run_id}:0" not in hook._step_contexts
