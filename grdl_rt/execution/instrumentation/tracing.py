# -*- coding: utf-8 -*-
"""
OpenTelemetry Tracing Hook — workflow traces with per-step spans.

Instruments the executor with OpenTelemetry traces:

- Each workflow execution is a root **trace**.
- Each processing step is a child **span** of the workflow span.
- Each global pass is a grandchild **span** of the step span.

Span attributes include processor name, step index, timing, memory,
GPU usage, and status.

Requires the optional ``opentelemetry-*`` packages::

    pip install grdl-runtime[tracing]

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from grdl_rt.execution.instrumentation import ExecutionHook

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry import context as _otel_context
except ImportError:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]
    _otel_context = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from grdl_rt.execution.context import ExecutionContext
    from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics


def _create_provider(
    service_name: str,
    exporter: str,
    endpoint: Optional[str] = None,
) -> Any:
    """Create an OpenTelemetry TracerProvider with the given exporter.

    Parameters
    ----------
    service_name : str
        OTel service name resource attribute.
    exporter : str
        One of ``"otlp"``, ``"console"``, ``"none"``.
    endpoint : str, optional
        OTLP endpoint URL (only used when ``exporter="otlp"``).

    Returns
    -------
    TracerProvider
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exp_kwargs: Dict[str, Any] = {}
        if endpoint:
            exp_kwargs["endpoint"] = endpoint
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(**exp_kwargs))
        )
    elif exporter == "console":
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
    # "none" — no exporter; useful with InMemorySpanExporter for tests

    return provider


class OtelHook(ExecutionHook):
    """OpenTelemetry tracing instrumentation hook.

    Parameters
    ----------
    tracer_provider : TracerProvider, optional
        Pre-configured TracerProvider.  If ``None``, one is created
        from *service_name* and *exporter*.
    service_name : str
        OTel service name (default ``"grdl-runtime"``).
    exporter : str
        Exporter type: ``"otlp"``, ``"console"``, or ``"none"``.
    endpoint : str, optional
        OTLP collector endpoint.

    Raises
    ------
    ImportError
        If ``opentelemetry-api`` or ``opentelemetry-sdk`` is not installed.
    """

    def __init__(
        self,
        tracer_provider: Any = None,
        service_name: str = "grdl-runtime",
        exporter: str = "none",
        endpoint: Optional[str] = None,
    ) -> None:
        if _otel_trace is None:
            raise ImportError(
                "opentelemetry-api and opentelemetry-sdk are required for "
                "OtelHook. Install with: pip install grdl-runtime[tracing]"
            )

        if tracer_provider is None:
            tracer_provider = _create_provider(service_name, exporter, endpoint)

        self._provider = tracer_provider
        self._tracer = tracer_provider.get_tracer("grdl_rt")

        # Active spans keyed by run_id or run_id:step_index
        self._workflow_spans: Dict[str, Any] = {}
        self._workflow_contexts: Dict[str, Any] = {}
        self._step_spans: Dict[str, Any] = {}
        self._step_contexts: Dict[str, Any] = {}
        self._global_pass_spans: Dict[str, Any] = {}

    @property
    def provider(self) -> Any:
        """Return the TracerProvider."""
        return self._provider

    def on_workflow_start(
        self,
        ctx: ExecutionContext,
        workflow_name: str,
        workflow_version: str,
    ) -> None:
        span = self._tracer.start_span(
            f"workflow:{workflow_name}",
            attributes={
                "workflow.name": workflow_name,
                "workflow.version": workflow_version,
                "workflow.run_id": ctx.run_id,
            },
        )
        self._workflow_spans[ctx.run_id] = span
        self._workflow_contexts[ctx.run_id] = _otel_trace.set_span_in_context(span)

    def on_step_start(
        self,
        ctx: ExecutionContext,
        step_index: int,
        processor_name: str,
    ) -> None:
        parent_context = self._workflow_contexts.get(ctx.run_id)
        span = self._tracer.start_span(
            f"step:{processor_name}",
            context=parent_context,
            attributes={
                "step.index": step_index,
                "step.processor": processor_name,
            },
        )
        key = f"{ctx.run_id}:{step_index}"
        self._step_spans[key] = span
        self._step_contexts[key] = _otel_trace.set_span_in_context(span)

    def on_step_end(
        self,
        ctx: ExecutionContext,
        step_index: int,
        step_metrics: StepMetrics,
    ) -> None:
        key = f"{ctx.run_id}:{step_index}"
        span = self._step_spans.pop(key, None)
        self._step_contexts.pop(key, None)
        if span is None:
            return

        span.set_attribute("step.wall_time_s", step_metrics.wall_time_s)
        span.set_attribute("step.cpu_time_s", step_metrics.cpu_time_s)
        span.set_attribute("step.peak_rss_bytes", step_metrics.peak_rss_bytes)
        span.set_attribute("step.gpu_used", step_metrics.gpu_used)
        span.set_attribute("step.status", step_metrics.status)
        if step_metrics.global_pass_duration is not None:
            span.set_attribute(
                "step.global_pass_duration_s",
                step_metrics.global_pass_duration,
            )
        span.end()

    def on_global_pass_start(
        self,
        ctx: ExecutionContext,
        step_index: int,
        processor_name: str,
    ) -> None:
        key = f"{ctx.run_id}:{step_index}"
        parent_context = self._step_contexts.get(key)
        if parent_context is None:
            parent_context = self._workflow_contexts.get(ctx.run_id)

        span = self._tracer.start_span(
            f"global_pass:{processor_name}",
            context=parent_context,
            attributes={
                "global_pass.step_index": step_index,
                "global_pass.processor": processor_name,
            },
        )
        self._global_pass_spans[key] = span

    def on_global_pass_end(
        self,
        ctx: ExecutionContext,
        step_index: int,
        duration_s: float,
        peak_memory_bytes: int,
    ) -> None:
        key = f"{ctx.run_id}:{step_index}"
        span = self._global_pass_spans.pop(key, None)
        if span is None:
            return

        span.set_attribute("global_pass.duration_s", duration_s)
        span.set_attribute("global_pass.peak_memory_bytes", peak_memory_bytes)
        span.end()

    def on_workflow_end(
        self,
        ctx: ExecutionContext,
        workflow_metrics: WorkflowMetrics,
    ) -> None:
        span = self._workflow_spans.pop(ctx.run_id, None)
        self._workflow_contexts.pop(ctx.run_id, None)
        if span is None:
            return

        span.set_attribute(
            "workflow.total_wall_time_s",
            workflow_metrics.total_wall_time_s,
        )
        span.set_attribute(
            "workflow.total_cpu_time_s",
            workflow_metrics.total_cpu_time_s,
        )
        span.set_attribute(
            "workflow.peak_rss_bytes",
            workflow_metrics.peak_rss_bytes,
        )
        span.set_attribute("workflow.status", workflow_metrics.status)
        span.set_attribute(
            "workflow.step_count",
            len(workflow_metrics.step_metrics),
        )
        span.end()

    def on_error(
        self,
        ctx: ExecutionContext,
        error: Exception,
        step_index: Optional[int] = None,
    ) -> None:
        if step_index is not None:
            key = f"{ctx.run_id}:{step_index}"
            span = self._step_spans.get(key)
        else:
            span = self._workflow_spans.get(ctx.run_id)

        if span is not None:
            span.set_status(
                _otel_trace.StatusCode.ERROR,
                str(error),
            )
            span.record_exception(error)


__all__ = ["OtelHook"]
