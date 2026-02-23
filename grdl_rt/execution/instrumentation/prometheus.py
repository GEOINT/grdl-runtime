"""
Prometheus Metrics Hook — workflow and step duration, errors, memory, GPU.

Instruments the executor with Prometheus metrics that the embedding
application can scrape via a ``CollectorRegistry``.  Uses a dedicated
registry by default so that the framework does not pollute the global
``REGISTRY``.

Requires the optional ``prometheus-client`` package::

    pip install grdl-runtime[metrics]

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

from typing import TYPE_CHECKING

from grdl_rt.execution.instrumentation import ExecutionHook

try:
    import prometheus_client as _prom
except ImportError:  # pragma: no cover
    _prom = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from grdl_rt.execution.context import ExecutionContext
    from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics


class PrometheusHook(ExecutionHook):
    """Prometheus metrics instrumentation hook.

    Creates the following metrics:

    - ``{prefix}_workflow_duration_seconds`` — Histogram of total
      workflow execution duration.
    - ``{prefix}_step_duration_seconds`` — Histogram of per-step
      execution duration, labeled by ``processor``.
    - ``{prefix}_errors_total`` — Counter of errors, labeled by
      ``error_type``.
    - ``{prefix}_memory_peak_bytes`` — Gauge of peak RSS memory.
    - ``{prefix}_gpu_used`` — Gauge indicating whether GPU was used
      for the most recent step (1.0 = yes, 0.0 = no).

    Parameters
    ----------
    registry : prometheus_client.CollectorRegistry, optional
        Registry to register metrics with.  If ``None``, creates a
        new dedicated registry (framework pattern).
    prefix : str
        Metric name prefix.  Defaults to ``"grdl_rt"``.

    Raises
    ------
    ImportError
        If ``prometheus_client`` is not installed.
    """

    def __init__(
        self,
        registry: object = None,
        prefix: str = "grdl_rt",
    ) -> None:
        if _prom is None:
            raise ImportError(
                "prometheus_client is required for PrometheusHook. "
                "Install it with: pip install grdl-runtime[metrics]"
            )

        if registry is None:
            registry = _prom.CollectorRegistry()

        self._registry = registry
        self._prefix = prefix

        self.workflow_duration = _prom.Histogram(
            f"{prefix}_workflow_duration_seconds",
            "Total workflow execution duration in seconds",
            registry=self._registry,
        )
        self.step_duration = _prom.Histogram(
            f"{prefix}_step_duration_seconds",
            "Per-step execution duration in seconds",
            labelnames=["processor"],
            registry=self._registry,
        )
        self.errors_total = _prom.Counter(
            f"{prefix}_errors_total",
            "Total error count by error type",
            labelnames=["error_type"],
            registry=self._registry,
        )
        self.memory_peak = _prom.Gauge(
            f"{prefix}_memory_peak_bytes",
            "Peak memory usage in bytes",
            registry=self._registry,
        )
        self.gpu_used = _prom.Gauge(
            f"{prefix}_gpu_used",
            "Whether GPU was used for the most recent step (1 = yes, 0 = no)",
            registry=self._registry,
        )

    @property
    def registry(self) -> object:
        """Return the Prometheus ``CollectorRegistry``."""
        return self._registry

    def on_step_end(
        self,
        ctx: ExecutionContext,
        step_index: int,
        step_metrics: StepMetrics,
    ) -> None:
        self.step_duration.labels(
            processor=step_metrics.processor_name,
        ).observe(step_metrics.wall_time_s)

        if step_metrics.peak_rss_bytes > 0:
            self.memory_peak.set(step_metrics.peak_rss_bytes)

        self.gpu_used.set(1.0 if step_metrics.gpu_used else 0.0)

    def on_workflow_end(
        self,
        ctx: ExecutionContext,
        workflow_metrics: WorkflowMetrics,
    ) -> None:
        self.workflow_duration.observe(workflow_metrics.total_wall_time_s)
        self.memory_peak.set(workflow_metrics.peak_rss_bytes)

    def on_error(
        self,
        ctx: ExecutionContext,
        error: Exception,
        step_index: int | None = None,
    ) -> None:
        self.errors_total.labels(
            error_type=type(error).__name__,
        ).inc()


__all__ = ["PrometheusHook"]
