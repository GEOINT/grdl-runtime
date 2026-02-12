# -*- coding: utf-8 -*-
"""
Instrumentation subpackage — execution hooks for metrics and tracing.

Provides the ``ExecutionHook`` base class and concrete implementations
for Prometheus metrics (``PrometheusHook``) and OpenTelemetry tracing
(``OtelHook``).  Hooks are opt-in and never crash the workflow — all
callbacks are wrapped in try/except by the executor.

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

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from grdl_rt.execution.context import ExecutionContext
    from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics


class ExecutionHook:
    """Base class for execution instrumentation hooks.

    Subclasses override the ``on_*`` methods they care about.
    All methods are no-ops by default so that partial implementations
    are safe.  The executor calls each method inside a try/except to
    guarantee that instrumentation failures never crash a workflow.
    """

    def on_workflow_start(
        self,
        ctx: ExecutionContext,
        workflow_name: str,
        workflow_version: str,
    ) -> None:
        """Called once at the start of a workflow execution."""

    def on_step_start(
        self,
        ctx: ExecutionContext,
        step_index: int,
        processor_name: str,
    ) -> None:
        """Called before a processing step begins."""

    def on_step_end(
        self,
        ctx: ExecutionContext,
        step_index: int,
        step_metrics: StepMetrics,
    ) -> None:
        """Called after a processing step completes (success or failure)."""

    def on_global_pass_start(
        self,
        ctx: ExecutionContext,
        step_index: int,
        processor_name: str,
    ) -> None:
        """Called before a global pass begins for a processor."""

    def on_global_pass_end(
        self,
        ctx: ExecutionContext,
        step_index: int,
        duration_s: float,
        peak_memory_bytes: int,
    ) -> None:
        """Called after a global pass completes."""

    def on_workflow_end(
        self,
        ctx: ExecutionContext,
        workflow_metrics: WorkflowMetrics,
    ) -> None:
        """Called after the workflow completes (success or failure)."""

    def on_error(
        self,
        ctx: ExecutionContext,
        error: Exception,
        step_index: Optional[int] = None,
    ) -> None:
        """Called when an error occurs during execution."""


__all__ = ["ExecutionHook"]
