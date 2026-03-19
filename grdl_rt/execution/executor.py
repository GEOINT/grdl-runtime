"""
Workflow Executor - Headless and interactive workflow execution.

Executes a compiled WorkflowDefinition by instantiating GRDL processors
and running them in sequence over input imagery. Supports both single-image
and batch execution with optional GPU acceleration.  Returns structured
``WorkflowResult`` objects containing both the processed array and timing
metrics.

Integrates resilience primitives: per-step retry with exponential backoff,
step timeouts via ``concurrent.futures``, memory-aware pre-flight checks,
and graceful shutdown on SIGTERM/SIGINT.

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
2026-02-06

Modified
--------
2026-02-11
"""

# Standard library
import contextlib
import json
import re
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Third-party
import numpy as np

from grdl_rt.execution.band_adaptation import BandExpansion, BandReduction, adapt_bands

# grdl-runtime internal
from grdl_rt.execution.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    CheckpointState,
    compute_workflow_hash,
)
from grdl_rt.execution.config import get_runtime_config
from grdl_rt.execution.context import ExecutionContext, get_logger
from grdl_rt.execution.discovery import resolve_processor_class
from grdl_rt.execution.errors import (
    StepRetryExhaustedError,
    StepTimeoutError,
)
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.hardware import LocalHardwareContext
from grdl_rt.execution.history import ExecutionHistoryDB
from grdl_rt.execution.instrumentation import ExecutionHook
from grdl_rt.execution.lineage import build_lineage, compute_array_hash
from grdl_rt.execution.memory_sampler import MemorySampler
from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics
from grdl_rt.execution.quota import QuotaEnforcer, ResourceQuota
from grdl_rt.execution.resilience import (
    CircuitBreaker,
    RetryPolicy,
    ShutdownCoordinator,
    execute_with_retry,
    execute_with_timeout,
    run_memory_preflight,
)
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.workflow import ProcessingStep, TapOutStepDef, WorkflowDefinition

logger = get_logger(__name__)

# GRDL exceptions (optional — graceful fallback if GRDL is old)
try:
    from grdl.exceptions import GrdlError
except ImportError:
    GrdlError = None  # type: ignore[misc,assignment]


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class WorkflowExecutor:
    """Execute a compiled workflow headlessly or interactively.

    Instantiates each processor in the workflow and runs them in
    sequence on input imagery.  Returns ``WorkflowResult`` objects
    containing both the result array and execution metrics.

    Supports per-step retry with exponential backoff, step timeouts,
    memory pre-flight checks, circuit breaker, and graceful shutdown
    on SIGTERM/SIGINT.

    Parameters
    ----------
    workflow : WorkflowDefinition
        Compiled workflow to execute.
    gpu : Optional[GpuBackend]
        GPU backend for acceleration. If None, CPU only.
    shutdown : Optional[ShutdownCoordinator]
        Shutdown coordinator for graceful signal handling.
        If None, one is created and registered automatically.
    circuit_breaker : Optional[CircuitBreaker]
        Circuit breaker for processor failure tracking.
        If None, circuit breaking is disabled.
    checkpoint_manager : Optional[CheckpointManager]
        Checkpoint manager for per-step checkpointing and resume.
        Required when ``enable_checkpointing=True`` is passed to
        ``execute()``.
    history_db : Optional[ExecutionHistoryDB]
        Execution history journal.  If provided, each execution is
        recorded at start and completion.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        gpu: GpuBackend | None = None,
        *,
        shutdown: ShutdownCoordinator | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        history_db: ExecutionHistoryDB | None = None,
        resource_quota: ResourceQuota | None = None,
        hooks: list[ExecutionHook] | None = None,
    ) -> None:
        self._workflow = workflow
        self._gpu = gpu or GpuBackend(prefer_gpu=False)
        self._shutdown = shutdown or ShutdownCoordinator()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._checkpoint_mgr = checkpoint_manager
        self._history_db = history_db
        self._resource_quota = resource_quota
        self._hooks: list[ExecutionHook] = list(hooks or [])

    def execute(
        self,
        source: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
        *,
        auto_tap_out: bool = False,
        tap_out_format: str = "npy",
        tap_out_dir: str | Path | None = None,
        run_folder: str | Path | None = None,
        enable_memory_check: bool = True,
        enable_shutdown_handler: bool = True,
        enable_checkpointing: bool = False,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Run the full pipeline on a single image.

        Parameters
        ----------
        source : np.ndarray
            Input image array.
        progress_callback : Optional[Callable[[float], None]]
            Called with progress fraction in [0.0, 1.0] as each step
            completes.
        auto_tap_out : bool
            If ``True``, write all intermediates to a timestamped
            directory alongside workflow params and a manifest.
        tap_out_format : str
            Format for auto tap-out files (default ``'npy'``).
        tap_out_dir : str or Path, optional
            Override the auto tap-out output directory.
        enable_memory_check : bool
            If ``True`` (default), run memory pre-flight check.
        enable_shutdown_handler : bool
            If ``True`` (default), register signal handlers for
            graceful shutdown.
        enable_checkpointing : bool
            If ``True``, write a checkpoint after every processing step.
            Requires a ``checkpoint_manager`` on the executor.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        WorkflowResult
            Result array and execution metrics.

        Raises
        ------
        MemoryThresholdError
            If estimated memory exceeds the abort threshold.
        StepRetryExhaustedError
            If a step fails after exhausting all retries.
        StepTimeoutError
            If a step exceeds its configured timeout.
        """
        run_id = str(uuid.uuid4())
        ctx = ExecutionContext(
            workflow_id=f"{self._workflow.name}:{self._workflow.version}",
            workflow_name=self._workflow.name,
            run_id=run_id,
        )
        logger.bind(**ctx.as_log_dict())
        get_runtime_config()

        # Register shutdown handler
        if enable_shutdown_handler:
            self._shutdown.register()

        try:
            if auto_tap_out:
                return self._execute_with_auto_tap_out(
                    source,
                    progress_callback=progress_callback,
                    tap_out_format=tap_out_format,
                    tap_out_dir=Path(tap_out_dir) if tap_out_dir else None,
                    ctx=ctx,
                    enable_memory_check=enable_memory_check,
                    **kwargs,
                )

            return self._execute_main(
                source,
                progress_callback=progress_callback,
                ctx=ctx,
                run_folder=Path(run_folder) if run_folder else None,
                enable_memory_check=enable_memory_check,
                enable_checkpointing=enable_checkpointing,
                **kwargs,
            )
        finally:
            if enable_shutdown_handler:
                self._shutdown.unregister()

    def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call a hook method on all registered hooks, swallowing errors."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                getattr(hook, method)(*args, **kwargs)

    def _execute_main(
        self,
        source: np.ndarray,
        *,
        progress_callback: Callable[[float], None] | None,
        ctx: ExecutionContext,
        run_folder: Path | None = None,
        enable_memory_check: bool,
        enable_checkpointing: bool = False,
        resume_state: CheckpointState | None = None,
        prior_metrics: list[StepMetrics] | None = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Core execution loop with resilience, checkpointing, and history.

        Parameters
        ----------
        resume_state : CheckpointState, optional
            If resuming, the loaded checkpoint state.  Steps up to
            ``resume_state.step_index`` are skipped.
        prior_metrics : list of StepMetrics, optional
            Metrics from the prior (interrupted) execution segment,
            merged into the final WorkflowMetrics.
        """
        log = logger.bind(**ctx.as_log_dict())
        cfg = get_runtime_config()
        started_at = _iso_now()
        wf_dict = self._workflow.to_dict()
        wf_hash = compute_workflow_hash(wf_dict)

        # Determine which steps to skip on resume
        skip_up_to = resume_state.step_index if resume_state else -1

        # Count processing steps for memory estimation
        processing_steps = [s for s in self._workflow.steps if isinstance(s, ProcessingStep)]
        n_steps = len(self._workflow.steps)

        # Memory pre-flight
        if enable_memory_check and processing_steps:
            run_memory_preflight(
                source,
                n_steps=len(processing_steps),
                multiplier=cfg.memory.estimation_multiplier,
                warn_threshold=cfg.memory.warn_threshold,
                abort_threshold=cfg.memory.abort_threshold,
                log=log,
            )

        # Resource quota pre-flight and monitoring
        quota_enforcer: QuotaEnforcer | None = None
        if self._resource_quota is not None:
            quota_enforcer = QuotaEnforcer(self._resource_quota)
            quota_enforcer.check_before_execution(
                source,
                len(processing_steps),
                multiplier=cfg.memory.estimation_multiplier,
            )
            quota_enforcer.mark_start()
            quota_enforcer.start_monitoring()

        # Notify hooks of workflow start
        self._call_hooks(
            "on_workflow_start",
            ctx,
            self._workflow.name,
            self._workflow.version,
        )

        # Run global pass before main execution (only on fresh runs)
        global_pass_processors: dict[int, Any] = {}
        if not resume_state:
            global_pass_processors = self._run_global_pass(
                source,
                ctx=ctx,
                log=log,
            )

        current = source
        step_metrics_list: list[StepMetrics] = list(prior_metrics or [])
        intermediate_files: list[str] = []
        if resume_state:
            intermediate_files = list(resume_state.intermediate_files)

        # Record start in history
        if self._history_db is not None:
            try:
                params_json = json.dumps(wf_dict, default=str)
                self._history_db.record_start(
                    workflow_id=ctx.workflow_id,
                    run_id=ctx.run_id,
                    workflow_hash=wf_hash,
                    parameters_json=params_json,
                )
            except Exception as e:
                log.warning("history_record_start_failed", error=str(e))

        # Write as_planned.json before execution begins
        if run_folder is not None:
            try:
                run_folder.mkdir(parents=True, exist_ok=True)
                planned = {
                    "workflow_name": self._workflow.name,
                    "workflow_version": self._workflow.version,
                    "workflow_hash": wf_hash,
                    "steps": [
                        {
                            "index": idx,
                            "processor_name": getattr(s, "processor_name", type(s).__name__),
                            "params": dict(getattr(s, "params", {})),
                        }
                        for idx, s in enumerate(self._workflow.steps)
                        if isinstance(s, ProcessingStep)
                    ],
                }
                (run_folder / "as_planned.json").write_text(
                    json.dumps(planned, indent=2, default=str),
                    encoding="utf-8",
                )
                log.info("as_planned_written", path=str(run_folder / "as_planned.json"))
            except Exception as e:
                log.warning("as_planned_write_failed", error=str(e))

        # Capture hardware snapshot once before execution
        try:
            _hardware_dict: dict[str, Any] | None = LocalHardwareContext().to_dict()
        except Exception:
            _hardware_dict = None

        # Extract step dependency graph for DAG workflows
        _dep_map: dict[str, list[str]] = {
            str(step.id): list(step.depends_on)
            for step in self._workflow.steps
            if step.id is not None and step.depends_on
        }
        _step_depends_on: dict[str, list[str]] | None = _dep_map or None

        _sampler = MemorySampler()
        _sampler.start()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        last_completed_index = skip_up_to

        try:
            for i, step in enumerate(self._workflow.steps):
                # Skip already-completed steps on resume
                if i <= skip_up_to:
                    log.debug("step_skipped_resume", step_index=i)
                    if progress_callback is not None and n_steps > 0:
                        progress_callback((i + 1) / n_steps)
                    continue

                # Check shutdown
                if self._shutdown.shutdown_requested:
                    log.info("shutdown_during_execution", step_index=i)
                    self._do_shutdown(
                        ctx,
                        last_completed_index,
                        current,
                        step_metrics_list,
                        intermediate_files=intermediate_files,
                        wf_dict=wf_dict,
                        wf_hash=wf_hash,
                    )

                # Quota enforcement between steps
                if quota_enforcer is not None:
                    quota_enforcer.check_wall_clock()
                    quota_enforcer.check_memory_violation()

                if isinstance(step, TapOutStepDef):
                    log.debug("tap_out", step_index=i, path=step.path)
                    try:
                        from grdl.IO import write as io_write

                        io_write(current, step.path, format=step.format)
                    except Exception as e:
                        log.warning(
                            "tap_out_failed",
                            step_index=i,
                            path=step.path,
                            error=str(e),
                        )
                else:
                    step_t0_wall = time.perf_counter()
                    step_t0_cpu = time.process_time()

                    # Build a rescaled callback for this step's internal progress
                    step_kwargs = dict(kwargs)
                    if progress_callback is not None and n_steps > 0:
                        base = i / n_steps
                        scale = 1.0 / n_steps
                        step_kwargs["progress_callback"] = (
                            lambda f, _b=base, _s=scale: progress_callback(_b + f * _s)
                        )

                    log.debug(
                        "step_start",
                        step_index=i,
                        processor_name=step.processor_name,
                    )

                    # Notify hooks
                    self._call_hooks(
                        "on_step_start",
                        ctx,
                        i,
                        step.processor_name,
                    )

                    # Check circuit breaker
                    if self._circuit_breaker.is_open(step.processor_name):
                        raise RuntimeError(
                            f"Circuit breaker open for processor " f"'{step.processor_name}'"
                        )

                    # Use pre-instantiated processor if global pass ran
                    gp_info = global_pass_processors.get(i)
                    pre_inst = gp_info[0] if gp_info else None

                    _in_shape = getattr(current, 'shape', None)
                    _in_dtype = str(current.dtype) if hasattr(current, 'dtype') else None

                    current = self._execute_step_resilient(
                        step,
                        current,
                        log=log,
                        _pre_instantiated=pre_inst,
                        **step_kwargs,
                    )

                    step_t_end = time.perf_counter()
                    step_wall = step_t_end - step_t0_wall
                    step_cpu = time.process_time() - step_t0_cpu

                    self._circuit_breaker.record_success(step.processor_name)

                    sm = StepMetrics(
                        step_index=i,
                        processor_name=step.processor_name,
                        wall_time_s=step_wall,
                        cpu_time_s=step_cpu,
                        peak_rss_bytes=0,
                        gpu_used=self._gpu.last_gpu_used,
                        gpu_memory_bytes=self._gpu.last_gpu_memory_bytes,
                        global_pass_duration=(gp_info[1] if gp_info else None),
                        global_pass_memory=(gp_info[2] if gp_info else None),
                        input_shape=_in_shape,
                        input_dtype=_in_dtype,
                        wall_start=step_t0_wall,
                        wall_end=step_t_end,
                    )
                    step_metrics_list.append(sm)
                    last_completed_index = i
                    log.debug(
                        "step_complete",
                        step_index=i,
                        processor_name=step.processor_name,
                        wall_time_s=round(step_wall, 4),
                    )

                    # Notify hooks of step completion
                    self._call_hooks("on_step_end", ctx, i, sm)

                    # Per-step checkpoint
                    if enable_checkpointing and self._checkpoint_mgr is not None:
                        try:
                            ipath = self._checkpoint_mgr.write_step_intermediate(
                                ctx.run_id,
                                i,
                                current,
                            )
                            intermediate_files.append(ipath)
                            ckpt_state = CheckpointState(
                                schema_version=CHECKPOINT_SCHEMA_VERSION,
                                workflow_definition_hash=wf_hash,
                                step_index=i,
                                intermediate_files=list(intermediate_files),
                                metrics_so_far=[m.to_dict() for m in step_metrics_list],
                                execution_context=ctx.as_log_dict(),
                                timestamp=_iso_now(),
                                workflow_dict=wf_dict,
                            )
                            self._checkpoint_mgr.write_checkpoint(
                                ckpt_state,
                                ctx.run_id,
                            )
                        except Exception as e:
                            log.warning(
                                "checkpoint_write_failed",
                                step_index=i,
                                error=str(e),
                            )

                if progress_callback is not None and n_steps > 0:
                    progress_callback((i + 1) / n_steps)

            # Check shutdown after last step
            if self._shutdown.shutdown_requested:
                self._do_shutdown(
                    ctx,
                    last_completed_index,
                    current,
                    step_metrics_list,
                    intermediate_files=intermediate_files,
                    wf_dict=wf_dict,
                    wf_hash=wf_hash,
                )

            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _timeline = _sampler.stop()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=_timeline.peak(),
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="success",
                hardware=_hardware_dict,
                step_depends_on=_step_depends_on,
                memory_timeline=_timeline,
            )
            # Notify hooks of successful completion
            self._call_hooks("on_workflow_end", ctx, wf_metrics)

            # Build data lineage
            lineage = None
            try:
                lineage = build_lineage(
                    source,
                    current,
                    self._workflow.steps,
                    step_metrics_list,
                )
            except Exception as e:
                log.warning("lineage_build_failed", error=str(e))

            log.info(
                "workflow_complete",
                status="success",
                total_wall_time_s=round(total_wall, 4),
                step_count=len(step_metrics_list),
            )

            # Compute output hash for history
            output_hash_val = None
            if lineage is not None:
                output_hash_val = lineage.output_hash
            else:
                with contextlib.suppress(Exception):
                    output_hash_val = compute_array_hash(current)

            # Record completion in history
            if self._history_db is not None:
                try:
                    ckpt_path = None
                    if self._checkpoint_mgr is not None and intermediate_files:
                        ckpt_path = str(self._checkpoint_mgr.run_dir(ctx.run_id))
                    self._history_db.record_completion(
                        run_id=ctx.run_id,
                        status="success",
                        step_count=len(step_metrics_list),
                        metrics_json=wf_metrics.to_json(),
                        output_hash=output_hash_val,
                        checkpoint_path=ckpt_path,
                    )
                except Exception as e:
                    log.warning("history_record_completion_failed", error=str(e))

            # Write as_executed.json after successful execution
            if run_folder is not None:
                try:
                    self._write_as_executed_linear(
                        run_folder,
                        ctx,
                        "success",
                        step_metrics_list,
                        wf_metrics,
                        lineage_dict=lineage.to_dict() if lineage else None,
                        log=log,
                    )
                except Exception as e:
                    log.warning("as_executed_write_failed", error=str(e))

            return WorkflowResult(
                result=current,
                metrics=wf_metrics,
                lineage=lineage,
            )

        except (StepRetryExhaustedError, StepTimeoutError) as exc:
            if isinstance(exc, StepRetryExhaustedError):
                self._circuit_breaker.record_failure(exc.step_name)
            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _err_timeline = _sampler.stop()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=_err_timeline.peak(),
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="failed",
                error_message=str(exc),
                hardware=_hardware_dict,
                memory_timeline=_err_timeline,
            )
            exc.__workflow_metrics__ = wf_metrics  # type: ignore[union-attr, attr-defined]
            self._call_hooks("on_error", ctx, exc)
            self._call_hooks("on_workflow_end", ctx, wf_metrics)
            self._record_failure(ctx, step_metrics_list, wf_metrics, log)
            if run_folder is not None:
                with contextlib.suppress(Exception):
                    self._write_as_executed_linear(
                        run_folder,
                        ctx,
                        "failed",
                        step_metrics_list,
                        wf_metrics,
                        error_message=str(exc),
                        log=log,
                    )
            raise

        except Exception as exc:
            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _err_timeline = _sampler.stop()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=_err_timeline.peak(),
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="failed",
                error_message=str(exc),
                hardware=_hardware_dict,
                memory_timeline=_err_timeline,
            )
            exc.__workflow_metrics__ = wf_metrics  # type: ignore[union-attr, attr-defined]
            self._call_hooks("on_error", ctx, exc)
            self._call_hooks("on_workflow_end", ctx, wf_metrics)
            self._record_failure(ctx, step_metrics_list, wf_metrics, log)
            if run_folder is not None:
                with contextlib.suppress(Exception):
                    self._write_as_executed_linear(
                        run_folder,
                        ctx,
                        "failed",
                        step_metrics_list,
                        wf_metrics,
                        error_message=str(exc),
                        log=log,
                    )
            raise
        finally:
            if quota_enforcer is not None:
                quota_enforcer.stop_monitoring()

    def _record_failure(
        self,
        ctx: ExecutionContext,
        step_metrics_list: list[StepMetrics],
        wf_metrics: WorkflowMetrics,
        log: Any,
    ) -> None:
        """Record a failed execution in the history DB."""
        if self._history_db is not None:
            try:
                self._history_db.record_completion(
                    run_id=ctx.run_id,
                    status="failed",
                    step_count=len(step_metrics_list),
                    error_message=wf_metrics.error_message,
                    metrics_json=wf_metrics.to_json(),
                )
            except Exception as e:
                log.warning("history_record_failure_failed", error=str(e))

    def _write_as_executed_linear(
        self,
        run_folder: Path,
        ctx: ExecutionContext,
        status: str,
        step_metrics_list: list[StepMetrics],
        wf_metrics: WorkflowMetrics,
        *,
        lineage_dict: dict[str, Any] | None = None,
        error_message: str | None = None,
        log: Any = None,
    ) -> None:
        """Write as_executed.json for linear workflow execution."""
        log = log or logger
        run_folder.mkdir(parents=True, exist_ok=True)
        executed = {
            "workflow_name": self._workflow.name,
            "workflow_version": self._workflow.version,
            "run_id": ctx.run_id,
            "started_at": wf_metrics.started_at,
            "completed_at": wf_metrics.completed_at,
            "status": status,
            "steps": [sm.to_dict() for sm in step_metrics_list],
            "total_wall_time_s": wf_metrics.total_wall_time_s,
            "total_cpu_time_s": wf_metrics.total_cpu_time_s,
            "peak_rss_bytes": wf_metrics.peak_rss_bytes,
        }
        if lineage_dict is not None:
            executed["data_lineage"] = lineage_dict
        if error_message is not None:
            executed["error_message"] = error_message

        (run_folder / "as_executed.json").write_text(
            json.dumps(executed, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("as_executed_written", path=str(run_folder / "as_executed.json"))

    def _execute_with_auto_tap_out(
        self,
        source: np.ndarray,
        *,
        progress_callback: Callable[[float], None] | None,
        tap_out_format: str,
        tap_out_dir: Path | None,
        ctx: ExecutionContext,
        enable_memory_check: bool,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute pipeline writing all intermediates to a directory."""
        log = logger.bind(**ctx.as_log_dict())
        cfg = get_runtime_config()
        started_at = _iso_now()

        # Count processing steps for memory estimation
        processing_steps = [s for s in self._workflow.steps if isinstance(s, ProcessingStep)]
        n_steps = len(self._workflow.steps)

        # Memory pre-flight
        if enable_memory_check and processing_steps:
            run_memory_preflight(
                source,
                n_steps=len(processing_steps),
                multiplier=cfg.memory.estimation_multiplier,
                warn_threshold=cfg.memory.warn_threshold,
                abort_threshold=cfg.memory.abort_threshold,
                log=log,
            )

        try:
            _hardware_dict: dict[str, Any] | None = LocalHardwareContext().to_dict()
        except Exception:
            _hardware_dict = None

        # Determine output directory
        if tap_out_dir is not None:
            run_dir = tap_out_dir
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            run_dir = Path.cwd() / f"grdl_run_{timestamp}"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Run global pass before main execution
        global_pass_processors = self._run_global_pass(source, ctx=ctx, log=log)

        ext = tap_out_format.lstrip(".")
        current = source
        manifest_entries: list[dict[str, Any]] = []
        step_metrics_list: list[StepMetrics] = []
        step_counter = 0
        last_completed_index = -1

        _sampler = MemorySampler()
        _sampler.start()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        try:
            for i, step_def in enumerate(self._workflow.steps):
                # Check shutdown
                if self._shutdown.shutdown_requested:
                    log.info("shutdown_during_execution", step_index=i)
                    self._do_shutdown(
                        ctx,
                        last_completed_index,
                        current,
                        step_metrics_list,
                    )

                if isinstance(step_def, TapOutStepDef):
                    t0 = time.perf_counter()
                    log.debug("tap_out", step_index=i, path=step_def.path)
                    try:
                        from grdl.IO import write as io_write

                        io_write(current, step_def.path, format=step_def.format)
                    except Exception as e:
                        log.warning(
                            "tap_out_failed",
                            step_index=i,
                            path=step_def.path,
                            error=str(e),
                        )
                    elapsed = time.perf_counter() - t0
                    manifest_entries.append(
                        {
                            "index": i,
                            "type": "tap_out",
                            "name": f"tap_out({step_def.path})",
                            "path": step_def.path,
                            "elapsed_s": round(elapsed, 4),
                            "shape": list(current.shape),
                            "dtype": str(current.dtype),
                        }
                    )
                else:
                    step_t0_wall = time.perf_counter()
                    step_t0_cpu = time.process_time()

                    step_kwargs = dict(kwargs)
                    if progress_callback is not None and n_steps > 0:
                        base = i / n_steps
                        scale = 1.0 / n_steps
                        step_kwargs["progress_callback"] = (
                            lambda f, _b=base, _s=scale: progress_callback(_b + f * _s)
                        )

                    # Check circuit breaker
                    if self._circuit_breaker.is_open(step_def.processor_name):
                        raise RuntimeError(
                            f"Circuit breaker open for processor " f"'{step_def.processor_name}'"
                        )

                    # Use pre-instantiated processor if global pass ran
                    gp_info = global_pass_processors.get(i)
                    pre_inst = gp_info[0] if gp_info else None

                    _in_shape = getattr(current, 'shape', None)
                    _in_dtype = str(current.dtype) if hasattr(current, 'dtype') else None

                    current = self._execute_step_resilient(
                        step_def,
                        current,
                        log=log,
                        _pre_instantiated=pre_inst,
                        **step_kwargs,
                    )

                    step_t_end = time.perf_counter()
                    step_wall = step_t_end - step_t0_wall
                    step_cpu = time.process_time() - step_t0_cpu
                    step_counter += 1

                    self._circuit_breaker.record_success(step_def.processor_name)

                    sm = StepMetrics(
                        step_index=i,
                        processor_name=step_def.processor_name,
                        wall_time_s=step_wall,
                        cpu_time_s=step_cpu,
                        peak_rss_bytes=0,
                        gpu_used=self._gpu.last_gpu_used,
                        gpu_memory_bytes=self._gpu.last_gpu_memory_bytes,
                        global_pass_duration=(gp_info[1] if gp_info else None),
                        global_pass_memory=(gp_info[2] if gp_info else None),
                        input_shape=_in_shape,
                        input_dtype=_in_dtype,
                        wall_start=step_t0_wall,
                        wall_end=step_t_end,
                    )
                    step_metrics_list.append(sm)
                    last_completed_index = i

                    # Write intermediate
                    safe_name = re.sub(r"[^\w\-]", "_", step_def.processor_name)
                    safe_name = re.sub(r"_+", "_", safe_name).strip("_") or "step"
                    filename = f"step_{step_counter:03d}_{safe_name}.{ext}"
                    out_path = run_dir / filename
                    try:
                        from grdl.IO import write as io_write

                        io_write(current, out_path)
                    except Exception as e:
                        log.warning(
                            "auto_tap_out_write_failed",
                            path=str(out_path),
                            error=str(e),
                        )

                    manifest_entries.append(
                        {
                            "index": i,
                            "type": "processing",
                            "name": step_def.processor_name,
                            "file": filename,
                            "elapsed_s": round(step_wall, 4),
                            "shape": list(current.shape),
                            "dtype": str(current.dtype),
                        }
                    )

                if progress_callback is not None and n_steps > 0:
                    progress_callback((i + 1) / n_steps)

            # Write final output
            final_filename = f"final_output.{ext}"
            try:
                from grdl.IO import write as io_write

                io_write(current, run_dir / final_filename)
            except Exception as e:
                log.warning("auto_tap_out_final_write_failed", error=str(e))

            # Write workflow_params.yaml
            try:
                import yaml

                params = self._workflow.to_dict()
                yaml_path = run_dir / "workflow_params.yaml"
                yaml_path.write_text(
                    yaml.dump(params, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )
            except Exception as e:
                log.warning("auto_tap_out_yaml_write_failed", error=str(e))

            # Write manifest.json
            try:
                manifest = {
                    "workflow_name": self._workflow.name,
                    "tap_out_format": tap_out_format,
                    "final_output": final_filename,
                    "steps": manifest_entries,
                }
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                log.warning("auto_tap_out_manifest_write_failed", error=str(e))

            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _timeline = _sampler.stop()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=_timeline.peak(),
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="success",
                hardware=_hardware_dict,
                memory_timeline=_timeline,
            )
            log.info("auto_tap_out_complete", run_dir=str(run_dir))
            return WorkflowResult(result=current, metrics=wf_metrics)

        except Exception as exc:
            if isinstance(exc, StepRetryExhaustedError):
                self._circuit_breaker.record_failure(exc.step_name)
            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _err_timeline = _sampler.stop()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=_err_timeline.peak(),
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="failed",
                error_message=str(exc),
                hardware=_hardware_dict,
                memory_timeline=_err_timeline,
            )
            exc.__workflow_metrics__ = wf_metrics  # type: ignore[union-attr, attr-defined]
            raise

    def execute_batch(
        self,
        sources: list[np.ndarray],
        **kwargs: Any,
    ) -> list[WorkflowResult]:
        """Run the pipeline on multiple images.

        Parameters
        ----------
        sources : List[np.ndarray]
            List of input image arrays.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        List[WorkflowResult]
            List of results with metrics.
        """
        return [self.execute(src, **kwargs) for src in sources]

    def execute_step(
        self,
        step_index: int,
        source: np.ndarray,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute a single step from the workflow.

        Useful for interactive preview of individual steps.

        Parameters
        ----------
        step_index : int
            Index of the step to execute.
        source : np.ndarray
            Input image array.
        **kwargs
            Additional arguments.

        Returns
        -------
        WorkflowResult
            Result of this step with metrics.
        """
        run_id = str(uuid.uuid4())
        ctx = ExecutionContext(
            workflow_id=f"{self._workflow.name}:{self._workflow.version}",
            workflow_name=self._workflow.name,
            run_id=run_id,
        )
        started_at = _iso_now()
        step = self._workflow.steps[step_index]

        _sampler = MemorySampler()
        _sampler.start()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        try:
            if isinstance(step, ProcessingStep):
                result = self._execute_step_resilient(step, source, **kwargs)
            else:
                result = self._execute_step(step, source, **kwargs)  # type: ignore[arg-type]

            step_t_end = time.perf_counter()
            step_wall = step_t_end - t0_wall
            step_cpu = time.process_time() - t0_cpu
            _timeline = _sampler.stop()

            processor_name = getattr(step, "processor_name", "unknown")
            sm = StepMetrics(
                step_index=step_index,
                processor_name=processor_name,
                wall_time_s=step_wall,
                cpu_time_s=step_cpu,
                peak_rss_bytes=_timeline.peak(),
                gpu_used=self._gpu.last_gpu_used,
                gpu_memory_bytes=self._gpu.last_gpu_memory_bytes,
                input_shape=getattr(source, 'shape', None),
                input_dtype=str(source.dtype) if hasattr(source, 'dtype') else None,
                wall_start=t0_wall,
                wall_end=step_t_end,
            )

            try:
                _hw_single_dict: dict[str, Any] | None = LocalHardwareContext().to_dict()
            except Exception:
                _hw_single_dict = None

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=step_wall,
                total_cpu_time_s=step_cpu,
                peak_rss_bytes=_timeline.peak(),
                step_metrics=[sm],
                started_at=started_at,
                completed_at=_iso_now(),
                status="success",
                hardware=_hw_single_dict,
                memory_timeline=_timeline,
            )
            return WorkflowResult(result=result, metrics=wf_metrics)
        except Exception:
            _sampler.stop()
            raise

    # ------------------------------------------------------------------
    # Global pass execution
    # ------------------------------------------------------------------

    def _run_global_pass(
        self,
        source: np.ndarray,
        *,
        ctx: ExecutionContext | None = None,
        log: Any = None,
    ) -> dict[int, Any]:
        """Run the global pass for processors that require it.

        Scans workflow steps for processors with ``__has_global_pass__``.
        For each, instantiates the processor, creates a read-only view
        of the source buffer, and streams it through the global callbacks.
        Returns a mapping of step index to pre-instantiated processor
        so the main execution loop can reuse them with accumulated state.

        Parameters
        ----------
        source : np.ndarray
            Full input image array.
        ctx : ExecutionContext, optional
            Execution context for hook calls.
        log : Any
            Structured logger instance.

        Returns
        -------
        Dict[int, Any]
            Mapping of step index to (processor_instance, global_pass_duration,
            global_pass_memory) for steps that required a global pass.
        """
        log = log or logger
        pre_instantiated: dict[int, Any] = {}

        # Identify steps requiring a global pass
        global_steps: list[tuple] = []
        for i, step in enumerate(self._workflow.steps):
            if not isinstance(step, ProcessingStep):
                continue
            try:
                processor_cls = resolve_processor_class(step.processor_name)
            except ImportError:
                continue
            if getattr(processor_cls, "__has_global_pass__", False):
                global_steps.append((i, step, processor_cls))

        if not global_steps:
            return pre_instantiated

        # Create read-only view of source (no copy — just flips flag)
        readonly_source = source.view()
        readonly_source.flags.writeable = False

        # Use a sampler per global-pass step to measure peak memory
        for i, step, processor_cls in global_steps:
            log.debug(
                "global_pass_start",
                step_index=i,
                processor_name=step.processor_name,
            )

            # Notify hooks
            if ctx is not None:
                self._call_hooks(
                    "on_global_pass_start",
                    ctx,
                    i,
                    step.processor_name,
                )

            gp_sampler = MemorySampler()
            gp_sampler.start()
            gp_t0 = time.perf_counter()

            try:
                processor = processor_cls()
            except Exception as e:
                gp_sampler.stop()
                log.warning(
                    "global_pass_instantiation_failed",
                    step_index=i,
                    processor_name=step.processor_name,
                    error=str(e),
                )
                continue

            # Run each global callback on the read-only buffer
            callbacks = getattr(processor_cls, "__global_callbacks__", ())
            for cb_name in callbacks:
                cb = getattr(processor, cb_name, None)
                if cb is None:
                    continue
                try:
                    cb(readonly_source)
                except ValueError as exc:
                    # Mutation attempt on read-only buffer
                    log.warning(
                        "global_pass_mutation_blocked",
                        processor_name=step.processor_name,
                        method=cb_name,
                        error=str(exc),
                    )
                except Exception as exc:
                    log.error(
                        "global_pass_callback_failed",
                        processor_name=step.processor_name,
                        method=cb_name,
                        error=str(exc),
                    )
                    raise

            gp_duration = time.perf_counter() - gp_t0
            gp_timeline = gp_sampler.stop()
            gp_peak = gp_timeline.peak()

            pre_instantiated[i] = (processor, gp_duration, gp_peak)

            # Notify hooks
            if ctx is not None:
                self._call_hooks(
                    "on_global_pass_end",
                    ctx,
                    i,
                    gp_duration,
                    gp_peak,
                )

            log.debug(
                "global_pass_complete",
                step_index=i,
                processor_name=step.processor_name,
                duration_s=round(gp_duration, 4),
            )

        return pre_instantiated

    # ------------------------------------------------------------------
    # Internal step execution
    # ------------------------------------------------------------------

    def _execute_step_resilient(
        self,
        step: ProcessingStep,
        source: np.ndarray,
        *,
        log: Any = None,
        _pre_instantiated: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a step with retry and timeout logic.

        Uses the step's own retry/timeout config if set, otherwise
        falls back to the runtime config defaults.
        """
        log = log or logger
        cfg = get_runtime_config()

        # Determine retry policy
        retry = step.retry
        if retry is None:
            retry = RetryPolicy(
                max_retries=cfg.retry.max_retries,
                backoff_base=cfg.retry.backoff_base,
                backoff_max=cfg.retry.backoff_max,
                retryable_exceptions=tuple(cfg.retry.retryable_exceptions),
            )

        timeout = step.timeout_seconds

        def _do_step() -> np.ndarray:
            def raw_fn():
                return self._execute_step(
                    step,
                    source,
                    _pre_instantiated=_pre_instantiated,
                    **kwargs,
                )

            if timeout is not None:
                return execute_with_timeout(
                    raw_fn,
                    timeout,
                    step.processor_name,
                )
            return raw_fn()

        if retry.max_retries > 0:
            return execute_with_retry(
                _do_step,
                retry,
                step.processor_name,
                log=log,
            )

        # No retry — run directly (timeout still applies)
        return _do_step()

    def _execute_step(
        self,
        step: ProcessingStep,
        source: np.ndarray,
        *,
        _pre_instantiated: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a single processing step (no resilience wrapping).

        Uses ``GpuBackend.apply_transform()`` which dispatches via the
        GRDL ``execute()`` protocol, with fallback to ``apply()`` for
        legacy processors.

        Parameters
        ----------
        step : ProcessingStep
        source : np.ndarray
        _pre_instantiated : Any, optional
            If provided, reuse this processor instance (which already
            has accumulated state from the global pass) instead of
            creating a new one.
        **kwargs

        Returns
        -------
        np.ndarray
        """
        log = logger.bind(processor_name=step.processor_name)

        if _pre_instantiated is not None:
            processor = _pre_instantiated
            log.debug("step_using_pre_instantiated")
        else:
            log.debug("step_resolving")
            try:
                processor_cls = resolve_processor_class(step.processor_name)
            except ImportError as e:
                raise ImportError(
                    f"Failed to resolve processor '{step.processor_name}': {e}"
                ) from e

            try:
                processor = processor_cls()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate processor '{step.processor_name}': {e}"
                ) from e

        # Band adaptation from @processor_tags
        tags = getattr(type(processor), "__processor_tags__", {})
        required_bands = tags.get("required_bands")
        if required_bands is not None and isinstance(source, np.ndarray):
            exp_str = getattr(step, "band_expansion", None)
            red_str = getattr(step, "band_reduction", None)
            exp = BandExpansion(exp_str) if exp_str else BandExpansion.REPEAT
            red = BandReduction(red_str) if red_str else BandReduction.FIRST_N
            source = adapt_bands(source, required_bands, expansion=exp, reduction=red)

        # Merge step params with kwargs (step params take precedence)
        merged_kwargs = {**kwargs, **step.params}

        try:
            # apply_transform dispatches via execute() protocol.
            # No metadata in YAML executor path — synthetic metadata
            # is created internally as needed.
            result = self._gpu.apply_transform(
                processor,
                source,
                **merged_kwargs,
            )
        except Exception as e:
            # Distinguish GRDL library errors from general Python errors
            if GrdlError is not None and isinstance(e, GrdlError):
                log.error(
                    "step_grdl_error",
                    error_type=type(e).__name__,
                    error=str(e),
                )
            else:
                log.error("step_failed", error=str(e))
            raise RuntimeError(f"Pipeline step '{step.processor_name}' failed: {e}") from e

        return result

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(
        self,
        checkpoint_path: str | Path,
        progress_callback: Callable[[float], None] | None = None,
        *,
        enable_memory_check: bool = True,
        enable_shutdown_handler: bool = True,
        enable_checkpointing: bool = True,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Resume a workflow from a checkpoint file.

        Validates that the workflow definition hash matches and that
        all intermediate files still exist, then continues execution
        from the first uncompleted step.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to ``checkpoint.json`` or its parent directory.
        progress_callback : Optional[Callable[[float], None]]
            Progress callback.
        enable_memory_check : bool
            Run memory pre-flight check (default True).
        enable_shutdown_handler : bool
            Register signal handlers (default True).
        enable_checkpointing : bool
            Continue writing checkpoints (default True).
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        WorkflowResult
            Result with merged metrics from both segments.

        Raises
        ------
        ResumeError
            If the checkpoint is incompatible with the current workflow.
        CheckpointError
            If the checkpoint file cannot be read.
        """
        state = CheckpointManager.load_checkpoint(Path(checkpoint_path))
        wf_dict = self._workflow.to_dict()
        CheckpointManager.validate_for_resume(state, wf_dict)

        # Load the last intermediate array as input
        source = CheckpointManager.load_last_intermediate(state)

        # Reconstruct prior metrics from checkpoint
        prior_metrics: list[StepMetrics] = []
        for m in state.metrics_so_far:
            _raw_shape = m.get("input_shape")
            prior_metrics.append(
                StepMetrics(
                    step_index=m.get("step_index", 0),
                    processor_name=m.get("processor_name", "unknown"),
                    wall_time_s=m.get("wall_time_s", 0.0),
                    cpu_time_s=m.get("cpu_time_s", 0.0),
                    peak_rss_bytes=m.get("peak_rss_bytes", 0),
                    gpu_used=m.get("gpu_used", False),
                    status=m.get("status", "success"),
                    error_message=m.get("error_message"),
                    input_shape=tuple(_raw_shape) if _raw_shape is not None else None,
                    input_dtype=m.get("input_dtype"),
                )
            )

        run_id = str(uuid.uuid4())
        ctx = ExecutionContext(
            workflow_id=f"{self._workflow.name}:{self._workflow.version}",
            workflow_name=self._workflow.name,
            run_id=run_id,
        )
        log = logger.bind(**ctx.as_log_dict())
        log.info(
            "workflow_resume",
            checkpoint_step=state.step_index,
            total_steps=len(self._workflow.steps),
        )

        if enable_shutdown_handler:
            self._shutdown.register()

        try:
            return self._execute_main(
                source,
                progress_callback=progress_callback,
                ctx=ctx,
                enable_memory_check=enable_memory_check,
                enable_checkpointing=enable_checkpointing,
                resume_state=state,
                prior_metrics=prior_metrics,
                **kwargs,
            )
        finally:
            if enable_shutdown_handler:
                self._shutdown.unregister()

    # ------------------------------------------------------------------
    # Shutdown handling
    # ------------------------------------------------------------------

    def _do_shutdown(
        self,
        ctx: ExecutionContext,
        last_completed_index: int,
        current_array: np.ndarray,
        step_metrics_list: list[StepMetrics],
        *,
        intermediate_files: list[str] | None = None,
        wf_dict: dict[str, Any] | None = None,
        wf_hash: str | None = None,
    ) -> None:
        """Write checkpoint and exit cleanly."""
        log = logger.bind(**ctx.as_log_dict())
        log.info(
            "shutdown_checkpoint",
            last_completed_index=last_completed_index,
        )

        # Use CheckpointManager if available (TG4 path)
        if self._checkpoint_mgr is not None and wf_dict is not None:
            try:
                if wf_hash is None:
                    wf_hash = compute_workflow_hash(wf_dict)
                # Save intermediate if not already saved
                if intermediate_files is None:
                    intermediate_files = []
                if last_completed_index >= 0:
                    ipath = self._checkpoint_mgr.write_step_intermediate(
                        ctx.run_id,
                        last_completed_index,
                        current_array,
                    )
                    if ipath not in intermediate_files:
                        intermediate_files.append(ipath)
                ckpt_state = CheckpointState(
                    schema_version=CHECKPOINT_SCHEMA_VERSION,
                    workflow_definition_hash=wf_hash,
                    step_index=last_completed_index,
                    intermediate_files=list(intermediate_files),
                    metrics_so_far=[m.to_dict() for m in step_metrics_list],
                    execution_context=ctx.as_log_dict(),
                    timestamp=_iso_now(),
                    workflow_dict=wf_dict,
                )
                self._checkpoint_mgr.write_checkpoint(ckpt_state, ctx.run_id)
            except Exception as e:
                log.error("checkpoint_write_failed", error=str(e))
        else:
            # Legacy path: use ShutdownCoordinator.write_checkpoint
            try:
                self._shutdown.write_checkpoint(
                    workflow_dict=wf_dict or self._workflow.to_dict(),
                    current_step_index=last_completed_index,
                    current_array=current_array,
                    run_id=ctx.run_id,
                    step_outputs=[sm.to_dict() for sm in step_metrics_list],
                )
            except Exception as e:
                log.error("checkpoint_write_failed", error=str(e))

        # Record cancellation in history
        if self._history_db is not None:
            try:
                self._history_db.record_completion(
                    run_id=ctx.run_id,
                    status="cancelled",
                    step_count=len(step_metrics_list),
                    checkpoint_path=(
                        str(self._checkpoint_mgr.run_dir(ctx.run_id))
                        if self._checkpoint_mgr
                        else None
                    ),
                )
            except Exception as e:
                log.warning("history_record_cancelled_failed", error=str(e))

        self._shutdown.cleanup_gpu()
        sys.exit(130)
