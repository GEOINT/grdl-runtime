# -*- coding: utf-8 -*-
"""
DAG Executor — Parallel workflow execution over a directed acyclic graph.

Executes a ``WorkflowDefinition`` whose steps form a DAG.  Steps at the
same topological level (no mutual dependencies) are executed concurrently
via a thread pool.  Supports conditional steps, fan-out/fan-in, per-step
resilience (retry, timeout, circuit breaker), and structured metrics.

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

# Standard library
import time
import tracemalloc
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.config import get_runtime_config
from grdl_rt.execution.context import ExecutionContext, get_logger
from grdl_rt.execution.dag import evaluate_condition
from grdl_rt.execution.discovery import resolve_processor_class
from grdl_rt.execution.errors import (
    ConditionError,
    StepRetryExhaustedError,
    StepTimeoutError,
)
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics
from grdl_rt.execution.resilience import (
    CircuitBreaker,
    RetryPolicy,
    execute_with_retry,
    execute_with_timeout,
    run_memory_preflight,
)
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.workflow import (
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
)

logger = get_logger(__name__)

# GRDL exceptions (optional)
try:
    from grdl.exceptions import GrdlError
except ImportError:
    GrdlError = None  # type: ignore[misc,assignment]


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class DAGExecutor:
    """Execute a workflow DAG with parallel level execution.

    Steps are topologically sorted into levels.  All steps within a
    level execute concurrently via a thread pool.  Each step's output
    is stored in a results map keyed by step ID, and downstream steps
    gather their inputs from this map.

    Parameters
    ----------
    workflow : WorkflowDefinition
        Compiled workflow with DAG structure.
    gpu : Optional[GpuBackend]
        GPU backend for acceleration.
    circuit_breaker : Optional[CircuitBreaker]
        Circuit breaker for processor failure tracking.
    max_workers : int, optional
        Maximum number of parallel threads.  Defaults to the number
        of steps in the widest level.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        gpu: Optional[GpuBackend] = None,
        *,
        circuit_breaker: Optional[CircuitBreaker] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self._workflow = workflow
        self._gpu = gpu or GpuBackend(prefer_gpu=False)
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._max_workers = max_workers

    def execute(
        self,
        source: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
        *,
        enable_memory_check: bool = True,
        execution_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute the DAG workflow on a single input.

        Parameters
        ----------
        source : np.ndarray
            Input image array fed to root steps (those with no deps).
        progress_callback : callable, optional
            Called with a float in ``[0.0, 1.0]`` after each level.
        enable_memory_check : bool
            If ``True`` (default), run memory pre-flight check.
        execution_context : dict, optional
            Additional context for condition evaluation.  Merged with
            step results and metadata to form the condition context.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        WorkflowResult
            Result containing the terminal step output and per-step
            metrics.
        """
        run_id = str(uuid.uuid4())
        ctx = ExecutionContext(
            workflow_id=f"{self._workflow.name}:{self._workflow.version}",
            workflow_name=self._workflow.name,
            run_id=run_id,
        )
        log = logger.bind(**ctx.as_log_dict())
        cfg = get_runtime_config()
        started_at = _iso_now()
        user_context = execution_context or {}

        # Validate DAG
        dag_errors = self._workflow.validate_dag()
        if dag_errors:
            raise ValueError(
                f"Invalid workflow DAG: {'; '.join(dag_errors)}"
            )

        # Topological sort into levels
        levels = self._workflow.topological_sort()
        total_steps = sum(len(level) for level in levels)

        # Memory pre-flight
        if enable_memory_check:
            processing_steps = [
                s for s in self._workflow.steps
                if isinstance(s, ProcessingStep)
            ]
            if processing_steps:
                # Estimate based on widest parallel level
                max_width = max(len(level) for level in levels) if levels else 1
                run_memory_preflight(
                    source,
                    n_steps=max_width,
                    multiplier=cfg.memory.estimation_multiplier,
                    warn_threshold=cfg.memory.warn_threshold,
                    abort_threshold=cfg.memory.abort_threshold,
                    log=log,
                )

        # Results map: step_id -> output array
        results: Dict[str, np.ndarray] = {}
        step_metrics_list: List[StepMetrics] = []
        completed_steps = 0

        # Determine max workers
        max_width = max((len(level) for level in levels), default=1)
        max_workers = self._max_workers or max_width

        tracemalloc.start()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        try:
            for level_idx, level in enumerate(levels):
                if len(level) == 1:
                    # Single step — no threading overhead
                    step_id = level[0]
                    sm = self._execute_single_step(
                        step_id, source, results,
                        user_context=user_context,
                        log=log, **kwargs,
                    )
                    step_metrics_list.append(sm)
                    completed_steps += 1
                else:
                    # Parallel execution of level
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {}
                        for step_id in level:
                            future = pool.submit(
                                self._execute_single_step,
                                step_id, source, results,
                                user_context=user_context,
                                log=log, **kwargs,
                            )
                            futures[future] = step_id

                        for future in as_completed(futures):
                            sm = future.result()
                            step_metrics_list.append(sm)
                            completed_steps += 1

                if progress_callback is not None and total_steps > 0:
                    progress_callback(completed_steps / total_steps)

            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _, total_peak = tracemalloc.get_traced_memory()

            # Determine terminal output
            terminal_ids = self._workflow.terminal_step_ids()
            if len(terminal_ids) == 1:
                final_result = results[terminal_ids[0]]
            elif terminal_ids:
                # Multiple terminals: use last in topological order
                last_level = levels[-1] if levels else []
                # Pick from terminals that are in the last level
                for sid in reversed(last_level):
                    if sid in terminal_ids:
                        final_result = results[sid]
                        break
                else:
                    final_result = results[terminal_ids[-1]]
            else:
                # Fallback: return source if no steps
                final_result = source

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=total_peak,
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="success",
            )

            log.info(
                "dag_workflow_complete", status="success",
                total_wall_time_s=round(total_wall, 4),
                step_count=len(step_metrics_list),
            )

            return WorkflowResult(
                result=final_result,
                metrics=wf_metrics,
                step_results=dict(results),
            )

        except Exception as exc:
            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu
            _, total_peak = tracemalloc.get_traced_memory()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._workflow.name,
                workflow_version=self._workflow.version,
                total_wall_time_s=total_wall,
                total_cpu_time_s=total_cpu,
                peak_rss_bytes=total_peak,
                step_metrics=step_metrics_list,
                started_at=started_at,
                completed_at=_iso_now(),
                status="failed",
                error_message=str(exc),
            )
            exc.__workflow_metrics__ = wf_metrics  # type: ignore[attr-defined]
            raise

        finally:
            tracemalloc.stop()

    def _execute_single_step(
        self,
        step_id: str,
        source: np.ndarray,
        results: Dict[str, np.ndarray],
        *,
        user_context: Dict[str, Any],
        log: Any,
        **kwargs: Any,
    ) -> StepMetrics:
        """Execute one step, gathering inputs and storing the output.

        Parameters
        ----------
        step_id : str
            ID of the step to execute.
        source : np.ndarray
            Original input (for root steps with no dependencies).
        results : Dict[str, np.ndarray]
            Shared results map.  Read for inputs, written with output.
        user_context : dict
            User-provided context for condition evaluation.
        log
            Structured logger.
        **kwargs
            Extra processor arguments.

        Returns
        -------
        StepMetrics
        """
        step = self._workflow.get_step(step_id)

        # Gather inputs
        if step.depends_on:
            dep_results = {
                dep_id: results[dep_id] for dep_id in step.depends_on
            }
            if len(dep_results) == 1:
                step_input = next(iter(dep_results.values()))
            else:
                step_input = dep_results
        else:
            step_input = source

        # Evaluate condition
        if isinstance(step, ProcessingStep) and step.condition is not None:
            cond_context = dict(user_context)
            cond_context['results'] = results
            try:
                cond_result = evaluate_condition(step.condition, cond_context)
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                raise ConditionError(step.condition, str(e)) from e

            if not cond_result:
                log.debug(
                    "step_skipped_condition", step_id=step_id,
                    condition=step.condition,
                )
                # Propagate input unchanged
                if isinstance(step_input, dict):
                    # For multi-dep, propagate first dep's output
                    output = next(iter(step_input.values()))
                else:
                    output = step_input
                results[step_id] = output
                return StepMetrics(
                    step_index=0,
                    processor_name=getattr(step, 'processor_name', 'tap_out'),
                    wall_time_s=0.0,
                    cpu_time_s=0.0,
                    peak_rss_bytes=0,
                    gpu_used=False,
                    status="skipped",
                    step_id=step_id,
                )

        # Execute
        step_t0_wall = time.perf_counter()
        step_t0_cpu = time.process_time()

        if isinstance(step, TapOutStepDef):
            # Tap-out: write to disk, pass through
            try:
                from grdl.IO import write as io_write
                if isinstance(step_input, dict):
                    write_data = next(iter(step_input.values()))
                else:
                    write_data = step_input
                io_write(write_data, step.path, format=step.format)
            except Exception as e:
                log.warning(
                    "tap_out_failed", step_id=step_id,
                    path=step.path, error=str(e),
                )
            if isinstance(step_input, dict):
                output = next(iter(step_input.values()))
            else:
                output = step_input

        elif isinstance(step, ProcessingStep):
            # Circuit breaker check
            if self._circuit_breaker.is_open(step.processor_name):
                raise RuntimeError(
                    f"Circuit breaker open for processor "
                    f"'{step.processor_name}'"
                )

            output = self._execute_processor_resilient(
                step, step_input, log=log, **kwargs,
            )
            self._circuit_breaker.record_success(step.processor_name)
        else:
            output = step_input

        step_wall = time.perf_counter() - step_t0_wall
        step_cpu = time.process_time() - step_t0_cpu

        results[step_id] = output

        processor_name = getattr(step, 'processor_name', 'tap_out')
        log.debug(
            "dag_step_complete", step_id=step_id,
            processor_name=processor_name,
            wall_time_s=round(step_wall, 4),
        )

        return StepMetrics(
            step_index=0,
            processor_name=processor_name,
            wall_time_s=step_wall,
            cpu_time_s=step_cpu,
            peak_rss_bytes=0,
            gpu_used=False,
            step_id=step_id,
        )

    def _execute_processor_resilient(
        self,
        step: ProcessingStep,
        step_input: Union[np.ndarray, Dict[str, np.ndarray]],
        *,
        log: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a processing step with retry and timeout.

        Parameters
        ----------
        step : ProcessingStep
        step_input : np.ndarray or Dict[str, np.ndarray]
            Single array for single-dependency steps, dict for
            multi-dependency (fan-in) steps.
        log
            Structured logger.
        **kwargs
            Extra processor arguments.

        Returns
        -------
        np.ndarray
        """
        log = log or logger
        cfg = get_runtime_config()

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
            raw_fn = lambda: self._execute_processor(
                step, step_input, **kwargs
            )
            if timeout is not None:
                return execute_with_timeout(
                    raw_fn, timeout, step.processor_name,
                )
            return raw_fn()

        if retry.max_retries > 0:
            return execute_with_retry(
                _do_step, retry, step.processor_name, log=log,
            )

        return _do_step()

    def _execute_processor(
        self,
        step: ProcessingStep,
        step_input: Union[np.ndarray, Dict[str, np.ndarray]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a single processing step (no resilience wrapping).

        Parameters
        ----------
        step : ProcessingStep
        step_input : np.ndarray or Dict[str, np.ndarray]
        **kwargs

        Returns
        -------
        np.ndarray
        """
        log = logger.bind(processor_name=step.processor_name)
        log.debug("step_resolving")

        try:
            processor_cls = resolve_processor_class(step.processor_name)
            processor = processor_cls()
        except (ImportError, Exception) as e:
            raise ImportError(
                f"Failed to resolve processor '{step.processor_name}': {e}"
            ) from e

        # Merge step params with kwargs
        merged_kwargs = {**kwargs, **step.params}

        try:
            result = self._gpu.apply_transform(
                processor, step_input, **merged_kwargs
            )
        except Exception as e:
            if GrdlError is not None and isinstance(e, GrdlError):
                log.error(
                    "step_grdl_error",
                    error_type=type(e).__name__, error=str(e),
                )
            else:
                log.error("step_failed", error=str(e))
            raise RuntimeError(
                f"DAG step '{step.processor_name}' ({step.id}) failed: {e}"
            ) from e

        return result
