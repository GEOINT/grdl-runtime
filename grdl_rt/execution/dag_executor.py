"""
DAG Executor — Parallel workflow execution over a directed acyclic graph.

Executes a ``WorkflowDefinition`` whose steps form a DAG.  Steps are
dispatched to a thread pool the instant their specific dependencies
complete, enabling true critical-path scheduling for asymmetric branch
DAGs.  Supports conditional steps, fan-out/fan-in, per-step resilience
(retry, timeout, circuit breaker), and structured metrics.

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
import contextlib
import json
import threading
import time
import tracemalloc
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.execution.config import get_runtime_config
from grdl_rt.execution.context import ExecutionContext, get_logger
from grdl_rt.execution.dag import evaluate_condition
from grdl_rt.execution.discovery import resolve_processor_class
from grdl_rt.execution.errors import (
    ConditionError,
    StepRetryExhaustedError,
)
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.instrumentation import ExecutionHook
from grdl_rt.execution.lineage import build_lineage
from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics
from grdl_rt.execution.plan import (
    AsExecutedManifest,
    ExecutedStepRecord,
    ResolvedExecutionPlan,
)
from grdl_rt.execution.quota import ResourceQuota
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
    return datetime.now(UTC).isoformat()


def run_dag_ready_dispatch(
    all_step_ids: list[str],
    deps_map: dict[str, list[str]],
    step_index_map: dict[str, int],
    execute_step: Callable[[str, Any, bool], tuple[StepMetrics, Any]],
    gather_input: Callable[[str], Any],
    results: dict[str, Any],
    *,
    max_workers: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[StepMetrics], int]:
    """Execute DAG steps via readiness-based dispatch.

    A step is submitted to the thread pool the instant **all** of its
    specific dependencies have completed.  This eliminates the
    level-synchronisation barrier and allows downstream steps on a fast
    branch to start before slower sibling branches finish.

    Total execution time equals the duration of the longest critical
    path through the DAG rather than the sum of per-level maxima.

    Thread safety
    -------------
    A single ``threading.Lock`` protects the shared state
    (``completed_ids``, ``in_flight``, ``pending``, ``results``).
    ``gather_input`` is always called while the lock is held so that
    dependency outputs are atomically visible to the step being
    dispatched.  ``execute_step`` runs in a worker thread, never while
    the lock is held.

    Memory tracking
    ---------------
    ``tracemalloc`` counters are process-wide, so true per-step isolation
    is impossible when threads overlap.  Two modes are used:

    * **Concurrent steps** (``concurrent=True``): ``peak_rss_bytes`` is
      set to the process-wide ``tracemalloc`` peak at completion time.
      The value reflects the shared high-water mark across all in-flight
      steps and is labelled "shared" in reports.
    * **Solo steps** (``concurrent=False``): the step resets the
      ``tracemalloc`` peak immediately before its processor runs (when no
      other threads are active) and records the peak immediately after.
      This gives an isolated, step-specific memory reading.

    Parameters
    ----------
    all_step_ids : list[str]
        All step IDs in topological order (determines root-node set and
        total step count).
    deps_map : dict[str, list[str]]
        ``{step_id: [dep_id, ...]}`` — dependency map.
    step_index_map : dict[str, int]
        ``{step_id: int}`` — deterministic index for each step.
    execute_step : callable
        ``(step_id, step_input, reset_mem_peak) -> (StepMetrics, output)``
        — executes a single step.  When ``reset_mem_peak`` is ``True``
        the callable must reset the ``tracemalloc`` peak before the
        processor runs and record the peak afterwards; the returned
        ``peak_rss_bytes`` is then used as-is.  When ``False``,
        ``peak_rss_bytes`` is overwritten in the completion callback.
    gather_input : callable
        ``(step_id) -> Any`` — returns the pre-gathered input for a
        step.  Called while the state lock is held; must read only from
        the shared ``results`` dict and the original source array.
    results : dict
        Shared results map.  Written under the lock after each step
        completes so downstream steps receive correct dependency data.
    max_workers : int, optional
        Maximum thread-pool size.  Defaults to total step count.
    progress_callback : callable, optional
        Called with a float in ``[0.0, 1.0]`` after each step completes.

    Returns
    -------
    tuple[list[StepMetrics], int]
        ``(step_metrics_list, overall_peak_bytes)``

    Raises
    ------
    Exception
        Re-raises the first exception from any failed step after the
        thread pool has drained.
    """
    total = len(all_step_ids)
    if total == 0:
        return [], 0

    # --- shared mutable state (all guarded by lock) -----------------------
    lock = threading.RLock()
    completed_ids: set[str] = set()
    in_flight: set[str] = set()
    concurrent_step_ids: set[str] = set()
    pending: set[str] = set(all_step_ids)
    step_metrics_list: list[StepMetrics] = []
    completed_count = 0
    overall_peak = 0
    first_exc: BaseException | None = None
    done_event = threading.Event()
    # ----------------------------------------------------------------------

    _max = max_workers or total

    def _make_callback(s: str, p: ThreadPoolExecutor) -> Callable[[Any], None]:
        def cb(f: Any) -> None:
            _on_done(f, s, p)

        return cb

    def _submit_ready(pool: ThreadPoolExecutor) -> None:
        """Submit all currently-ready steps.  Must be called under *lock*."""
        ready = [s for s in pending if all(d in completed_ids for d in deps_map.get(s, []))]
        already_in_flight = len(in_flight)
        for sid in ready:
            pending.discard(sid)
            in_flight.add(sid)
            # Mark all currently in-flight steps as concurrent at submission
            # time, before any of them can complete and leave in_flight.
            if len(in_flight) > 1:
                concurrent_step_ids.update(in_flight)
            step_input = gather_input(sid)  # atomic: lock held, results stable
            # A step is solo when it is the only one in this dispatch batch
            # AND no other steps were already running.  Solo steps get an
            # isolated tracemalloc measurement; concurrent steps share one.
            reset_mem_peak = len(ready) == 1 and already_in_flight == 0
            fut = pool.submit(execute_step, sid, step_input, reset_mem_peak)
            fut.add_done_callback(_make_callback(sid, pool))

    def _on_done(fut: Any, sid: str, pool: ThreadPoolExecutor) -> None:
        """Completion callback — runs in a worker thread."""
        nonlocal completed_count, overall_peak, first_exc

        with lock:
            if first_exc is not None:
                # A prior step already failed; drain in-flight but dispatch nothing.
                in_flight.discard(sid)
                if not in_flight and not pending:
                    done_event.set()
                return

            try:
                sm, output = fut.result()
            except Exception as exc:  # noqa: BLE001
                first_exc = exc
                in_flight.discard(sid)
                done_event.set()
                return

            # Concurrent detection: a step is concurrent if it was in-flight
            # at the same time as another step (recorded at submission time).
            sm.concurrent = sid in concurrent_step_ids
            sm.step_index = step_index_map[sid]

            _, current_peak = tracemalloc.get_traced_memory()
            overall_peak = max(overall_peak, current_peak)
            if sm.concurrent:
                # Can't isolate: report the shared process-wide high-water mark.
                sm.peak_rss_bytes = current_peak
            # else: solo step already measured and set peak_rss_bytes itself.

            in_flight.discard(sid)
            completed_ids.add(sid)
            results[sid] = output
            step_metrics_list.append(sm)
            completed_count += 1

            if progress_callback is not None and total > 0:
                progress_callback(completed_count / total)

            _submit_ready(pool)  # dispatch any newly unblocked steps

            if not pending and not in_flight:
                done_event.set()

    with ThreadPoolExecutor(max_workers=_max) as pool:
        with lock:
            _submit_ready(pool)  # kick off root nodes (zero dependencies)
        done_event.wait()        # block until all steps finish or first failure
    # pool.__exit__ calls shutdown(wait=True), draining remaining futures

    if first_exc is not None:
        raise first_exc

    return step_metrics_list, overall_peak


class DAGExecutor:
    """Execute a workflow DAG with readiness-based scheduling.

    Steps are dispatched to a thread pool the instant their specific
    dependencies complete.  Total execution time equals the duration of
    the longest critical path through the DAG — not the sum of the
    longest steps at each topological level.  Each step's output is
    stored in a results map keyed by step ID, and downstream steps
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
        Maximum number of parallel threads.  Defaults to total step count.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        gpu: GpuBackend | None = None,
        *,
        catalog: ArtifactCatalogBase | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_workers: int | None = None,
        resource_quota: ResourceQuota | None = None,
        hooks: list[ExecutionHook] | None = None,
    ) -> None:
        self._workflow = workflow
        self._gpu = gpu or GpuBackend(prefer_gpu=False)
        self._catalog = catalog
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._max_workers = max_workers
        self._resource_quota = resource_quota
        self._hooks: list[ExecutionHook] = list(hooks or [])
        self._runtime_substitutions: list[dict[str, Any]] = []
        self._runtime_subs_lock = threading.Lock()

    def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call a hook method on all registered hooks, swallowing errors."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                getattr(hook, method)(*args, **kwargs)

    def execute(
        self,
        source: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
        *,
        enable_memory_check: bool = True,
        execution_context: dict[str, Any] | None = None,
        resolved_plan: ResolvedExecutionPlan | None = None,
        run_folder: Path | None = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute the DAG workflow on a single input.

        Steps are dispatched to the thread pool the instant all of their
        specific dependencies complete.  This readiness-based approach
        eliminates level-synchronisation barriers, so total runtime equals
        the duration of the longest critical path through the DAG.

        .. note:: **Readiness-based scheduling**

           Each step starts as soon as its direct dependencies finish,
           regardless of unrelated branches.  For asymmetric DAGs (e.g.,
           Branch A: 10 s, Branch B: 20 s) a step that depends only on
           Branch A starts after 10 s even while Branch B is still running.

        Parameters
        ----------
        source : np.ndarray
            Input image array fed to root steps (those with no deps).
        progress_callback : callable, optional
            Called with a float in ``[0.0, 1.0]`` after each step completes.
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
            raise ValueError(f"Invalid workflow DAG: {'; '.join(dag_errors)}")

        # Topological sort into levels
        levels = self._workflow.topological_sort()
        total_steps = sum(len(level) for level in levels)

        # Memory pre-flight
        if enable_memory_check:
            processing_steps = [s for s in self._workflow.steps if isinstance(s, ProcessingStep)]
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

        # Write as_planned.json before execution begins
        if run_folder is not None and resolved_plan is not None:
            _run_folder = Path(run_folder)
            _run_folder.mkdir(parents=True, exist_ok=True)
            planned_path = _run_folder / "as_planned.json"
            planned_path.write_text(
                json.dumps(resolved_plan.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            log.info("as_planned_written", path=str(planned_path))

        # Reset runtime substitutions for this execution
        self._runtime_substitutions = []

        # Results map: step_id -> output array
        results: dict[str, np.ndarray] = {}
        step_metrics_list: list[StepMetrics] = []
        completed_steps = 0

        # Pre-compute stable step indices from topological order so that
        # parallel steps get unique, deterministic indices regardless of
        # completion order.
        step_index_map: dict[str, int] = {}
        _idx = 0
        for _lvl in levels:
            for _sid in _lvl:
                step_index_map[_sid] = _idx
                _idx += 1

        # Determine max workers
        max_width = max((len(level) for level in levels), default=1)
        max_workers = self._max_workers or max_width

        tracemalloc.start()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        try:
            # Build flat step list and dependency map for readiness dispatch
            deps_map: dict[str, list[str]] = {
                _s.id: list(_s.depends_on or [])
                for _s in self._workflow.steps
                if _s.id is not None
            }
            all_step_ids = [sid for lvl in levels for sid in lvl]

            def _gather(sid: str) -> Any:
                _s = self._workflow.get_step(sid)
                if _s.depends_on:
                    dep_res = {d: results[d] for d in _s.depends_on}
                    return next(iter(dep_res.values())) if len(dep_res) == 1 else dep_res
                return source

            def _exec_step(sid: str, step_input: Any, reset_mem_peak: bool = False) -> tuple[StepMetrics, Any]:
                return self._execute_single_step(
                    sid, step_input, results,
                    reset_mem_peak=reset_mem_peak,
                    user_context=user_context, log=log, **kwargs,
                )

            step_metrics_list, total_peak = run_dag_ready_dispatch(
                all_step_ids,
                deps_map,
                step_index_map,
                _exec_step,
                _gather,
                results,
                max_workers=max_workers,
                progress_callback=progress_callback,
            )

            total_wall = time.perf_counter() - t0_wall
            total_cpu = time.process_time() - t0_cpu

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

            # Build data lineage
            lineage = None
            try:
                lineage = build_lineage(
                    source,
                    final_result,
                    self._workflow.steps,
                    step_metrics_list,
                )
            except Exception as e:
                log.warning("lineage_build_failed", error=str(e))

            log.info(
                "dag_workflow_complete",
                status="success",
                total_wall_time_s=round(total_wall, 4),
                step_count=len(step_metrics_list),
            )

            # Write as_executed.json
            if run_folder is not None:
                self._write_as_executed(
                    run_id=run_id,
                    run_folder=Path(run_folder),
                    started_at=started_at,
                    status="success",
                    resolved_plan=resolved_plan,
                    step_metrics_list=step_metrics_list,
                    total_wall=total_wall,
                    total_cpu=total_cpu,
                    total_peak=total_peak,
                    lineage_dict=(lineage.to_dict() if lineage else None),
                    log=log,
                )

            return WorkflowResult(
                result=final_result,
                metrics=wf_metrics,
                step_results=dict(results),
                lineage=lineage,
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

            # Write as_executed.json on failure too
            if run_folder is not None:
                try:
                    self._write_as_executed(
                        run_id=run_id,
                        run_folder=Path(run_folder),
                        started_at=started_at,
                        status="failed",
                        resolved_plan=resolved_plan,
                        step_metrics_list=step_metrics_list,
                        total_wall=total_wall,
                        total_cpu=total_cpu,
                        total_peak=total_peak,
                        error_message=str(exc),
                        log=log,
                    )
                except Exception:
                    log.warning("as_executed_write_failed")

            raise

        finally:
            tracemalloc.stop()

    def _execute_single_step(
        self,
        step_id: str,
        step_input: Any,
        results: dict[str, Any],
        *,
        reset_mem_peak: bool = False,
        user_context: dict[str, Any],
        log: Any,
        **kwargs: Any,
    ) -> tuple[StepMetrics, Any]:
        """Execute one step with a pre-gathered input.

        For solo (non-concurrent) steps ``reset_mem_peak=True``: the
        tracemalloc peak is reset immediately before the processor runs
        and recorded immediately after, giving an isolated per-step
        reading.  For concurrent steps the completion callback in
        :func:`run_dag_ready_dispatch` overwrites ``peak_rss_bytes``
        with the shared process-wide peak instead.

        Parameters
        ----------
        step_id : str
            ID of the step to execute.
        step_input : Any
            Pre-gathered input (single array, dict, or source).  Must be
            assembled by the caller under the state lock before this
            method is invoked in a worker thread.
        results : dict
            Shared results map.  Read-only here; used only to populate
            the condition-evaluation context.
        user_context : dict
            User-provided context for condition evaluation.
        log
            Structured logger.
        **kwargs
            Extra processor arguments.

        Returns
        -------
        Tuple[StepMetrics, Any]
            ``(metrics, output)`` — output is the step's result array.
        """
        step = self._workflow.get_step(step_id)

        # Evaluate condition
        if isinstance(step, ProcessingStep) and step.condition is not None:
            cond_context = dict(user_context)
            cond_context["results"] = results
            try:
                cond_result = evaluate_condition(step.condition, cond_context)
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                raise ConditionError(step.condition, str(e)) from e

            if not cond_result:
                log.debug(
                    "step_skipped_condition",
                    step_id=step_id,
                    condition=step.condition,
                )
                # Propagate input unchanged
                if isinstance(step_input, dict):
                    # For multi-dep, propagate first dep's output
                    output = next(iter(step_input.values()))
                else:
                    output = step_input
                return StepMetrics(
                    step_index=0,
                    processor_name=getattr(step, "processor_name", "tap_out"),
                    wall_time_s=0.0,
                    cpu_time_s=0.0,
                    peak_rss_bytes=0,
                    gpu_used=False,
                    status="skipped",
                    step_id=step_id,
                ), output

        # For solo steps: reset the tracemalloc peak so the measurement
        # below reflects only this step, not any preceding parallel phase.
        if reset_mem_peak:
            tracemalloc.reset_peak()

        # Execute
        step_t0_wall = time.perf_counter()
        step_t0_cpu = time.thread_time()

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
                    "tap_out_failed",
                    step_id=step_id,
                    path=step.path,
                    error=str(e),
                )
            output = next(iter(step_input.values())) if isinstance(step_input, dict) else step_input
            step_status = "success"

        elif isinstance(step, ProcessingStep):
            # Circuit breaker check
            if self._circuit_breaker.is_open(step.processor_name):
                raise RuntimeError(
                    f"Circuit breaker open for processor " f"'{step.processor_name}'"
                )

            try:
                output = self._execute_processor_resilient(
                    step,
                    step_input,
                    log=log,
                    **kwargs,
                )
                self._circuit_breaker.record_success(step.processor_name)
                step_status = "success"
            except (StepRetryExhaustedError, RuntimeError) as exc:
                self._circuit_breaker.record_failure(step.processor_name)
                fallback_output = self._attempt_fallback(
                    step,
                    step_input,
                    exc,
                    log=log,
                    **kwargs,
                )
                if fallback_output is not None:
                    output = fallback_output
                    step_status = "fallback"
                else:
                    raise
        else:
            output = step_input
            step_status = "success"

        step_wall = time.perf_counter() - step_t0_wall
        step_cpu = time.thread_time() - step_t0_cpu

        # For solo steps, record peak now while no other threads are active.
        # For concurrent steps, leave 0; _on_done will overwrite with the
        # shared process-wide peak.
        step_peak = tracemalloc.get_traced_memory()[1] if reset_mem_peak else 0

        processor_name = getattr(step, "processor_name", "tap_out")
        log.debug(
            "dag_step_complete",
            step_id=step_id,
            processor_name=processor_name,
            wall_time_s=round(step_wall, 4),
        )

        return StepMetrics(
            step_index=0,
            processor_name=processor_name,
            wall_time_s=step_wall,
            cpu_time_s=step_cpu,
            peak_rss_bytes=step_peak,
            gpu_used=False,
            status=step_status,
            step_id=step_id,
        ), output

    def _execute_processor_resilient(
        self,
        step: ProcessingStep,
        step_input: np.ndarray | dict[str, np.ndarray],
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
            def raw_fn():
                return self._execute_processor(step, step_input, **kwargs)

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

        return _do_step()

    def _execute_processor(
        self,
        step: ProcessingStep,
        step_input: np.ndarray | dict[str, np.ndarray],
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
        except ImportError as e:
            raise ImportError(f"Failed to resolve processor '{step.processor_name}': {e}") from e

        try:
            processor = processor_cls()
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate processor '{step.processor_name}': {e}"
            ) from e

        # Merge step params with kwargs
        merged_kwargs = {**kwargs, **step.params}

        try:
            result = self._gpu.apply_transform(processor, step_input, **merged_kwargs)  # type: ignore[arg-type]
        except Exception as e:
            if GrdlError is not None and isinstance(e, GrdlError):
                log.error(
                    "step_grdl_error",
                    error_type=type(e).__name__,
                    error=str(e),
                )
            else:
                log.error("step_failed", error=str(e))
            raise RuntimeError(f"DAG step '{step.processor_name}' ({step.id}) failed: {e}") from e

        return result

    def _attempt_fallback(
        self,
        step: ProcessingStep,
        step_input: np.ndarray | dict[str, np.ndarray],
        original_error: Exception,
        *,
        log: Any = None,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """Attempt to execute a fallback processor for a failed step.

        Queries the catalog for alternatives, tries the first compatible
        one.  Only one level of fallback is attempted.

        Parameters
        ----------
        step : ProcessingStep
            The step that failed.
        step_input : np.ndarray or dict
            The input to the step.
        original_error : Exception
            The original error.
        log
            Structured logger.
        **kwargs
            Extra processor arguments.

        Returns
        -------
        Optional[np.ndarray]
            Fallback output, or None if no fallback succeeded.
        """
        log = log or logger
        alternatives = self._get_step_alternatives(step.processor_name)
        if not alternatives:
            log.warning(
                "step_no_alternatives",
                step_id=step.id,
                processor=step.processor_name,
            )
            return None

        for alt in alternatives:
            alt_name = alt.get("processor_name", "")
            if not alt_name:
                continue

            log.warning(
                "step_fallback_attempt",
                step_id=step.id,
                original=step.processor_name,
                fallback=alt_name,
            )

            try:
                alt_cls = resolve_processor_class(alt_name)
                alt_processor = alt_cls()
                merged_kwargs = {**kwargs, **step.params}
                output = self._gpu.apply_transform(
                    alt_processor,
                    step_input,  # type: ignore[arg-type]
                    **merged_kwargs,
                )

                with self._runtime_subs_lock:
                    self._runtime_substitutions.append(
                        {
                            "step_id": step.id,
                            "original_processor": step.processor_name,
                            "replacement_processor": alt_name,
                            "reason": (f"Primary processor failed: {original_error}"),
                        }
                    )

                log.info(
                    "step_fallback_success",
                    step_id=step.id,
                    fallback=alt_name,
                )
                return output

            except Exception as fallback_exc:
                log.warning(
                    "step_fallback_failed",
                    step_id=step.id,
                    fallback=alt_name,
                    error=str(fallback_exc),
                )
                continue

        log.error(
            "step_fallback_exhausted",
            step_id=step.id,
            processor=step.processor_name,
            tried=[a.get("processor_name", "") for a in alternatives],
        )
        return None

    def _get_step_alternatives(
        self,
        processor_name: str,
    ) -> list[dict[str, Any]]:
        """Get alternative processors for a step from the catalog.

        Parameters
        ----------
        processor_name : str
            Name of the processor to find alternatives for.

        Returns
        -------
        List[Dict[str, Any]]
            Alternative entries sorted by priority.
        """
        if self._catalog is None:
            return []

        # Search catalog for artifacts matching this processor
        for artifact in self._catalog.list_artifacts(
            artifact_type="grdl_processor",
        ):
            if artifact.name == processor_name:
                alts = list(artifact.alternatives)
                alts.sort(key=lambda a: a.get("priority", 0))
                return alts
            if artifact.processor_class:
                short = artifact.processor_class.rsplit(".", 1)[-1]
                if short == processor_name:
                    alts = list(artifact.alternatives)
                    alts.sort(key=lambda a: a.get("priority", 0))
                    return alts

        # Try get_alternatives directly
        try:
            alts = self._catalog.get_alternatives(processor_name, "")
            alts.sort(key=lambda a: a.get("priority", 0))
            return alts
        except Exception:
            return []

    def _write_as_executed(
        self,
        run_id: str,
        run_folder: Path,
        started_at: str,
        status: str,
        resolved_plan: ResolvedExecutionPlan | None,
        step_metrics_list: list[StepMetrics],
        total_wall: float,
        total_cpu: float,
        total_peak: int,
        error_message: str | None = None,
        lineage_dict: dict[str, Any] | None = None,
        log: Any = None,
    ) -> None:
        """Write as_executed.json to the run folder."""
        log = log or logger
        executed_records: list[ExecutedStepRecord] = []
        for sm in step_metrics_list:
            sub = next(
                (s for s in self._runtime_substitutions if s["step_id"] == sm.step_id),
                None,
            )
            executed_records.append(
                ExecutedStepRecord(
                    step_id=sm.step_id or "",
                    processor_name=sm.processor_name,
                    status=sm.status,
                    wall_time_s=sm.wall_time_s,
                    cpu_time_s=sm.cpu_time_s,
                    peak_rss_bytes=sm.peak_rss_bytes,
                    gpu_used=sm.gpu_used,
                    fallback_processor=(sub["replacement_processor"] if sub else None),
                    fallback_reason=sub["reason"] if sub else None,
                    error_message=(error_message if sm.status == "failed" else None),
                )
            )

        manifest = AsExecutedManifest(
            workflow_name=self._workflow.name,
            workflow_version=self._workflow.version,
            run_id=run_id,
            started_at=started_at,
            completed_at=_iso_now(),
            status=status,
            hardware_context=(resolved_plan.hardware_context if resolved_plan else {}),
            planned_steps=(
                {k: v.to_dict() for k, v in resolved_plan.steps.items()} if resolved_plan else {}
            ),
            executed_steps=executed_records,
            runtime_substitutions=list(self._runtime_substitutions),
            total_wall_time_s=total_wall,
            total_cpu_time_s=total_cpu,
            peak_rss_bytes=total_peak,
            data_lineage=lineage_dict,
        )

        run_folder.mkdir(parents=True, exist_ok=True)
        executed_path = run_folder / "as_executed.json"
        executed_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        log.info("as_executed_written", path=str(executed_path))
