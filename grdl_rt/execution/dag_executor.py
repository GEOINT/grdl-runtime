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

        # Determine max workers
        max_width = max((len(level) for level in levels), default=1)
        max_workers = self._max_workers or max_width

        tracemalloc.start()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        try:
            for _level_idx, level in enumerate(levels):
                if len(level) == 1:
                    # Single step — no threading overhead
                    step_id = level[0]
                    sm = self._execute_single_step(
                        step_id,
                        source,
                        results,
                        user_context=user_context,
                        log=log,
                        **kwargs,
                    )
                    sm.step_index = completed_steps
                    step_metrics_list.append(sm)
                    completed_steps += 1
                else:
                    # Parallel execution of level
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {}
                        for step_id in level:
                            future = pool.submit(
                                self._execute_single_step,
                                step_id,
                                source,
                                results,
                                user_context=user_context,
                                log=log,
                                **kwargs,
                            )
                            futures[future] = step_id

                        for future in as_completed(futures):
                            sm = future.result()
                            sm.step_index = completed_steps
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
        source: np.ndarray,
        results: dict[str, np.ndarray],
        *,
        user_context: dict[str, Any],
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
            dep_results = {dep_id: results[dep_id] for dep_id in step.depends_on}
            step_input = next(iter(dep_results.values())) if len(dep_results) == 1 else dep_results
        else:
            step_input = source

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
                results[step_id] = output
                return StepMetrics(
                    step_index=0,
                    processor_name=getattr(step, "processor_name", "tap_out"),
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
        step_cpu = time.process_time() - step_t0_cpu

        results[step_id] = output

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
            peak_rss_bytes=0,
            gpu_used=False,
            status=step_status,
            step_id=step_id,
        )

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
