"""
Execution Metrics — Structured timing and resource usage data.

Provides ``StepMetrics`` and ``WorkflowMetrics`` dataclasses that capture
wall-clock time, CPU time, peak memory, GPU usage, and a continuous
memory timeline for every workflow execution.  These are returned as
part of ``WorkflowResult`` from all execution paths.

Memory is measured by a background ``MemorySampler`` thread that polls
``tracemalloc`` at 1 ms intervals.  The resulting ``MemoryTimeline``
on ``WorkflowMetrics`` can be overlaid with each step's
``wall_start`` / ``wall_end`` timestamps for visualization.

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

Modified
--------
2026-02-11
"""

# Standard library
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepMetrics:
    """Timing and resource metrics for a single workflow step.

    Memory is tracked at the workflow level via a ``MemorySampler``
    background thread that produces a continuous ``MemoryTimeline``
    of corrected ``tracemalloc`` readings.  Per-step memory fields
    (``peak_rss_bytes``, ``peak_overhead_bytes``,
    ``end_of_step_footprint_bytes``) are retained for backward
    compatibility but are **not populated** for pipeline steps.
    Use ``wall_start`` / ``wall_end`` to correlate steps with the
    workflow-level ``memory_timeline`` for visualization.

    For single-component benchmarks (``execute_single_step``),
    ``peak_rss_bytes`` reflects the timeline peak for that step.

    Attributes
    ----------
    step_index : int
        Zero-based position within the workflow.
    processor_name : str
        Fully-qualified or short name of the processor executed.
    wall_time_s : float
        Wall-clock duration in seconds (``time.perf_counter`` delta).
    cpu_time_s : float
        CPU time in seconds (``time.process_time`` delta).
    peak_rss_bytes : int
        Peak memory in bytes.  For single-component benchmarks this
        is the timeline peak for the step.  For pipeline steps this
        is 0 — use the workflow-level ``memory_timeline`` instead.
    gpu_used : bool
        Whether this step ran on GPU.
    gpu_memory_bytes : Optional[int]
        GPU memory allocated in bytes during this step, if applicable.
    status : str
        ``"success"`` or ``"failed"``.
    error_message : Optional[str]
        If status is ``"failed"``, the exception message.
    concurrent : bool
        Whether this step ran concurrently with other steps in a
        parallel DAG level.  Used by the benchmarking layer for
        topology classification and report rendering.
    peak_overhead_bytes : int
        Legacy field retained for backward compatibility.  No longer
        populated by the runtime — always 0 for pipeline steps.
    end_of_step_footprint_bytes : int
        Legacy field retained for backward compatibility.  No longer
        populated by the runtime — always 0 for pipeline steps.
    input_shape : tuple of int, optional
        Shape of the NumPy array passed as input to this step.
        ``None`` when the input is not an ndarray (e.g., a file path
        or dict).  Used by the benchmarking layer to compute
        throughput (elements/sec).
    input_dtype : str, optional
        String representation of the input array's dtype (e.g.,
        ``"float32"``).  ``None`` when the input is not an ndarray.
    wall_start : float
        Absolute ``time.perf_counter()`` value when this step began.
        Used to correlate with the workflow-level ``memory_timeline``.
    wall_end : float
        Absolute ``time.perf_counter()`` value when this step ended.
    """

    step_index: int
    processor_name: str
    wall_time_s: float
    cpu_time_s: float
    peak_rss_bytes: int
    gpu_used: bool
    status: str = "success"
    error_message: str | None = None
    step_id: str | None = None
    gpu_memory_bytes: int | None = None
    global_pass_duration: float | None = None
    global_pass_memory: int | None = None
    concurrent: bool = False
    peak_overhead_bytes: int = 0
    end_of_step_footprint_bytes: int = 0
    input_shape: tuple[int, ...] | None = None
    input_dtype: str | None = None
    wall_start: float = 0.0
    wall_end: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d = {
            "step_index": self.step_index,
            "processor_name": self.processor_name,
            "wall_time_s": self.wall_time_s,
            "cpu_time_s": self.cpu_time_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "gpu_used": self.gpu_used,
            "status": self.status,
            "error_message": self.error_message,
            "concurrent": self.concurrent,
            "peak_overhead_bytes": self.peak_overhead_bytes,
            "end_of_step_footprint_bytes": self.end_of_step_footprint_bytes,
        }
        if self.step_id is not None:
            d["step_id"] = self.step_id
        if self.gpu_memory_bytes is not None:
            d["gpu_memory_bytes"] = self.gpu_memory_bytes
        if self.global_pass_duration is not None:
            d["global_pass_duration"] = self.global_pass_duration
        if self.global_pass_memory is not None:
            d["global_pass_memory"] = self.global_pass_memory
        if self.input_shape is not None:
            d["input_shape"] = list(self.input_shape)
        if self.input_dtype is not None:
            d["input_dtype"] = self.input_dtype
        if self.wall_start > 0.0:
            d["wall_start"] = self.wall_start
        if self.wall_end > 0.0:
            d["wall_end"] = self.wall_end
        return d


@dataclass
class WorkflowMetrics:
    """Aggregate metrics for a complete workflow execution.

    Memory is measured via a ``MemorySampler`` background thread that
    polls ``tracemalloc.get_traced_memory()`` at 1 ms intervals and
    subtracts its own storage overhead from every reading.  The
    resulting ``memory_timeline`` is a continuous time-series of
    corrected memory values that can be overlaid with step
    ``wall_start`` / ``wall_end`` timestamps for visualization.

    Attributes
    ----------
    workflow_id : str
        Identifier for the workflow definition (e.g., ``name:version``).
    run_id : str
        Unique run identifier (UUID4).
    workflow_name : str
        Human-readable workflow name.
    workflow_version : str
        Semantic version of the workflow.
    total_wall_time_s : float
        Total wall-clock duration in seconds.
    total_cpu_time_s : float
        Total CPU time in seconds.
    peak_rss_bytes : int
        Peak memory usage in bytes across the entire run, derived from
        the ``memory_timeline``.
    step_metrics : List[StepMetrics]
        Per-step metrics, one entry per processing step.
    started_at : str
        ISO 8601 UTC timestamp when execution started.
    completed_at : str
        ISO 8601 UTC timestamp when execution completed.
    status : str
        ``"success"``, ``"failed"``, or ``"cancelled"``.
    error_message : Optional[str]
        If status is ``"failed"``, the exception message.
    hardware : dict, optional
        Serialized hardware snapshot captured at execution start (cpu_count,
        total_memory_bytes, gpu_available, gpu_devices, gpu_memory_bytes,
        platform_info, python_version, hostname, captured_at).  ``None``
        for runs produced before this field was added.
    step_depends_on : dict, optional
        Maps each step_id to its list of dependency step_ids.  Populated
        for DAG (``WorkflowExecutor``) workflows where
        ``ProcessingStep.depends_on`` is set.  ``None`` for linear pipelines
        (``Workflow`` builder) and for runs produced before this field was added.
    memory_timeline : MemoryTimeline, optional
        Continuous time-series of corrected ``tracemalloc`` memory
        readings sampled at 1 ms intervals over the pipeline lifetime.
        Correlate with each step's ``wall_start`` / ``wall_end`` for
        per-step memory visualization.  ``None`` for runs produced
        before this field was added.
    """

    workflow_id: str
    run_id: str
    workflow_name: str
    workflow_version: str
    total_wall_time_s: float
    total_cpu_time_s: float
    peak_rss_bytes: int
    step_metrics: list[StepMetrics] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    status: str = "success"
    error_message: str | None = None
    hardware: dict[str, Any] | None = None
    step_depends_on: dict[str, list[str]] | None = None
    memory_timeline: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "total_wall_time_s": self.total_wall_time_s,
            "total_cpu_time_s": self.total_cpu_time_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "step_metrics": [s.to_dict() for s in self.step_metrics],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "error_message": self.error_message,
        }
        if self.hardware is not None:
            d["hardware"] = self.hardware
        if self.step_depends_on is not None:
            d["step_depends_on"] = self.step_depends_on
        if self.memory_timeline is not None:
            d["memory_timeline"] = self.memory_timeline.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level.

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), indent=indent)
