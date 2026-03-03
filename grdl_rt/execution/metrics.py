"""
Execution Metrics — Structured timing and resource usage data.

Provides ``StepMetrics`` and ``WorkflowMetrics`` dataclasses that capture
wall-clock time, CPU time, peak memory, and GPU usage for every step and
the overall workflow execution.  These are returned as part of
``WorkflowResult`` from all execution paths.

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
        Peak memory delta in bytes (``tracemalloc``).
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
        parallel DAG level.  When ``True``, ``peak_rss_bytes``
        reflects the shared level-wide peak (not isolated to this
        step) because per-thread memory isolation is not possible
        with ``tracemalloc``.
    peak_overhead_bytes : int
        Extra RAM the step allocated and then released during its
        execution.  Computed as the historical peak captured
        immediately after the step minus the live allocation
        measured after intermediate results are freed.  This is
        "The Spike" shown in the memory profile table.
    end_of_step_footprint_bytes : int
        Total live RAM occupied by workflow data at the moment this
        step completed.  This is the "End-of-Step Footprint (Live)"
        column — the running total that accumulates as the workflow
        produces outputs.
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
        return d


@dataclass
class WorkflowMetrics:
    """Aggregate metrics for a complete workflow execution.

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
        Peak memory usage in bytes across the entire run.
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {
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
