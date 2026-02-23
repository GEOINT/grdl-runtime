"""
Execution Plan Models — Data models for resolved execution plans.

Provides ``ResolvedExecutionPlan``, ``ResolvedStep``, ``ParallelGroup``,
``Substitution``, ``AsExecutedManifest``, and ``ExecutedStepRecord`` —
the data structures produced by the resolver and consumed by the executor
to write ``as_planned.json`` and ``as_executed.json``.

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
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResolvedStep:
    """A single step in a resolved execution plan.

    Attributes
    ----------
    step_id : str
        Unique step identifier within the workflow.
    original_processor : str
        Processor name as specified in the workflow definition.
    resolved_processor : str
        Processor name after resolution (may differ if substituted).
    processor_class_fqn : str
        Fully-qualified class name of the resolved processor.
    params : Dict[str, Any]
        Tunable parameter values for this step.
    gpu_capability : str
        One of ``"required"``, ``"preferred"``, ``"cpu_only"``.
    will_use_gpu : bool
        Whether this step is expected to use GPU.
    requires_global_pass : bool
        Whether this processor requires a global pass.
    substitution_reason : Optional[str]
        If the processor was substituted, why.
    estimated_memory_bytes : int
        Heuristic memory estimate for this step.
    retry : Optional[Dict[str, Any]]
        Serialized retry policy, or None.
    timeout_seconds : Optional[float]
        Per-step timeout, or None.
    depends_on : List[str]
        Step IDs this step depends on.
    """

    step_id: str
    original_processor: str
    resolved_processor: str
    processor_class_fqn: str
    params: dict[str, Any]
    gpu_capability: str
    will_use_gpu: bool
    requires_global_pass: bool
    substitution_reason: str | None
    estimated_memory_bytes: int
    retry: dict[str, Any] | None
    timeout_seconds: float | None
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "original_processor": self.original_processor,
            "resolved_processor": self.resolved_processor,
            "processor_class_fqn": self.processor_class_fqn,
            "params": self.params,
            "gpu_capability": self.gpu_capability,
            "will_use_gpu": self.will_use_gpu,
            "requires_global_pass": self.requires_global_pass,
            "substitution_reason": self.substitution_reason,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "retry": self.retry,
            "timeout_seconds": self.timeout_seconds,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedStep":
        """Deserialize from dictionary."""
        return cls(
            step_id=data["step_id"],
            original_processor=data["original_processor"],
            resolved_processor=data["resolved_processor"],
            processor_class_fqn=data["processor_class_fqn"],
            params=data.get("params", {}),
            gpu_capability=data["gpu_capability"],
            will_use_gpu=data["will_use_gpu"],
            requires_global_pass=data["requires_global_pass"],
            substitution_reason=data.get("substitution_reason"),
            estimated_memory_bytes=data.get("estimated_memory_bytes", 0),
            retry=data.get("retry"),
            timeout_seconds=data.get("timeout_seconds"),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class ParallelGroup:
    """A set of steps that can execute concurrently.

    Attributes
    ----------
    level : int
        Zero-based topological level index.
    step_ids : List[str]
        Step IDs in this parallel group.
    estimated_peak_memory_bytes : int
        Sum of per-step memory estimates (worst-case concurrent).
    """

    level: int
    step_ids: list[str]
    estimated_peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "level": self.level,
            "step_ids": self.step_ids,
            "estimated_peak_memory_bytes": self.estimated_peak_memory_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelGroup":
        """Deserialize from dictionary."""
        return cls(
            level=data["level"],
            step_ids=data["step_ids"],
            estimated_peak_memory_bytes=data.get("estimated_peak_memory_bytes", 0),
        )


@dataclass
class Substitution:
    """Record of a processor substitution during resolution.

    Attributes
    ----------
    step_id : str
        Step that was substituted.
    original_processor : str
        Original processor name.
    replacement_processor : str
        Replacement processor name.
    reason : str
        Human-readable reason for the substitution.
    """

    step_id: str
    original_processor: str
    replacement_processor: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "original_processor": self.original_processor,
            "replacement_processor": self.replacement_processor,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Substitution":
        """Deserialize from dictionary."""
        return cls(
            step_id=data["step_id"],
            original_processor=data["original_processor"],
            replacement_processor=data["replacement_processor"],
            reason=data["reason"],
        )


@dataclass
class ResolvedExecutionPlan:
    """The complete output of the resolver.

    Serialized to ``as_planned.json`` before execution begins.

    Attributes
    ----------
    workflow_name : str
        Name of the workflow being resolved.
    workflow_version : str
        Version of the workflow being resolved.
    resolved_at : str
        ISO 8601 timestamp when resolution completed.
    hardware_context : Dict[str, Any]
        Serialized hardware context snapshot.
    steps : Dict[str, ResolvedStep]
        Resolved steps keyed by step ID.
    parallel_groups : List[ParallelGroup]
        Execution levels from topological sort.
    global_pass_steps : List[str]
        Step IDs requiring a global pass.
    substitutions : List[Substitution]
        Processor substitutions made during resolution.
    estimated_total_memory_bytes : int
        Peak estimated memory across all parallel groups.
    warnings : List[str]
        Non-fatal resolution warnings.
    """

    workflow_name: str
    workflow_version: str
    resolved_at: str
    hardware_context: dict[str, Any]
    steps: dict[str, ResolvedStep]
    parallel_groups: list[ParallelGroup]
    global_pass_steps: list[str]
    substitutions: list[Substitution]
    estimated_total_memory_bytes: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "resolved_at": self.resolved_at,
            "hardware_context": self.hardware_context,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "parallel_groups": [g.to_dict() for g in self.parallel_groups],
            "global_pass_steps": self.global_pass_steps,
            "substitutions": [s.to_dict() for s in self.substitutions],
            "estimated_total_memory_bytes": self.estimated_total_memory_bytes,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedExecutionPlan":
        """Deserialize from dictionary."""
        return cls(
            workflow_name=data["workflow_name"],
            workflow_version=data["workflow_version"],
            resolved_at=data["resolved_at"],
            hardware_context=data["hardware_context"],
            steps={k: ResolvedStep.from_dict(v) for k, v in data["steps"].items()},
            parallel_groups=[ParallelGroup.from_dict(g) for g in data["parallel_groups"]],
            global_pass_steps=data["global_pass_steps"],
            substitutions=[Substitution.from_dict(s) for s in data["substitutions"]],
            estimated_total_memory_bytes=data["estimated_total_memory_bytes"],
            warnings=data.get("warnings", []),
        )


# ---------------------------------------------------------------------------
# As-Executed artifacts
# ---------------------------------------------------------------------------


@dataclass
class ExecutedStepRecord:
    """Record of a single step's execution.

    Attributes
    ----------
    step_id : str
        Step identifier.
    processor_name : str
        Processor that actually ran (may differ from plan if fallback).
    status : str
        One of ``"success"``, ``"failed"``, ``"skipped"``, ``"fallback"``.
    wall_time_s : float
        Wall-clock duration in seconds.
    cpu_time_s : float
        CPU time in seconds.
    peak_rss_bytes : int
        Peak memory usage in bytes.
    gpu_used : bool
        Whether GPU was used for this step.
    fallback_processor : Optional[str]
        If a fallback was used, the fallback processor name.
    fallback_reason : Optional[str]
        If a fallback was used, why.
    error_message : Optional[str]
        If the step failed, the error message.
    """

    step_id: str
    processor_name: str
    status: str
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    peak_rss_bytes: int = 0
    gpu_used: bool = False
    fallback_processor: str | None = None
    fallback_reason: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {
            "step_id": self.step_id,
            "processor_name": self.processor_name,
            "status": self.status,
            "wall_time_s": self.wall_time_s,
            "cpu_time_s": self.cpu_time_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "gpu_used": self.gpu_used,
        }
        if self.fallback_processor is not None:
            d["fallback_processor"] = self.fallback_processor
            d["fallback_reason"] = self.fallback_reason
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d


@dataclass
class AsExecutedManifest:
    """Complete execution record written as ``as_executed.json``.

    Attributes
    ----------
    workflow_name : str
    workflow_version : str
    run_id : str
    started_at : str
    completed_at : str
    status : str
        ``"success"``, ``"failed"``, or ``"partial"``.
    hardware_context : Dict[str, Any]
    planned_steps : Dict[str, Any]
        The original resolved plan (snapshot).
    executed_steps : List[ExecutedStepRecord]
    runtime_substitutions : List[Dict[str, Any]]
    total_wall_time_s : float
    total_cpu_time_s : float
    peak_rss_bytes : int
    data_lineage : Optional[Dict[str, Any]]
        Serialized data lineage record, or ``None``.
    """

    workflow_name: str
    workflow_version: str
    run_id: str
    started_at: str
    completed_at: str
    status: str
    hardware_context: dict[str, Any]
    planned_steps: dict[str, Any]
    executed_steps: list[ExecutedStepRecord]
    runtime_substitutions: list[dict[str, Any]]
    total_wall_time_s: float
    total_cpu_time_s: float
    peak_rss_bytes: int
    data_lineage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        d: dict[str, Any] = {
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "hardware_context": self.hardware_context,
            "planned_steps": self.planned_steps,
            "executed_steps": [s.to_dict() for s in self.executed_steps],
            "runtime_substitutions": self.runtime_substitutions,
            "total_wall_time_s": self.total_wall_time_s,
            "total_cpu_time_s": self.total_cpu_time_s,
            "peak_rss_bytes": self.peak_rss_bytes,
        }
        if self.data_lineage is not None:
            d["data_lineage"] = self.data_lineage
        return d
