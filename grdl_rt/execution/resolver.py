# -*- coding: utf-8 -*-
"""
Execution Resolver — Plans execution paths through a workflow DAG.

The ``Resolver`` takes a ``WorkflowDefinition`` and a ``HardwareContext``,
then produces a ``ResolvedExecutionPlan`` that accounts for hardware
capabilities, processor alternatives, resource constraints, and global
pass requirements.  This is the "brain" of the executor — it plans
before executing.

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.execution.context import get_logger
from grdl_rt.execution.discovery import (
    get_gpu_capability,
    has_global_pass,
    resolve_processor_class,
)
from grdl_rt.execution.errors import ResolutionError
from grdl_rt.execution.hardware import HardwareContext
from grdl_rt.execution.plan import (
    ParallelGroup,
    ResolvedExecutionPlan,
    ResolvedStep,
    Substitution,
)
from grdl_rt.execution.workflow import (
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
)

logger = get_logger(__name__)

# Default multiplier for memory estimation heuristic
_DEFAULT_MEMORY_MULTIPLIER = 3.0


class Resolver:
    """Plans execution paths through a workflow DAG.

    Takes a catalog for processor resolution and alternative lookup,
    and produces a ``ResolvedExecutionPlan`` for a given workflow and
    hardware context.

    Parameters
    ----------
    catalog : ArtifactCatalogBase
        Catalog for processor resolution and alternative lookup.
    memory_multiplier : float
        Heuristic multiplier for per-step memory estimation.
        Applied as ``estimated_bytes = input_size * multiplier``.
    """

    def __init__(
        self,
        catalog: ArtifactCatalogBase,
        memory_multiplier: float = _DEFAULT_MEMORY_MULTIPLIER,
    ) -> None:
        self._catalog = catalog
        self._memory_multiplier = memory_multiplier

    def resolve(
        self,
        workflow: WorkflowDefinition,
        hardware: HardwareContext,
    ) -> ResolvedExecutionPlan:
        """Resolve a workflow against a hardware context.

        Parameters
        ----------
        workflow : WorkflowDefinition
            The workflow to resolve.
        hardware : HardwareContext
            Available hardware resources.

        Returns
        -------
        ResolvedExecutionPlan
            The resolved execution plan.

        Raises
        ------
        ResolutionError
            If a step cannot be resolved and no alternative exists.
        ValueError
            If the workflow DAG is invalid.
        """
        # Validate DAG first
        dag_errors = workflow.validate_dag()
        if dag_errors:
            raise ValueError(
                f"Invalid workflow DAG: {'; '.join(dag_errors)}"
            )

        levels = workflow.topological_sort()
        hw_dict = hardware.to_dict()

        resolved_steps: Dict[str, ResolvedStep] = {}
        substitutions: List[Substitution] = []
        global_pass_steps: List[str] = []
        warnings: List[str] = []

        for step in workflow.steps:
            if isinstance(step, TapOutStepDef):
                # Tap-out steps pass through without resolution
                continue

            assert isinstance(step, ProcessingStep)
            resolved = self._resolve_step(
                step, hardware, substitutions, warnings,
            )
            resolved_steps[step.id] = resolved

            if resolved.requires_global_pass:
                global_pass_steps.append(step.id)

        # Build parallel groups from topological levels
        parallel_groups: List[ParallelGroup] = []
        for level_idx, level_ids in enumerate(levels):
            group_memory = sum(
                resolved_steps[sid].estimated_memory_bytes
                for sid in level_ids
                if sid in resolved_steps
            )
            parallel_groups.append(ParallelGroup(
                level=level_idx,
                step_ids=level_ids,
                estimated_peak_memory_bytes=group_memory,
            ))

        # Peak memory is the max across parallel groups
        estimated_total = max(
            (g.estimated_peak_memory_bytes for g in parallel_groups),
            default=0,
        )

        plan = ResolvedExecutionPlan(
            workflow_name=workflow.name,
            workflow_version=workflow.version,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            hardware_context=hw_dict,
            steps=resolved_steps,
            parallel_groups=parallel_groups,
            global_pass_steps=global_pass_steps,
            substitutions=substitutions,
            estimated_total_memory_bytes=estimated_total,
            warnings=warnings,
        )

        logger.info(
            "workflow_resolved",
            workflow=workflow.name,
            steps=len(resolved_steps),
            substitutions=len(substitutions),
            warnings=len(warnings),
        )

        return plan

    def _resolve_step(
        self,
        step: ProcessingStep,
        hardware: HardwareContext,
        substitutions: List[Substitution],
        warnings: List[str],
    ) -> ResolvedStep:
        """Resolve a single processing step.

        Parameters
        ----------
        step : ProcessingStep
            The step to resolve.
        hardware : HardwareContext
            Available hardware.
        substitutions : List[Substitution]
            Accumulator for substitutions (mutated).
        warnings : List[str]
            Accumulator for warnings (mutated).

        Returns
        -------
        ResolvedStep

        Raises
        ------
        ResolutionError
            If the processor cannot be resolved.
        """
        # Resolve the processor class
        try:
            processor_cls = resolve_processor_class(step.processor_name)
        except ImportError as exc:
            raise ResolutionError(
                step.id, step.processor_name,
                f"Processor not found: {exc}",
            ) from exc

        gpu_cap = get_gpu_capability(processor_cls)
        fqn = f"{processor_cls.__module__}.{processor_cls.__qualname__}"
        global_pass = has_global_pass(processor_cls)

        resolved_name = step.processor_name
        substitution_reason: Optional[str] = None

        # Check GPU compatibility
        if gpu_cap == "required" and not hardware.gpu_available:
            # Need an alternative
            alt_result = self._find_alternative(
                step, hardware, warnings,
            )
            if alt_result is not None:
                alt_cls, alt_name, alt_fqn, alt_gpu_cap = alt_result
                substitution_reason = (
                    f"GPU required but not available; "
                    f"substituted with '{alt_name}'"
                )
                substitutions.append(Substitution(
                    step_id=step.id,
                    original_processor=step.processor_name,
                    replacement_processor=alt_name,
                    reason=substitution_reason,
                ))
                processor_cls = alt_cls
                resolved_name = alt_name
                fqn = alt_fqn
                gpu_cap = alt_gpu_cap
                global_pass = has_global_pass(alt_cls)
            else:
                raise ResolutionError(
                    step.id, step.processor_name,
                    "GPU required but not available, and no "
                    "compatible alternative exists in the catalog",
                )

        elif gpu_cap == "preferred" and not hardware.gpu_available:
            warnings.append(
                f"Step '{step.id}': processor '{step.processor_name}' "
                f"prefers GPU but none available; will run on CPU"
            )

        will_use_gpu = hardware.gpu_available and gpu_cap != "cpu_only"

        # Memory estimation heuristic
        estimated_memory = int(self._memory_multiplier * 1024 * 1024)  # 3 MB default

        retry_dict = step.retry.to_dict() if step.retry else None

        return ResolvedStep(
            step_id=step.id,
            original_processor=step.processor_name,
            resolved_processor=resolved_name,
            processor_class_fqn=fqn,
            params=dict(step.params),
            gpu_capability=gpu_cap,
            will_use_gpu=will_use_gpu,
            requires_global_pass=global_pass,
            substitution_reason=substitution_reason,
            estimated_memory_bytes=estimated_memory,
            retry=retry_dict,
            timeout_seconds=step.timeout_seconds,
            depends_on=list(step.depends_on),
        )

    def _find_alternative(
        self,
        step: ProcessingStep,
        hardware: HardwareContext,
        warnings: List[str],
    ) -> Optional[tuple]:
        """Find a compatible alternative for a step.

        Queries the catalog for alternatives to the step's processor,
        sorted by priority, and returns the first one whose GPU
        capability is compatible with the hardware.

        Returns
        -------
        Optional[tuple]
            ``(processor_cls, name, fqn, gpu_capability)`` or ``None``.
        """
        # Try to find alternatives from the catalog by searching
        # all versions of the processor artifact
        alternatives = self._get_alternatives_for_processor(
            step.processor_name,
        )

        # Sort by priority (lower = higher priority)
        alternatives.sort(key=lambda a: a.get("priority", 0))

        for alt in alternatives:
            alt_name = alt.get("processor_name", "")
            if not alt_name:
                continue

            try:
                alt_cls = resolve_processor_class(alt_name)
            except ImportError:
                warnings.append(
                    f"Step '{step.id}': alternative '{alt_name}' "
                    f"could not be resolved; skipping"
                )
                continue

            alt_gpu_cap = get_gpu_capability(alt_cls)

            # Check compatibility: alternative must not require GPU
            # if GPU is not available
            if alt_gpu_cap == "required" and not hardware.gpu_available:
                continue

            alt_fqn = f"{alt_cls.__module__}.{alt_cls.__qualname__}"
            return alt_cls, alt_name, alt_fqn, alt_gpu_cap

        return None

    def _get_alternatives_for_processor(
        self, processor_name: str,
    ) -> List[Dict[str, Any]]:
        """Get alternatives for a processor from the catalog.

        Searches the catalog for artifacts matching the processor name
        and returns their alternatives list.
        """
        # Search catalog for artifacts matching this processor name
        for artifact in self._catalog.list_artifacts(
            artifact_type="grdl_processor",
        ):
            if artifact.name == processor_name:
                return list(artifact.alternatives)
            # Also check by processor class short name
            if artifact.processor_class:
                short = artifact.processor_class.rsplit(".", 1)[-1]
                if short == processor_name:
                    return list(artifact.alternatives)

        # Try get_alternatives method directly
        try:
            return self._catalog.get_alternatives(processor_name, "")
        except Exception:
            return []
