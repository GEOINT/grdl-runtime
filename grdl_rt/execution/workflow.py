# -*- coding: utf-8 -*-
"""
Workflow Models - Data models for image processing workflow definitions.

A workflow is a DAG (directed acyclic graph) of processing steps, each
referencing a GRDL image processor class with specific tunable parameter
values.  Linear pipelines are a degenerate case where each step depends
on its predecessor.  Workflows carry tags, execution phase annotations,
and can be serialized to Python DSL or YAML.

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
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    from grdl_rt.catalog.base import ArtifactCatalogBase
    from grdl_rt.execution.validation import ValidationError

# grdl-runtime internal
from grdl_rt.execution.resilience import RetryPolicy
from grdl_rt.execution.tags import WorkflowTags


SCHEMA_VERSION = "2.0"


class WorkflowState(Enum):
    """State of a workflow within a project."""

    DRAFT = "draft"
    TESTING = "testing"
    PUBLISHED = "published"


class ExecutionPhase(Enum):
    """Canonical execution phases that constrain step ordering.

    Processors declare their phase via a ``__phase__`` attribute.
    Phase annotations are validated during workflow validation —
    out-of-phase steps produce a warning but do not block execution.
    """

    IO = "io"
    GLOBAL_PROCESSING = "global_processing"
    DATA_PREP = "data_prep"
    TILING = "tiling"
    TILE_PROCESSING = "tile_processing"
    EXTRACTION = "extraction"
    VECTOR_PROCESSING = "vector_processing"
    FINALIZATION = "finalization"


# Canonical ordering for phase validation (lower index = earlier phase)
PHASE_ORDER = {phase: i for i, phase in enumerate(ExecutionPhase)}


class ProcessingStep:
    """A single step in a processing workflow.

    Represents a GRDL image processor with a specific version and
    a set of tunable parameter values, plus optional resilience config,
    DAG wiring, conditional execution, and phase annotation.

    Parameters
    ----------
    processor_name : str
        Fully-qualified class name or short name of the processor.
    processor_version : str
        Semantic version of the processor (from @processor_version).
    params : Dict[str, Any]
        Tunable parameter values for this step.
    retry : RetryPolicy, optional
        Per-step retry configuration.
    timeout_seconds : float, optional
        Maximum wall-clock seconds for this step.
    id : str, optional
        Unique step identifier within the workflow.  Auto-assigned
        if not provided.
    depends_on : List[str], optional
        Step IDs this step depends on.  For linear workflows these
        are inferred automatically.
    condition : str, optional
        Expression evaluated at runtime to decide whether to execute
        this step (e.g., ``"metadata.band_count > 1"``).  If the
        expression evaluates to ``False``, the step is skipped and
        its input is propagated unchanged.
    phase : str, optional
        Execution phase annotation (one of :class:`ExecutionPhase`
        values).  Used for validation warnings.
    """

    def __init__(
        self,
        processor_name: str,
        processor_version: str = "",
        params: Optional[Dict[str, Any]] = None,
        *,
        retry: Optional[RetryPolicy] = None,
        timeout_seconds: Optional[float] = None,
        id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        condition: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        self.processor_name = processor_name
        self.processor_version = processor_version
        self.params = params or {}
        self.retry = retry
        self.timeout_seconds = timeout_seconds
        self.id = id
        self.depends_on: List[str] = list(depends_on) if depends_on else []
        self.condition = condition
        self.phase = phase

    def to_dict(self) -> dict:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        d: dict = {
            'processor': self.processor_name,
            'version': self.processor_version,
        }
        if self.id is not None:
            d['id'] = self.id
        if self.params:
            d['params'] = self.params
        if self.depends_on:
            d['depends_on'] = list(self.depends_on)
        if self.condition is not None:
            d['condition'] = self.condition
        if self.phase is not None:
            d['phase'] = self.phase
        if self.retry is not None and self.retry.max_retries > 0:
            d['retry'] = self.retry.to_dict()
        if self.timeout_seconds is not None:
            d['timeout_seconds'] = self.timeout_seconds
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingStep':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        ProcessingStep
        """
        retry = None
        if 'retry' in data:
            retry = RetryPolicy.from_dict(data['retry'])
        return cls(
            processor_name=data['processor'],
            processor_version=data.get('version', ''),
            params=data.get('params', {}),
            retry=retry,
            timeout_seconds=data.get('timeout_seconds'),
            id=data.get('id'),
            depends_on=data.get('depends_on'),
            condition=data.get('condition'),
            phase=data.get('phase'),
        )


class TapOutStepDef:
    """A tap-out step in a serializable workflow definition.

    Writes the current intermediate to disk without modifying it.
    Used in ``WorkflowDefinition`` for YAML/dict serialization.

    Parameters
    ----------
    path : str
        Output file path.
    format : str, optional
        Writer format override.  If ``None``, auto-detected from
        the file extension.
    id : str, optional
        Unique step identifier within the workflow.
    depends_on : List[str], optional
        Step IDs this tap-out depends on.
    """

    def __init__(
        self,
        path: str,
        format: Optional[str] = None,
        *,
        id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> None:
        self.path = path
        self.format = format
        self.id = id
        self.depends_on: List[str] = list(depends_on) if depends_on else []

    def to_dict(self) -> dict:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        d: dict = {'tap_out': self.path}
        if self.id is not None:
            d['id'] = self.id
        if self.format:
            d['format'] = self.format
        if self.depends_on:
            d['depends_on'] = list(self.depends_on)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'TapOutStepDef':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        TapOutStepDef
        """
        return cls(
            path=data['tap_out'],
            format=data.get('format'),
            id=data.get('id'),
            depends_on=data.get('depends_on'),
        )


class WorkflowDefinition:
    """A complete image processing workflow definition.

    Steps form a directed acyclic graph (DAG) via ``depends_on``
    references.  Linear pipelines are represented as a chain where
    each step depends on its predecessor (inferred automatically
    when no explicit ``depends_on`` is provided).

    Parameters
    ----------
    name : str
        Human-readable workflow name.
    version : str
        Semantic version of the workflow definition.
    description : str
        Description of what the workflow does.
    steps : List[Union[ProcessingStep, TapOutStepDef]]
        Processing and tap-out steps.
    tags : Optional[WorkflowTags]
        Workflow classification tags.
    state : WorkflowState
        Current state of the workflow.
    schema_version : str
        Schema version for serialization migration.
    """

    def __init__(
        self,
        name: str,
        version: str = "0.1.0",
        description: str = "",
        steps: Optional[List[Union[ProcessingStep, TapOutStepDef]]] = None,
        tags: Optional[WorkflowTags] = None,
        state: WorkflowState = WorkflowState.DRAFT,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.steps = steps or []
        self.tags = tags or WorkflowTags()
        self.state = state
        self.schema_version = schema_version
        self._ensure_dag_consistency()

    def _ensure_dag_consistency(self) -> None:
        """Assign step IDs and infer linear deps if needed."""
        self._assign_step_ids()
        self._infer_linear_deps()

    def _assign_step_ids(self) -> None:
        """Auto-assign IDs to steps that don't have one."""
        for i, step in enumerate(self.steps):
            if step.id is None:
                step.id = f"step_{i}"

    def _infer_linear_deps(self) -> None:
        """If no step has explicit depends_on, chain them linearly.

        This preserves backward compatibility: old-style workflows
        with no depends_on are treated as linear pipelines where
        step N depends on step N-1.
        """
        has_any_deps = any(step.depends_on for step in self.steps)
        if has_any_deps:
            return
        for i in range(1, len(self.steps)):
            self.steps[i].depends_on = [self.steps[i - 1].id]  # type: ignore[union-attr]

    def add_step(self, step: Union[ProcessingStep, TapOutStepDef]) -> None:
        """Append a processing or tap-out step to the workflow.

        Parameters
        ----------
        step : ProcessingStep or TapOutStepDef
        """
        self.steps.append(step)
        self._ensure_dag_consistency()

    def remove_step(self, index: int) -> Union[ProcessingStep, TapOutStepDef]:
        """Remove and return a step by index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ProcessingStep or TapOutStepDef
            The removed step.
        """
        removed = self.steps.pop(index)
        self._ensure_dag_consistency()
        return removed

    def move_step(self, from_index: int, to_index: int) -> None:
        """Move a processing step from one position to another.

        Parameters
        ----------
        from_index : int
            Current position.
        to_index : int
            Target position.
        """
        step = self.steps.pop(from_index)
        self.steps.insert(to_index, step)

    def get_step(self, step_id: str) -> Union[ProcessingStep, TapOutStepDef]:
        """Look up a step by its ID.

        Parameters
        ----------
        step_id : str

        Returns
        -------
        ProcessingStep or TapOutStepDef

        Raises
        ------
        KeyError
            If no step with the given ID exists.
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"No step with id '{step_id}'")

    def terminal_step_ids(self) -> List[str]:
        """Return IDs of steps that have no dependents (DAG sinks).

        Returns
        -------
        List[str]
        """
        all_ids = {s.id for s in self.steps}
        depended_on: Set[str] = set()
        for step in self.steps:
            for dep in step.depends_on:
                depended_on.add(dep)
        return [sid for sid in all_ids if sid not in depended_on]

    def topological_sort(self) -> List[List[str]]:
        """Topological sort returning parallelizable levels.

        Each level is a list of step IDs whose dependencies are all
        satisfied by previous levels.  Steps within a level can
        execute in parallel.

        Returns
        -------
        List[List[str]]
            Levels from root to terminal.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        # Build adjacency and in-degree maps
        step_ids = [s.id for s in self.steps]
        in_degree: Dict[str, int] = {sid: 0 for sid in step_ids}
        dependents: Dict[str, List[str]] = {sid: [] for sid in step_ids}

        for step in self.steps:
            for dep in step.depends_on:
                if dep in dependents:
                    dependents[dep].append(step.id)
                    in_degree[step.id] += 1

        # Kahn's algorithm with level tracking
        queue: deque = deque()
        for sid in step_ids:
            if in_degree[sid] == 0:
                queue.append(sid)

        levels: List[List[str]] = []
        visited = 0

        while queue:
            level = list(queue)
            queue.clear()
            levels.append(level)
            visited += len(level)

            for sid in level:
                for dep_id in dependents[sid]:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        queue.append(dep_id)

        if visited != len(step_ids):
            raise ValueError(
                "Workflow DAG contains a cycle — cannot determine execution order"
            )

        return levels

    def validate_dag(self) -> List[str]:
        """Validate DAG structure.

        Checks:
        1. No duplicate step IDs.
        2. All depends_on references resolve to existing step IDs.
        3. Graph is acyclic.

        Returns
        -------
        List[str]
            List of error messages.  Empty if valid.
        """
        errors: List[str] = []

        # Check duplicate IDs
        seen_ids: Dict[str, int] = {}
        for i, step in enumerate(self.steps):
            if step.id in seen_ids:
                errors.append(
                    f"Duplicate step id '{step.id}' at indices "
                    f"{seen_ids[step.id]} and {i}"
                )
            else:
                seen_ids[step.id] = i

        # Check all depends_on references resolve
        all_ids = {s.id for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in all_ids:
                    errors.append(
                        f"Step '{step.id}' depends on '{dep}' which "
                        f"does not exist"
                    )

        # Check acyclic
        if not errors:
            try:
                self.topological_sort()
            except ValueError as e:
                errors.append(str(e))

        return errors

    def validate(
        self,
        catalog: Optional['ArtifactCatalogBase'] = None,
    ) -> List['ValidationError']:
        """Validate this workflow definition before execution.

        Checks that all processors can be resolved, required parameters
        are present, parameter values satisfy type and constraint
        checks, and the DAG structure is valid.

        Parameters
        ----------
        catalog : ArtifactCatalogBase, optional
            Catalog for processor resolution.  If ``None``, uses the
            default discovery catalog.

        Returns
        -------
        List[ValidationError]
            Empty list if valid; otherwise, list of problems found.
        """
        from grdl_rt.execution.validation import validate_workflow
        return validate_workflow(self, catalog=catalog)

    def to_dict(self) -> dict:
        """Serialize to dictionary for YAML/JSON storage.

        Returns
        -------
        dict
        """
        return {
            'schema_version': self.schema_version,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'state': self.state.value,
            'tags': self.tags.to_dict(),
            'steps': [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorkflowDefinition':
        """Deserialize from dictionary.

        Handles both v1 (no schema_version) and v2 formats.

        Parameters
        ----------
        data : dict

        Returns
        -------
        WorkflowDefinition
        """
        schema_version = data.get('schema_version', '1.0')
        tags = WorkflowTags.from_dict(data.get('tags', {}))
        raw_steps = data.get('steps', [])
        steps: List[Union[ProcessingStep, TapOutStepDef]] = []
        for s in raw_steps:
            if 'tap_out' in s:
                steps.append(TapOutStepDef.from_dict(s))
            else:
                steps.append(ProcessingStep.from_dict(s))
        return cls(
            name=data['name'],
            version=data.get('version', '0.1.0'),
            description=data.get('description', ''),
            steps=steps,
            tags=tags,
            state=WorkflowState(data.get('state', 'draft')),
            schema_version=schema_version,
        )
