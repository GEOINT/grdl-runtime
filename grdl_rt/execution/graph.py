"""
Workflow Graph — Introspection and manipulation API for visual workflow builders.

Provides ``WorkflowGraph``, ``NodeInfo``, and ``EdgeInfo`` for programmatic
construction and inspection of DAG-based workflows.  This is the backend
API that a visual workflow builder (e.g., tkgis) uses to add/remove/connect
nodes and validate type compatibility between steps.

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
2026-03-25
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from grdl_rt.execution.workflow import (
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
)

if TYPE_CHECKING:
    from grdl_rt.catalog.base import ArtifactCatalogBase

# Valid data types for type-compatibility checking
VALID_DATA_TYPES = {"raster", "feature_set", "detection_set"}

# Type compatibility matrix: (source_type, target_type) -> compatible
# None means "any" and is always compatible.
# DetectionSet is a specialization of FeatureSet and is compatible.
_COMPATIBLE_PAIRS: set[tuple[str, str]] = {
    ("raster", "raster"),
    ("feature_set", "feature_set"),
    ("detection_set", "detection_set"),
    ("detection_set", "feature_set"),  # detection is a kind of feature
}


def types_compatible(source_type: str | None, target_type: str | None) -> bool:
    """Check whether two data types are compatible for connection.

    Parameters
    ----------
    source_type : str or None
        Output type of the source step.  ``None`` means any.
    target_type : str or None
        Input type of the target step.  ``None`` means any.

    Returns
    -------
    bool
    """
    if source_type is None or target_type is None:
        return True
    return (source_type, target_type) in _COMPATIBLE_PAIRS


@dataclass
class NodeInfo:
    """Metadata about a single node (step) in the workflow graph."""

    step_id: str
    processor_name: str
    processor_version: str | None
    display_name: str
    category: str | None
    input_type: str | None
    output_type: str | None
    output_ports: dict[str, str] | None
    params: dict[str, Any]
    param_specs: dict[str, dict]
    depends_on: list[str]
    phase: str | None
    position: tuple[float, float] | None


@dataclass
class EdgeInfo:
    """Metadata about a single edge (connection) in the workflow graph."""

    source_id: str
    source_port: str | None
    target_id: str
    target_port: str | None
    data_type: str | None


class WorkflowGraph:
    """Introspection and manipulation API for workflow DAGs.

    Wraps a ``WorkflowDefinition`` and provides node/edge CRUD operations,
    type-compatibility validation, and topological analysis.

    Parameters
    ----------
    workflow : WorkflowDefinition
        The underlying workflow definition.
    catalog : ArtifactCatalogBase, optional
        Catalog for processor metadata lookup.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        catalog: ArtifactCatalogBase | None = None,
    ) -> None:
        self._workflow = workflow
        self._catalog = catalog
        self._id_counter = itertools.count(len(workflow.steps))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_nodes(self) -> list[NodeInfo]:
        """Return info for all nodes in the graph.

        Returns
        -------
        list[NodeInfo]
        """
        nodes: list[NodeInfo] = []
        for s in self._workflow.steps:
            if isinstance(s, ProcessingStep):
                nodes.append(self._step_to_node(s))
        return nodes

    def get_edges(self) -> list[EdgeInfo]:
        """Return info for all edges in the graph.

        Returns
        -------
        list[EdgeInfo]
        """
        edges: list[EdgeInfo] = []
        step_map = {s.id: s for s in self._workflow.steps}
        for step in self._workflow.steps:
            if not isinstance(step, ProcessingStep):
                continue
            for dep_id in step.depends_on:
                dep = step_map.get(dep_id)
                source_type = None
                if dep is not None and isinstance(dep, ProcessingStep):
                    source_type = dep.output_type
                edges.append(
                    EdgeInfo(
                        source_id=dep_id,
                        source_port=None,
                        target_id=step.id,  # type: ignore[arg-type]
                        target_port=None,
                        data_type=source_type,
                    )
                )
        return edges

    def get_node(self, step_id: str) -> NodeInfo | None:
        """Return info for a single node, or ``None`` if not found.

        Parameters
        ----------
        step_id : str

        Returns
        -------
        NodeInfo or None
        """
        for step in self._workflow.steps:
            if isinstance(step, ProcessingStep) and step.id == step_id:
                return self._step_to_node(step)
        return None

    # ------------------------------------------------------------------
    # Mutation operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        processor_name: str,
        params: dict[str, Any] | None = None,
        position: tuple[float, float] | None = None,
        *,
        input_type: str | None = None,
        output_type: str | None = None,
    ) -> str:
        """Add a new processing node to the graph.

        Parameters
        ----------
        processor_name : str
            Processor class name.
        params : dict, optional
            Initial parameter values.
        position : tuple, optional
            GUI layout position ``(x, y)``.
        input_type : str, optional
            Expected input data type.
        output_type : str, optional
            Produced output data type.

        Returns
        -------
        str
            Auto-generated step ID.
        """
        step_id = f"step_{next(self._id_counter)}"
        step = ProcessingStep(
            processor_name=processor_name,
            params=params or {},
            id=step_id,
            position=position,
            input_type=input_type,
            output_type=output_type,
        )
        self._workflow.steps.append(step)
        return step_id

    def remove_node(self, step_id: str) -> None:
        """Remove a node and all edges referencing it.

        Parameters
        ----------
        step_id : str

        Raises
        ------
        KeyError
            If no node with the given ID exists.
        """
        found = False
        new_steps = []
        for step in self._workflow.steps:
            if step.id == step_id:
                found = True
                continue
            # Remove references to the deleted node from depends_on
            if hasattr(step, "depends_on"):
                step.depends_on = [d for d in step.depends_on if d != step_id]
            new_steps.append(step)
        if not found:
            raise KeyError(f"No node with id '{step_id}'")
        self._workflow.steps = new_steps

    def connect(
        self,
        source_id: str,
        target_id: str,
        source_port: str | None = None,
        target_port: str | None = None,
    ) -> None:
        """Connect two nodes (add a dependency edge).

        Validates type compatibility between the source output and
        target input before creating the connection.

        Parameters
        ----------
        source_id : str
            ID of the upstream node.
        target_id : str
            ID of the downstream node.
        source_port : str, optional
            Output port name on the source (for multi-output steps).
        target_port : str, optional
            Input port name on the target.

        Raises
        ------
        KeyError
            If either node does not exist.
        ValueError
            If the connection would create a type mismatch.
        """
        source_step = self._find_step(source_id)
        target_step = self._find_step(target_id)

        # Determine source output type
        source_out = None
        if isinstance(source_step, ProcessingStep):
            if source_port and source_step.output_ports:
                source_out = source_step.output_ports.get(source_port, source_step.output_type)
            else:
                source_out = source_step.output_type

        # Determine target input type
        target_in = None
        if isinstance(target_step, ProcessingStep):
            target_in = target_step.input_type

        if not types_compatible(source_out, target_in):
            raise ValueError(
                f"Type mismatch: step '{source_id}' outputs '{source_out}' "
                f"but step '{target_id}' expects '{target_in}'"
            )

        if source_id not in target_step.depends_on:
            target_step.depends_on.append(source_id)

    def disconnect(self, source_id: str, target_id: str) -> None:
        """Remove a dependency edge between two nodes.

        Parameters
        ----------
        source_id : str
        target_id : str

        Raises
        ------
        KeyError
            If the target node does not exist.
        """
        target_step = self._find_step(target_id)
        target_step.depends_on = [d for d in target_step.depends_on if d != source_id]

    def update_node_params(self, step_id: str, params: dict) -> None:
        """Update parameters on an existing node.

        Parameters
        ----------
        step_id : str
        params : dict
            New parameter values (merged with existing).

        Raises
        ------
        KeyError
            If no node with the given ID exists.
        """
        step = self._find_step(step_id)
        if isinstance(step, ProcessingStep):
            step.params.update(params)

    def update_node_position(self, step_id: str, position: tuple[float, float]) -> None:
        """Update the GUI layout position of a node.

        Parameters
        ----------
        step_id : str
        position : tuple[float, float]

        Raises
        ------
        KeyError
            If no node with the given ID exists.
        """
        step = self._find_step(step_id)
        if isinstance(step, ProcessingStep):
            step.position = position

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the graph structure and type compatibility.

        Checks for:
        - Cycles in the DAG
        - Type mismatches between connected steps
        - Missing dependency references

        Returns
        -------
        list[str]
            Error messages.  Empty if valid.
        """
        errors: list[str] = []

        # Delegate structural validation to WorkflowDefinition
        dag_errors = self._workflow.validate_dag()
        errors.extend(dag_errors)

        # Type compatibility checks
        errors.extend(validate_type_compatibility(self._workflow))

        return errors

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def topological_levels(self) -> list[list[str]]:
        """Return topological levels for parallel execution.

        Returns
        -------
        list[list[str]]
            Each inner list contains step IDs that can execute in parallel.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        return self._workflow.topological_sort()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_workflow_definition(self) -> WorkflowDefinition:
        """Return the underlying WorkflowDefinition.

        Returns
        -------
        WorkflowDefinition
        """
        return self._workflow

    @classmethod
    def from_workflow_definition(
        cls,
        workflow: WorkflowDefinition,
        catalog: ArtifactCatalogBase | None = None,
    ) -> WorkflowGraph:
        """Create a WorkflowGraph from an existing WorkflowDefinition.

        Parameters
        ----------
        workflow : WorkflowDefinition
        catalog : ArtifactCatalogBase, optional

        Returns
        -------
        WorkflowGraph
        """
        return cls(workflow, catalog=catalog)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_step(self, step_id: str) -> ProcessingStep | TapOutStepDef:
        """Look up a step by ID.

        Raises
        ------
        KeyError
            If not found.
        """
        for step in self._workflow.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"No step with id '{step_id}'")

    def _step_to_node(self, step: ProcessingStep) -> NodeInfo:
        """Convert a ProcessingStep to a NodeInfo."""
        # Try to get param_specs from the catalog or the processor class
        param_specs: dict[str, dict] = {}
        category: str | None = None

        if self._catalog is not None:
            try:
                artifact = self._catalog.get_artifact(
                    step.processor_name, step.processor_version or ""
                )
                if artifact is not None:
                    category = getattr(artifact, "processor_type", None)
                    if artifact.param_schema and "properties" in artifact.param_schema:
                        param_specs = artifact.param_schema["properties"]
            except Exception:
                pass

        return NodeInfo(
            step_id=step.id or "",
            processor_name=step.processor_name,
            processor_version=step.processor_version or None,
            display_name=step.processor_name.rsplit(".", 1)[-1],
            category=category,
            input_type=step.input_type,
            output_type=step.output_type,
            output_ports=step.output_ports,
            params=dict(step.params),
            param_specs=param_specs,
            depends_on=list(step.depends_on),
            phase=step.phase,
            position=step.position,
        )


def validate_type_compatibility(workflow: WorkflowDefinition) -> list[str]:
    """Check type compatibility between connected steps.

    Parameters
    ----------
    workflow : WorkflowDefinition

    Returns
    -------
    list[str]
        Error messages for type mismatches.
    """
    errors: list[str] = []
    step_map: dict[str, ProcessingStep | TapOutStepDef] = {}
    for s in workflow.steps:
        if s.id is not None:
            step_map[s.id] = s

    for step in workflow.steps:
        if not isinstance(step, ProcessingStep):
            continue
        target_in = step.input_type
        for dep_id in step.depends_on:
            dep = step_map.get(dep_id)
            if dep is None:
                continue
            source_out = None
            if isinstance(dep, ProcessingStep):
                source_out = dep.output_type
            if not types_compatible(source_out, target_in):
                errors.append(
                    f"Type mismatch: step '{dep_id}' outputs '{source_out}' "
                    f"but step '{step.id}' expects '{target_in}'"
                )
    return errors
