# -*- coding: utf-8 -*-
"""
Workflow Models - Data models for image processing workflow definitions.

A workflow is an ordered sequence of processing steps, each referencing
a GRDL image processor class with specific tunable parameter values.
Workflows carry tags and can be serialized to Python DSL or YAML.

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
2026-02-06
"""

# Standard library
from enum import Enum
from typing import Any, Dict, List, Optional

# grdl-runtime internal
from grdl_rt.execution.tags import WorkflowTags


class WorkflowState(Enum):
    """State of a workflow within a project."""

    DRAFT = "draft"
    TESTING = "testing"
    PUBLISHED = "published"


class ProcessingStep:
    """A single step in a processing workflow.

    Represents a GRDL image processor with a specific version and
    a set of tunable parameter values.

    Parameters
    ----------
    processor_name : str
        Fully-qualified class name or short name of the processor
        (e.g., "PauliDecomposition" or
        "grdl.image_processing.decomposition.pauli.PauliDecomposition").
    processor_version : str
        Semantic version of the processor (from @processor_version).
    params : Dict[str, Any]
        Tunable parameter values for this step.
    """

    def __init__(
        self,
        processor_name: str,
        processor_version: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.processor_name = processor_name
        self.processor_version = processor_version
        self.params = params or {}

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
        if self.params:
            d['params'] = self.params
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
        return cls(
            processor_name=data['processor'],
            processor_version=data.get('version', ''),
            params=data.get('params', {}),
        )


class WorkflowDefinition:
    """A complete image processing workflow definition.

    An ordered sequence of processing steps with metadata and tags.

    Parameters
    ----------
    name : str
        Human-readable workflow name.
    version : str
        Semantic version of the workflow definition.
    description : str
        Description of what the workflow does.
    steps : List[ProcessingStep]
        Ordered processing steps.
    tags : Optional[WorkflowTags]
        Workflow classification tags.
    state : WorkflowState
        Current state of the workflow.
    """

    def __init__(
        self,
        name: str,
        version: str = "0.1.0",
        description: str = "",
        steps: Optional[List[ProcessingStep]] = None,
        tags: Optional[WorkflowTags] = None,
        state: WorkflowState = WorkflowState.DRAFT,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.steps = steps or []
        self.tags = tags or WorkflowTags()
        self.state = state

    def add_step(self, step: ProcessingStep) -> None:
        """Append a processing step to the workflow.

        Parameters
        ----------
        step : ProcessingStep
        """
        self.steps.append(step)

    def remove_step(self, index: int) -> ProcessingStep:
        """Remove and return a processing step by index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ProcessingStep
            The removed step.
        """
        return self.steps.pop(index)

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

    def to_dict(self) -> dict:
        """Serialize to dictionary for YAML/JSON storage.

        Returns
        -------
        dict
        """
        return {
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

        Parameters
        ----------
        data : dict

        Returns
        -------
        WorkflowDefinition
        """
        tags = WorkflowTags.from_dict(data.get('tags', {}))
        steps = [ProcessingStep.from_dict(s) for s in data.get('steps', [])]
        return cls(
            name=data['name'],
            version=data.get('version', '0.1.0'),
            description=data.get('description', ''),
            steps=steps,
            tags=tags,
            state=WorkflowState(data.get('state', 'draft')),
        )
