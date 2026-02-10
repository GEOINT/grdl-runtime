# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.workflow — Workflow and ProcessingStep models.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06
"""

import pytest

from grdl_rt.execution.tags import ImageModality, WorkflowTags
from grdl_rt.execution.workflow import (
    ProcessingStep,
    WorkflowDefinition,
    WorkflowState,
)


class TestProcessingStep:

    def test_basic_construction(self):
        step = ProcessingStep("PauliDecomposition", "0.1.0")
        assert step.processor_name == "PauliDecomposition"
        assert step.processor_version == "0.1.0"
        assert step.params == {}

    def test_with_params(self):
        step = ProcessingStep(
            "AdaptiveThreshold", "1.0.0",
            params={'threshold': 0.65, 'kernel_size': 7},
        )
        assert step.params['threshold'] == 0.65
        assert step.params['kernel_size'] == 7

    def test_roundtrip(self):
        step = ProcessingStep(
            "MyFilter", "2.0.0",
            params={'alpha': 0.5, 'mode': 'reflect'},
        )
        d = step.to_dict()
        restored = ProcessingStep.from_dict(d)
        assert restored.processor_name == "MyFilter"
        assert restored.processor_version == "2.0.0"
        assert restored.params == {'alpha': 0.5, 'mode': 'reflect'}

    def test_to_dict_no_params(self):
        step = ProcessingStep("Orthorectifier", "0.1.0")
        d = step.to_dict()
        assert 'params' not in d


class TestWorkflowDefinition:

    def test_basic_construction(self):
        wf = WorkflowDefinition(name="Test Workflow")
        assert wf.name == "Test Workflow"
        assert wf.version == "0.1.0"
        assert wf.state == WorkflowState.DRAFT
        assert wf.steps == []

    def test_add_step(self):
        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("A", "1.0"))
        wf.add_step(ProcessingStep("B", "1.0"))
        assert len(wf.steps) == 2
        assert wf.steps[0].processor_name == "A"
        assert wf.steps[1].processor_name == "B"

    def test_remove_step(self):
        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("A", "1.0"))
        wf.add_step(ProcessingStep("B", "1.0"))
        removed = wf.remove_step(0)
        assert removed.processor_name == "A"
        assert len(wf.steps) == 1

    def test_move_step(self):
        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("A", "1.0"))
        wf.add_step(ProcessingStep("B", "1.0"))
        wf.add_step(ProcessingStep("C", "1.0"))
        wf.move_step(2, 0)
        assert wf.steps[0].processor_name == "C"
        assert wf.steps[1].processor_name == "A"
        assert wf.steps[2].processor_name == "B"

    def test_roundtrip(self):
        tags = WorkflowTags(
            modalities=[ImageModality.SAR],
            niirs_range=(3.0, 6.0),
        )
        wf = WorkflowDefinition(
            name="SAR Pipeline",
            version="1.0.0",
            description="A test pipeline",
            steps=[
                ProcessingStep("PauliDecomposition", "0.1.0"),
                ProcessingStep("Threshold", "1.0.0", params={'t': 0.5}),
            ],
            tags=tags,
            state=WorkflowState.TESTING,
        )
        d = wf.to_dict()
        restored = WorkflowDefinition.from_dict(d)
        assert restored.name == "SAR Pipeline"
        assert restored.version == "1.0.0"
        assert restored.state == WorkflowState.TESTING
        assert len(restored.steps) == 2
        assert restored.steps[1].params == {'t': 0.5}
        assert restored.tags.modalities == [ImageModality.SAR]


class TestWorkflowState:

    def test_values(self):
        assert WorkflowState.DRAFT.value == "draft"
        assert WorkflowState.TESTING.value == "testing"
        assert WorkflowState.PUBLISHED.value == "published"
