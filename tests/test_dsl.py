# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.dsl — DSL compiler and Python DSL decorators.

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

from grdl_rt.execution.dsl import DslCompiler, step, workflow
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition, WorkflowState


class TestPythonDsl:

    def test_workflow_decorator_captures_steps(self):

        @workflow(name="Test Pipeline", version="1.0.0")
        def my_pipeline():
            step("FilterA", version="0.1.0", threshold=0.5)
            step("FilterB", version="0.2.0")

        wf = my_pipeline._workflow_definition
        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "Test Pipeline"
        assert wf.version == "1.0.0"
        assert wf.state == WorkflowState.PUBLISHED
        assert len(wf.steps) == 2
        assert wf.steps[0].processor_name == "FilterA"
        assert wf.steps[0].params == {"threshold": 0.5}
        assert wf.steps[1].processor_name == "FilterB"

    def test_workflow_with_tags(self):

        @workflow(
            name="SAR Detector",
            version="2.0.0",
            modalities=["SAR"],
            niirs_range=(3.0, 6.0),
            day_capable=True,
            night_capable=True,
            detection_types=["classification"],
        )
        def sar_detector():
            step("PauliDecomposition", version="0.1.0")

        wf = sar_detector._workflow_definition
        assert len(wf.tags.modalities) == 1
        assert wf.tags.modalities[0].value == "SAR"
        assert wf.tags.niirs_range == (3.0, 6.0)
        assert wf.tags.night_capable is True

    def test_empty_workflow(self):

        @workflow(name="Empty")
        def empty_pipeline():
            pass

        wf = empty_pipeline._workflow_definition
        assert len(wf.steps) == 0


class TestDslCompiler:

    def test_yaml_roundtrip(self, tmp_path):
        compiler = DslCompiler()

        wf = WorkflowDefinition(name="YAML Test", version="1.0.0")
        wf.add_step(ProcessingStep("FilterA", "0.1.0", params={"k": 3}))

        yaml_str = compiler.to_yaml(wf)
        assert "YAML Test" in yaml_str
        assert "FilterA" in yaml_str

        # Write to file and compile back
        yaml_path = tmp_path / "workflow.yaml"
        yaml_path.write_text(yaml_str)
        restored = compiler.compile_yaml(yaml_path)
        assert restored.name == "YAML Test"
        assert len(restored.steps) == 1
        assert restored.steps[0].params == {"k": 3}

    def test_yaml_string_roundtrip(self):
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="String Test", version="0.1.0")
        wf.add_step(ProcessingStep("X", "1.0"))
        yaml_str = compiler.to_yaml(wf)
        restored = compiler.compile_yaml_string(yaml_str)
        assert restored.name == "String Test"

    def test_to_python_generates_valid_source(self):
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="Python Gen", version="1.0.0")
        wf.add_step(ProcessingStep("Threshold", "1.0.0", params={"t": 0.5, "mode": "reflect"}))
        source = compiler.to_python(wf)
        assert "from grdl_rt.execution.dsl import workflow, step" in source
        assert "@workflow(" in source
        assert "def python_gen():" in source
        assert 'step("Threshold"' in source
        assert "t=0.5" in source

    def test_python_generation_empty_workflow(self):
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="Empty Gen")
        source = compiler.to_python(wf)
        assert "pass" in source
