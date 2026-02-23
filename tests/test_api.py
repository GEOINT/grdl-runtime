# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.api — load_workflow() and execute_workflow().

Covers:
- load_workflow with YAML file (Path), string path, YAML string, dict
- load_workflow type error for unsupported input
- execute_workflow with GPU on/off and progress callback

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-09
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from grdl_rt.api import load_workflow, execute_workflow
from grdl_rt.execution.dsl import DslCompiler
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

# ---------------------------------------------------------------------------
# Helper mock processor
# ---------------------------------------------------------------------------


class _ScaleTransform:
    """Trivial processor that multiplies by a scale factor."""

    def apply(self, source, **kwargs):
        kwargs.pop("progress_callback", None)
        return source * kwargs.get("scale", 1.0)


# ---------------------------------------------------------------------------
# Helper: build a simple workflow dict / object for reuse
# ---------------------------------------------------------------------------


def _sample_workflow_dict():
    return {
        "name": "TestWorkflow",
        "version": "1.0.0",
        "description": "A test workflow",
        "state": "draft",
        "tags": {},
        "steps": [
            {"processor": "ScaleTransform", "version": "1.0", "params": {"scale": 2.0}},
        ],
    }


def _write_yaml_file(tmp_path):
    """Write a sample YAML workflow file and return its path."""
    wf = WorkflowDefinition(
        name="FileWorkflow",
        version="1.0.0",
        steps=[ProcessingStep("ScaleTransform", "1.0", params={"scale": 3.0})],
    )
    compiler = DslCompiler()
    yaml_text = compiler.to_yaml(wf)
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# TestLoadWorkflow
# ---------------------------------------------------------------------------


class TestLoadWorkflow:
    def test_load_from_yaml_file(self, tmp_path):
        """Path object pointing to a YAML file loads correctly."""
        yaml_path = _write_yaml_file(tmp_path)
        wf = load_workflow(yaml_path)

        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "FileWorkflow"
        assert len(wf.steps) == 1
        assert wf.steps[0].processor_name == "ScaleTransform"

    def test_load_from_string_path(self, tmp_path):
        """String that is an existing file path loads the file."""
        yaml_path = _write_yaml_file(tmp_path)
        wf = load_workflow(str(yaml_path))

        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "FileWorkflow"

    def test_load_from_yaml_string(self):
        """YAML text content is parsed as inline YAML."""
        wf_orig = WorkflowDefinition(
            name="InlineWorkflow",
            version="0.2.0",
            steps=[ProcessingStep("Identity", "1.0")],
        )
        yaml_text = DslCompiler().to_yaml(wf_orig)

        wf = load_workflow(yaml_text)

        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "InlineWorkflow"
        assert len(wf.steps) == 1

    def test_load_from_dict(self):
        """Dict input is deserialized via WorkflowDefinition.from_dict."""
        data = _sample_workflow_dict()
        wf = load_workflow(data)

        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "TestWorkflow"
        assert len(wf.steps) == 1
        assert wf.steps[0].params["scale"] == 2.0

    def test_load_invalid_type_raises(self):
        """Non-str/Path/dict input raises TypeError."""
        with pytest.raises(TypeError, match="source must be str, Path, or dict"):
            load_workflow(42)

    def test_load_nonexistent_string_as_yaml(self):
        """A string that is not an existing file is treated as YAML content."""
        yaml_text = (
            "name: QuickWorkflow\n" "version: '1.0'\n" "steps: []\n" "state: draft\n" "tags: {}\n"
        )
        wf = load_workflow(yaml_text)

        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "QuickWorkflow"
        assert wf.steps == []


# ---------------------------------------------------------------------------
# TestExecuteWorkflow
# ---------------------------------------------------------------------------


class TestExecuteWorkflow:
    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_execute_basic(self, mock_resolve):
        """Basic execution with a single-step workflow."""
        mock_resolve.return_value = _ScaleTransform

        wf = WorkflowDefinition(
            name="Test",
            steps=[ProcessingStep("ScaleTransform", "1.0", params={"scale": 2.0})],
        )
        source = np.ones((4, 4), dtype=np.float64)
        wr = execute_workflow(wf, source, prefer_gpu=False)

        np.testing.assert_array_almost_equal(wr.result, source * 2.0)

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_execute_gpu_off(self, mock_resolve):
        """Explicit prefer_gpu=False still produces correct results."""
        mock_resolve.return_value = _ScaleTransform

        wf = WorkflowDefinition(
            name="Test",
            steps=[ProcessingStep("ScaleTransform", "1.0", params={"scale": 5.0})],
        )
        source = np.ones((2, 2), dtype=np.float64)
        wr = execute_workflow(wf, source, prefer_gpu=False)

        np.testing.assert_array_almost_equal(wr.result, source * 5.0)

    @patch("grdl_rt.execution.executor.resolve_processor_class")
    def test_execute_with_progress_callback(self, mock_resolve):
        """Progress callback receives 0.5 and 1.0 for a two-step workflow."""
        mock_resolve.return_value = _ScaleTransform

        wf = WorkflowDefinition(
            name="Test",
            steps=[
                ProcessingStep("ScaleTransform", "1.0", params={"scale": 2.0}),
                ProcessingStep("ScaleTransform", "1.0", params={"scale": 3.0}),
            ],
        )
        source = np.ones((2, 2), dtype=np.float64)
        progress_values = []
        wr = execute_workflow(
            wf,
            source,
            prefer_gpu=False,
            progress_callback=lambda f: progress_values.append(round(f, 4)),
        )

        assert 0.5 in progress_values
        assert 1.0 in progress_values
        np.testing.assert_array_almost_equal(wr.result, source * 6.0)

    def test_execute_empty_workflow(self):
        """Empty workflow returns the source unchanged."""
        wf = WorkflowDefinition(name="Empty")
        source = np.arange(9, dtype=np.float32).reshape(3, 3)
        wr = execute_workflow(wf, source, prefer_gpu=False)

        np.testing.assert_array_equal(wr.result, source)
