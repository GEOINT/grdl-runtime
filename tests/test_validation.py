"""
Tests for grdl_rt.execution.validation — workflow validation.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

from unittest.mock import patch

from grdl_rt.execution.validation import validate_workflow
from grdl_rt.execution.workflow import (
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleProcessor:
    """Processor with no required params."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def apply(self, source, **kwargs):
        return source * self.scale


class _RequiredParamProcessor:
    """Processor with a required param (no default)."""

    def __init__(self, window_size: int, metadata=None):
        self.window_size = window_size

    def apply(self, source, **kwargs):
        return source


# ---------------------------------------------------------------------------
# Empty workflow
# ---------------------------------------------------------------------------


class TestEmptyWorkflow:
    def test_empty_workflow_error(self):
        wf = WorkflowDefinition(name="Empty")
        errors = validate_workflow(wf)
        assert len(errors) == 1
        assert errors[0].code == "EMPTY_WORKFLOW"
        assert errors[0].step_index is None

    def test_tap_out_only_is_empty(self):
        wf = WorkflowDefinition(name="TapOnly")
        wf.add_step(TapOutStepDef(path="./out.npy"))
        errors = validate_workflow(wf)
        codes = [e.code for e in errors]
        assert "EMPTY_WORKFLOW" in codes


# ---------------------------------------------------------------------------
# Processor resolution
# ---------------------------------------------------------------------------


class TestProcessorResolution:
    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_unresolvable_processor(self, mock_resolve):
        mock_resolve.side_effect = ImportError("Cannot resolve 'NoSuch'")

        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("NoSuch", "1.0"))
        errors = validate_workflow(wf)

        assert len(errors) == 1
        assert errors[0].code == "PROCESSOR_NOT_FOUND"
        assert errors[0].step_index == 0
        assert errors[0].processor_name == "NoSuch"

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_resolvable_processor_no_error(self, mock_resolve):
        mock_resolve.return_value = _SimpleProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("SimpleProcessor", "1.0"))
        errors = validate_workflow(wf)

        assert len(errors) == 0

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_fqn_processor_resolves(self, mock_resolve):
        mock_resolve.return_value = _SimpleProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("some.module.SimpleProcessor", "1.0"))
        errors = validate_workflow(wf)

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Required parameters
# ---------------------------------------------------------------------------


class TestRequiredParams:
    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_missing_required_param(self, mock_resolve):
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("RequiredParam", "1.0", params={}))
        errors = validate_workflow(wf)

        assert len(errors) == 1
        assert errors[0].code == "MISSING_REQUIRED_PARAM"
        assert "window_size" in errors[0].message

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_required_param_satisfied(self, mock_resolve):
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredParam",
                "1.0",
                params={"window_size": 5},
            )
        )
        errors = validate_workflow(wf)

        assert len(errors) == 0

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_metadata_param_excluded(self, mock_resolve):
        """'metadata' parameter should not be treated as required."""
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        # Only provide window_size, not metadata — should be fine
        wf.add_step(
            ProcessingStep(
                "RequiredParam",
                "1.0",
                params={"window_size": 3},
            )
        )
        errors = validate_workflow(wf)

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# TapOutStepDef skipped
# ---------------------------------------------------------------------------


class TestTapOutSkipped:
    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_tap_out_not_validated(self, mock_resolve):
        mock_resolve.return_value = _SimpleProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("Simple", "1.0"))
        wf.add_step(TapOutStepDef(path="./debug.npy"))

        errors = validate_workflow(wf)
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# WorkflowDefinition.validate()
# ---------------------------------------------------------------------------


class TestWorkflowDefinitionValidate:
    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_validate_method_delegates(self, mock_resolve):
        mock_resolve.return_value = _SimpleProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("Simple", "1.0"))
        errors = wf.validate()

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_empty_returns_errors(self):
        wf = WorkflowDefinition(name="Empty")
        errors = wf.validate()
        assert len(errors) > 0
        assert errors[0].code == "EMPTY_WORKFLOW"
