"""
Tests for type validation in grdl_rt.execution — type compatibility checks.

Covers:
- Compatible raster-to-raster connections
- Compatible feature-to-feature connections
- Incompatible raster-to-vector connections
- DetectionSet compatibility with FeatureSet
- None type accepts anything
- v2.0 YAML backward compatibility

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-03-25
"""

from grdl_rt.execution.graph import validate_type_compatibility
from grdl_rt.execution.workflow import (
    ProcessingStep,
    WorkflowDefinition,
)


class TestTypeValidation:
    """Test validate_type_compatibility."""

    def test_compatible_raster_to_raster(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", depends_on=["a"], input_type="raster")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])

        errors = validate_type_compatibility(wf)
        assert len(errors) == 0

    def test_compatible_feature_to_feature(self):
        step_a = ProcessingStep("A", id="a", output_type="feature_set")
        step_b = ProcessingStep("B", id="b", depends_on=["a"], input_type="feature_set")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])

        errors = validate_type_compatibility(wf)
        assert len(errors) == 0

    def test_incompatible_raster_to_vector(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", depends_on=["a"], input_type="feature_set")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])

        errors = validate_type_compatibility(wf)
        assert len(errors) == 1
        assert "Type mismatch" in errors[0]

    def test_detection_compatible_with_feature(self):
        step_a = ProcessingStep("A", id="a", output_type="detection_set")
        step_b = ProcessingStep("B", id="b", depends_on=["a"], input_type="feature_set")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])

        errors = validate_type_compatibility(wf)
        assert len(errors) == 0

    def test_none_type_accepts_anything(self):
        # Source has output_type, target has None input_type
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", depends_on=["a"])
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        assert len(validate_type_compatibility(wf)) == 0

        # Source has None output_type, target has input_type
        step_c = ProcessingStep("C", id="c")
        step_d = ProcessingStep("D", id="d", depends_on=["c"], input_type="raster")
        wf2 = WorkflowDefinition(name="test2", steps=[step_c, step_d])
        assert len(validate_type_compatibility(wf2)) == 0

        # Both None
        step_e = ProcessingStep("E", id="e")
        step_f = ProcessingStep("F", id="f", depends_on=["e"])
        wf3 = WorkflowDefinition(name="test3", steps=[step_e, step_f])
        assert len(validate_type_compatibility(wf3)) == 0

    def test_v2_yaml_backward_compatible(self):
        """v2.0 schema YAML (no type fields) loads without errors."""
        v2_data = {
            "schema_version": "2.0",
            "name": "legacy_workflow",
            "version": "1.0.0",
            "description": "A v2.0 workflow",
            "state": "draft",
            "tags": {},
            "steps": [
                {
                    "processor": "ProcA",
                    "version": "1.0",
                    "params": {"sigma": 1.5},
                    "id": "step_0",
                },
                {
                    "processor": "ProcB",
                    "version": "1.0",
                    "params": {},
                    "id": "step_1",
                    "depends_on": ["step_0"],
                },
            ],
        }
        wf = WorkflowDefinition.from_dict(v2_data)

        # Should load without error
        assert wf.name == "legacy_workflow"
        assert len(wf.steps) == 2

        # Type fields should be None (backward compatible defaults)
        s0 = wf.steps[0]
        assert isinstance(s0, ProcessingStep)
        assert s0.input_type is None
        assert s0.output_type is None
        assert s0.output_ports is None
        assert s0.position is None

        # Type validation should pass (None types are permissive)
        errors = validate_type_compatibility(wf)
        assert len(errors) == 0


class TestProcessingStepSerialization:
    """Test that new fields serialize and deserialize correctly."""

    def test_roundtrip_with_types(self):
        step = ProcessingStep(
            "MyProc",
            id="s1",
            input_type="raster",
            output_type="detection_set",
            output_ports={"detections": "detection_set", "mask": "raster"},
            position=(100.0, 200.0),
        )
        d = step.to_dict()
        assert d["input_type"] == "raster"
        assert d["output_type"] == "detection_set"
        assert d["output_ports"] == {"detections": "detection_set", "mask": "raster"}
        assert d["position"] == [100.0, 200.0]

        restored = ProcessingStep.from_dict(d)
        assert restored.input_type == "raster"
        assert restored.output_type == "detection_set"
        assert restored.output_ports == {"detections": "detection_set", "mask": "raster"}
        assert restored.position == (100.0, 200.0)

    def test_roundtrip_without_types(self):
        """Steps without type annotations serialize cleanly."""
        step = ProcessingStep("BasicProc", id="s1", params={"x": 1})
        d = step.to_dict()
        assert "input_type" not in d
        assert "output_type" not in d
        assert "output_ports" not in d
        assert "position" not in d

        restored = ProcessingStep.from_dict(d)
        assert restored.input_type is None
        assert restored.output_type is None
        assert restored.output_ports is None
        assert restored.position is None
