# -*- coding: utf-8 -*-
"""
Tests for DAG utilities, workflow DAG structure, and condition evaluator.

Covers:
- Safe expression evaluator (dag.py)
- Workflow DAG model: step IDs, depends_on, topological sort, validation
- Execution phases and phase ordering validation
- YAML round-trip with DAG fields
- DSL with DAG fields

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-11
"""

import pytest
import numpy as np

from grdl_rt.execution.dag import evaluate_condition
from grdl_rt.execution.dsl import DslCompiler, step, workflow
from grdl_rt.execution.errors import DAGCycleError, ConditionError
from grdl_rt.execution.validation import ValidationError, validate_workflow
from grdl_rt.execution.workflow import (
    SCHEMA_VERSION,
    ExecutionPhase,
    PHASE_ORDER,
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
    WorkflowState,
)


# ---------------------------------------------------------------------------
# Condition Evaluator — Safe evaluation
# ---------------------------------------------------------------------------

class TestEvaluateCondition:

    def test_simple_comparison_true(self):
        assert evaluate_condition("x > 1", {"x": 5}) is True

    def test_simple_comparison_false(self):
        assert evaluate_condition("x > 10", {"x": 5}) is False

    def test_equality(self):
        assert evaluate_condition("x == 3", {"x": 3}) is True

    def test_arithmetic_in_condition(self):
        assert evaluate_condition("x + y > 10", {"x": 7, "y": 5}) is True

    def test_dotted_attribute_dict(self):
        """Dotted access resolves dict keys: metadata.band_count."""
        ctx = {"metadata": {"band_count": 3}}
        assert evaluate_condition("metadata.band_count > 1", ctx) is True

    def test_dotted_attribute_dict_false(self):
        ctx = {"metadata": {"band_count": 1}}
        assert evaluate_condition("metadata.band_count > 1", ctx) is False

    def test_dotted_attribute_object(self):
        """Dotted access falls back to getattr for objects."""
        class Meta:
            band_count = 4
        ctx = {"metadata": Meta()}
        assert evaluate_condition("metadata.band_count > 2", ctx) is True

    def test_subscript_access(self):
        ctx = {"data": {"key": 42}}
        assert evaluate_condition('data["key"] == 42', ctx) is True

    def test_boolean_and(self):
        ctx = {"a": True, "b": True}
        assert evaluate_condition("a and b", ctx) is True

    def test_boolean_or(self):
        ctx = {"a": False, "b": True}
        assert evaluate_condition("a or b", ctx) is True

    def test_boolean_not(self):
        ctx = {"skip": False}
        assert evaluate_condition("not skip", ctx) is True

    def test_combined_boolean(self):
        ctx = {"x": 5, "y": 10}
        assert evaluate_condition("x > 0 and y < 20", ctx) is True

    def test_in_operator(self):
        ctx = {"x": "SAR", "modes": ["SAR", "EO"]}
        assert evaluate_condition("x in modes", ctx) is True

    def test_not_in_operator(self):
        ctx = {"x": "LIDAR", "modes": ["SAR", "EO"]}
        assert evaluate_condition("x not in modes", ctx) is True

    def test_in_literal_tuple(self):
        ctx = {"x": 2}
        assert evaluate_condition("x in (1, 2, 3)", ctx) is True

    def test_in_literal_list(self):
        ctx = {"x": "b"}
        assert evaluate_condition('x in ["a", "b", "c"]', ctx) is True

    def test_ternary_expression(self):
        ctx = {"flag": True}
        result = evaluate_condition("1 if flag else 0", ctx)
        assert result is True  # bool(1) == True

    def test_chained_comparison(self):
        ctx = {"x": 5}
        assert evaluate_condition("1 < x < 10", ctx) is True

    def test_chained_comparison_false(self):
        ctx = {"x": 15}
        assert evaluate_condition("1 < x < 10", ctx) is False

    def test_constant_true(self):
        assert evaluate_condition("True", {}) is True

    def test_constant_false(self):
        assert evaluate_condition("False", {}) is False

    def test_negation_arithmetic(self):
        ctx = {"x": -5}
        assert evaluate_condition("x < 0", ctx) is True

    def test_unary_minus(self):
        ctx = {"x": 5}
        assert evaluate_condition("-x < 0", ctx) is True

    def test_modulo(self):
        ctx = {"x": 10}
        assert evaluate_condition("x % 3 == 1", ctx) is True

    def test_floor_division(self):
        ctx = {"x": 7}
        assert evaluate_condition("x // 2 == 3", ctx) is True


# ---------------------------------------------------------------------------
# Condition Evaluator — Safety checks
# ---------------------------------------------------------------------------

class TestEvaluateConditionSafety:

    def test_rejects_function_call(self):
        with pytest.raises(ValueError, match="Disallowed"):
            evaluate_condition("len([1, 2])", {})

    def test_rejects_import(self):
        with pytest.raises(ValueError, match="Disallowed"):
            evaluate_condition("__import__('os')", {})

    def test_rejects_lambda(self):
        with pytest.raises(ValueError, match="Disallowed"):
            evaluate_condition("(lambda: 1)()", {})

    def test_rejects_system_call(self):
        with pytest.raises(ValueError, match="Disallowed"):
            evaluate_condition("__import__('os').system('rm -rf /')", {})

    def test_rejects_nested_getattr_call(self):
        """Attempting getattr via a call is still blocked."""
        with pytest.raises(ValueError, match="Disallowed"):
            evaluate_condition("getattr(x, 'y')", {"x": {}})

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            evaluate_condition("nonexistent > 0", {})

    def test_syntax_error_raises(self):
        with pytest.raises(ValueError, match="Invalid condition"):
            evaluate_condition("x >>>> 0", {})


# ---------------------------------------------------------------------------
# Workflow DAG — Step ID assignment and linear inference
# ---------------------------------------------------------------------------

class TestWorkflowDAGStructure:

    def test_step_ids_auto_assigned(self):
        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("A", "1.0"))
        wf.add_step(ProcessingStep("B", "1.0"))
        assert wf.steps[0].id == "step_0"
        assert wf.steps[1].id == "step_1"

    def test_explicit_ids_preserved(self):
        s1 = ProcessingStep("A", "1.0", id="read")
        s2 = ProcessingStep("B", "1.0", id="process")
        wf = WorkflowDefinition(name="Test", steps=[s1, s2])
        assert wf.steps[0].id == "read"
        assert wf.steps[1].id == "process"

    def test_linear_deps_inferred(self):
        """Steps without explicit depends_on are chained linearly."""
        s1 = ProcessingStep("A", "1.0")
        s2 = ProcessingStep("B", "1.0")
        s3 = ProcessingStep("C", "1.0")
        wf = WorkflowDefinition(name="Test", steps=[s1, s2, s3])
        # First step has no deps, subsequent steps depend on predecessor
        assert wf.steps[0].depends_on == []
        assert wf.steps[1].depends_on == [wf.steps[0].id]
        assert wf.steps[2].depends_on == [wf.steps[1].id]

    def test_explicit_deps_not_overridden(self):
        """If any step has depends_on, linear inference is skipped."""
        s1 = ProcessingStep("A", "1.0", id="root")
        s2 = ProcessingStep("B", "1.0", id="branch1", depends_on=["root"])
        s3 = ProcessingStep("C", "1.0", id="branch2", depends_on=["root"])
        wf = WorkflowDefinition(name="Test", steps=[s1, s2, s3])
        assert wf.steps[0].depends_on == []
        assert wf.steps[1].depends_on == ["root"]
        assert wf.steps[2].depends_on == ["root"]

    def test_get_step_by_id(self):
        s = ProcessingStep("A", "1.0", id="my_step")
        wf = WorkflowDefinition(name="Test", steps=[s])
        assert wf.get_step("my_step") is s

    def test_get_step_not_found(self):
        wf = WorkflowDefinition(name="Test")
        with pytest.raises(KeyError, match="no_such"):
            wf.get_step("no_such")

    def test_terminal_step_ids(self):
        s1 = ProcessingStep("A", "1.0", id="root")
        s2 = ProcessingStep("B", "1.0", id="b1", depends_on=["root"])
        s3 = ProcessingStep("C", "1.0", id="b2", depends_on=["root"])
        wf = WorkflowDefinition(name="Test", steps=[s1, s2, s3])
        terminals = wf.terminal_step_ids()
        assert set(terminals) == {"b1", "b2"}

    def test_terminal_step_ids_linear(self):
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0"),
            ProcessingStep("B", "1.0"),
        ])
        terminals = wf.terminal_step_ids()
        assert len(terminals) == 1
        assert terminals[0] == wf.steps[-1].id


# ---------------------------------------------------------------------------
# Topological Sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:

    def test_linear_pipeline(self):
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0"),
            ProcessingStep("B", "1.0"),
            ProcessingStep("C", "1.0"),
        ])
        levels = wf.topological_sort()
        # Linear: each level has one step
        assert len(levels) == 3
        assert len(levels[0]) == 1
        assert len(levels[1]) == 1
        assert len(levels[2]) == 1

    def test_branching_dag(self):
        """root → [b1, b2] → merge"""
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0", id="root"),
            ProcessingStep("B", "1.0", id="b1", depends_on=["root"]),
            ProcessingStep("C", "1.0", id="b2", depends_on=["root"]),
            ProcessingStep("D", "1.0", id="merge", depends_on=["b1", "b2"]),
        ])
        levels = wf.topological_sort()
        # Level 0: root; Level 1: b1, b2; Level 2: merge
        assert len(levels) == 3
        assert levels[0] == ["root"]
        assert set(levels[1]) == {"b1", "b2"}
        assert levels[2] == ["merge"]

    def test_diamond_dag(self):
        """Diamond shape: root → [a, b] → merge"""
        wf = WorkflowDefinition(name="Diamond", steps=[
            ProcessingStep("Root", "1.0", id="root"),
            ProcessingStep("Left", "1.0", id="left", depends_on=["root"]),
            ProcessingStep("Right", "1.0", id="right", depends_on=["root"]),
            ProcessingStep("Merge", "1.0", id="merge", depends_on=["left", "right"]),
        ])
        levels = wf.topological_sort()
        assert len(levels) == 3

    def test_single_step(self):
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0"),
        ])
        levels = wf.topological_sort()
        assert len(levels) == 1
        assert len(levels[0]) == 1

    def test_empty_workflow(self):
        wf = WorkflowDefinition(name="Test")
        levels = wf.topological_sort()
        assert levels == []

    def test_cycle_detection(self):
        """Manually create a cycle: A→B→C→A."""
        s1 = ProcessingStep("A", "1.0", id="a", depends_on=["c"])
        s2 = ProcessingStep("B", "1.0", id="b", depends_on=["a"])
        s3 = ProcessingStep("C", "1.0", id="c", depends_on=["b"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Cycle"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1, s2, s3]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        with pytest.raises(ValueError, match="cycle"):
            wf.topological_sort()


# ---------------------------------------------------------------------------
# DAG Validation
# ---------------------------------------------------------------------------

class TestDAGValidation:

    def test_valid_dag(self):
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0", id="root"),
            ProcessingStep("B", "1.0", id="child", depends_on=["root"]),
        ])
        errors = wf.validate_dag()
        assert errors == []

    def test_duplicate_step_id(self):
        """Two steps with same ID should be detected."""
        s1 = ProcessingStep("A", "1.0", id="dup")
        s2 = ProcessingStep("B", "1.0", id="dup")
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Test"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1, s2]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        errors = wf.validate_dag()
        assert len(errors) == 1
        assert "Duplicate" in errors[0]

    def test_unresolved_dependency(self):
        s1 = ProcessingStep("A", "1.0", id="a", depends_on=["nonexistent"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Test"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        errors = wf.validate_dag()
        assert len(errors) >= 1
        assert "does not exist" in errors[0]

    def test_cycle_in_validate_dag(self):
        s1 = ProcessingStep("A", "1.0", id="a", depends_on=["c"])
        s2 = ProcessingStep("B", "1.0", id="b", depends_on=["a"])
        s3 = ProcessingStep("C", "1.0", id="c", depends_on=["b"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Cycle"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1, s2, s3]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        errors = wf.validate_dag()
        assert len(errors) >= 1
        assert "cycle" in errors[0].lower()


# ---------------------------------------------------------------------------
# Execution Phases
# ---------------------------------------------------------------------------

class TestExecutionPhases:

    def test_phase_enum_values(self):
        assert ExecutionPhase.IO.value == "io"
        assert ExecutionPhase.FINALIZATION.value == "finalization"
        assert ExecutionPhase.DATA_PREP.value == "data_prep"

    def test_phase_order_consistent(self):
        """IO < GLOBAL_PROCESSING < ... < FINALIZATION."""
        assert PHASE_ORDER[ExecutionPhase.IO] < PHASE_ORDER[ExecutionPhase.FINALIZATION]
        assert PHASE_ORDER[ExecutionPhase.DATA_PREP] < PHASE_ORDER[ExecutionPhase.EXTRACTION]

    def test_all_phases_in_order(self):
        assert len(PHASE_ORDER) == len(ExecutionPhase)


# ---------------------------------------------------------------------------
# Validation — Phase ordering and conditions
# ---------------------------------------------------------------------------

class TestValidationExtended:

    def test_phase_order_violation(self):
        """Step in IO phase depends on step in FINALIZATION phase = warning."""
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0", id="final", phase="finalization"),
            ProcessingStep("B", "1.0", id="io_step", phase="io",
                           depends_on=["final"]),
        ])
        errors = validate_workflow(wf)
        phase_errors = [e for e in errors if e.code == "PHASE_ORDER_VIOLATION"]
        assert len(phase_errors) >= 1

    def test_valid_phase_ordering(self):
        """IO → FINALIZATION is valid ordering."""
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0", id="io_step", phase="io"),
            ProcessingStep("B", "1.0", id="final", phase="finalization",
                           depends_on=["io_step"]),
        ])
        errors = validate_workflow(wf)
        phase_errors = [e for e in errors if e.code == "PHASE_ORDER_VIOLATION"]
        assert len(phase_errors) == 0

    def test_invalid_condition_syntax(self):
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0", id="s1",
                           condition="x >>>> 0"),
        ])
        errors = validate_workflow(wf)
        cond_errors = [e for e in errors if e.code == "INVALID_CONDITION"]
        assert len(cond_errors) >= 1

    def test_valid_condition_passes(self):
        wf = WorkflowDefinition(name="Test", steps=[
            ProcessingStep("A", "1.0", id="s1",
                           condition="x > 0"),
        ])
        errors = validate_workflow(wf)
        cond_errors = [e for e in errors if e.code == "INVALID_CONDITION"]
        assert len(cond_errors) == 0

    def test_dag_cycle_validation_code(self):
        """validate_workflow catches cycle and returns proper code."""
        s1 = ProcessingStep("A", "1.0", id="a", depends_on=["b"])
        s2 = ProcessingStep("B", "1.0", id="b", depends_on=["a"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Test"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1, s2]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        errors = validate_workflow(wf)
        cycle_errors = [e for e in errors if e.code == "DAG_CYCLE"]
        assert len(cycle_errors) >= 1

    def test_duplicate_id_validation_code(self):
        s1 = ProcessingStep("A", "1.0", id="dup")
        s2 = ProcessingStep("B", "1.0", id="dup")
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Test"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1, s2]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        errors = validate_workflow(wf)
        dup_errors = [e for e in errors if e.code == "DUPLICATE_STEP_ID"]
        assert len(dup_errors) >= 1

    def test_unresolved_dep_validation_code(self):
        s1 = ProcessingStep("A", "1.0", id="a", depends_on=["ghost"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Test"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [s1]
        wf.tags = None
        wf.state = WorkflowState.DRAFT
        wf.schema_version = SCHEMA_VERSION
        errors = validate_workflow(wf)
        dep_errors = [e for e in errors if e.code == "UNRESOLVED_DEPENDENCY"]
        assert len(dep_errors) >= 1


# ---------------------------------------------------------------------------
# Serialization — to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestWorkflowSerialization:

    def test_processing_step_round_trip(self):
        s = ProcessingStep(
            "FilterA", "1.0",
            params={"k": 3},
            id="read",
            depends_on=["prev"],
            condition="x > 0",
            phase="io",
        )
        d = s.to_dict()
        restored = ProcessingStep.from_dict(d)
        assert restored.processor_name == "FilterA"
        assert restored.id == "read"
        assert restored.depends_on == ["prev"]
        assert restored.condition == "x > 0"
        assert restored.phase == "io"
        assert restored.params == {"k": 3}

    def test_tap_out_step_round_trip(self):
        s = TapOutStepDef(
            path="output.tif",
            format="geotiff",
            id="tap",
            depends_on=["prev"],
        )
        d = s.to_dict()
        restored = TapOutStepDef.from_dict(d)
        assert restored.path == "output.tif"
        assert restored.format == "geotiff"
        assert restored.id == "tap"
        assert restored.depends_on == ["prev"]

    def test_workflow_dict_round_trip(self):
        wf = WorkflowDefinition(name="Test DAG", version="1.0.0", steps=[
            ProcessingStep("A", "1.0", id="root"),
            ProcessingStep("B", "1.0", id="b1", depends_on=["root"],
                           condition="x > 1", phase="data_prep"),
            ProcessingStep("C", "1.0", id="b2", depends_on=["root"]),
            ProcessingStep("D", "1.0", id="merge", depends_on=["b1", "b2"]),
        ])
        d = wf.to_dict()
        assert d['schema_version'] == SCHEMA_VERSION

        restored = WorkflowDefinition.from_dict(d)
        assert restored.name == "Test DAG"
        assert restored.schema_version == SCHEMA_VERSION
        assert len(restored.steps) == 4
        assert restored.steps[1].condition == "x > 1"
        assert restored.steps[1].phase == "data_prep"
        assert restored.steps[3].depends_on == ["b1", "b2"]

    def test_old_v1_format_loads(self):
        """v1 YAML (no schema_version) still loads correctly."""
        data = {
            'name': 'Old Pipeline',
            'version': '1.0.0',
            'state': 'draft',
            'tags': {},
            'steps': [
                {'processor': 'FilterA', 'version': '1.0', 'params': {'k': 3}},
                {'processor': 'FilterB', 'version': '1.0'},
            ],
        }
        wf = WorkflowDefinition.from_dict(data)
        assert wf.name == "Old Pipeline"
        assert wf.schema_version == "1.0"
        assert len(wf.steps) == 2
        # Linear deps should be inferred
        assert wf.steps[1].depends_on == [wf.steps[0].id]


# ---------------------------------------------------------------------------
# YAML Round-trip with DAG fields
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:

    def test_dag_yaml_round_trip(self):
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="DAG Test", version="2.0.0", steps=[
            ProcessingStep("Read", "1.0", id="read", phase="io"),
            ProcessingStep("Band1", "1.0", id="band1",
                           depends_on=["read"], phase="data_prep"),
            ProcessingStep("Band2", "1.0", id="band2",
                           depends_on=["read"],
                           condition="metadata.band_count > 1",
                           phase="data_prep"),
            ProcessingStep("Merge", "1.0", id="merge",
                           depends_on=["band1", "band2"],
                           phase="global_processing"),
        ])
        yaml_str = compiler.to_yaml(wf)
        assert "schema_version" in yaml_str
        assert "depends_on" in yaml_str
        assert "condition" in yaml_str

        restored = compiler.compile_yaml_string(yaml_str)
        assert restored.name == "DAG Test"
        assert len(restored.steps) == 4
        assert restored.steps[0].id == "read"
        assert restored.steps[1].depends_on == ["read"]
        assert restored.steps[2].condition == "metadata.band_count > 1"
        assert restored.steps[3].depends_on == ["band1", "band2"]

    def test_old_yaml_still_loads(self, tmp_path):
        """v1 YAML (no schema_version, no IDs) loads and chains linearly."""
        compiler = DslCompiler()
        yaml_content = """
name: Legacy Pipeline
version: "1.0.0"
state: draft
tags: {}
steps:
  - processor: FilterA
    version: "1.0"
    params:
      k: 3
  - processor: FilterB
    version: "1.0"
"""
        yaml_path = tmp_path / "legacy.yaml"
        yaml_path.write_text(yaml_content)
        wf = compiler.compile_yaml(yaml_path)
        assert wf.name == "Legacy Pipeline"
        assert len(wf.steps) == 2
        # Auto-assigned IDs and linear deps
        assert wf.steps[0].id is not None
        assert wf.steps[1].depends_on == [wf.steps[0].id]

    def test_dag_yaml_with_tap_out(self):
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="Tap DAG", version="1.0.0", steps=[
            ProcessingStep("A", "1.0", id="root"),
            TapOutStepDef("intermediate.tif", id="tap",
                          depends_on=["root"]),
            ProcessingStep("B", "1.0", id="final",
                           depends_on=["tap"]),
        ])
        yaml_str = compiler.to_yaml(wf)
        restored = compiler.compile_yaml_string(yaml_str)
        assert len(restored.steps) == 3
        assert isinstance(restored.steps[1], TapOutStepDef)
        assert restored.steps[1].id == "tap"


# ---------------------------------------------------------------------------
# Python DSL with DAG fields
# ---------------------------------------------------------------------------

class TestDSLWithDAG:

    def test_step_with_dag_fields(self):
        @workflow(name="DAG DSL", version="2.0.0")
        def dag_pipeline():
            step("Read", version="1.0", id="read", phase="io")
            step("Process", version="1.0", id="proc",
                 depends_on=["read"], condition="x > 0")

        wf = dag_pipeline._workflow_definition
        assert len(wf.steps) == 2
        assert wf.steps[0].id == "read"
        assert wf.steps[0].phase == "io"
        assert wf.steps[1].depends_on == ["read"]
        assert wf.steps[1].condition == "x > 0"

    def test_to_python_includes_dag_fields(self):
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="Python Gen", version="1.0.0", steps=[
            ProcessingStep("A", "1.0", id="root", phase="io"),
            ProcessingStep("B", "1.0", id="child",
                           depends_on=["root"],
                           condition="x > 1",
                           phase="data_prep"),
        ])
        source = compiler.to_python(wf)
        assert 'id="root"' in source
        assert 'depends_on=["root"]' in source
        assert 'condition="x > 1"' in source
        assert 'phase="io"' in source


# ---------------------------------------------------------------------------
# Schema Version
# ---------------------------------------------------------------------------

class TestSchemaVersion:

    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "2.0"

    def test_new_workflow_has_v2(self):
        wf = WorkflowDefinition(name="Test")
        assert wf.schema_version == "2.0"

    def test_serialized_includes_schema_version(self):
        wf = WorkflowDefinition(name="Test")
        d = wf.to_dict()
        assert d['schema_version'] == "2.0"
