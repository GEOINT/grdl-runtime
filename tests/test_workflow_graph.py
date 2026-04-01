"""
Tests for grdl_rt.execution.graph — WorkflowGraph introspection and manipulation.

Covers:
- Adding and removing nodes
- Connecting with type validation
- Disconnecting edges
- Topological levels
- Roundtrip to/from WorkflowDefinition
- Cycle detection
- Type mismatch detection
- Updating params and position

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

import pytest

from grdl_rt.execution.graph import (
    WorkflowGraph,
    types_compatible,
)
from grdl_rt.execution.workflow import (
    ProcessingStep,
    WorkflowDefinition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow(*steps: ProcessingStep, name: str = "test") -> WorkflowDefinition:
    """Create a WorkflowDefinition from steps without auto-linear-dep inference."""
    # Give each step explicit depends_on (even if empty) to prevent
    # WorkflowDefinition._infer_linear_deps from chaining them.
    for s in steps:
        if not s.depends_on:
            # Mark as explicitly empty by setting a sentinel, then clear it
            pass
    wf = WorkflowDefinition(name=name, steps=list(steps))
    return wf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphAddRemoveNodes:
    """Test adding and removing nodes."""

    def test_add_node(self):
        wf = WorkflowDefinition(name="empty")
        graph = WorkflowGraph(wf)

        step_id = graph.add_node("MyProcessor", params={"sigma": 1.5})

        assert step_id is not None
        node = graph.get_node(step_id)
        assert node is not None
        assert node.processor_name == "MyProcessor"
        assert node.params == {"sigma": 1.5}

    def test_add_node_with_position(self):
        wf = WorkflowDefinition(name="empty")
        graph = WorkflowGraph(wf)

        step_id = graph.add_node("Proc", position=(100.0, 200.0))
        node = graph.get_node(step_id)
        assert node is not None
        assert node.position == (100.0, 200.0)

    def test_add_node_with_types(self):
        wf = WorkflowDefinition(name="empty")
        graph = WorkflowGraph(wf)

        step_id = graph.add_node(
            "Detector",
            input_type="raster",
            output_type="detection_set",
        )
        node = graph.get_node(step_id)
        assert node is not None
        assert node.input_type == "raster"
        assert node.output_type == "detection_set"

    def test_remove_node(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        sid = graph.add_node("A")
        graph.remove_node(sid)

        assert graph.get_node(sid) is None
        assert len(graph.get_nodes()) == 0

    def test_remove_node_cleans_deps(self):
        """Removing a node removes it from other steps' depends_on."""
        step_a = ProcessingStep("A", id="a")
        step_b = ProcessingStep("B", id="b", depends_on=["a"])
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        graph.remove_node("a")
        node_b = graph.get_node("b")
        assert node_b is not None
        assert "a" not in node_b.depends_on

    def test_remove_nonexistent_raises(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        with pytest.raises(KeyError):
            graph.remove_node("nonexistent")

    def test_get_nodes_returns_all(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        assert len(graph.get_nodes()) == 3

    def test_get_node_not_found(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        assert graph.get_node("missing") is None


class TestGraphConnect:
    """Test connecting nodes with type validation."""

    def test_connect_compatible_types(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", input_type="raster")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        graph.connect("a", "b")
        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0].source_id == "a"
        assert edges[0].target_id == "b"

    def test_connect_incompatible_types_raises(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", input_type="feature_set")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        with pytest.raises(ValueError, match="Type mismatch"):
            graph.connect("a", "b")

    def test_connect_none_type_accepts_anything(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b")  # input_type=None
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        graph.connect("a", "b")  # should not raise
        assert len(graph.get_edges()) == 1

    def test_connect_detection_to_feature(self):
        step_a = ProcessingStep("A", id="a", output_type="detection_set")
        step_b = ProcessingStep("B", id="b", input_type="feature_set")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        graph.connect("a", "b")  # detection_set -> feature_set is valid
        assert len(graph.get_edges()) == 1

    def test_connect_nonexistent_raises(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        with pytest.raises(KeyError):
            graph.connect("a", "b")

    def test_connect_idempotent(self):
        step_a = ProcessingStep("A", id="a")
        step_b = ProcessingStep("B", id="b")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        graph.connect("a", "b")
        graph.connect("a", "b")  # second call should not duplicate
        node_b = graph.get_node("b")
        assert node_b is not None
        assert node_b.depends_on.count("a") == 1


class TestGraphDisconnect:
    """Test disconnecting edges."""

    def test_disconnect(self):
        step_a = ProcessingStep("A", id="a")
        step_b = ProcessingStep("B", id="b", depends_on=["a"])
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        graph.disconnect("a", "b")
        node_b = graph.get_node("b")
        assert node_b is not None
        assert "a" not in node_b.depends_on

    def test_disconnect_nonexistent_target_raises(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        with pytest.raises(KeyError):
            graph.disconnect("a", "nonexistent")


class TestGraphTopologicalLevels:
    """Test topological level computation."""

    def test_linear_chain(self):
        step_a = ProcessingStep("A", id="a")
        step_b = ProcessingStep("B", id="b", depends_on=["a"])
        step_c = ProcessingStep("C", id="c", depends_on=["b"])
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b, step_c])
        graph = WorkflowGraph(wf)

        levels = graph.topological_levels()
        assert len(levels) == 3
        assert levels[0] == ["a"]
        assert levels[1] == ["b"]
        assert levels[2] == ["c"]

    def test_parallel_branches(self):
        step_a = ProcessingStep("A", id="a")
        step_b = ProcessingStep("B", id="b", depends_on=["a"])
        step_c = ProcessingStep("C", id="c", depends_on=["a"])
        step_d = ProcessingStep("D", id="d", depends_on=["b", "c"])
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b, step_c, step_d])
        graph = WorkflowGraph(wf)

        levels = graph.topological_levels()
        assert len(levels) == 3
        assert levels[0] == ["a"]
        assert set(levels[1]) == {"b", "c"}
        assert levels[2] == ["d"]


class TestGraphRoundtrip:
    """Test roundtrip WorkflowDefinition <-> WorkflowGraph."""

    def test_roundtrip(self):
        step_a = ProcessingStep(
            "ProcA",
            id="a",
            params={"x": 1},
            input_type="raster",
            output_type="raster",
            position=(10.0, 20.0),
        )
        step_b = ProcessingStep(
            "ProcB",
            id="b",
            params={"y": 2},
            depends_on=["a"],
            input_type="raster",
            output_type="feature_set",
        )
        wf = WorkflowDefinition(name="roundtrip", steps=[step_a, step_b])

        graph = WorkflowGraph.from_workflow_definition(wf)
        wf2 = graph.to_workflow_definition()

        assert wf2.name == "roundtrip"
        assert len(wf2.steps) == 2
        s0 = wf2.steps[0]
        assert isinstance(s0, ProcessingStep)
        assert s0.input_type == "raster"
        assert s0.output_type == "raster"
        assert s0.position == (10.0, 20.0)

    def test_from_workflow_definition(self):
        wf = WorkflowDefinition(
            name="test",
            steps=[
                ProcessingStep("A", id="a"),
            ],
        )
        graph = WorkflowGraph.from_workflow_definition(wf)
        nodes = graph.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].processor_name == "A"


class TestGraphValidation:
    """Test graph validation."""

    def test_validate_catches_cycles(self):
        step_a = ProcessingStep("A", id="a", depends_on=["b"])
        step_b = ProcessingStep("B", id="b", depends_on=["a"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "cyclic"
        wf.version = "0.1.0"
        wf.description = ""
        wf.steps = [step_a, step_b]
        wf.tags = __import__("grdl_rt.execution.tags", fromlist=["WorkflowTags"]).WorkflowTags()
        wf.state = __import__(
            "grdl_rt.execution.workflow", fromlist=["WorkflowState"]
        ).WorkflowState.DRAFT
        wf.schema_version = "3.0"

        graph = WorkflowGraph(wf)
        errors = graph.validate()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_catches_type_mismatch(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", depends_on=["a"], input_type="feature_set")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        errors = graph.validate()
        assert any("Type mismatch" in e for e in errors)

    def test_validate_passes_for_valid_graph(self):
        step_a = ProcessingStep("A", id="a", output_type="raster")
        step_b = ProcessingStep("B", id="b", depends_on=["a"], input_type="raster")
        wf = WorkflowDefinition(name="test", steps=[step_a, step_b])
        graph = WorkflowGraph(wf)

        errors = graph.validate()
        assert len(errors) == 0


class TestGraphUpdateParams:
    """Test parameter and position updates."""

    def test_update_params(self):
        step_a = ProcessingStep("A", id="a", params={"x": 1})
        wf = WorkflowDefinition(name="test", steps=[step_a])
        graph = WorkflowGraph(wf)

        graph.update_node_params("a", {"x": 5, "y": 10})
        node = graph.get_node("a")
        assert node is not None
        assert node.params["x"] == 5
        assert node.params["y"] == 10

    def test_update_params_nonexistent_raises(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        with pytest.raises(KeyError):
            graph.update_node_params("missing", {"x": 1})


class TestGraphUpdatePosition:
    """Test position updates."""

    def test_update_position(self):
        step_a = ProcessingStep("A", id="a")
        wf = WorkflowDefinition(name="test", steps=[step_a])
        graph = WorkflowGraph(wf)

        graph.update_node_position("a", (50.0, 75.0))
        node = graph.get_node("a")
        assert node is not None
        assert node.position == (50.0, 75.0)

    def test_update_position_nonexistent_raises(self):
        wf = WorkflowDefinition(name="test")
        graph = WorkflowGraph(wf)
        with pytest.raises(KeyError):
            graph.update_node_position("missing", (0, 0))


class TestTypesCompatible:
    """Test the types_compatible helper."""

    def test_none_accepts_anything(self):
        assert types_compatible(None, "raster")
        assert types_compatible("raster", None)
        assert types_compatible(None, None)

    def test_same_types(self):
        assert types_compatible("raster", "raster")
        assert types_compatible("feature_set", "feature_set")
        assert types_compatible("detection_set", "detection_set")

    def test_detection_to_feature(self):
        assert types_compatible("detection_set", "feature_set")

    def test_incompatible(self):
        assert not types_compatible("raster", "feature_set")
        assert not types_compatible("feature_set", "raster")
        assert not types_compatible("raster", "detection_set")
        assert not types_compatible("feature_set", "detection_set")
