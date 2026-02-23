"""
Tests for grdl_rt.execution.resolver — Resolver and plan data models.

Covers:
- Resolver resolves linear workflow (all steps keep original processors)
- GPU-required step substituted when no GPU available
- GPU-required step, no GPU, no alternative → ResolutionError
- GPU-preferred step, no GPU → warning only
- Parallel groups match topological_sort output
- Global pass steps correctly flagged
- ResolvedExecutionPlan round-trip serialization
- Substitutions record what was swapped and why

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

from unittest.mock import MagicMock, patch

import pytest

from grdl_rt.execution.errors import ResolutionError
from grdl_rt.execution.hardware import GpuDeviceInfo, HardwareContext
from grdl_rt.execution.plan import (
    ParallelGroup,
    ResolvedExecutionPlan,
    ResolvedStep,
    Substitution,
)
from grdl_rt.execution.resolver import Resolver
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

# ---------------------------------------------------------------------------
# Mock HardwareContext
# ---------------------------------------------------------------------------


class _MockHW(HardwareContext):
    def __init__(self, *, gpu: bool = True):
        self._gpu = gpu

    @property
    def cpu_count(self):
        return 4

    @property
    def total_memory_bytes(self):
        return 16_000_000_000

    @property
    def available_memory_bytes(self):
        return 8_000_000_000

    @property
    def gpu_available(self):
        return self._gpu

    @property
    def gpu_devices(self):
        if self._gpu:
            return [GpuDeviceInfo("MockGPU", 8_000_000_000)]
        return []

    def to_dict(self):
        return {
            "cpu_count": self.cpu_count,
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "gpu_available": self.gpu_available,
            "gpu_devices": [
                {"name": d.name, "memory_bytes": d.memory_bytes} for d in self.gpu_devices
            ],
        }


# ---------------------------------------------------------------------------
# Mock processor classes
# ---------------------------------------------------------------------------


class _CpuOnlyProcessor:
    __processor_tags__ = {"gpu_capability": MagicMock(value="cpu_only")}


class _PreferredProcessor:
    __processor_tags__ = {"gpu_capability": MagicMock(value="preferred")}


class _RequiredProcessor:
    __processor_tags__ = {"gpu_capability": MagicMock(value="required")}


class _CpuFallbackProcessor:
    __processor_tags__ = {"gpu_capability": MagicMock(value="cpu_only")}


class _GlobalPassProcessor:
    __processor_tags__ = {"gpu_capability": MagicMock(value="preferred")}
    __has_global_pass__ = True


_PROCESSOR_MAP = {
    "CpuOnly": _CpuOnlyProcessor,
    "Preferred": _PreferredProcessor,
    "Required": _RequiredProcessor,
    "CpuFallback": _CpuFallbackProcessor,
    "GlobalPass": _GlobalPassProcessor,
}


def _mock_resolve(name):
    if name in _PROCESSOR_MAP:
        return _PROCESSOR_MAP[name]
    raise ImportError(f"Cannot resolve '{name}'")


# ---------------------------------------------------------------------------
# Mock catalog
# ---------------------------------------------------------------------------


def _make_catalog(alternatives_map=None):
    """Create a mock catalog with optional alternatives."""
    alternatives_map = alternatives_map or {}
    cat = MagicMock()

    artifacts = []
    for name, alts in alternatives_map.items():
        art = MagicMock()
        art.name = name
        art.processor_class = f"mock.module.{name}"
        art.alternatives = alts
        artifacts.append(art)

    cat.list_artifacts.return_value = artifacts
    cat.get_alternatives.return_value = []
    return cat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolverBasic:

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_linear_workflow_gpu_available(self, mock_resolve):
        """All steps keep original processors when GPU is available."""
        wf = WorkflowDefinition(
            name="Linear",
            version="1.0.0",
            steps=[
                ProcessingStep("CpuOnly", "1.0", id="s1"),
                ProcessingStep("Preferred", "1.0", id="s2", depends_on=["s1"]),
            ],
        )
        cat = _make_catalog()
        resolver = Resolver(cat)
        hw = _MockHW(gpu=True)

        plan = resolver.resolve(wf, hw)

        assert plan.workflow_name == "Linear"
        assert plan.workflow_version == "1.0.0"
        assert len(plan.steps) == 2
        assert plan.steps["s1"].resolved_processor == "CpuOnly"
        assert plan.steps["s2"].resolved_processor == "Preferred"
        assert len(plan.substitutions) == 0

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_three_step_parallel_groups(self, mock_resolve):
        """Parallel groups match topological sort levels."""
        wf = WorkflowDefinition(
            name="Parallel",
            version="1.0.0",
            steps=[
                ProcessingStep("CpuOnly", "1.0", id="root"),
                ProcessingStep("Preferred", "1.0", id="left", depends_on=["root"]),
                ProcessingStep("CpuOnly", "1.0", id="right", depends_on=["root"]),
            ],
        )
        cat = _make_catalog()
        resolver = Resolver(cat)
        plan = resolver.resolve(wf, _MockHW(gpu=True))

        assert len(plan.parallel_groups) == 2
        assert plan.parallel_groups[0].step_ids == ["root"]
        assert set(plan.parallel_groups[1].step_ids) == {"left", "right"}


class TestResolverGpuSubstitution:

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_gpu_required_no_gpu_substitutes_alternative(self, mock_resolve):
        """GPU-required step substituted when no GPU available."""
        wf = WorkflowDefinition(
            name="SubTest",
            version="1.0.0",
            steps=[
                ProcessingStep("Required", "1.0", id="s1"),
            ],
        )
        cat = _make_catalog(
            {
                "Required": [{"processor_name": "CpuFallback", "priority": 1}],
            }
        )
        resolver = Resolver(cat)
        plan = resolver.resolve(wf, _MockHW(gpu=False))

        assert plan.steps["s1"].resolved_processor == "CpuFallback"
        assert len(plan.substitutions) == 1
        sub = plan.substitutions[0]
        assert sub.original_processor == "Required"
        assert sub.replacement_processor == "CpuFallback"
        assert "GPU" in sub.reason

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_gpu_required_no_gpu_no_alternative_raises(self, mock_resolve):
        """GPU-required step with no alternative raises ResolutionError."""
        wf = WorkflowDefinition(
            name="NoAlt",
            version="1.0.0",
            steps=[
                ProcessingStep("Required", "1.0", id="s1"),
            ],
        )
        cat = _make_catalog()
        resolver = Resolver(cat)

        with pytest.raises(ResolutionError, match="GPU required"):
            resolver.resolve(wf, _MockHW(gpu=False))

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_gpu_preferred_no_gpu_warns(self, mock_resolve):
        """GPU-preferred step without GPU keeps original but warns."""
        wf = WorkflowDefinition(
            name="Warn",
            version="1.0.0",
            steps=[
                ProcessingStep("Preferred", "1.0", id="s1"),
            ],
        )
        cat = _make_catalog()
        resolver = Resolver(cat)
        plan = resolver.resolve(wf, _MockHW(gpu=False))

        assert plan.steps["s1"].resolved_processor == "Preferred"
        assert len(plan.substitutions) == 0
        assert len(plan.warnings) == 1
        assert "GPU" in plan.warnings[0]


class TestResolverGlobalPass:

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_global_pass_steps_flagged(self, mock_resolve):
        wf = WorkflowDefinition(
            name="GP",
            version="1.0.0",
            steps=[
                ProcessingStep("CpuOnly", "1.0", id="s1"),
                ProcessingStep("GlobalPass", "1.0", id="s2", depends_on=["s1"]),
            ],
        )
        cat = _make_catalog()
        resolver = Resolver(cat)
        plan = resolver.resolve(wf, _MockHW(gpu=True))

        assert "s2" in plan.global_pass_steps
        assert "s1" not in plan.global_pass_steps
        assert plan.steps["s2"].requires_global_pass is True
        assert plan.steps["s1"].requires_global_pass is False


class TestResolverInvalidDAG:

    @patch("grdl_rt.execution.resolver.resolve_processor_class", side_effect=_mock_resolve)
    def test_invalid_dag_raises(self, mock_resolve):
        s1 = ProcessingStep("CpuOnly", "1.0", id="a", depends_on=["b"])
        s2 = ProcessingStep("CpuOnly", "1.0", id="b", depends_on=["a"])
        wf = WorkflowDefinition.__new__(WorkflowDefinition)
        wf.name = "Cycle"
        wf.version = "1.0.0"
        wf.description = ""
        wf.steps = [s1, s2]
        wf.tags = None
        wf.state = None
        wf.schema_version = "2.0"

        cat = _make_catalog()
        resolver = Resolver(cat)

        with pytest.raises(ValueError, match="Invalid workflow DAG"):
            resolver.resolve(wf, _MockHW(gpu=True))


class TestPlanRoundTrip:

    def test_resolved_step_roundtrip(self):
        step = ResolvedStep(
            step_id="s1",
            original_processor="Foo",
            resolved_processor="Bar",
            processor_class_fqn="mod.Bar",
            params={"k": 1},
            gpu_capability="preferred",
            will_use_gpu=True,
            requires_global_pass=False,
            substitution_reason="test",
            estimated_memory_bytes=1024,
            retry=None,
            timeout_seconds=30.0,
            depends_on=["s0"],
        )
        d = step.to_dict()
        restored = ResolvedStep.from_dict(d)

        assert restored.step_id == "s1"
        assert restored.resolved_processor == "Bar"
        assert restored.params == {"k": 1}
        assert restored.depends_on == ["s0"]
        assert restored.timeout_seconds == 30.0

    def test_resolved_plan_roundtrip(self):
        plan = ResolvedExecutionPlan(
            workflow_name="Test",
            workflow_version="1.0.0",
            resolved_at="2026-02-11T00:00:00+00:00",
            hardware_context={"cpu_count": 4},
            steps={
                "s1": ResolvedStep(
                    step_id="s1",
                    original_processor="Foo",
                    resolved_processor="Foo",
                    processor_class_fqn="mod.Foo",
                    params={},
                    gpu_capability="cpu_only",
                    will_use_gpu=False,
                    requires_global_pass=False,
                    substitution_reason=None,
                    estimated_memory_bytes=512,
                    retry=None,
                    timeout_seconds=None,
                ),
            },
            parallel_groups=[
                ParallelGroup(level=0, step_ids=["s1"], estimated_peak_memory_bytes=512),
            ],
            global_pass_steps=[],
            substitutions=[
                Substitution(
                    step_id="s1",
                    original_processor="X",
                    replacement_processor="Y",
                    reason="test",
                ),
            ],
            estimated_total_memory_bytes=512,
            warnings=["warn1"],
        )

        d = plan.to_dict()
        restored = ResolvedExecutionPlan.from_dict(d)

        assert restored.workflow_name == "Test"
        assert len(restored.steps) == 1
        assert restored.steps["s1"].step_id == "s1"
        assert len(restored.parallel_groups) == 1
        assert len(restored.substitutions) == 1
        assert restored.substitutions[0].reason == "test"
        assert restored.warnings == ["warn1"]
