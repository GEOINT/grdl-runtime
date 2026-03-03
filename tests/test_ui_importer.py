"""
Tests for grdl_rt.ui._importer — processor/workflow discovery and param extraction.
"""

import textwrap
from pathlib import Path

import pytest

from grdl_rt.ui._importer import (
    _params_from_schema,
    _params_from_signature,
    classify_py_file,
    discover_processors_in_module,
    discover_workflow_in_module,
    extract_tunable_params,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def tmp_component_file(tmp_path: Path) -> Path:
    """Create a temporary .py file with a fake processor class."""
    src = textwrap.dedent("""\
        import numpy as np

        class FakeFilter:
            __param_specs__ = None  # no specs, but has apply

            def __init__(self, sigma: float = 1.0, iterations: int = 3):
                self.sigma = sigma
                self.iterations = iterations

            def apply(self, source):
                return source * self.sigma
    """)
    p = tmp_path / "fake_filter.py"
    p.write_text(src, encoding="utf-8")
    return p


@pytest.fixture
def tmp_multi_processor_file(tmp_path: Path) -> Path:
    """Create a .py file with multiple processor classes."""
    src = textwrap.dedent("""\
        import numpy as np

        class FilterA:
            def apply(self, source):
                return source + 1

        class FilterB:
            def __init__(self, gain: float = 2.0):
                self.gain = gain

            def apply(self, source):
                return source * self.gain

        class NotAProcessor:
            \"\"\"No apply method.\"\"\"
            def run(self):
                pass
    """)
    p = tmp_path / "multi_procs.py"
    p.write_text(src, encoding="utf-8")
    return p


@pytest.fixture
def tmp_workflow_file(tmp_path: Path) -> Path:
    """Create a .py file that defines a workflow via build_workflow()."""
    src = textwrap.dedent("""\
        from grdl_rt.execution.builder import Workflow

        class SimpleStep:
            def apply(self, source):
                return source * 2

        def build_workflow():
            return Workflow("TestWorkflow").step(SimpleStep)
    """)
    p = tmp_path / "wf_builder.py"
    p.write_text(src, encoding="utf-8")
    return p


@pytest.fixture
def tmp_empty_file(tmp_path: Path) -> Path:
    """Create a .py file with no processors or workflows."""
    p = tmp_path / "empty.py"
    p.write_text("x = 42\n", encoding="utf-8")
    return p


# ── Tests: discover_processors_in_module ─────────────────────────────


class TestDiscoverProcessors:
    def test_single_processor(self, tmp_component_file: Path):
        procs = discover_processors_in_module(tmp_component_file)
        assert len(procs) == 1
        name, cls = procs[0]
        assert name == "FakeFilter"
        assert hasattr(cls, "apply")

    def test_multiple_processors(self, tmp_multi_processor_file: Path):
        procs = discover_processors_in_module(tmp_multi_processor_file)
        names = {n for n, _ in procs}
        assert "FilterA" in names
        assert "FilterB" in names
        # NotAProcessor has no apply() method
        assert "NotAProcessor" not in names

    def test_empty_file(self, tmp_empty_file: Path):
        procs = discover_processors_in_module(tmp_empty_file)
        assert procs == []

    def test_nonexistent_file(self, tmp_path: Path):
        with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
            discover_processors_in_module(tmp_path / "nonexistent.py")


# ── Tests: discover_workflow_in_module ───────────────────────────────


class TestDiscoverWorkflow:
    def test_workflow_via_build_function(self, tmp_workflow_file: Path):
        wf = discover_workflow_in_module(tmp_workflow_file)
        assert wf is not None
        from grdl_rt.execution.builder import Workflow

        assert isinstance(wf, Workflow)

    def test_no_workflow_in_component(self, tmp_component_file: Path):
        wf = discover_workflow_in_module(tmp_component_file)
        assert wf is None

    def test_empty_file(self, tmp_empty_file: Path):
        wf = discover_workflow_in_module(tmp_empty_file)
        assert wf is None


# ── Tests: classify_py_file ──────────────────────────────────────────


class TestClassifyPyFile:
    def test_component(self, tmp_component_file: Path):
        assert classify_py_file(tmp_component_file) == "component"

    def test_workflow(self, tmp_workflow_file: Path):
        assert classify_py_file(tmp_workflow_file) == "workflow"


# ── Tests: extract_tunable_params ────────────────────────────────────


class TestExtractTunableParams:
    def test_from_init_signature(self, tmp_component_file: Path):
        procs = discover_processors_in_module(tmp_component_file)
        _, cls = procs[0]
        params = extract_tunable_params(cls)

        assert "sigma" in params
        assert "iterations" in params

        sigma = params["sigma"]
        assert sigma.param_type == "float"
        assert sigma.default == 1.0
        assert sigma.required is False

        iters = params["iterations"]
        assert iters.param_type == "int"
        assert iters.default == 3

    def test_no_params(self, tmp_multi_processor_file: Path):
        procs = discover_processors_in_module(tmp_multi_processor_file)
        # FilterA has no __init__ params
        cls_a = dict(procs)["FilterA"]
        params = extract_tunable_params(cls_a)
        assert params == {}

    def test_class_with_metadata_param_excluded(self):
        """Parameters named 'metadata' should be excluded."""

        class _FakeWithMeta:
            def __init__(self, metadata, alpha: float = 0.5):
                pass

        params = _params_from_signature(_FakeWithMeta)
        assert "metadata" not in params
        assert "alpha" in params

    def test_required_param_detection(self):
        class _Required:
            def __init__(self, name: str, value: int = 10):
                pass

        params = _params_from_signature(_Required)
        assert params["name"].required is True
        assert params["value"].required is False


# ── Tests: _params_from_schema ───────────────────────────────────────


class TestParamsFromSchema:
    def test_basic_schema(self):
        props = {
            "sigma": {
                "type": "number",
                "default": 1.5,
                "description": "Gaussian sigma",
                "minimum": 0.1,
                "maximum": 10.0,
            },
            "mode": {
                "type": "string",
                "enum": ["reflect", "nearest", "wrap"],
                "default": "reflect",
            },
        }
        result = _params_from_schema(props, {"sigma"})

        assert result["sigma"].param_type == "float"
        assert result["sigma"].required is True
        assert result["sigma"].min_value == 0.1
        assert result["sigma"].max_value == 10.0
        assert result["sigma"].description == "Gaussian sigma"

        assert result["mode"].choices == ["reflect", "nearest", "wrap"]
        assert result["mode"].required is False


# ── Tests: _load_module edge cases ───────────────────────────────────


class TestLoadModule:
    def test_load_module_spec_none_raises(self, tmp_path: Path):
        """_load_module raises ImportError when spec_from_file_location returns None."""
        from unittest.mock import patch

        from grdl_rt.ui._importer import _load_module

        p = tmp_path / "fake.py"
        p.write_text("x = 1", encoding="utf-8")
        with (
            patch("importlib.util.spec_from_file_location", return_value=None),
            pytest.raises(ImportError, match="Cannot create import spec"),
        ):
            _load_module(p)

    def test_load_module_exec_error_cleans_up(self, tmp_path: Path):
        """_load_module removes the module from sys.modules if exec_module raises."""
        import sys

        from grdl_rt.ui._importer import _load_module

        p = tmp_path / "bad_module.py"
        p.write_text("raise RuntimeError('boom')", encoding="utf-8")
        with pytest.raises(RuntimeError, match="boom"):
            _load_module(p)
        # module should not linger in sys.modules
        leaked = [k for k in sys.modules if "bad_module" in k]
        assert leaked == []


# ── Tests: discover_workflow_in_module additional paths ──────────────


class TestDiscoverWorkflowAdditionalPaths:
    def test_module_level_workflow_instance(self, tmp_path: Path):
        """A module-level Workflow instance is detected as a workflow."""
        src = textwrap.dedent("""\
            from grdl_rt.execution.builder import Workflow

            class _Step:
                def apply(self, source):
                    return source * 2

            my_wf = Workflow("module_level").step(_Step)
        """)
        p = tmp_path / "mod_wf.py"
        p.write_text(src, encoding="utf-8")
        from grdl_rt.execution.builder import Workflow
        from grdl_rt.ui._importer import discover_workflow_in_module

        result = discover_workflow_in_module(p)
        assert isinstance(result, Workflow)

    def test_build_workflow_raises_returns_none(self, tmp_path: Path):
        """discover_workflow_in_module returns None if build_workflow() raises."""
        src = textwrap.dedent("""\
            def build_workflow():
                raise RuntimeError("intentional failure")
        """)
        p = tmp_path / "bad_wf.py"
        p.write_text(src, encoding="utf-8")
        from grdl_rt.ui._importer import discover_workflow_in_module

        result = discover_workflow_in_module(p)
        assert result is None


# ── Tests: _params_from_signature additional paths ───────────────────


class TestParamsFromSignatureEdgeCases:
    def test_param_with_no_annotation_no_default(self):
        """Parameter with no type annotation and no default → type 'str'."""

        class _NoAnnotation:
            def __init__(self, x):
                pass

        params = _params_from_signature(_NoAnnotation)
        assert "x" in params
        assert params["x"].param_type == "str"
        assert params["x"].required is True

    def test_var_args_and_kwargs_skipped(self):
        """*args and **kwargs are skipped in parameter extraction."""

        class _VarArgs:
            def __init__(self, *args, **kwargs):
                pass

        params = _params_from_signature(_VarArgs)
        assert params == {}
