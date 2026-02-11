# -*- coding: utf-8 -*-
"""
Tap-Out Tests - Unit tests for tap-out steps and auto tap-out mode.

Covers TapOutStep dataclass, fluent builder .tap_out(), pipeline execution
with tap-out, YAML round-trip, and auto tap-out directory structure.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from grdl_rt.execution.builder import Workflow, TapOutStep, WorkflowStep
from grdl_rt.execution.workflow import (
    ProcessingStep, TapOutStepDef, WorkflowDefinition,
)
from grdl_rt.execution.dsl import DslCompiler
from grdl_rt.execution.result import WorkflowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _double(source: np.ndarray) -> np.ndarray:
    return source * 2.0


def _add_one(source: np.ndarray) -> np.ndarray:
    return source + 1.0


# ---------------------------------------------------------------------------
# TapOutStep dataclass
# ---------------------------------------------------------------------------

class TestTapOutStep:
    def test_construction(self):
        ts = TapOutStep(path="out.npy", format="numpy", name="tap_out(out.npy)")
        assert ts.path == "out.npy"
        assert ts.format == "numpy"
        assert ts.name == "tap_out(out.npy)"

    def test_default_name(self):
        ts = TapOutStep(path="/tmp/test.tif")
        assert ts.name == "tap_out"

    def test_default_format_is_none(self):
        ts = TapOutStep(path="x.tif")
        assert ts.format is None


# ---------------------------------------------------------------------------
# TapOutStepDef (serializable model)
# ---------------------------------------------------------------------------

class TestTapOutStepDef:
    def test_construction(self):
        td = TapOutStepDef(path="./out.tif", format="geotiff")
        assert td.path == "./out.tif"
        assert td.format == "geotiff"

    def test_to_dict(self):
        td = TapOutStepDef(path="./out.npy")
        d = td.to_dict()
        assert d == {'tap_out': './out.npy'}

    def test_to_dict_with_format(self):
        td = TapOutStepDef(path="./out.tif", format="geotiff")
        d = td.to_dict()
        assert d == {'tap_out': './out.tif', 'format': 'geotiff'}

    def test_from_dict_roundtrip(self):
        original = TapOutStepDef(path="./debug.npy", format="numpy")
        restored = TapOutStepDef.from_dict(original.to_dict())
        assert restored.path == original.path
        assert restored.format == original.format


# ---------------------------------------------------------------------------
# Fluent builder .tap_out()
# ---------------------------------------------------------------------------

class TestWorkflowTapOut:
    def test_tap_out_adds_step(self):
        wf = Workflow("test").tap_out("debug.npy")
        assert len(wf) == 1
        assert isinstance(wf.steps[0], TapOutStep)
        assert wf.steps[0].path == "debug.npy"

    def test_tap_out_fluent_chaining(self):
        wf = Workflow("test")
        ret = wf.tap_out("debug.npy")
        assert ret is wf

    def test_tap_out_name_from_filename(self):
        wf = Workflow("test").tap_out("/some/path/debug.tif")
        assert "debug.tif" in wf.steps[0].name

    def test_tap_out_with_format(self):
        wf = Workflow("test").tap_out("out.dat", format="numpy")
        assert wf.steps[0].format == "numpy"

    def test_mixed_steps_and_tap_outs(self):
        wf = (
            Workflow("mixed")
            .step(_double)
            .tap_out("after_double.npy")
            .step(_add_one)
        )
        assert len(wf) == 3
        assert isinstance(wf.steps[0], WorkflowStep)
        assert isinstance(wf.steps[1], TapOutStep)
        assert isinstance(wf.steps[2], WorkflowStep)


# ---------------------------------------------------------------------------
# Tap-out pipeline execution
# ---------------------------------------------------------------------------

class TestTapOutExecution:
    def test_tap_out_writes_file(self, tmp_path):
        """Tap-out step writes data to disk."""
        out_path = tmp_path / "intermediate.npy"
        wf = (
            Workflow("test")
            .step(_double)
            .tap_out(str(out_path))
            .step(_add_one)
        )

        source = np.array([1.0, 2.0, 3.0])
        wr = wf.execute(source)

        # Pipeline result: (source * 2) + 1
        np.testing.assert_array_equal(wr.result, np.array([3.0, 5.0, 7.0]))

        # Tap-out file contains the intermediate (source * 2)
        tapped = np.load(str(out_path))
        np.testing.assert_array_equal(tapped, np.array([2.0, 4.0, 6.0]))

    def test_tap_out_passes_data_through_unchanged(self, tmp_path):
        """Tap-out does not modify the data flowing through the pipeline."""
        out_path = tmp_path / "tap.npy"
        wf = (
            Workflow("passthrough")
            .step(_double)
            .tap_out(str(out_path))
        )

        source = np.array([5.0])
        wr = wf.execute(source)

        # Result is from _double, not modified by tap-out
        np.testing.assert_array_equal(wr.result, np.array([10.0]))

    def test_tap_out_failure_does_not_abort_pipeline(self, tmp_path):
        """Bad tap-out path logs warning but pipeline continues."""
        wf = (
            Workflow("resilient")
            .step(_double)
            .tap_out("/nonexistent/path/bad.npy")
            .step(_add_one)
        )

        source = np.array([3.0])
        wr = wf.execute(source)

        # Pipeline completes despite bad tap-out
        np.testing.assert_array_equal(wr.result, np.array([7.0]))

    def test_multiple_tap_outs(self, tmp_path):
        """Multiple tap-out steps each write their intermediate."""
        tap1 = tmp_path / "tap1.npy"
        tap2 = tmp_path / "tap2.npy"
        wf = (
            Workflow("multi tap")
            .step(_double)
            .tap_out(str(tap1))
            .step(_add_one)
            .tap_out(str(tap2))
        )

        source = np.array([1.0])
        wr = wf.execute(source)

        np.testing.assert_array_equal(wr.result, np.array([3.0]))
        np.testing.assert_array_equal(np.load(str(tap1)), np.array([2.0]))
        np.testing.assert_array_equal(np.load(str(tap2)), np.array([3.0]))

    def test_progress_callback_includes_tap_out_steps(self, tmp_path):
        """Progress callback fires for tap-out steps too."""
        progress: list = []
        out_path = tmp_path / "tap.npy"
        wf = (
            Workflow("progress")
            .step(_double)
            .tap_out(str(out_path))
            .step(_add_one)
        )
        wf.execute(np.array([1.0]), progress_callback=progress.append)
        assert len(progress) == 3
        assert progress[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# YAML DSL round-trip with tap-out
# ---------------------------------------------------------------------------

class TestTapOutYaml:
    def test_workflow_definition_with_tap_out_serializes(self):
        """WorkflowDefinition with TapOutStepDef serializes to dict."""
        wf = WorkflowDefinition(name="Test")
        wf.add_step(ProcessingStep("FilterA", "1.0"))
        wf.add_step(TapOutStepDef(path="./debug.npy"))
        wf.add_step(ProcessingStep("FilterB", "1.0"))

        d = wf.to_dict()
        assert d['steps'][0]['processor'] == 'FilterA'
        assert d['steps'][0]['version'] == '1.0'
        assert d['steps'][1]['tap_out'] == './debug.npy'
        assert d['steps'][2]['processor'] == 'FilterB'
        assert d['steps'][2]['version'] == '1.0'

    def test_yaml_roundtrip(self, tmp_path):
        """YAML serialization preserves tap-out steps."""
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="Tap Test", version="1.0.0")
        wf.add_step(ProcessingStep("FilterA", "0.1.0", params={'k': 3}))
        wf.add_step(TapOutStepDef(path="./intermediates/after_a.tif"))
        wf.add_step(ProcessingStep("FilterB", "0.2.0"))

        yaml_str = compiler.to_yaml(wf)
        assert 'tap_out' in yaml_str

        yaml_path = tmp_path / "workflow.yaml"
        yaml_path.write_text(yaml_str)
        restored = compiler.compile_yaml(yaml_path)

        assert len(restored.steps) == 3
        assert isinstance(restored.steps[0], ProcessingStep)
        assert isinstance(restored.steps[1], TapOutStepDef)
        assert restored.steps[1].path == "./intermediates/after_a.tif"
        assert isinstance(restored.steps[2], ProcessingStep)

    def test_python_codegen_includes_tap_out(self):
        """Python DSL generation includes tap_out() calls."""
        compiler = DslCompiler()
        wf = WorkflowDefinition(name="CodeGen Test")
        wf.add_step(ProcessingStep("FilterA"))
        wf.add_step(TapOutStepDef(path="./debug.npy"))

        source = compiler.to_python(wf)
        assert 'tap_out("./debug.npy")' in source
        assert 'import workflow, step, tap_out' in source

    def test_dsl_decorator_captures_tap_out(self):
        """@workflow decorator captures tap_out() calls."""
        from grdl_rt.execution.dsl import workflow, step, tap_out

        @workflow(name="DSL Tap Test")
        def my_pipeline():
            step("FilterA")
            tap_out("./debug.npy")
            step("FilterB")

        wf = my_pipeline._workflow_definition
        assert len(wf.steps) == 3
        assert isinstance(wf.steps[1], TapOutStepDef)
        assert wf.steps[1].path == "./debug.npy"


# ---------------------------------------------------------------------------
# WorkflowExecutor with tap-out
# ---------------------------------------------------------------------------

class TestExecutorTapOut:
    def test_executor_handles_tap_out_step(self, tmp_path):
        """WorkflowExecutor handles TapOutStepDef in workflow."""
        from unittest.mock import patch
        from grdl_rt.execution.executor import WorkflowExecutor

        class _FakeTransform:
            def apply(self, source, **kwargs):
                return source * 2.0

        tap_path = tmp_path / "executor_tap.npy"
        wf = WorkflowDefinition(name="Executor Tap Test")
        wf.add_step(ProcessingStep("FakeTransform", "1.0"))
        wf.add_step(TapOutStepDef(path=str(tap_path)))

        with patch('grdl_rt.execution.executor.resolve_processor_class') as mock:
            mock.return_value = _FakeTransform
            executor = WorkflowExecutor(wf)
            source = np.ones((4, 4), dtype=np.float64)
            wr = executor.execute(source)

        np.testing.assert_array_almost_equal(wr.result, np.ones((4, 4)) * 2.0)
        tapped = np.load(str(tap_path))
        np.testing.assert_array_almost_equal(tapped, np.ones((4, 4)) * 2.0)


# ---------------------------------------------------------------------------
# Auto tap-out mode (builder)
# ---------------------------------------------------------------------------

class TestAutoTapOut:
    def test_auto_tap_out_creates_directory(self, tmp_path):
        """auto_tap_out=True creates an output directory."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("auto").step(_double).step(_add_one)
        source = np.array([1.0, 2.0, 3.0])
        wr = wf.execute(
            source,
            auto_tap_out=True,
            tap_out_dir=str(out_dir),
        )

        np.testing.assert_array_equal(wr.result, np.array([3.0, 5.0, 7.0]))
        assert out_dir.exists()

    def test_auto_tap_out_writes_intermediates(self, tmp_path):
        """Each processing step writes an intermediate file."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("auto").step(_double).step(_add_one)
        source = np.array([1.0, 2.0])
        wf.execute(source, auto_tap_out=True, tap_out_dir=str(out_dir))

        # Two processing steps → two intermediate files
        npy_files = sorted(out_dir.glob("step_*.npy"))
        assert len(npy_files) == 2

        # First intermediate: source * 2
        np.testing.assert_array_equal(
            np.load(str(npy_files[0])), np.array([2.0, 4.0]),
        )
        # Second intermediate: (source * 2) + 1
        np.testing.assert_array_equal(
            np.load(str(npy_files[1])), np.array([3.0, 5.0]),
        )

    def test_auto_tap_out_writes_final_output(self, tmp_path):
        """Auto tap-out writes a final_output file."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("auto").step(_double)
        source = np.array([4.0])
        wf.execute(source, auto_tap_out=True, tap_out_dir=str(out_dir))

        final = np.load(str(out_dir / "final_output.npy"))
        np.testing.assert_array_equal(final, np.array([8.0]))

    def test_auto_tap_out_writes_manifest(self, tmp_path):
        """Auto tap-out writes a manifest.json file."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("Manifest Test").step(_double).step(_add_one)
        source = np.array([1.0])
        wf.execute(source, auto_tap_out=True, tap_out_dir=str(out_dir))

        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest['workflow_name'] == "Manifest Test"
        assert manifest['final_output'] == "final_output.npy"
        assert len(manifest['steps']) == 2
        assert manifest['steps'][0]['type'] == 'processing'
        assert manifest['steps'][1]['type'] == 'processing'

    def test_auto_tap_out_writes_workflow_params(self, tmp_path):
        """Auto tap-out writes workflow_params.yaml."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("Params Test").step(_double)
        source = np.array([1.0])
        wf.execute(source, auto_tap_out=True, tap_out_dir=str(out_dir))

        import yaml
        yaml_path = out_dir / "workflow_params.yaml"
        assert yaml_path.exists()
        params = yaml.safe_load(yaml_path.read_text())
        assert params['name'] == "Params Test"

    def test_auto_tap_out_default_directory(self, tmp_path, monkeypatch):
        """Without tap_out_dir, creates a timestamped dir in cwd."""
        monkeypatch.chdir(tmp_path)
        wf = Workflow("cwd test").step(_double)
        source = np.array([1.0])
        wf.execute(source, auto_tap_out=True)

        run_dirs = list(tmp_path.glob("grdl_run_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "manifest.json").exists()

    def test_auto_tap_out_with_explicit_tap_out_step(self, tmp_path):
        """Explicit tap_out() steps still write to their own path."""
        out_dir = tmp_path / "run_output"
        explicit_tap = tmp_path / "explicit.npy"
        wf = (
            Workflow("mixed auto")
            .step(_double)
            .tap_out(str(explicit_tap))
            .step(_add_one)
        )
        source = np.array([2.0])
        wr = wf.execute(
            source, auto_tap_out=True, tap_out_dir=str(out_dir),
        )

        np.testing.assert_array_equal(wr.result, np.array([5.0]))
        # Explicit tap-out wrote to its path
        np.testing.assert_array_equal(
            np.load(str(explicit_tap)), np.array([4.0]),
        )
        # Auto tap-out wrote processing step intermediates
        npy_files = sorted(out_dir.glob("step_*.npy"))
        assert len(npy_files) == 2

    def test_auto_tap_out_manifest_includes_timing(self, tmp_path):
        """Manifest entries include elapsed_s field."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("timing").step(_double)
        wf.execute(np.array([1.0]), auto_tap_out=True, tap_out_dir=str(out_dir))

        manifest = json.loads((out_dir / "manifest.json").read_text())
        for entry in manifest['steps']:
            assert 'elapsed_s' in entry
            assert isinstance(entry['elapsed_s'], float)

    def test_auto_tap_out_manifest_includes_shape_dtype(self, tmp_path):
        """Manifest entries include shape and dtype."""
        out_dir = tmp_path / "run_output"
        wf = Workflow("shape").step(_double)
        source = np.ones((4, 4), dtype=np.float32)
        wf.execute(source, auto_tap_out=True, tap_out_dir=str(out_dir))

        manifest = json.loads((out_dir / "manifest.json").read_text())
        entry = manifest['steps'][0]
        assert entry['shape'] == [4, 4]
        assert entry['dtype'] == 'float32'


# ---------------------------------------------------------------------------
# Auto tap-out mode (WorkflowExecutor)
# ---------------------------------------------------------------------------

class TestExecutorAutoTapOut:
    def test_executor_auto_tap_out(self, tmp_path):
        """WorkflowExecutor auto_tap_out writes intermediates."""
        from grdl_rt.execution.executor import WorkflowExecutor

        class _FakeDouble:
            def apply(self, source, **kwargs):
                return source * 2.0

        class _FakeAddOne:
            def apply(self, source, **kwargs):
                return source + 1.0

        out_dir = tmp_path / "executor_auto"
        wf = WorkflowDefinition(name="Executor Auto Test")
        wf.add_step(ProcessingStep("FakeDouble", "1.0"))
        wf.add_step(ProcessingStep("FakeAddOne", "1.0"))

        def _mock_resolve(name):
            return {'FakeDouble': _FakeDouble, 'FakeAddOne': _FakeAddOne}[name]

        with patch('grdl_rt.execution.executor.resolve_processor_class', side_effect=_mock_resolve):
            executor = WorkflowExecutor(wf)
            source = np.ones((4, 4), dtype=np.float64)
            wr = executor.execute(
                source,
                auto_tap_out=True,
                tap_out_dir=str(out_dir),
            )

        np.testing.assert_array_almost_equal(wr.result, np.ones((4, 4)) * 2.0 + 1.0)
        assert out_dir.exists()

        # Two processing step intermediates
        npy_files = sorted(out_dir.glob("step_*.npy"))
        assert len(npy_files) == 2

        # Manifest exists
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest['workflow_name'] == "Executor Auto Test"
        assert len(manifest['steps']) == 2

        # workflow_params.yaml exists
        assert (out_dir / "workflow_params.yaml").exists()

        # final_output.npy exists
        final = np.load(str(out_dir / "final_output.npy"))
        np.testing.assert_array_almost_equal(final, np.ones((4, 4)) * 3.0)
