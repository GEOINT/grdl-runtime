# -*- coding: utf-8 -*-
"""
CLI Tests - Unit tests for the grdl-rt command-line interface.

Covers argument parsing, help output, run subcommand with mock
workflows and synthetic input, output writing, auto-tap-out,
validate subcommand, and list-processors subcommand.

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
import yaml

from grdl_rt.cli import main, _build_parser, _read_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_workflow_yaml(tmp_path, steps=None):
    """Create a minimal workflow YAML file for testing."""
    if steps is None:
        steps = [{'processor': 'FakeProcessor', 'version': '1.0'}]
    wf = {
        'name': 'Test CLI Workflow',
        'version': '1.0.0',
        'description': 'Test workflow for CLI',
        'state': 'published',
        'tags': {},
        'steps': steps,
    }
    yaml_path = tmp_path / 'workflow.yaml'
    yaml_path.write_text(yaml.dump(wf, default_flow_style=False))
    return yaml_path


def _create_npy_input(tmp_path, shape=(8, 8)):
    """Create a synthetic .npy input file."""
    data = np.random.rand(*shape).astype(np.float32)
    npy_path = tmp_path / 'input.npy'
    np.save(str(npy_path), data)
    return npy_path, data


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_no_args_returns_zero(self, capsys):
        """No arguments prints help and returns 0."""
        result = main([])
        assert result == 0

    def test_version_flag(self, capsys):
        """--version prints version and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main(['--version'])
        assert exc_info.value.code == 0

    def test_run_requires_workflow_and_input(self):
        """'run' subcommand requires workflow and input args."""
        with pytest.raises(SystemExit) as exc_info:
            main(['run'])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Read input
# ---------------------------------------------------------------------------

class TestReadInput:
    def test_read_npy_input(self, tmp_path):
        """_read_input reads .npy files directly."""
        npy_path, data = _create_npy_input(tmp_path)
        result = _read_input(npy_path)
        np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# Run subcommand
# ---------------------------------------------------------------------------

class TestRunSubcommand:
    def test_run_with_npy_input(self, tmp_path):
        """Run subcommand executes workflow on .npy input."""
        yaml_path = _create_workflow_yaml(tmp_path)
        npy_path, data = _create_npy_input(tmp_path)

        class _FakeProcessor:
            def apply(self, source, **kwargs):
                return source * 2.0

        with patch('grdl_rt.execution.executor.resolve_processor_class') as mock_exec, \
             patch('grdl_rt.execution.discovery.resolve_processor_class') as mock_disc:
            mock_exec.return_value = _FakeProcessor
            mock_disc.return_value = _FakeProcessor
            result = main(['run', str(yaml_path), str(npy_path)])

        assert result == 0

    def test_run_with_output_flag(self, tmp_path):
        """--output flag writes result to disk."""
        yaml_path = _create_workflow_yaml(tmp_path)
        npy_path, data = _create_npy_input(tmp_path)
        out_path = tmp_path / 'output.npy'

        class _FakeProcessor:
            def apply(self, source, **kwargs):
                return source * 3.0

        with patch('grdl_rt.execution.executor.resolve_processor_class') as mock_exec, \
             patch('grdl_rt.execution.discovery.resolve_processor_class') as mock_disc:
            mock_exec.return_value = _FakeProcessor
            mock_disc.return_value = _FakeProcessor
            result = main([
                'run', str(yaml_path), str(npy_path),
                '--output', str(out_path),
            ])

        assert result == 0
        assert out_path.exists()
        output = np.load(str(out_path))
        np.testing.assert_array_almost_equal(output, data * 3.0)

    def test_run_missing_workflow(self, tmp_path):
        """Run with nonexistent workflow file returns input error."""
        npy_path, _ = _create_npy_input(tmp_path)
        result = main(['run', str(tmp_path / 'missing.yaml'), str(npy_path)])
        assert result == 3  # EXIT_INPUT_ERROR

    def test_run_missing_input(self, tmp_path):
        """Run with nonexistent input file returns input error."""
        yaml_path = _create_workflow_yaml(tmp_path)
        result = main(['run', str(yaml_path), str(tmp_path / 'missing.npy')])
        assert result == 3  # EXIT_INPUT_ERROR

    def test_run_with_auto_tap_out(self, tmp_path):
        """--auto-tap-out creates directory structure."""
        yaml_path = _create_workflow_yaml(tmp_path, steps=[
            {'processor': 'FakeDouble', 'version': '1.0'},
            {'processor': 'FakeAddOne', 'version': '1.0'},
        ])
        npy_path, data = _create_npy_input(tmp_path)
        tap_dir = tmp_path / 'tap_output'

        class _FakeDouble:
            def apply(self, source, **kwargs):
                return source * 2.0

        class _FakeAddOne:
            def apply(self, source, **kwargs):
                return source + 1.0

        def _mock_resolve(name):
            return {'FakeDouble': _FakeDouble, 'FakeAddOne': _FakeAddOne}[name]

        with patch('grdl_rt.execution.executor.resolve_processor_class', side_effect=_mock_resolve), \
             patch('grdl_rt.execution.discovery.resolve_processor_class', side_effect=_mock_resolve):
            result = main([
                'run', str(yaml_path), str(npy_path),
                '--auto-tap-out',
                '--tap-out-dir', str(tap_dir),
            ])

        assert result == 0
        assert tap_dir.exists()
        assert (tap_dir / 'manifest.json').exists()
        assert (tap_dir / 'workflow_params.yaml').exists()
        assert (tap_dir / 'final_output.npy').exists()

        npy_files = sorted(tap_dir.glob('step_*.npy'))
        assert len(npy_files) == 2

    def test_run_with_prefer_gpu_flag(self, tmp_path):
        """--prefer-gpu flag is accepted without error."""
        yaml_path = _create_workflow_yaml(tmp_path)
        npy_path, _ = _create_npy_input(tmp_path)

        class _FakeProcessor:
            def apply(self, source, **kwargs):
                return source

        with patch('grdl_rt.execution.executor.resolve_processor_class') as mock_exec, \
             patch('grdl_rt.execution.discovery.resolve_processor_class') as mock_disc:
            mock_exec.return_value = _FakeProcessor
            mock_disc.return_value = _FakeProcessor
            result = main([
                'run', str(yaml_path), str(npy_path),
                '--prefer-gpu',
            ])

        assert result == 0

    def test_run_verbose_flag(self, tmp_path):
        """-v flag increases logging verbosity."""
        yaml_path = _create_workflow_yaml(tmp_path)
        npy_path, _ = _create_npy_input(tmp_path)

        class _FakeProcessor:
            def apply(self, source, **kwargs):
                return source

        with patch('grdl_rt.execution.executor.resolve_processor_class') as mock_exec, \
             patch('grdl_rt.execution.discovery.resolve_processor_class') as mock_disc:
            mock_exec.return_value = _FakeProcessor
            mock_disc.return_value = _FakeProcessor
            result = main([
                '-v', 'run', str(yaml_path), str(npy_path),
            ])

        assert result == 0

    def test_run_validation_failure(self, tmp_path):
        """Unresolvable processor in workflow returns validation failure."""
        yaml_path = _create_workflow_yaml(tmp_path)
        npy_path, _ = _create_npy_input(tmp_path)

        # Don't patch resolve — it will fail to find 'FakeProcessor'
        result = main(['run', str(yaml_path), str(npy_path)])
        assert result == 1  # EXIT_VALIDATION_FAILURE


# ---------------------------------------------------------------------------
# Validate subcommand
# ---------------------------------------------------------------------------

class TestValidateSubcommand:
    def test_validate_valid_workflow(self, tmp_path):
        """Valid workflow returns 0."""
        yaml_path = _create_workflow_yaml(tmp_path)

        class _FakeProcessor:
            def apply(self, source, **kwargs):
                return source

        with patch('grdl_rt.execution.discovery.resolve_processor_class') as mock:
            mock.return_value = _FakeProcessor
            result = main(['validate', str(yaml_path)])

        assert result == 0

    def test_validate_missing_file(self, tmp_path):
        """Missing workflow file returns input error."""
        result = main(['validate', str(tmp_path / 'missing.yaml')])
        assert result == 3  # EXIT_INPUT_ERROR

    def test_validate_bad_processor(self, tmp_path):
        """Unresolvable processor returns validation failure."""
        yaml_path = _create_workflow_yaml(tmp_path)
        result = main(['validate', str(yaml_path)])
        assert result == 1  # EXIT_VALIDATION_FAILURE

    def test_validate_empty_workflow(self, tmp_path):
        """Empty workflow (no steps) returns validation failure."""
        yaml_path = _create_workflow_yaml(tmp_path, steps=[])
        result = main(['validate', str(yaml_path)])
        assert result == 1  # EXIT_VALIDATION_FAILURE


# ---------------------------------------------------------------------------
# List-processors subcommand
# ---------------------------------------------------------------------------

class TestListProcessorsSubcommand:
    def test_list_processors_returns_zero(self):
        """list-processors returns 0."""
        result = main(['list-processors'])
        assert result == 0

    def test_list_processors_json(self, capsys):
        """--json produces valid JSON output."""
        result = main(['list-processors', '--json'])
        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)

    def test_list_processors_with_modality_filter(self):
        """--modality flag is accepted."""
        result = main(['list-processors', '--modality', 'SAR'])
        assert result == 0

    def test_list_processors_with_category_filter(self):
        """--category flag is accepted."""
        result = main(['list-processors', '--category', 'spatial_filter'])
        assert result == 0


# ---------------------------------------------------------------------------
# python -m grdl_rt
# ---------------------------------------------------------------------------

class TestPythonDashM:
    def test_main_module_exists(self):
        """grdl_rt.__main__ module can be imported."""
        import grdl_rt.__main__  # noqa: F401
