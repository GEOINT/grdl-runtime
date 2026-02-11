# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.executor — WorkflowExecutor and processor resolution.

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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from grdl_rt.execution.executor import WorkflowExecutor
from grdl_rt.execution.discovery import resolve_processor_class, discover_processors
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition


# ---------------------------------------------------------------------------
# discover_processors
# ---------------------------------------------------------------------------

class TestDiscoverProcessors:
    def test_returns_dict(self):
        result = discover_processors()
        assert isinstance(result, dict)

    def test_processors_have_apply_or_estimate(self):
        processors = discover_processors()
        for name, cls in processors.items():
            assert hasattr(cls, 'apply') or hasattr(cls, 'estimate'), (
                f"{name} has neither apply nor estimate"
            )


# ---------------------------------------------------------------------------
# resolve_processor_class
# ---------------------------------------------------------------------------

class TestResolveProcessorClass:
    def test_fully_qualified_import(self):
        # Use a known stdlib class to test the FQ path
        cls = resolve_processor_class('collections.OrderedDict')
        from collections import OrderedDict
        assert cls is OrderedDict

    def test_unknown_short_name_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve"):
            resolve_processor_class("NonExistentProcessor12345")

    def test_bad_fully_qualified_raises(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            resolve_processor_class("no.such.module.ClassName")


# ---------------------------------------------------------------------------
# WorkflowExecutor
# ---------------------------------------------------------------------------

class _FakeTransform:
    """Minimal mock processor for testing."""

    def apply(self, source, **kwargs):
        scale = kwargs.get('scale', 2.0)
        return source * scale


class TestWorkflowExecutor:
    def _make_workflow(self, steps=None):
        wf = WorkflowDefinition(name="Test")
        for s in (steps or []):
            wf.add_step(s)
        return wf

    def test_empty_workflow_returns_source(self):
        wf = self._make_workflow()
        executor = WorkflowExecutor(wf)
        source = np.ones((4, 4))
        result = executor.execute(source)
        np.testing.assert_array_equal(result, source)

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_single_step_execution(self, mock_resolve):
        mock_resolve.return_value = _FakeTransform

        step = ProcessingStep(
            processor_name='FakeTransform',
            processor_version='1.0',
            params={'scale': 3.0},
        )
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((4, 4), dtype=np.float64)
        result = executor.execute(source)
        np.testing.assert_array_almost_equal(result, np.ones((4, 4)) * 3.0)

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_multi_step_pipeline(self, mock_resolve):
        mock_resolve.return_value = _FakeTransform

        steps = [
            ProcessingStep('FakeTransform', '1.0', params={'scale': 2.0}),
            ProcessingStep('FakeTransform', '1.0', params={'scale': 3.0}),
        ]
        wf = self._make_workflow(steps)
        executor = WorkflowExecutor(wf)

        source = np.ones((4, 4), dtype=np.float64)
        result = executor.execute(source)
        np.testing.assert_array_almost_equal(result, np.ones((4, 4)) * 6.0)

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_execute_batch(self, mock_resolve):
        mock_resolve.return_value = _FakeTransform

        step = ProcessingStep('FakeTransform', '1.0', params={'scale': 5.0})
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        sources = [np.ones((2, 2)) * i for i in range(3)]
        results = executor.execute_batch(sources)
        assert len(results) == 3
        for i, r in enumerate(results):
            np.testing.assert_array_almost_equal(r, np.ones((2, 2)) * i * 5.0)

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_execute_step_by_index(self, mock_resolve):
        mock_resolve.return_value = _FakeTransform

        steps = [
            ProcessingStep('FakeTransform', '1.0', params={'scale': 2.0}),
            ProcessingStep('FakeTransform', '1.0', params={'scale': 10.0}),
        ]
        wf = self._make_workflow(steps)
        executor = WorkflowExecutor(wf)

        source = np.ones((2, 2))
        # Execute only step 1 (scale=10)
        result = executor.execute_step(1, source)
        np.testing.assert_array_almost_equal(result, np.ones((2, 2)) * 10.0)

    def test_unresolvable_processor_raises(self):
        step = ProcessingStep('NoSuchProcessor999', '1.0')
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        with pytest.raises(ImportError):
            executor.execute(np.ones((2, 2)))
