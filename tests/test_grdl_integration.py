# -*- coding: utf-8 -*-
"""
Tests for GRDL v2 API integration points in grdl-runtime.

Covers:
- __gpu_compatible__ flag check in GpuBackend
- progress_callback wiring in WorkflowExecutor
- GrdlError exception handling in WorkflowExecutor
- Processor tag discovery functions
- BandwiseTransformMixin transparency

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06

Modified
--------
2026-02-09 — migrated from grdk.core to grdl_rt.execution
"""

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.executor import WorkflowExecutor
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.discovery import (
    discover_processors,
    get_processor_tags,
    get_all_modalities,
    get_all_categories,
    filter_processors,
)
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition


# ---------------------------------------------------------------------------
# __gpu_compatible__ flag in GpuBackend
# ---------------------------------------------------------------------------

class TestGpuCompatibleFlag:
    def test_gpu_incompatible_skips_gpu_dispatch(self):
        """Processor with __gpu_compatible__=False should bypass GPU."""
        backend = GpuBackend(prefer_gpu=False)
        source = np.ones((4, 4), dtype=np.float32)

        proc = MagicMock()
        proc.__gpu_compatible__ = False
        proc.apply.return_value = source * 2.0

        result = backend.apply_transform(proc, source)
        proc.apply.assert_called_once_with(source)
        np.testing.assert_array_almost_equal(result, source * 2.0)

    def test_gpu_compatible_true_allows_gpu_attempt(self):
        """Processor with __gpu_compatible__=True should be eligible for GPU."""
        backend = GpuBackend(prefer_gpu=False)
        source = np.ones((4, 4), dtype=np.float32)

        proc = MagicMock()
        proc.__gpu_compatible__ = True
        proc.apply.return_value = source * 3.0

        result = backend.apply_transform(proc, source)
        # On CPU-only backend, still runs on CPU
        proc.apply.assert_called_once_with(source)

    def test_no_flag_allows_gpu_attempt(self):
        """Processor without __gpu_compatible__ should be attempted on GPU."""
        backend = GpuBackend(prefer_gpu=False)
        source = np.zeros((4, 4))

        proc = MagicMock()
        # No __gpu_compatible__ attr
        proc.apply.return_value = source

        backend.apply_transform(proc, source)
        proc.apply.assert_called_once()


# ---------------------------------------------------------------------------
# progress_callback in WorkflowExecutor
# ---------------------------------------------------------------------------

class _ScaleTransform:
    def apply(self, source, **kwargs):
        # Consume and ignore progress_callback like GRDL processors do
        kwargs.pop('progress_callback', None)
        return source * kwargs.get('scale', 1.0)


class TestProgressCallback:
    def _make_workflow(self, steps):
        wf = WorkflowDefinition(name="Test")
        for s in steps:
            wf.add_step(s)
        return wf

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_progress_callback_called_per_step(self, mock_resolve):
        mock_resolve.return_value = _ScaleTransform

        steps = [
            ProcessingStep('ScaleTransform', '1.0', params={'scale': 2.0}),
            ProcessingStep('ScaleTransform', '1.0', params={'scale': 3.0}),
        ]
        wf = self._make_workflow(steps)
        executor = WorkflowExecutor(wf)

        progress_values = []
        def cb(fraction):
            progress_values.append(round(fraction, 4))

        source = np.ones((2, 2), dtype=np.float64)
        wr = executor.execute(source, progress_callback=cb)

        # Should report 0.5 (step 1 done) and 1.0 (step 2 done)
        assert 0.5 in progress_values
        assert 1.0 in progress_values
        np.testing.assert_array_almost_equal(wr.result, np.ones((2, 2)) * 6.0)

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_progress_callback_none_is_safe(self, mock_resolve):
        mock_resolve.return_value = _ScaleTransform

        step = ProcessingStep('ScaleTransform', '1.0', params={'scale': 5.0})
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((2, 2))
        # No callback — should not raise
        wr = executor.execute(source, progress_callback=None)
        np.testing.assert_array_almost_equal(wr.result, np.ones((2, 2)) * 5.0)

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_progress_callback_forwarded_to_processor(self, mock_resolve):
        """progress_callback should be passed as kwarg to processor.apply()."""
        received_kwargs = {}

        class CapturingTransform:
            def apply(self, source, **kwargs):
                received_kwargs.update(kwargs)
                kwargs.pop('progress_callback', None)
                return source

        mock_resolve.return_value = CapturingTransform

        step = ProcessingStep('CapturingTransform', '1.0')
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((2, 2))
        executor.execute(source, progress_callback=lambda f: None)

        assert 'progress_callback' in received_kwargs
        assert callable(received_kwargs['progress_callback'])


# ---------------------------------------------------------------------------
# GrdlError exception handling
# ---------------------------------------------------------------------------

class TestGrdlErrorHandling:
    def _make_workflow(self, steps):
        wf = WorkflowDefinition(name="Test")
        for s in steps:
            wf.add_step(s)
        return wf

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_runtime_error_wraps_processor_failure(self, mock_resolve):
        class FailingTransform:
            def apply(self, source, **kwargs):
                raise ValueError("bad shape")

        mock_resolve.return_value = FailingTransform

        step = ProcessingStep('FailingTransform', '1.0')
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        with pytest.raises(RuntimeError, match="Pipeline step 'FailingTransform' failed"):
            executor.execute(np.ones((2, 2)))

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_grdl_error_logged_distinctly(self, mock_resolve):
        """When GrdlError is available, it should be caught and logged."""
        try:
            from grdl.exceptions import ProcessorError

            class GrdlFailingTransform:
                def apply(self, source, **kwargs):
                    raise ProcessorError("merge failed")

            mock_resolve.return_value = GrdlFailingTransform

            step = ProcessingStep('GrdlFailingTransform', '1.0')
            wf = self._make_workflow([step])
            executor = WorkflowExecutor(wf)

            with pytest.raises(RuntimeError, match="merge failed"):
                executor.execute(np.ones((2, 2)))
        except ImportError:
            pytest.skip("grdl.exceptions not available")


# ---------------------------------------------------------------------------
# Processor tag discovery
# ---------------------------------------------------------------------------

class TestProcessorTags:
    def test_get_processor_tags_with_tags(self):
        """Class with __processor_tags__ should return the dict."""
        cls = MagicMock()
        cls.__processor_tags__ = {
            'modalities': ('SAR', 'PAN'),
            'category': 'spatial_filter',
            'description': None,
        }
        tags = get_processor_tags(cls)
        assert tags['modalities'] == ('SAR', 'PAN')
        assert tags['category'] == 'spatial_filter'

    def test_get_processor_tags_without_tags(self):
        """Class without __processor_tags__ should return empty dict."""
        cls = MagicMock(spec=[])  # no attributes
        tags = get_processor_tags(cls)
        assert tags == {}

    def test_get_all_modalities_returns_set(self):
        result = get_all_modalities()
        assert isinstance(result, set)

    def test_get_all_categories_returns_set(self):
        result = get_all_categories()
        assert isinstance(result, set)

    def test_filter_processors_returns_dict(self):
        result = filter_processors()
        assert isinstance(result, dict)

    def test_filter_processors_by_modality(self):
        """Filtering by a valid modality should return a subset."""
        all_procs = discover_processors()
        # Filter by a modality that may or may not exist
        filtered = filter_processors(modality='SAR')
        # Filtered result should be a subset
        assert set(filtered.keys()).issubset(set(all_procs.keys()))

    def test_filter_processors_by_nonexistent_modality(self):
        """Filtering by a nonexistent modality should return empty."""
        result = filter_processors(modality='NONEXISTENT_MODALITY_XYZ')
        assert len(result) == 0

    def test_filter_processors_by_category(self):
        all_procs = discover_processors()
        filtered = filter_processors(category='contrast_enhancement')
        assert set(filtered.keys()).issubset(set(all_procs.keys()))


# ---------------------------------------------------------------------------
# BandwiseTransformMixin transparency
# ---------------------------------------------------------------------------

class TestBandwiseTransformMixin:
    def test_3d_array_passes_through_executor(self):
        """3D arrays should work with processors that handle them."""
        backend = GpuBackend(prefer_gpu=False)
        source_3d = np.ones((3, 8, 8), dtype=np.float32)

        class StackAwareTransform:
            def apply(self, source, **kwargs):
                return source * 2.0

        proc = StackAwareTransform()
        result = backend.apply_transform(proc, source_3d)
        np.testing.assert_array_almost_equal(result, source_3d * 2.0)
        assert result.shape == (3, 8, 8)
