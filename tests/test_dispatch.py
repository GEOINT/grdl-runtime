# -*- coding: utf-8 -*-
"""
Dispatch Module Tests.

Tests for ``execute_processor()`` and ``supports_gpu_transfer()`` from
``grdl_rt.execution.dispatch``.

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
2026-02-12
"""

from typing import Any, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from shapely.geometry import box

from grdl.IO.models.base import ImageMetadata
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import Detection, DetectionSet
from grdl_rt.execution.dispatch import (
    execute_processor,
    supports_gpu_transfer,
    _minimal_metadata,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class DoubleTransform(ImageTransform):
    def apply(self, source, **kwargs):
        return source * 2.0


class GpuTransform(ImageTransform):
    __gpu_compatible__ = True

    def apply(self, source, **kwargs):
        return source + 1


class StubDetector(ImageDetector):
    def detect(self, source, geolocation=None, **kwargs):
        det = Detection(
            pixel_geometry=box(0, 0, 5, 5),
            properties={"test": True},
            confidence=0.8,
        )
        return DetectionSet(
            detections=[det],
            detector_name="StubDetector",
            detector_version="1.0.0",
        )

    @property
    def output_fields(self) -> Tuple[str, ...]:
        return ()


class LegacyProcessor:
    """Non-GRDL processor with only apply()."""

    def apply(self, source, **kwargs):
        return source - 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def meta():
    return ImageMetadata(
        format="test",
        rows=10,
        cols=12,
        dtype="float32",
        bands=3,
    )


@pytest.fixture
def source():
    return np.ones((10, 12, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# execute_processor() tests
# ---------------------------------------------------------------------------


class TestExecuteProcessor:

    def test_execute_transform(self, meta, source):
        """ImageTransform dispatches via execute() protocol."""
        t = DoubleTransform()
        result, out_meta = execute_processor(t, meta, source)
        np.testing.assert_allclose(result, source * 2.0)
        assert isinstance(out_meta, ImageMetadata)
        assert out_meta.dtype == "float32"

    def test_execute_detector(self, meta, source):
        """ImageDetector dispatches via execute(), returns DetectionSet."""
        d = StubDetector()
        result, out_meta = execute_processor(d, meta, source)
        assert isinstance(result, DetectionSet)
        assert len(result) == 1
        assert out_meta is meta  # unchanged

    def test_execute_legacy_processor(self, meta, source):
        """Non-GRDL processor with apply() dispatches via legacy path."""
        p = LegacyProcessor()
        result, out_meta = execute_processor(p, meta, source)
        np.testing.assert_allclose(result, source - 1)
        assert out_meta is meta

    def test_execute_raw_callable(self, meta, source):
        """Raw callable dispatches via callable path."""
        fn = lambda src: src * 3
        result, out_meta = execute_processor(fn, meta, source)
        np.testing.assert_allclose(result, source * 3)
        assert out_meta is meta

    def test_execute_mock_uses_apply(self, meta, source):
        """MagicMock with apply() dispatches via legacy .apply() path."""
        mock = MagicMock()
        mock.apply.return_value = source + 5
        result, out_meta = execute_processor(mock, meta, source)
        np.testing.assert_allclose(result, source + 5)
        mock.apply.assert_called_once()

    def test_metadata_propagates_through_transform(self, meta, source):
        """Transform chain updates metadata at each step."""
        t = DoubleTransform()
        result1, meta1 = execute_processor(t, meta, source)
        assert meta1.dtype == "float32"
        assert meta1.rows == 10
        assert meta1.cols == 12
        result2, meta2 = execute_processor(t, meta1, result1)
        assert meta2.dtype == "float32"

    def test_metadata_unchanged_for_detector(self, meta, source):
        """Detector returns input metadata unchanged."""
        d = StubDetector()
        _, out_meta = execute_processor(d, meta, source)
        assert out_meta is meta
        assert out_meta.rows == 10
        assert out_meta.cols == 12

    def test_not_dispatchable_raises(self, meta, source):
        """Object with no dispatch method raises TypeError."""
        with pytest.raises(TypeError, match="Cannot dispatch"):
            execute_processor(42, meta, source)


# ---------------------------------------------------------------------------
# supports_gpu_transfer() tests
# ---------------------------------------------------------------------------


class TestSupportsGpuTransfer:

    def test_gpu_compatible_transform(self):
        """ImageTransform with __gpu_compatible__=True → True."""
        assert supports_gpu_transfer(GpuTransform()) is True

    def test_non_gpu_transform(self):
        """ImageTransform with default __gpu_compatible__=False → False."""
        assert supports_gpu_transfer(DoubleTransform()) is False

    def test_detector_returns_false(self):
        """Detectors are never GPU-transferable."""
        assert supports_gpu_transfer(StubDetector()) is False

    def test_raw_callable_returns_false(self):
        """Raw callables are not GPU-transferable."""
        assert supports_gpu_transfer(lambda x: x) is False


# ---------------------------------------------------------------------------
# _minimal_metadata() tests
# ---------------------------------------------------------------------------


class TestMinimalMetadata:

    def test_2d_array(self):
        meta = _minimal_metadata(np.zeros((10, 20), dtype=np.float32))
        assert meta.rows == 10
        assert meta.cols == 20
        assert meta.bands == 1
        assert meta.dtype == "float32"

    def test_3d_array(self):
        meta = _minimal_metadata(np.zeros((10, 20, 4), dtype=np.complex64))
        assert meta.rows == 10
        assert meta.cols == 20
        assert meta.bands == 4
        assert meta.dtype == "complex64"

    def test_1d_array(self):
        meta = _minimal_metadata(np.zeros(5, dtype=np.float64))
        assert meta.rows == 1
        assert meta.cols == 5
        assert meta.bands == 1
