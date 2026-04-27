"""
Operator Base Class Tests.

Tests for ``WorkflowOperator`` and ``DetectionAggregator`` from
``grdl_rt.execution.operators``.

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

from typing import Any

import numpy as np
import pytest

shapely = pytest.importorskip("shapely", reason="shapely not installed")
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.detection.models import Detection, DetectionSet
from grdl.image_processing.versioning import processor_version
from grdl.IO.models.base import ImageMetadata
from shapely.geometry import box  # noqa: E402

from grdl_rt.execution.operators import (
    DetectionAggregator,
    WorkflowOperator,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@processor_version("1.0.0")
class SumOperator(WorkflowOperator):
    """Operator that sums dict values."""

    def operate(self, metadata, source, **kwargs):
        result = sum(source.values()) if isinstance(source, dict) else source
        return result, metadata


@processor_version("1.0.0")
class UnionAggregator(DetectionAggregator):
    """Aggregator that unions all detections."""

    def aggregate(self, inputs: dict[str, Any], **kwargs) -> DetectionSet:
        all_dets = []
        for ds in inputs.values():
            all_dets.extend(ds.detections)
        return DetectionSet(
            detections=all_dets,
            detector_name="UnionAggregator",
            detector_version="1.0.0",
        )


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
    )


def _make_detection_set(n, name="test"):
    dets = [
        Detection(
            pixel_geometry=box(i * 10, 0, (i + 1) * 10, 10),
            properties={"idx": i},
            confidence=0.9,
        )
        for i in range(n)
    ]
    return DetectionSet(
        detections=dets,
        detector_name=name,
        detector_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# WorkflowOperator tests
# ---------------------------------------------------------------------------


class TestWorkflowOperator:

    def test_is_image_processor(self):
        """WorkflowOperator is an ImageProcessor subclass."""
        op = SumOperator()
        assert isinstance(op, ImageProcessor)

    def test_has_execute(self, meta):
        """WorkflowOperator.execute() dispatches to operate()."""
        op = SumOperator()
        source = {"a": np.array([1.0]), "b": np.array([2.0])}
        result, out_meta = op.execute(meta, source)
        np.testing.assert_allclose(result, np.array([3.0]))
        assert out_meta is meta

    def test_metadata_available(self, meta):
        """self.metadata is set during operate()."""
        op = SumOperator()
        op.execute(meta, {"a": np.array([1.0])})
        assert op.metadata is meta


# ---------------------------------------------------------------------------
# DetectionAggregator tests
# ---------------------------------------------------------------------------


class TestDetectionAggregator:

    def test_receives_dict(self, meta):
        """Fan-in dict is unpacked correctly."""
        ds1 = _make_detection_set(2, "sn4")
        ds2 = _make_detection_set(3, "sn2")
        agg = UnionAggregator()
        result, out_meta = agg.execute(meta, {"sn4": ds1, "sn2": ds2})
        assert isinstance(result, DetectionSet)
        assert len(result) == 5

    def test_non_dict_raises(self, meta):
        """Non-dict input raises TypeError."""
        agg = UnionAggregator()
        with pytest.raises(TypeError, match="expects a dict"):
            agg.execute(meta, np.zeros((10, 10)))

    def test_is_image_processor(self):
        """DetectionAggregator is a valid ImageProcessor."""
        agg = UnionAggregator()
        assert isinstance(agg, ImageProcessor)
        assert isinstance(agg, WorkflowOperator)
