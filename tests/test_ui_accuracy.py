"""
Tests for grdl_rt.ui._accuracy — ground truth comparison and accuracy metrics.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from grdl_rt.ui._accuracy import (
    _bbox_iou,
    _extract_detections,
    _geometry_to_bbox,
    _match_greedy,
    compute_accuracy,
    load_geojson,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def gt_geojson(tmp_path: Path) -> Path:
    """Create a ground truth GeoJSON file with 3 bounding-box features."""
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            },
            "properties": {"id": 1},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]],
            },
            "properties": {"id": 2},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]],
            },
            "properties": {"id": 3},
        },
    ]
    fc = {"type": "FeatureCollection", "features": features}
    p = tmp_path / "ground_truth.geojson"
    p.write_text(json.dumps(fc), encoding="utf-8")
    return p


@pytest.fixture
def empty_geojson(tmp_path: Path) -> Path:
    fc = {"type": "FeatureCollection", "features": []}
    p = tmp_path / "empty.geojson"
    p.write_text(json.dumps(fc), encoding="utf-8")
    return p


# ── Tests: load_geojson ─────────────────────────────────────────────


class TestLoadGeojson:
    def test_feature_collection(self, gt_geojson: Path):
        features = load_geojson(gt_geojson)
        assert len(features) == 3
        assert features[0]["geometry"]["type"] == "Polygon"

    def test_single_feature(self, tmp_path: Path):
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            "properties": {},
        }
        p = tmp_path / "single.geojson"
        p.write_text(json.dumps(feat), encoding="utf-8")
        result = load_geojson(p)
        assert len(result) == 1

    def test_bare_geometry(self, tmp_path: Path):
        geom = {"type": "Point", "coordinates": [5.0, 5.0]}
        p = tmp_path / "bare.geojson"
        p.write_text(json.dumps(geom), encoding="utf-8")
        result = load_geojson(p)
        assert len(result) == 1
        assert result[0]["type"] == "Feature"

    def test_empty(self, empty_geojson: Path):
        features = load_geojson(empty_geojson)
        assert features == []


# ── Tests: _geometry_to_bbox ─────────────────────────────────────────


class TestGeometryToBbox:
    def test_polygon(self):
        geom = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        }
        bbox = _geometry_to_bbox(geom)
        assert bbox == (0.0, 0.0, 10.0, 10.0)

    def test_point(self):
        geom = {"type": "Point", "coordinates": [5.0, 3.0]}
        bbox = _geometry_to_bbox(geom)
        assert bbox == (5.0, 3.0, 5.0, 3.0)

    def test_unsupported(self):
        bbox = _geometry_to_bbox({"type": "GeometryCollection"})
        assert bbox is None


# ── Tests: _bbox_iou ─────────────────────────────────────────────────


class TestBboxIou:
    def test_perfect_overlap(self):
        box = (0, 0, 10, 10)
        assert _bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert _bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        # Intersection = 5x5 = 25
        # Union = 100 + 100 - 25 = 175
        assert _bbox_iou(a, b) == pytest.approx(25.0 / 175.0)

    def test_contained(self):
        a = (0, 0, 20, 20)
        b = (5, 5, 10, 10)
        inter = 5 * 5
        union = 400 + 25 - 25
        assert _bbox_iou(a, b) == pytest.approx(inter / union)


# ── Tests: _extract_detections ───────────────────────────────────────


class TestExtractDetections:
    def test_dict_with_geometry(self):
        results = [
            [{"geometry": {"type": "Point", "coordinates": [1, 2]}}],
        ]
        dets = _extract_detections(results)
        assert len(dets) == 1

    def test_dict_with_bbox(self):
        results = [
            [{"bbox": [0, 0, 10, 10]}],
        ]
        dets = _extract_detections(results)
        assert len(dets) == 1
        assert dets[0]["geometry"]["type"] == "Polygon"

    def test_numpy_array_skipped(self):
        results = [np.zeros((10, 10))]
        dets = _extract_detections(results)
        assert len(dets) == 0

    def test_workflow_result_unwrap(self):
        """Objects with .result attribute should be unwrapped."""

        class FakeWR:
            def __init__(self):
                self.result = [{"geometry": {"type": "Point", "coordinates": [0, 0]}}]

        dets = _extract_detections([FakeWR()])
        assert len(dets) == 1


# ── Tests: _match_greedy ─────────────────────────────────────────────


class TestMatchGreedy:
    def test_perfect_match(self):
        iou_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        matches = _match_greedy(iou_mat, 0.5)
        assert len(matches) == 2

    def test_below_threshold(self):
        iou_mat = np.array([[0.3]])
        matches = _match_greedy(iou_mat, 0.5)
        assert len(matches) == 0

    def test_single_match(self):
        iou_mat = np.array([[0.8, 0.1], [0.2, 0.6]])
        matches = _match_greedy(iou_mat, 0.5)
        assert len(matches) == 2


# ── Tests: compute_accuracy ──────────────────────────────────────────


class TestComputeAccuracy:
    def test_perfect_detection(self, gt_geojson: Path):
        """Detections that exactly match ground truth → P=R=F1=1."""
        detections = [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                }
            },
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]],
                }
            },
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]],
                }
            },
        ]
        report = compute_accuracy([detections], gt_geojson)
        assert report.true_positives == 3
        assert report.false_positives == 0
        assert report.false_negatives == 0
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.f1 == pytest.approx(1.0)

    def test_no_detections(self, gt_geojson: Path):
        """No detections → FN = number of GT features."""
        report = compute_accuracy([[]], gt_geojson)
        assert report.true_positives == 0
        assert report.false_negatives == 3
        assert report.precision == pytest.approx(0.0)

    def test_no_ground_truth(self, empty_geojson: Path):
        """Detections with no GT → all FP."""
        detections = [
            {"geometry": {"type": "Point", "coordinates": [1, 1]}},
        ]
        report = compute_accuracy([detections], empty_geojson)
        assert report.false_positives == 1
        assert report.recall == pytest.approx(0.0)

    def test_both_empty(self, empty_geojson: Path):
        """No detections, no GT → perfect score."""
        report = compute_accuracy([[]], empty_geojson)
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.f1 == pytest.approx(1.0)

    def test_partial_match(self, gt_geojson: Path):
        """One matching detection + one false positive → partial scores."""
        detections = [
            # Matches GT feature 1
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                }
            },
            # False positive (no GT match)
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80, 80], [90, 80], [90, 90], [80, 90], [80, 80]]],
                }
            },
        ]
        report = compute_accuracy([detections], gt_geojson)
        assert report.true_positives == 1
        assert report.false_positives == 1
        assert report.false_negatives == 2
        # P = 1/2 = 0.5, R = 1/3
        assert report.precision == pytest.approx(0.5)
        assert report.recall == pytest.approx(1.0 / 3.0)
