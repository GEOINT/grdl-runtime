# -*- coding: utf-8 -*-
"""
Accuracy Module — Ground truth comparison for detection workflows.

Computes aggregate precision, recall, F1, and IoU metrics by comparing
workflow detection results against a GeoJSON ground truth file.  Uses
only the standard library (``json``) and ``numpy``; polygon IoU falls
back to ``matplotlib.path.Path`` rasterisation when needed.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-13
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class AccuracyReport:
    """Aggregate accuracy metrics across all images in a batch.

    Attributes
    ----------
    true_positives : int
        Number of correctly matched detections (IoU >= threshold).
    false_positives : int
        Number of detections with no ground truth match.
    false_negatives : int
        Number of ground truth features with no detection match.
    precision : float
        TP / (TP + FP), or 0.0 if no detections.
    recall : float
        TP / (TP + FN), or 0.0 if no ground truth.
    f1 : float
        Harmonic mean of precision and recall.
    iou_scores : list of float
        Per-matched-detection IoU values.
    mean_iou : float
        Mean IoU across matched detections.
    details : list of dict
        Per-match details (indices, IoU, match status).
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou_scores: List[float] = field(default_factory=list)
    mean_iou: float = 0.0
    details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
            "iou_scores": self.iou_scores,
            "details": self.details,
        }


# ── GeoJSON loading ──────────────────────────────────────────────────


def load_geojson(path: Path) -> List[Dict]:
    """Load features from a GeoJSON file.

    Supports both ``FeatureCollection`` and single ``Feature`` objects.

    Parameters
    ----------
    path : Path
        Path to a ``.geojson`` or ``.json`` file.

    Returns
    -------
    list of dict
        Each dict is a GeoJSON Feature with ``geometry`` and optional
        ``properties``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("type") == "FeatureCollection":
        return data.get("features", [])
    if data.get("type") == "Feature":
        return [data]
    # Bare geometry — wrap it
    if "type" in data and "coordinates" in data:
        return [{"type": "Feature", "geometry": data, "properties": {}}]
    return []


# ── Bounding box helpers ─────────────────────────────────────────────


def _geometry_to_bbox(geometry: Dict) -> Optional[Tuple[float, float, float, float]]:
    """Extract an axis-aligned bounding box from a GeoJSON geometry.

    Returns ``(min_x, min_y, max_x, max_y)`` or ``None`` if the
    geometry type is unsupported.
    """
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates")
    if coords is None:
        return None

    if gtype == "Point":
        x, y = coords[0], coords[1]
        return (x, y, x, y)

    if gtype == "Polygon":
        ring = np.array(coords[0])
        return (
            float(ring[:, 0].min()), float(ring[:, 1].min()),
            float(ring[:, 0].max()), float(ring[:, 1].max()),
        )

    if gtype == "MultiPolygon":
        all_pts = np.concatenate([np.array(poly[0]) for poly in coords])
        return (
            float(all_pts[:, 0].min()), float(all_pts[:, 1].min()),
            float(all_pts[:, 0].max()), float(all_pts[:, 1].max()),
        )

    if gtype in ("LineString", "MultiPoint"):
        pts = np.array(coords)
        return (
            float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()),
        )

    return None


def _bbox_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two axis-aligned bounding boxes.

    Each box is ``(min_x, min_y, max_x, max_y)``.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


# ── Polygon IoU (rasterisation fallback) ─────────────────────────────


def _polygon_iou(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    resolution: int = 256,
) -> float:
    """Approximate IoU between two polygons via rasterisation.

    Uses ``matplotlib.path.Path.contains_points`` for mask generation.

    Parameters
    ----------
    poly_a, poly_b : np.ndarray
        Polygon vertices as (N, 2) arrays (x, y coords).
    resolution : int
        Rasterisation grid side length.

    Returns
    -------
    float
        Approximate IoU.
    """
    try:
        from matplotlib.path import Path as MplPath
    except ImportError:
        # Cannot compute polygon IoU without matplotlib — fall back to bbox
        bbox_a = (
            float(poly_a[:, 0].min()), float(poly_a[:, 1].min()),
            float(poly_a[:, 0].max()), float(poly_a[:, 1].max()),
        )
        bbox_b = (
            float(poly_b[:, 0].min()), float(poly_b[:, 1].min()),
            float(poly_b[:, 0].max()), float(poly_b[:, 1].max()),
        )
        return _bbox_iou(bbox_a, bbox_b)

    # Compute bounding box of both polygons
    all_pts = np.concatenate([poly_a, poly_b])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    if x_max - x_min < 1e-12 or y_max - y_min < 1e-12:
        return 0.0

    # Build sample grid
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    mask_a = MplPath(poly_a).contains_points(points)
    mask_b = MplPath(poly_b).contains_points(points)

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# ── Detection extraction ─────────────────────────────────────────────


def _extract_detections(results: List[Any]) -> List[Dict]:
    """Pull detection geometries out of workflow results.

    Handles several common result shapes:

    * ``list`` of dicts with ``"geometry"`` or ``"bbox"`` keys
    * Objects with ``.geometry`` or ``.bbox`` attributes
    * ``WorkflowResult`` whose ``.result`` is one of the above
    """
    detections: List[Dict] = []

    for r in results:
        # Unwrap WorkflowResult
        data = getattr(r, "result", r)

        if isinstance(data, (list, tuple)):
            for item in data:
                det = _coerce_detection(item)
                if det is not None:
                    detections.append(det)
        elif isinstance(data, dict):
            det = _coerce_detection(data)
            if det is not None:
                detections.append(det)
        # numpy arrays and other non-detection types are skipped

    return detections


def _coerce_detection(item: Any) -> Optional[Dict]:
    """Coerce a single item to a detection dict with ``geometry``."""
    if isinstance(item, dict):
        if "geometry" in item:
            return item
        if "bbox" in item:
            bbox = item["bbox"]
            return {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]],
                },
                "properties": item.get("properties", {}),
            }
    # Object with attributes
    geom = getattr(item, "geometry", None)
    if geom is not None:
        if isinstance(geom, dict):
            return {"geometry": geom, "properties": {}}
    bbox = getattr(item, "bbox", None)
    if bbox is not None:
        return _coerce_detection({"bbox": bbox})
    return None


# ── Matching (greedy or Hungarian) ───────────────────────────────────


def _build_iou_matrix(
    det_bboxes: List[Tuple[float, float, float, float]],
    gt_bboxes: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Build an (N_det, N_gt) IoU matrix."""
    n_det = len(det_bboxes)
    n_gt = len(gt_bboxes)
    iou_mat = np.zeros((n_det, n_gt), dtype=np.float64)
    for i in range(n_det):
        for j in range(n_gt):
            iou_mat[i, j] = _bbox_iou(det_bboxes[i], gt_bboxes[j])
    return iou_mat


def _match_hungarian(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """Hungarian matching on a cost matrix derived from IoU.

    Falls back to greedy matching if ``scipy`` is unavailable.

    Returns
    -------
    list of (det_idx, gt_idx, iou)
        Matched pairs with IoU >= threshold.
    """
    try:
        from scipy.optimize import linear_sum_assignment

        # Cost = 1 - IoU (minimise)
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            iou_val = float(iou_matrix[r, c])
            if iou_val >= iou_threshold:
                matches.append((int(r), int(c), iou_val))
        return matches
    except ImportError:
        return _match_greedy(iou_matrix, iou_threshold)


def _match_greedy(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """Greedy matching: iteratively pick the highest IoU pair."""
    mat = iou_matrix.copy()
    matches: List[Tuple[int, int, float]] = []
    used_dets: set = set()
    used_gts: set = set()

    while True:
        if mat.size == 0:
            break
        idx = int(np.argmax(mat))
        r, c = divmod(idx, mat.shape[1])
        iou_val = float(mat[r, c])
        if iou_val < iou_threshold:
            break
        matches.append((r, c, iou_val))
        used_dets.add(r)
        used_gts.add(c)
        mat[r, :] = -1.0
        mat[:, c] = -1.0

    return matches


# ── Public API ───────────────────────────────────────────────────────


def compute_accuracy(
    results: List[Any],
    ground_truth_path: Path,
    iou_threshold: float = 0.5,
) -> AccuracyReport:
    """Compare detection results against ground truth (aggregate).

    Parameters
    ----------
    results : list
        Workflow results (``WorkflowResult`` instances or raw detection
        lists).  All detections are pooled together.
    ground_truth_path : Path
        Path to a GeoJSON file with ground truth features.
    iou_threshold : float
        Minimum IoU to count as a true positive (default 0.5).

    Returns
    -------
    AccuracyReport
        Aggregate precision, recall, F1, and IoU statistics.
    """
    # Load ground truth
    gt_features = load_geojson(ground_truth_path)
    gt_bboxes: List[Tuple[float, float, float, float]] = []
    for feat in gt_features:
        geom = feat.get("geometry")
        if geom is None:
            continue
        bbox = _geometry_to_bbox(geom)
        if bbox is not None:
            gt_bboxes.append(bbox)

    # Extract detections from all results
    det_features = _extract_detections(results)
    det_bboxes: List[Tuple[float, float, float, float]] = []
    for det in det_features:
        geom = det.get("geometry")
        if geom is None:
            continue
        bbox = _geometry_to_bbox(geom)
        if bbox is not None:
            det_bboxes.append(bbox)

    n_det = len(det_bboxes)
    n_gt = len(gt_bboxes)

    # Edge cases
    if n_det == 0 and n_gt == 0:
        return AccuracyReport(precision=1.0, recall=1.0, f1=1.0)
    if n_det == 0:
        return AccuracyReport(false_negatives=n_gt)
    if n_gt == 0:
        return AccuracyReport(false_positives=n_det)

    # Build IoU matrix and match
    iou_matrix = _build_iou_matrix(det_bboxes, gt_bboxes)
    matches = _match_hungarian(iou_matrix, iou_threshold)

    tp = len(matches)
    fp = n_det - tp
    fn = n_gt - tp
    iou_scores = [m[2] for m in matches]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    details = [
        {"det_idx": m[0], "gt_idx": m[1], "iou": m[2]}
        for m in matches
    ]

    return AccuracyReport(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        iou_scores=iou_scores,
        mean_iou=mean_iou,
        details=details,
    )
