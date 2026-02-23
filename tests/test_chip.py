# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.chip — ChipLabel, PolygonRegion, Chip, ChipSet.

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

import numpy as np
import pytest

from grdl_rt.execution.chip import Chip, ChipLabel, ChipSet, PolygonRegion

# ---------------------------------------------------------------------------
# ChipLabel
# ---------------------------------------------------------------------------


class TestChipLabel:
    def test_enum_values(self):
        assert ChipLabel.POSITIVE.value == "positive"
        assert ChipLabel.NEGATIVE.value == "negative"
        assert ChipLabel.UNKNOWN.value == "unknown"

    def test_all_members(self):
        assert set(ChipLabel) == {
            ChipLabel.POSITIVE,
            ChipLabel.NEGATIVE,
            ChipLabel.UNKNOWN,
        }


# ---------------------------------------------------------------------------
# PolygonRegion
# ---------------------------------------------------------------------------


class TestPolygonRegion:
    def test_basic_construction(self):
        verts = np.array([[10, 20], [10, 80], [50, 80], [50, 20]])
        region = PolygonRegion(vertices=verts, name="ROI-1")
        assert region.name == "ROI-1"
        assert region.vertices.shape == (4, 2)

    def test_bounding_box(self):
        verts = np.array([[10.5, 20.3], [10.5, 80.7], [50.8, 80.7], [50.8, 20.3]])
        region = PolygonRegion(vertices=verts)
        bb = region.bounding_box
        assert bb["row_start"] == 10
        assert bb["row_end"] == 51
        assert bb["col_start"] == 20
        assert bb["col_end"] == 81

    def test_bounding_box_single_point(self):
        verts = np.array([[5.0, 10.0]])
        region = PolygonRegion(vertices=verts)
        bb = region.bounding_box
        assert bb["row_start"] == 5
        assert bb["row_end"] == 5
        assert bb["col_start"] == 10
        assert bb["col_end"] == 10

    def test_bounding_box_negative_coords(self):
        verts = np.array([[-10.5, -20.3], [-5.2, -3.1]])
        region = PolygonRegion(vertices=verts)
        bb = region.bounding_box
        assert bb["row_start"] == -11
        assert bb["row_end"] == -5
        assert bb["col_start"] == -21
        assert bb["col_end"] == -3

    def test_to_dict_from_dict_roundtrip(self):
        verts = np.array([[10, 20], [30, 40]])
        region = PolygonRegion(vertices=verts, name="test")
        data = region.to_dict()
        restored = PolygonRegion.from_dict(data)
        np.testing.assert_array_almost_equal(restored.vertices, region.vertices)
        assert restored.name == "test"

    def test_vertices_coerced_to_float64(self):
        verts = [[1, 2], [3, 4]]  # plain list
        region = PolygonRegion(vertices=verts)
        assert region.vertices.dtype == np.float64


# ---------------------------------------------------------------------------
# Chip
# ---------------------------------------------------------------------------


class TestChip:
    def _make_region(self):
        return PolygonRegion(np.array([[0, 0], [0, 8], [8, 8], [8, 0]]))

    def test_basic_construction(self):
        region = self._make_region()
        data = np.zeros((8, 8), dtype=np.uint8)
        chip = Chip(
            image_data=data,
            source_image_index=0,
            source_image_name="test.tif",
            polygon_region=region,
        )
        assert chip.label == ChipLabel.UNKNOWN
        assert chip.source_image_index == 0
        assert chip.source_image_name == "test.tif"
        assert chip.metadata == {}
        assert chip.timestamp is None

    def test_with_label_and_metadata(self):
        region = self._make_region()
        data = np.ones((4, 4), dtype=np.float32)
        chip = Chip(
            image_data=data,
            source_image_index=2,
            source_image_name="img_002.tif",
            polygon_region=region,
            label=ChipLabel.POSITIVE,
            timestamp="2026-01-15T12:00:00Z",
            metadata={"sensor": "SAR"},
        )
        assert chip.label == ChipLabel.POSITIVE
        assert chip.timestamp == "2026-01-15T12:00:00Z"
        assert chip.metadata["sensor"] == "SAR"


# ---------------------------------------------------------------------------
# ChipSet
# ---------------------------------------------------------------------------


class TestChipSet:
    def _make_chip(self, region, label=ChipLabel.UNKNOWN, idx=0):
        return Chip(
            image_data=np.zeros((4, 4), dtype=np.uint8),
            source_image_index=idx,
            source_image_name=f"img_{idx}.tif",
            polygon_region=region,
            label=label,
        )

    def test_empty(self):
        cs = ChipSet()
        assert len(cs) == 0
        assert list(cs) == []

    def test_len_and_iter(self):
        region = PolygonRegion(np.array([[0, 0], [0, 4], [4, 4], [4, 0]]))
        cs = ChipSet(
            chips=[
                self._make_chip(region, idx=0),
                self._make_chip(region, idx=1),
            ]
        )
        assert len(cs) == 2
        assert all(isinstance(c, Chip) for c in cs)

    def test_getitem(self):
        region = PolygonRegion(np.array([[0, 0], [0, 4], [4, 4], [4, 0]]))
        chip0 = self._make_chip(region, idx=0)
        chip1 = self._make_chip(region, idx=1)
        cs = ChipSet(chips=[chip0, chip1])
        assert cs[0] is chip0
        assert cs[1] is chip1

    def test_add_chip(self):
        cs = ChipSet()
        region = PolygonRegion(np.array([[0, 0], [0, 4], [4, 4], [4, 0]]))
        chip = self._make_chip(region)
        cs.add_chip(chip)
        assert len(cs) == 1

    def test_label_counts(self):
        region = PolygonRegion(np.array([[0, 0], [0, 4], [4, 4], [4, 0]]))
        cs = ChipSet(
            chips=[
                self._make_chip(region, ChipLabel.POSITIVE, 0),
                self._make_chip(region, ChipLabel.POSITIVE, 1),
                self._make_chip(region, ChipLabel.NEGATIVE, 2),
                self._make_chip(region, ChipLabel.UNKNOWN, 3),
            ]
        )
        counts = cs.label_counts
        assert counts["positive"] == 2
        assert counts["negative"] == 1
        assert counts["unknown"] == 1

    def test_chips_for_region_identity(self):
        region_a = PolygonRegion(np.array([[0, 0], [0, 4], [4, 4], [4, 0]]), name="A")
        region_b = PolygonRegion(np.array([[10, 10], [10, 14], [14, 14], [14, 10]]), name="B")
        cs = ChipSet(
            chips=[
                self._make_chip(region_a, idx=0),
                self._make_chip(region_b, idx=1),
                self._make_chip(region_a, idx=2),
            ],
            polygon_regions=[region_a, region_b],
        )
        a_chips = cs.chips_for_region(region_a)
        assert len(a_chips) == 2
        assert all(c.polygon_region is region_a for c in a_chips)

        b_chips = cs.chips_for_region(region_b)
        assert len(b_chips) == 1
