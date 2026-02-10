# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.tags — Tag taxonomy models.

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

import pytest

from grdl_rt.execution.tags import (
    DetectionType,
    ImageModality,
    ProjectTags,
    SegmentationType,
    WorkflowTags,
)


class TestImageModality:

    def test_enum_values(self):
        assert ImageModality.PAN.value == "PAN"
        assert ImageModality.SAR.value == "SAR"
        assert ImageModality.MSI.value == "MSI"

    def test_ir_sub_bands(self):
        assert ImageModality.SWIR.value == "SWIR"
        assert ImageModality.MWIR.value == "MWIR"
        assert ImageModality.LWIR.value == "LWIR"

    def test_from_string(self):
        assert ImageModality("SAR") == ImageModality.SAR

    def test_canonical_source_is_grdl(self):
        """Enums are imported from grdl.vocabulary, not defined locally."""
        from grdl.vocabulary import ImageModality as canonical
        assert ImageModality is canonical


class TestDetectionType:

    def test_enum_values(self):
        assert DetectionType.CLASSIFICATION.value == "classification"
        assert DetectionType.CHARACTERIZATION.value == "characterization"


class TestSegmentationType:

    def test_enum_values(self):
        assert SegmentationType.INSTANCE.value == "instance"
        assert SegmentationType.PANOPTIC.value == "panoptic"


class TestProjectTags:

    def test_default(self):
        tags = ProjectTags()
        assert tags.intended_target is None

    def test_with_target(self):
        tags = ProjectTags(intended_target="vehicle")
        assert tags.intended_target == "vehicle"

    def test_roundtrip(self):
        tags = ProjectTags(intended_target="building")
        d = tags.to_dict()
        restored = ProjectTags.from_dict(d)
        assert restored.intended_target == "building"


class TestWorkflowTags:

    def test_defaults(self):
        tags = WorkflowTags()
        assert tags.modalities == []
        assert tags.niirs_range == (0.0, 9.0)
        assert tags.day_capable is True
        assert tags.night_capable is False
        assert tags.detection_types == []
        assert tags.segmentation_types == []

    def test_full_construction(self):
        tags = WorkflowTags(
            modalities=[ImageModality.SAR, ImageModality.PAN],
            niirs_range=(3.0, 6.0),
            day_capable=True,
            night_capable=True,
            detection_types=[DetectionType.CLASSIFICATION],
            segmentation_types=[SegmentationType.INSTANCE],
        )
        assert len(tags.modalities) == 2
        assert tags.niirs_range == (3.0, 6.0)
        assert tags.night_capable is True

    def test_roundtrip(self):
        tags = WorkflowTags(
            modalities=[ImageModality.SAR],
            niirs_range=(2.0, 5.0),
            day_capable=False,
            night_capable=True,
            detection_types=[
                DetectionType.PHENOMENON_SIGNATURE,
                DetectionType.CLASSIFICATION,
            ],
            segmentation_types=[SegmentationType.PANOPTIC],
        )
        d = tags.to_dict()
        restored = WorkflowTags.from_dict(d)
        assert restored.modalities == [ImageModality.SAR]
        assert restored.niirs_range == (2.0, 5.0)
        assert restored.day_capable is False
        assert restored.night_capable is True
        assert len(restored.detection_types) == 2
        assert restored.segmentation_types == [SegmentationType.PANOPTIC]

    def test_to_dict_format(self):
        tags = WorkflowTags(
            modalities=[ImageModality.SAR],
            detection_types=[DetectionType.CLASSIFICATION],
        )
        d = tags.to_dict()
        assert d['modalities'] == ['SAR']
        assert d['detection_types'] == ['classification']
