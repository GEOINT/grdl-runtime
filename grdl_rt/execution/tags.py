"""
Tag Taxonomy - Enumerated tags for GRDK projects and workflows.

Provides container classes for tagging projects and workflows with their
intended targets, image modalities, quality levels, detection types, and
segmentation strategies. Tags enable filtering and discovery in the
catalog, and some can be auto-derived from processors at publish time.

The canonical enum definitions (``ImageModality``, ``DetectionType``,
``SegmentationType``) live in :mod:`grdl.vocabulary` and are re-exported
here for backward compatibility.

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
2026-02-06

Modified
--------
2026-02-10
"""

# Standard library

# Canonical vocabulary — single source of truth lives in grdl
from grdl.vocabulary import (  # noqa: F401 — re-exported
    DetectionType,
    ExecutionPhase,
    GpuCapability,
    ImageModality,
    OutputFormat,
    PolarimetricMode,
    SegmentationType,
)


class ProjectTags:
    """Tags applied to a GRDK project.

    Parameters
    ----------
    intended_target : Optional[str]
        Free-text description of the target the project addresses
        (e.g., "vehicle", "building", "ship", "runway").
    """

    def __init__(self, intended_target: str | None = None) -> None:
        self.intended_target = intended_target

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {"intended_target": self.intended_target}

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectTags":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with tag fields.

        Returns
        -------
        ProjectTags
        """
        return cls(intended_target=data.get("intended_target"))


class WorkflowTags:
    """Tags applied to a GRDK workflow definition.

    Workflows are tagged with modalities, quality requirements,
    temporal capabilities, and processing characteristics. Some
    tags can be auto-derived from the processors in the workflow
    at publish time.

    Parameters
    ----------
    modalities : List[ImageModality]
        Required input image modalities (boolean AND).
    niirs_range : Tuple[float, float]
        Minimum and maximum NIIRS quality range (e.g., (3.0, 6.0)).
    day_capable : bool
        Whether the workflow works on daytime imagery.
    night_capable : bool
        Whether the workflow works on nighttime imagery.
    detection_types : List[DetectionType]
        Types of detection performed.
    segmentation_types : List[SegmentationType]
        Types of segmentation produced.
    pol_mode : Optional[PolarimetricMode]
        Required polarimetric collection mode.  ``None`` means the
        workflow places no constraint on polarimetric diversity.
        Set to ``PolarimetricMode.QUAD_POL`` for workflows that require
        HH, HV, VH, and VV channels simultaneously.
    """

    def __init__(
        self,
        modalities: list[ImageModality] | None = None,
        niirs_range: tuple[float, float] | None = None,
        day_capable: bool = True,
        night_capable: bool = False,
        detection_types: list[DetectionType] | None = None,
        segmentation_types: list[SegmentationType] | None = None,
        pol_mode: PolarimetricMode | None = None,
    ) -> None:
        self.modalities = modalities or []
        self.niirs_range = niirs_range or (0.0, 9.0)
        self.day_capable = day_capable
        self.night_capable = night_capable
        self.detection_types = detection_types or []
        self.segmentation_types = segmentation_types or []
        self.pol_mode = pol_mode

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/YAML storage.

        Returns
        -------
        dict
            Dictionary representation with enum values as strings.
        """
        return {
            "modalities": [m.value for m in self.modalities],
            "niirs_range": list(self.niirs_range),
            "day_capable": self.day_capable,
            "night_capable": self.night_capable,
            "detection_types": [d.value for d in self.detection_types],
            "segmentation_types": [s.value for s in self.segmentation_types],
            "pol_mode": self.pol_mode.value if self.pol_mode is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowTags":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with tag fields. Enum values as strings.

        Returns
        -------
        WorkflowTags
        """
        return cls(
            modalities=[ImageModality(m) for m in data.get("modalities", [])],
            niirs_range=tuple(data.get("niirs_range", [0.0, 9.0])),
            day_capable=data.get("day_capable", True),
            night_capable=data.get("night_capable", False),
            detection_types=[DetectionType(d) for d in data.get("detection_types", [])],
            segmentation_types=[SegmentationType(s) for s in data.get("segmentation_types", [])],
            pol_mode=(
                PolarimetricMode(data["pol_mode"]) if data.get("pol_mode") is not None else None
            ),
        )
