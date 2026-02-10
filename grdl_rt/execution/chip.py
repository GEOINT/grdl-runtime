# -*- coding: utf-8 -*-
"""
Chip Models - Data models for image chips and labeled chip collections.

A chip is a spatial subset of an image extracted at a polygon region
of interest. Chips carry labels (positive, negative, unknown) indicating
whether the object/signature of interest is present in the chip.

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
2026-02-06
"""

# Standard library
from enum import Enum
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np


class ChipLabel(Enum):
    """Label for a chip indicating presence of the target signature."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class PolygonRegion:
    """A polygon drawn on the image stack defining a region of interest.

    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices in pixel coordinates. Shape (N, 2) where
        columns are (row, col).
    name : Optional[str]
        Human-readable name for this region.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        name: Optional[str] = None,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.name = name

    @property
    def bounding_box(self) -> Dict[str, int]:
        """Axis-aligned bounding box enclosing the polygon.

        Returns
        -------
        Dict[str, int]
            Keys: row_start, row_end, col_start, col_end (inclusive start,
            exclusive end, suitable for array slicing).
        """
        row_min = int(np.floor(self.vertices[:, 0].min()))
        row_max = int(np.ceil(self.vertices[:, 0].max()))
        col_min = int(np.floor(self.vertices[:, 1].min()))
        col_max = int(np.ceil(self.vertices[:, 1].max()))
        return {
            'row_start': row_min,
            'row_end': row_max,
            'col_start': col_min,
            'col_end': col_max,
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            'vertices': self.vertices.tolist(),
            'name': self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PolygonRegion':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        PolygonRegion
        """
        return cls(
            vertices=np.array(data['vertices']),
            name=data.get('name'),
        )


class Chip:
    """A single image chip extracted from a polygon region.

    Parameters
    ----------
    image_data : np.ndarray
        Chip pixel data. Shape (rows, cols) or (rows, cols, bands).
    source_image_index : int
        Index of the source image in the image stack.
    source_image_name : str
        Name or path of the source image.
    polygon_region : PolygonRegion
        The polygon from which this chip was extracted.
    label : ChipLabel
        Label for this chip.
    timestamp : Optional[str]
        Acquisition timestamp of the source image (ISO 8601).
    metadata : Optional[Dict[str, Any]]
        Additional metadata (sensor, band info, etc.).
    """

    def __init__(
        self,
        image_data: np.ndarray,
        source_image_index: int,
        source_image_name: str,
        polygon_region: PolygonRegion,
        label: ChipLabel = ChipLabel.UNKNOWN,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_data = image_data
        self.source_image_index = source_image_index
        self.source_image_name = source_image_name
        self.polygon_region = polygon_region
        self.label = label
        self.timestamp = timestamp
        self.metadata = metadata or {}


class ChipSet:
    """A collection of chips with their polygon regions and labels.

    Parameters
    ----------
    chips : List[Chip]
        Ordered list of chips.
    polygon_regions : List[PolygonRegion]
        The polygon regions from which chips were extracted.
    """

    def __init__(
        self,
        chips: Optional[List[Chip]] = None,
        polygon_regions: Optional[List[PolygonRegion]] = None,
    ) -> None:
        self.chips = chips or []
        self.polygon_regions = polygon_regions or []

    def __len__(self) -> int:
        return len(self.chips)

    def __iter__(self):
        return iter(self.chips)

    def __getitem__(self, index: int) -> Chip:
        return self.chips[index]

    def add_chip(self, chip: Chip) -> None:
        """Add a chip to the collection.

        Parameters
        ----------
        chip : Chip
        """
        self.chips.append(chip)

    def chips_for_region(self, region: PolygonRegion) -> List[Chip]:
        """Get all chips extracted from a specific polygon region.

        Parameters
        ----------
        region : PolygonRegion

        Returns
        -------
        List[Chip]
        """
        return [c for c in self.chips if c.polygon_region is region]

    @property
    def label_counts(self) -> Dict[str, int]:
        """Count of chips by label.

        Returns
        -------
        Dict[str, int]
            Keys are ChipLabel values, values are counts.
        """
        counts: Dict[str, int] = {label.value: 0 for label in ChipLabel}
        for chip in self.chips:
            counts[chip.label.value] += 1
        return counts
