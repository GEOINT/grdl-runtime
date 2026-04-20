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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np


class ChipLabel(Enum):
    """Label for a chip indicating presence of the target signature."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


@dataclass
class ChipProvenance:
    """Typed extraction-provenance record for a single chip.

    Carries only the facts that describe *where* and *when* a chip was
    extracted from an image stack.  Channel/band information (polarization,
    frequency sub-band, etc.) must be read from the originating
    ``ImageReader.metadata.get_channel(band_index)`` at extraction time.

    Parameters
    ----------
    source_image_index : int
        Zero-based index of the source image in the image stack.
    source_image_name : str
        Human-readable name or file path of the source image.
    row_start : int
        Top row of the extracted bounding box in the source image.
    col_start : int
        Left column of the extracted bounding box in the source image.
    row_end : int
        Exclusive bottom row of the extracted bounding box.
    col_end : int
        Exclusive right column of the extracted bounding box.
    timestamp : str, optional
        ISO 8601 acquisition timestamp of the source image.
    extras : Dict[str, Any]
        Catch-all for additional provenance not covered by typed fields.
    """

    source_image_index: int = 0
    source_image_name: str = ""
    row_start: int = 0
    col_start: int = 0
    row_end: int = 0
    col_end: int = 0
    timestamp: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


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
        name: str | None = None,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.name = name

    @property
    def bounding_box(self) -> dict[str, int]:
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
            "row_start": row_min,
            "row_end": row_max,
            "col_start": col_min,
            "col_end": col_max,
        }


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
    provenance : ChipProvenance, optional
        Typed extraction-provenance record.  When not provided a
        ``ChipProvenance`` is auto-constructed from the chip's
        bounding box and positional parameters.
    annotations : list, optional
        Zero or more annotation objects (typically
        :class:`grdl.vector.models.Feature` instances from a GeoJSON
        sidecar loaded via :class:`grdl.vector.io.VectorReader`) describing
        labeled regions *within* this chip.  Empty list means the chip
        has not been annotated.
    modality : ImageModality, optional
        Image modality of this chip's source imagery.  Populated by the
        sidecar loader (from the GeoJSON FeatureCollection's ``properties``
        dict) or by :func:`~grdl.discovery.extract_modality` applied to
        the source reader's metadata.  ``None`` means unspecified — the
        chip's modality is not known or was not declared.
    """

    def __init__(
        self,
        image_data: np.ndarray,
        source_image_index: int,
        source_image_name: str,
        polygon_region: PolygonRegion,
        label: ChipLabel = ChipLabel.UNKNOWN,
        timestamp: str | None = None,
        provenance: ChipProvenance | None = None,
        annotations: Optional[list] = None,
        modality: Optional[Any] = None,  # ImageModality | None
    ) -> None:
        self.image_data = image_data
        self.source_image_index = source_image_index
        self.source_image_name = source_image_name
        self.polygon_region = polygon_region
        self.label = label
        self.timestamp = timestamp
        self.annotations: list = list(annotations) if annotations is not None else []
        self.modality = modality
        if provenance is not None:
            self.provenance = provenance
        else:
            bb = polygon_region.bounding_box
            self.provenance = ChipProvenance(
                source_image_index=source_image_index,
                source_image_name=source_image_name,
                row_start=bb['row_start'],
                col_start=bb['col_start'],
                row_end=bb['row_end'],
                col_end=bb['col_end'],
                timestamp=timestamp,
            )


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
        chips: list[Chip] | None = None,
        polygon_regions: list[PolygonRegion] | None = None,
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

    def chips_for_region(self, region: PolygonRegion) -> list[Chip]:
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
    def label_counts(self) -> dict[str, int]:
        """Count of chips by label.

        Returns
        -------
        Dict[str, int]
            Keys are ChipLabel values, values are counts.
        """
        counts: dict[str, int] = {label.value: 0 for label in ChipLabel}
        for chip in self.chips:
            counts[chip.label.value] += 1
        return counts
