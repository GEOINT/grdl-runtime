# -*- coding: utf-8 -*-
"""
Data Lineage — input-to-output provenance tracking for workflow executions.

Records the complete chain of transforms from input to output: input file
hash, each processing step (processor name, version, params), and output
file hash.  Lineage is stored in ``as_executed.json`` and optionally
embedded in GeoTIFF metadata as the ``GRDL_LINEAGE`` tag.

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
2026-02-11
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)


def compute_array_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of an array's raw bytes.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash.  Uses the contiguous byte representation.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class LineageTransform:
    """Record of a single processing step in the lineage chain.

    Attributes
    ----------
    step_index : int
        Zero-based position within the workflow.
    processor_name : str
        Processor name (short or fully-qualified).
    processor_version : str, optional
        Processor ``__processor_version__``, or ``None`` if unknown.
    params : Dict[str, Any]
        Tunable parameter values used for this step.
    """

    step_index: int
    processor_name: str
    processor_version: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d: Dict[str, Any] = {
            "step_index": self.step_index,
            "processor_name": self.processor_name,
            "params": self.params,
        }
        if self.processor_version is not None:
            d["processor_version"] = self.processor_version
        return d


@dataclass
class DataLineage:
    """Complete data lineage record for a workflow execution.

    Attributes
    ----------
    input_hash : str
        SHA-256 hash of the input array bytes.
    input_shape : List[int]
        Shape of the input array.
    input_dtype : str
        Data type of the input array.
    transforms : List[LineageTransform]
        Ordered list of processing steps applied.
    output_hash : str
        SHA-256 hash of the output array bytes.
    output_shape : List[int]
        Shape of the output array.
    output_dtype : str
        Data type of the output array.
    """

    input_hash: str
    input_shape: List[int]
    input_dtype: str
    transforms: List[LineageTransform]
    output_hash: str
    output_shape: List[int]
    output_dtype: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "input_hash": self.input_hash,
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "transforms": [t.to_dict() for t in self.transforms],
            "output_hash": self.output_hash,
            "output_shape": self.output_shape,
            "output_dtype": self.output_dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataLineage:
        """Deserialize from dictionary."""
        return cls(
            input_hash=data["input_hash"],
            input_shape=data["input_shape"],
            input_dtype=data["input_dtype"],
            transforms=[
                LineageTransform(
                    step_index=t["step_index"],
                    processor_name=t["processor_name"],
                    processor_version=t.get("processor_version"),
                    params=t.get("params", {}),
                )
                for t in data["transforms"]
            ],
            output_hash=data["output_hash"],
            output_shape=data["output_shape"],
            output_dtype=data["output_dtype"],
        )


def build_lineage(
    source: np.ndarray,
    result: np.ndarray,
    workflow_steps: list,
    step_metrics: list,
) -> DataLineage:
    """Build a ``DataLineage`` from execution artifacts.

    Parameters
    ----------
    source : np.ndarray
        Original input array.
    result : np.ndarray
        Final output array.
    workflow_steps : list
        Workflow step definitions (``ProcessingStep`` or ``TapOutStepDef``).
    step_metrics : list
        ``StepMetrics`` objects from the execution.

    Returns
    -------
    DataLineage
    """
    from grdl_rt.execution.workflow import ProcessingStep

    transforms: List[LineageTransform] = []
    for sm in step_metrics:
        # Find the matching workflow step for params
        matching_step = None
        for ws in workflow_steps:
            if isinstance(ws, ProcessingStep) and ws.processor_name == sm.processor_name:
                matching_step = ws
                break

        # Try to get processor version
        proc_version = None
        try:
            from grdl_rt.execution.discovery import resolve_processor_class
            proc_cls = resolve_processor_class(sm.processor_name)
            proc_version = getattr(proc_cls, '__processor_version__', None)
        except Exception:
            pass

        transforms.append(LineageTransform(
            step_index=sm.step_index,
            processor_name=sm.processor_name,
            processor_version=proc_version,
            params=dict(matching_step.params) if matching_step else {},
        ))

    return DataLineage(
        input_hash=compute_array_hash(source),
        input_shape=list(source.shape),
        input_dtype=str(source.dtype),
        transforms=transforms,
        output_hash=compute_array_hash(result),
        output_shape=list(result.shape),
        output_dtype=str(result.dtype),
    )


def embed_lineage_geotiff(output_path: str, lineage: DataLineage) -> None:
    """Embed lineage as a ``GRDL_LINEAGE`` metadata tag in a GeoTIFF.

    Requires ``rasterio``.  If the file is not a GeoTIFF or rasterio
    is not installed, logs a warning and returns silently.

    Parameters
    ----------
    output_path : str
        Path to the GeoTIFF file to update.
    lineage : DataLineage
        Lineage data to embed.
    """
    try:
        import rasterio
    except ImportError:
        logger.debug("rasterio not available; skipping GeoTIFF lineage embed")
        return

    try:
        with rasterio.open(output_path, "r+") as ds:
            lineage_json = json.dumps(lineage.to_dict(), separators=(",", ":"))
            ds.update_tags(GRDL_LINEAGE=lineage_json)
        logger.debug("geotiff_lineage_embedded", path=output_path)
    except Exception as e:
        logger.warning(
            "geotiff_lineage_embed_failed",
            path=output_path, error=str(e),
        )
