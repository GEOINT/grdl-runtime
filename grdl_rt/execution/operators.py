# -*- coding: utf-8 -*-
"""
Workflow Operators — Base classes for aggregation, conditionals, and routing.

Provides ``WorkflowOperator`` and ``DetectionAggregator``, which are
``ImageProcessor`` subtypes defined in grdl-runtime (not grdl) because
they are execution framework concepts, not image processing primitives.

These classes enable first-class operator nodes in DAG workflows:
fan-in aggregators, conditional gates, and routing steps that combine
or filter intermediate results.

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

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, TYPE_CHECKING

from grdl.image_processing.base import ImageProcessor

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata
    from grdl.image_processing.detection.models import DetectionSet


class WorkflowOperator(ImageProcessor):
    """Base class for workflow-level operations.

    Workflow operators perform structural operations on pipeline data:
    aggregation, conditional routing, fan-in/fan-out, etc.  They inherit
    from ``ImageProcessor`` so they can be used as DAG steps with the
    same ``execute()`` protocol.

    Subclasses implement ``operate()`` which receives the current
    metadata and source (which may be a dict for fan-in steps) and
    returns ``(result, updated_metadata)``.

    Examples
    --------
    >>> class MyAggregator(WorkflowOperator):
    ...     def operate(self, metadata, source, **kwargs):
    ...         merged = combine(source)
    ...         return merged, metadata
    """

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: Any,
        **kwargs: Any,
    ) -> tuple:
        """Dispatch to ``operate()`` with metadata context.

        Parameters
        ----------
        metadata : ImageMetadata
            Pipeline metadata.
        source : Any
            Input data — may be a dict ``{step_id: result}`` for
            fan-in steps, or an ndarray for linear steps.

        Returns
        -------
        tuple[Any, ImageMetadata]
        """
        self._metadata = metadata
        return self.operate(metadata, source, **kwargs)

    @abstractmethod
    def operate(
        self,
        metadata: 'ImageMetadata',
        source: Any,
        **kwargs: Any,
    ) -> tuple:
        """Perform the workflow operation.

        Parameters
        ----------
        metadata : ImageMetadata
            Pipeline metadata.
        source : Any
            Input data.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple[Any, ImageMetadata]
            ``(result, updated_metadata)``
        """
        ...


class DetectionAggregator(WorkflowOperator):
    """Base class for merging multiple ``DetectionSet`` results.

    Fan-in steps in a DAG receive a ``dict`` mapping step IDs to their
    results.  ``DetectionAggregator`` unpacks this dict and delegates
    to ``aggregate()`` which performs the merge logic.

    Subclasses implement ``aggregate(inputs, **kwargs)`` where *inputs*
    is the ``{step_id: DetectionSet}`` dict.

    Examples
    --------
    >>> class UnionAggregator(DetectionAggregator):
    ...     def aggregate(self, inputs, **kwargs):
    ...         all_dets = []
    ...         for ds in inputs.values():
    ...             all_dets.extend(ds.detections)
    ...         return DetectionSet(all_dets, ...)
    """

    def operate(
        self,
        metadata: 'ImageMetadata',
        source: Any,
        **kwargs: Any,
    ) -> tuple:
        """Unpack fan-in dict and delegate to ``aggregate()``.

        Parameters
        ----------
        metadata : ImageMetadata
        source : dict
            ``{step_id: DetectionSet}`` from DAG fan-in.

        Returns
        -------
        tuple[DetectionSet, ImageMetadata]

        Raises
        ------
        TypeError
            If source is not a dict.
        """
        if not isinstance(source, dict):
            raise TypeError(
                f"{type(self).__name__} expects a dict input from a "
                f"fan-in step, got {type(source).__name__}. Ensure this "
                f"step has multiple depends_on entries in the DAG."
            )
        result = self.aggregate(source, **kwargs)
        return result, metadata

    @abstractmethod
    def aggregate(
        self,
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> 'DetectionSet':
        """Merge multiple detection sets into one.

        Parameters
        ----------
        inputs : Dict[str, Any]
            ``{step_id: DetectionSet}`` from fan-in.
        **kwargs
            Additional parameters.

        Returns
        -------
        DetectionSet
            Merged detection set.
        """
        ...
