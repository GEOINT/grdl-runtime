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
from typing import TYPE_CHECKING, Any

from grdl.image_processing.base import ImageProcessor

if TYPE_CHECKING:
    from grdl.image_processing.detection.models import DetectionSet
    from grdl.IO.models.base import ImageMetadata


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
        metadata: ImageMetadata,
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
        self._metadata = metadata  # type: ignore[assignment]
        return self.operate(metadata, source, **kwargs)

    @abstractmethod
    def operate(
        self,
        metadata: ImageMetadata,
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
        metadata: ImageMetadata,
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
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> DetectionSet:
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


class FeatureSetAggregator(WorkflowOperator):
    """Aggregates multiple FeatureSet results using union or intersection.

    Fan-in steps in a DAG receive a ``dict`` mapping step IDs to their
    results.  ``FeatureSetAggregator`` merges FeatureSets according to
    the configured strategy.

    Parameters
    ----------
    strategy : str
        Merge strategy: ``'union'`` (default) keeps all features from
        all inputs, ``'intersection'`` keeps only features whose
        geometries overlap across all inputs.

    Examples
    --------
    >>> agg = FeatureSetAggregator(strategy='union')
    >>> result, meta = agg.execute(metadata, {
    ...     'step_a': feature_set_a,
    ...     'step_b': feature_set_b,
    ... })
    """

    def __init__(self, strategy: str = "union") -> None:
        if strategy not in ("union", "intersection"):
            raise ValueError(f"strategy must be 'union' or 'intersection', got {strategy!r}")
        super().__init__()
        self.strategy = strategy

    def operate(
        self,
        metadata: ImageMetadata,
        source: Any,
        **kwargs: Any,
    ) -> tuple:
        """Merge FeatureSet inputs per the configured strategy.

        Parameters
        ----------
        metadata : ImageMetadata
        source : dict
            ``{step_id: FeatureSet}`` from DAG fan-in.

        Returns
        -------
        tuple[dict, ImageMetadata]
            Merged result and metadata.

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

        result = self._union(source) if self.strategy == "union" else self._intersection(source)
        return result, metadata

    def _union(self, inputs: dict[str, Any]) -> dict:
        """Merge all features from all inputs.

        Concatenates feature lists from all FeatureSet-like inputs.
        Works with dict-based FeatureSets (``{'features': [...]}``).
        """
        all_features: list = []
        for value in inputs.values():
            if isinstance(value, dict) and "features" in value:
                all_features.extend(value["features"])
            elif hasattr(value, "features"):
                all_features.extend(value.features)
            else:
                all_features.append(value)
        return {"features": all_features, "type": "FeatureSet"}

    def _intersection(self, inputs: dict[str, Any]) -> dict:
        """Keep only features present in all inputs.

        Uses feature ID matching for intersection: a feature is kept
        if its ``id`` appears in every input FeatureSet.
        """
        if not inputs:
            return {"features": [], "type": "FeatureSet"}

        # Extract feature lists
        feature_lists: list[list] = []
        for value in inputs.values():
            if isinstance(value, dict) and "features" in value:
                feature_lists.append(value["features"])
            elif hasattr(value, "features"):
                feature_lists.append(list(value.features))
            else:
                feature_lists.append([value])

        if not feature_lists:
            return {"features": [], "type": "FeatureSet"}

        # Find IDs common to all inputs
        def _get_ids(features: list) -> set:
            ids = set()
            for f in features:
                fid = f.get("id") if isinstance(f, dict) else getattr(f, "id", id(f))
                ids.add(fid)
            return ids

        common_ids = _get_ids(feature_lists[0])
        for fl in feature_lists[1:]:
            common_ids &= _get_ids(fl)

        # Keep features from the first input that match common IDs
        result_features = []
        for f in feature_lists[0]:
            fid = f.get("id") if isinstance(f, dict) else getattr(f, "id", id(f))
            if fid in common_ids:
                result_features.append(f)

        return {"features": result_features, "type": "FeatureSet"}
