# -*- coding: utf-8 -*-
"""
Workflow Builder - Fluent, IDE-friendly API for typed processing workflows.

Provides the ``Workflow`` class, a builder that holds live callable references
(bound methods, functions, ``ImageTransform`` instances) rather than string
processor names.  This is the primary Python API for defining workflows in
code where IDE tooling (IntelliSense, autocomplete, type checking) matters.

For serializable / YAML-based workflows, use ``WorkflowDefinition`` and
``ProcessingStep`` from ``grdl_rt.execution.workflow``.

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
2026-02-11
"""

# Standard library
import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.tags import ImageModality, WorkflowTags

logger = logging.getLogger(__name__)

# GRDL base class (optional — graceful fallback if grdl is unavailable)
try:
    from grdl.image_processing.base import ImageTransform
except ImportError:
    ImageTransform = None  # type: ignore[misc,assignment]

# GRDL exceptions (optional — graceful fallback if grdl is old)
try:
    from grdl.exceptions import GrdlError
except ImportError:
    GrdlError = None  # type: ignore[misc,assignment]


@dataclass
class WorkflowStep:
    """A single step in a live :class:`Workflow`.

    Holds a callable reference along with display metadata.  The callable
    is fully bound at construction time — it accepts exactly one positional
    ``np.ndarray`` argument and returns an ``np.ndarray``.

    Attributes
    ----------
    fn : Callable[[np.ndarray], np.ndarray]
        The step callable.
    name : str
        Human-readable step name for logging and error messages.
    gpu_compatible : bool
        Whether this step may be accelerated via CuPy GPU transfer.
    """

    fn: Callable[[np.ndarray], np.ndarray]
    name: str
    gpu_compatible: bool


class Workflow:
    """Fluent builder for typed image processing workflows.

    Unlike :class:`~grdl_rt.execution.workflow.WorkflowDefinition` (which
    uses string-based processor references for serialization), ``Workflow``
    holds live callable references providing full IDE support for
    autocompletion, type checking, and inline documentation.

    Parameters
    ----------
    name : str
        Human-readable workflow name.
    version : str
        Semantic version of this workflow definition.
    description : str
        Description of what the workflow does.
    modalities : Optional[List[str]]
        Image modality tags (e.g., ``["SAR"]``).  Ignored when
        *tags* is provided.
    tags : Optional[WorkflowTags]
        Full workflow tags.  If provided, *modalities* is ignored.

    Examples
    --------
    >>> from grdl.image_processing.sar import SublookDecomposition
    >>> from grdl.data_prep import Normalizer
    >>> from grdl_rt import Workflow
    >>>
    >>> sublook = SublookDecomposition(metadata, num_looks=3)
    >>> normalizer = Normalizer(method='percentile')
    >>>
    >>> wf = (
    ...     Workflow("Sublook Pipeline", modalities=["SAR"])
    ...     .step(sublook.decompose, name="Decompose")
    ...     .step(sublook.to_db, name="To dB")
    ...     .step(normalizer.normalize, name="Normalize")
    ... )
    >>> result = wf.execute(image, prefer_gpu=True)
    """

    def __init__(
        self,
        name: str,
        *,
        version: str = "0.1.0",
        description: str = "",
        modalities: Optional[List[str]] = None,
        tags: Optional[WorkflowTags] = None,
    ) -> None:
        self._name = name
        self._version = version
        self._description = description
        self._steps: List[WorkflowStep] = []
        self._source: Optional[Callable[[], np.ndarray]] = None

        if tags is not None:
            self._tags = tags
        elif modalities:
            self._tags = WorkflowTags(
                modalities=[ImageModality(m) for m in modalities],
            )
        else:
            self._tags = WorkflowTags()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable workflow name."""
        return self._name

    @property
    def version(self) -> str:
        """Semantic version string."""
        return self._version

    @property
    def description(self) -> str:
        """Workflow description."""
        return self._description

    @property
    def tags(self) -> WorkflowTags:
        """Workflow classification tags."""
        return self._tags

    @property
    def steps(self) -> List[WorkflowStep]:
        """Shallow copy of the step list."""
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [s.name for s in self._steps]
        return f"Workflow({self._name!r}, steps={step_names})"

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def source(
        self,
        fn: Callable[..., np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> 'Workflow':
        """Set the data source for this workflow.

        The source callable is invoked when :meth:`execute` is called
        without a *source* array.  All positional and keyword arguments
        are captured and forwarded to *fn* at execution time.

        Parameters
        ----------
        fn : callable
            A callable that returns an ``np.ndarray`` when invoked.
        *args, **kwargs
            Arguments forwarded to *fn* at execution time.

        Returns
        -------
        Workflow
            Self, for fluent chaining.
        """
        self._source = functools.partial(fn, *args, **kwargs)
        return self

    def step(
        self,
        callable_or_transform: Union[Callable, Any],
        *,
        name: Optional[str] = None,
    ) -> 'Workflow':
        """Add a processing step to the workflow.

        Accepts any callable that takes a single ``np.ndarray`` and returns
        an ``np.ndarray``.  Also accepts ``ImageTransform`` instances
        (auto-wrapped to call ``.apply()``).

        Parameters
        ----------
        callable_or_transform : callable or ImageTransform
            The step operation.  Bound methods (e.g., ``sublook.decompose``),
            plain functions, lambdas, and ``ImageTransform`` instances are
            all supported.
        name : str, optional
            Display name for logging and error messages.  Auto-derived
            from the callable if not provided.

        Returns
        -------
        Workflow
            Self, for fluent chaining.

        Raises
        ------
        TypeError
            If *callable_or_transform* is a class (not an instance) or
            is not callable.
        """
        obj = callable_or_transform

        # Reject classes passed without instantiation
        if isinstance(obj, type):
            raise TypeError(
                f"step() expects an instance or callable, got class "
                f"'{obj.__name__}'.  Did you mean {obj.__name__}(...)?"
            )

        # ImageTransform instance → wrap .apply()
        if ImageTransform is not None and isinstance(obj, ImageTransform):
            fn = obj.apply
            step_name = name or type(obj).__name__
            gpu_ok = getattr(obj, '__gpu_compatible__', False)

        # Callable (bound method, function, lambda, etc.)
        elif callable(obj):
            fn = obj
            step_name = name or _infer_step_name(obj)
            gpu_ok = _infer_gpu_compatible(obj)

        else:
            raise TypeError(
                f"step() requires a callable or ImageTransform instance, "
                f"got {type(obj).__name__}"
            )

        self._steps.append(WorkflowStep(fn=fn, name=step_name, gpu_compatible=gpu_ok))
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        source: Optional[np.ndarray] = None,
        *,
        prefer_gpu: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Execute the workflow pipeline on a single input.

        Runs each step in sequence, passing the output of one step as
        the input to the next.  Optionally attempts GPU acceleration
        for steps that declare compatibility.

        If *source* is ``None``, the workflow's configured source
        (set via :meth:`source`) is called to obtain the input array.

        Parameters
        ----------
        source : np.ndarray, optional
            Input array.  If ``None``, uses the configured source.
        prefer_gpu : bool
            If ``True``, attempt GPU acceleration for compatible steps
            with automatic CPU fallback.  Default ``False``.
        progress_callback : callable, optional
            Called with a float in ``[0.0, 1.0]`` after each step
            completes.

        Returns
        -------
        np.ndarray
            Result after all steps have been applied.

        Raises
        ------
        ValueError
            If no source array is provided and no source is configured.
        RuntimeError
            If any step fails.
        """
        if source is None:
            if self._source is None:
                raise ValueError(
                    "No source provided.  Pass an array to execute() "
                    "or set a source with .source()."
                )
            source = self._source()

        if not self._steps:
            return source

        gpu = GpuBackend(prefer_gpu=prefer_gpu)
        n_steps = len(self._steps)
        current = source

        for i, ws in enumerate(self._steps):
            logger.debug(
                "Workflow '%s' step %d/%d: %s",
                self._name, i + 1, n_steps, ws.name,
            )
            current = self._execute_step_gpu_aware(ws, current, gpu, i, n_steps)
            if progress_callback is not None:
                progress_callback((i + 1) / n_steps)

        return current

    def execute_batch(
        self,
        sources: List[np.ndarray],
        *,
        prefer_gpu: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Execute the workflow on multiple inputs.

        Runs the full pipeline on each source array in order.

        Parameters
        ----------
        sources : List[np.ndarray]
            List of input arrays.
        prefer_gpu : bool
            If ``True``, attempt GPU acceleration.  Default ``False``.
        progress_callback : callable, optional
            Called with a float in ``[0.0, 1.0]`` reflecting aggregate
            progress across all sources and all steps.

        Returns
        -------
        List[np.ndarray]
            List of results, one per source.
        """
        n_sources = len(sources)
        n_steps = len(self._steps) or 1
        total_units = n_sources * n_steps
        results: List[np.ndarray] = []

        for src_idx, src in enumerate(sources):
            def _batch_progress(
                step_frac: float,
                _si: int = src_idx,
            ) -> None:
                if progress_callback is not None:
                    completed = _si * n_steps + step_frac * n_steps
                    progress_callback(completed / total_units)

            results.append(
                self.execute(
                    src,
                    prefer_gpu=prefer_gpu,
                    progress_callback=_batch_progress,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_step_gpu_aware(
        self,
        ws: WorkflowStep,
        source: np.ndarray,
        gpu: GpuBackend,
        step_index: int,
        n_steps: int,
    ) -> np.ndarray:
        """Execute a single step with optional GPU acceleration.

        Parameters
        ----------
        ws : WorkflowStep
        source : np.ndarray
        gpu : GpuBackend
        step_index : int
            Zero-based index of this step (for error messages).
        n_steps : int
            Total step count (for error messages).

        Returns
        -------
        np.ndarray
        """
        try:
            # Attempt GPU path for compatible steps
            if gpu.cupy_available and ws.gpu_compatible:
                try:
                    gpu_source = gpu.to_gpu(source)
                    result = ws.fn(gpu_source)
                    return gpu.to_cpu(result)
                except Exception as gpu_err:
                    logger.warning(
                        "GPU execution failed for step '%s', "
                        "falling back to CPU: %s",
                        ws.name, gpu_err,
                    )

            # CPU path (default or fallback)
            return ws.fn(source)

        except Exception as e:
            if GrdlError is not None and isinstance(e, GrdlError):
                logger.error(
                    "Workflow '%s' step %d/%d '%s' GRDL error (%s): %s",
                    self._name, step_index + 1, n_steps, ws.name,
                    type(e).__name__, e,
                )
            else:
                logger.error(
                    "Workflow '%s' step %d/%d '%s' failed: %s",
                    self._name, step_index + 1, n_steps, ws.name, e,
                )
            raise RuntimeError(
                f"Workflow '{self._name}' step {step_index + 1}/{n_steps} "
                f"'{ws.name}' failed: {e}"
            ) from e


# ======================================================================
# Module-level helpers
# ======================================================================


def _infer_step_name(obj: Any) -> str:
    """Derive a human-readable step name from a callable.

    Parameters
    ----------
    obj : callable
        A bound method, function, or other callable.

    Returns
    -------
    str
        Inferred name.
    """
    if inspect.ismethod(obj):
        cls_name = type(obj.__self__).__name__
        return f"{cls_name}.{obj.__name__}"

    qualname = getattr(obj, '__qualname__', None)
    if qualname:
        return qualname

    name = getattr(obj, '__name__', None)
    if name:
        return name

    return repr(obj)


def _infer_gpu_compatible(obj: Any) -> bool:
    """Check whether a callable's owning instance is GPU-compatible.

    Parameters
    ----------
    obj : callable

    Returns
    -------
    bool
    """
    # Bound method → check __self__
    owner = getattr(obj, '__self__', None)
    if owner is not None:
        return bool(getattr(owner, '__gpu_compatible__', False))

    return False
