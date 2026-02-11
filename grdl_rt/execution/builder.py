# -*- coding: utf-8 -*-
"""
Workflow Builder - Fluent, framework-level API for typed processing workflows.

Provides the ``Workflow`` class, a builder that holds live callable references
(bound methods, functions, ``ImageTransform`` instances) **or** deferred
processor classes that are constructed at execute time with automatic metadata
injection.  This is the primary Python API for defining workflows in code
where IDE tooling (IntelliSense, autocomplete, type checking) matters.

When a reader and chip strategy are configured, ``Workflow.execute()``
orchestrates the entire pipeline: opening the reader, extracting metadata,
planning chips, constructing metadata-dependent processors, and running the
processing steps — all from a single filepath argument.

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

Modified
--------
2026-02-11
"""

# Standard library
import functools
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

# GRDL chip extractor (optional — only needed for chip strategies)
try:
    from grdl.data_prep import ChipExtractor
except ImportError:
    ChipExtractor = None  # type: ignore[misc,assignment]


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


@dataclass
class DeferredStep:
    """A step whose processor is constructed at execute time.

    When a processor class is passed to :meth:`Workflow.step` instead of
    an instance or callable, it is stored as a ``DeferredStep``.  At
    execution time the framework inspects the constructor and injects
    ``metadata`` from the reader if the constructor accepts it.

    Attributes
    ----------
    processor_cls : type
        The processor class to instantiate.
    kwargs : Dict[str, Any]
        Keyword arguments forwarded to the constructor.
    name : str
        Human-readable step name for logging and error messages.
    """

    processor_cls: type
    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""


class Workflow:
    """Fluent builder and framework for typed image processing workflows.

    ``Workflow`` acts as both a recipe builder and an execution framework.
    Declare the reader, chip strategy, and processing steps, then call
    :meth:`execute` with a filepath — the framework handles reader
    lifecycle, metadata extraction, chip planning, processor construction,
    and GPU-accelerated pipeline execution.

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
    Framework-driven (recommended):

    >>> from grdl.IO import SICDReader
    >>> from grdl.image_processing.sar import SublookDecomposition
    >>> from grdl.image_processing.intensity import ToDecibels, PercentileStretch
    >>> from grdl_rt import Workflow
    >>>
    >>> wf = (
    ...     Workflow("Sublook Pipeline", modalities=["SAR"])
    ...     .reader(SICDReader)
    ...     .chip("center", size=5000)
    ...     .step(SublookDecomposition, num_looks=3, dimension='azimuth')
    ...     .step(ToDecibels)
    ...     .step(PercentileStretch, plow=2.0, phigh=98.0)
    ... )
    >>> result = wf.execute("image.nitf", prefer_gpu=True)

    Direct array mode (backward compatible):

    >>> wf = Workflow("Display").step(ToDecibels()).step(PercentileStretch())
    >>> result = wf.execute(my_array)
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
        self._steps: List[Union[WorkflowStep, DeferredStep]] = []
        self._source: Optional[Callable[[], np.ndarray]] = None
        self._reader_cls: Optional[type] = None
        self._chip_strategy: Optional[str] = None
        self._chip_kwargs: Dict[str, Any] = {}

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
    def steps(self) -> List[Union[WorkflowStep, DeferredStep]]:
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

    def reader(self, reader_cls: type) -> 'Workflow':
        """Declare the reader type for this workflow.

        When :meth:`execute` is called with a filepath, the framework
        opens this reader, extracts metadata, and manages its lifecycle.

        Parameters
        ----------
        reader_cls : type
            An ``ImageReader`` subclass (e.g., ``SICDReader``).

        Returns
        -------
        Workflow
            Self, for fluent chaining.
        """
        self._reader_cls = reader_cls
        return self

    def chip(self, strategy: str = 'center', **kwargs: Any) -> 'Workflow':
        """Declare the chip extraction strategy.

        Controls how the framework extracts pixel data from the reader.

        Parameters
        ----------
        strategy : str
            Chip strategy name:

            - ``"center"`` — extract a center chip of the given ``size``
              (default 5000).
            - ``"full"`` — read the entire image (no chipping).
        **kwargs
            Strategy-specific parameters (e.g., ``size=2048``).

        Returns
        -------
        Workflow
            Self, for fluent chaining.
        """
        self._chip_strategy = strategy
        self._chip_kwargs = kwargs
        return self

    def source(
        self,
        fn: Callable[..., np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> 'Workflow':
        """Set the data source for this workflow.

        The source callable is invoked when :meth:`execute` is called
        without a *source* array and no reader is configured.

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
        callable_or_class: Union[Callable, Any],
        *,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> 'Workflow':
        """Add a processing step to the workflow.

        Accepts three kinds of arguments:

        **Processor class** (deferred construction) — the class is stored
        and instantiated at execute time.  If the constructor accepts a
        ``metadata`` parameter, it is automatically injected from the
        reader.  All ``**kwargs`` are forwarded to the constructor::

            .step(SublookDecomposition, num_looks=3, dimension='azimuth')

        **ImageTransform instance** — auto-wrapped to call ``.apply()``::

            .step(ToDecibels(floor_db=-50.0))

        **Callable** — bound methods, plain functions, lambdas::

            .step(sublook.decompose, name="Decompose")

        Parameters
        ----------
        callable_or_class : callable, ImageTransform, or class
            The step operation.
        name : str, optional
            Display name for logging and error messages.
        **kwargs
            Constructor arguments for class-type steps.  Not allowed
            for instance or callable steps.

        Returns
        -------
        Workflow
            Self, for fluent chaining.

        Raises
        ------
        TypeError
            If ``**kwargs`` are provided for a non-class step, or if the
            argument is not callable.
        """
        obj = callable_or_class

        # Class → deferred construction
        if isinstance(obj, type):
            step_name = name or obj.__name__
            self._steps.append(DeferredStep(
                processor_cls=obj,
                kwargs=kwargs,
                name=step_name,
            ))
            return self

        # Non-class steps must not receive **kwargs
        if kwargs:
            raise TypeError(
                "step() keyword arguments are only supported for class-type "
                "steps (deferred construction).  For instances and callables, "
                "bind arguments before passing to step()."
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
                f"step() requires a callable, ImageTransform instance, or "
                f"processor class, got {type(obj).__name__}"
            )

        self._steps.append(WorkflowStep(fn=fn, name=step_name, gpu_compatible=gpu_ok))
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        source: Optional[Union[np.ndarray, str, Path]] = None,
        *,
        prefer_gpu: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
        metadata: Optional[Any] = None,
    ) -> np.ndarray:
        """Execute the workflow pipeline.

        Supports three modes of operation:

        **File mode** — pass a filepath string or ``Path``.  Requires a
        reader configured via :meth:`reader`.  The framework opens the
        reader, extracts metadata, plans chips, constructs deferred
        processors, and runs the pipeline::

            result = wf.execute("image.nitf")

        **Array mode** — pass an ``np.ndarray`` directly.  Deferred
        steps are resolved using the *metadata* keyword if provided::

            result = wf.execute(my_array, metadata=meta)

        **Source mode** — call with no arguments to use the configured
        :meth:`source` factory::

            result = wf.execute()

        Parameters
        ----------
        source : np.ndarray, str, Path, or None
            Input data.
        prefer_gpu : bool
            If ``True``, attempt GPU acceleration for compatible steps.
        progress_callback : callable, optional
            Called with a float in ``[0.0, 1.0]`` after each step.
        metadata : optional
            Explicit metadata for resolving deferred steps when
            executing in array mode.

        Returns
        -------
        np.ndarray
            Result after all steps have been applied.

        Raises
        ------
        ValueError
            If no source is available.
        RuntimeError
            If any step fails.
        """
        # File mode → framework orchestration
        if isinstance(source, (str, Path)):
            return self._execute_from_file(
                Path(source),
                prefer_gpu=prefer_gpu,
                progress_callback=progress_callback,
                metadata_override=metadata,
            )

        # Array mode → direct pipeline
        if isinstance(source, np.ndarray):
            steps = self._resolve_steps(metadata)
            return self._run_pipeline(
                source, steps,
                prefer_gpu=prefer_gpu,
                progress_callback=progress_callback,
            )

        # No source → fall back to stored source or reader
        if source is None:
            if self._source is not None:
                array = self._source()
                steps = self._resolve_steps(metadata)
                return self._run_pipeline(
                    array, steps,
                    prefer_gpu=prefer_gpu,
                    progress_callback=progress_callback,
                )
            raise ValueError(
                "No source provided.  Pass a filepath, array, or "
                "configure a source with .reader() or .source()."
            )

        raise TypeError(
            f"execute() expects a filepath, np.ndarray, or None, "
            f"got {type(source).__name__}"
        )

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
    # Framework internals
    # ------------------------------------------------------------------

    def _execute_from_file(
        self,
        filepath: Path,
        *,
        prefer_gpu: bool,
        progress_callback: Optional[Callable[[float], None]],
        metadata_override: Optional[Any],
    ) -> np.ndarray:
        """Open reader, read chip, resolve steps, and run pipeline."""
        if self._reader_cls is None:
            raise ValueError(
                f"Workflow '{self._name}' received a filepath but no reader "
                f"is configured.  Call .reader(ReaderClass) first."
            )

        with self._reader_cls(filepath) as rdr:
            meta = metadata_override if metadata_override is not None else rdr.metadata

            # Read chip according to strategy
            chip = self._read_chip(rdr)

            # Resolve deferred steps with metadata injection
            resolved = self._resolve_steps(meta)

            # Run the pipeline
            return self._run_pipeline(
                chip, resolved,
                prefer_gpu=prefer_gpu,
                progress_callback=progress_callback,
            )

    def _read_chip(self, reader: Any) -> np.ndarray:
        """Plan and read a chip from the reader.

        Parameters
        ----------
        reader
            An open ``ImageReader`` instance.

        Returns
        -------
        np.ndarray
            The chip pixel data.
        """
        rows, cols = reader.get_shape()

        if self._chip_strategy is None or self._chip_strategy == 'full':
            return reader.read_full()

        if self._chip_strategy == 'center':
            if ChipExtractor is None:
                raise ImportError(
                    "grdl.data_prep.ChipExtractor is required for chip "
                    "strategies but grdl is not installed."
                )
            size = self._chip_kwargs.get('size', 5000)
            extractor = ChipExtractor(nrows=rows, ncols=cols)
            region = extractor.chip_at_point(
                rows // 2, cols // 2,
                row_width=size, col_width=size,
            )
            return reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end,
            )

        raise ValueError(
            f"Unknown chip strategy '{self._chip_strategy}'.  "
            f"Supported: 'center', 'full'."
        )

    def _resolve_steps(
        self,
        metadata: Optional[Any],
    ) -> List[WorkflowStep]:
        """Resolve deferred steps into callable WorkflowSteps.

        Parameters
        ----------
        metadata : optional
            Reader metadata for constructor injection.

        Returns
        -------
        List[WorkflowStep]
            Fully resolved step list.
        """
        resolved: List[WorkflowStep] = []
        for step in self._steps:
            if isinstance(step, DeferredStep):
                resolved.append(self._resolve_deferred(step, metadata))
            else:
                resolved.append(step)
        return resolved

    def _resolve_deferred(
        self,
        ds: DeferredStep,
        metadata: Optional[Any],
    ) -> WorkflowStep:
        """Construct a processor from a DeferredStep.

        Inspects the processor constructor.  If it has a ``metadata``
        parameter and one was not provided in the step kwargs, injects
        the reader metadata automatically.

        Parameters
        ----------
        ds : DeferredStep
        metadata : optional

        Returns
        -------
        WorkflowStep
        """
        instance = _construct_processor(ds.processor_cls, ds.kwargs, metadata)

        # Determine the callable and GPU flag
        if ImageTransform is not None and isinstance(instance, ImageTransform):
            fn = instance.apply
        elif callable(instance):
            fn = instance
        elif hasattr(instance, 'apply') and callable(instance.apply):
            fn = instance.apply
        else:
            raise TypeError(
                f"Deferred step '{ds.name}' produced a {type(instance).__name__} "
                f"that is not callable and has no apply() method."
            )

        gpu_ok = getattr(instance, '__gpu_compatible__', False)
        return WorkflowStep(fn=fn, name=ds.name, gpu_compatible=gpu_ok)

    def _run_pipeline(
        self,
        source: np.ndarray,
        steps: List[WorkflowStep],
        *,
        prefer_gpu: bool,
        progress_callback: Optional[Callable[[float], None]],
    ) -> np.ndarray:
        """Execute resolved steps sequentially on source data.

        Parameters
        ----------
        source : np.ndarray
        steps : List[WorkflowStep]
        prefer_gpu : bool
        progress_callback : optional

        Returns
        -------
        np.ndarray
        """
        if not steps:
            return source

        gpu = GpuBackend(prefer_gpu=prefer_gpu)
        n_steps = len(steps)
        current = source

        for i, ws in enumerate(steps):
            logger.debug(
                "Workflow '%s' step %d/%d: %s",
                self._name, i + 1, n_steps, ws.name,
            )
            current = self._execute_step_gpu_aware(ws, current, gpu, i, n_steps)
            if progress_callback is not None:
                progress_callback((i + 1) / n_steps)

        return current

    def _execute_step_gpu_aware(
        self,
        ws: WorkflowStep,
        source: np.ndarray,
        gpu: GpuBackend,
        step_index: int,
        n_steps: int,
    ) -> np.ndarray:
        """Execute a single step with optional GPU acceleration."""
        try:
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


def _construct_processor(
    cls: type,
    kwargs: Dict[str, Any],
    metadata: Optional[Any],
) -> Any:
    """Instantiate a processor class with convention-based metadata injection.

    If the constructor has a ``metadata`` parameter and one was not
    provided in *kwargs*, injects *metadata* automatically.

    Parameters
    ----------
    cls : type
        Processor class.
    kwargs : Dict[str, Any]
        User-provided constructor arguments.
    metadata : optional
        Reader metadata to inject.

    Returns
    -------
    object
        The instantiated processor.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    if 'metadata' in params and 'metadata' not in kwargs:
        if metadata is None:
            raise ValueError(
                f"Processor '{cls.__name__}' requires metadata but none is "
                f"available.  Configure a reader with .reader() and execute "
                f"from a filepath, or pass metadata= to execute()."
            )
        return cls(metadata, **kwargs)

    return cls(**kwargs)


def _infer_step_name(obj: Any) -> str:
    """Derive a human-readable step name from a callable."""
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
    """Check whether a callable's owning instance is GPU-compatible."""
    owner = getattr(obj, '__self__', None)
    if owner is not None:
        return bool(getattr(owner, '__gpu_compatible__', False))

    return False
