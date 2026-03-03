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
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.band_adaptation import (
    BandExpansion,
    BandReduction,
    adapt_bands,
)
from grdl_rt.execution.config import get_runtime_config
from grdl_rt.execution.context import ExecutionContext, get_logger
from grdl_rt.execution.dag_executor import run_dag_ready_dispatch
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics
from grdl_rt.execution.resilience import (
    RetryPolicy,
    execute_with_retry,
    execute_with_timeout,
    run_memory_preflight,
)
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.tags import ExecutionPhase, ImageModality, OutputFormat, WorkflowTags

logger = get_logger(__name__)

# GRDL base classes (optional — graceful fallback if grdl is unavailable)
try:
    from grdl.image_processing.base import ImageProcessor, ImageTransform
except ImportError:
    ImageProcessor = None  # type: ignore[misc,assignment]
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

    Holds either a processor instance (dispatched via ``execute_processor()``)
    or a raw callable.  The ``fn`` field stores the processor instance or
    callable; ``execute_processor()`` handles both transparently.

    Attributes
    ----------
    fn : Any
        Processor instance (``ImageProcessor`` subclass) or raw callable.
        Dispatched via ``execute_processor()`` during pipeline execution.
    name : str
        Human-readable step name for logging and error messages.
    gpu_compatible : bool
        Whether this step may be accelerated via CuPy GPU transfer.
    retry : Optional[RetryPolicy]
        Per-step retry configuration.
    timeout_seconds : Optional[float]
        Maximum wall-clock seconds for this step.
    id : Optional[str]
        Unique step identifier for DAG wiring.
    depends_on : Optional[List[str]]
        Step IDs this step depends on.
    condition : Optional[str]
        Condition expression for conditional execution.
    phase : Optional[str]
        Execution phase annotation.
    """

    fn: Any
    name: str
    gpu_compatible: bool
    retry: RetryPolicy | None = None
    timeout_seconds: float | None = None
    id: str | None = None
    depends_on: list[str] | None = None
    condition: str | None = None
    phase: str | None = None
    required_bands: int | None = None
    band_expansion: str | None = None
    band_reduction: str | None = None


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
    retry : Optional[RetryPolicy]
        Per-step retry configuration.
    timeout_seconds : Optional[float]
        Maximum wall-clock seconds for this step.
    id : Optional[str]
        Unique step identifier for DAG wiring.
    depends_on : Optional[List[str]]
        Step IDs this step depends on.
    condition : Optional[str]
        Condition expression for conditional execution.
    phase : Optional[str]
        Execution phase annotation.
    """

    processor_cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    retry: RetryPolicy | None = None
    timeout_seconds: float | None = None
    id: str | None = None
    depends_on: list[str] | None = None
    condition: str | None = None
    phase: str | None = None
    required_bands: int | None = None
    band_expansion: str | None = None
    band_reduction: str | None = None


@dataclass
class TapOutStep:
    """A non-processing step that writes current data to disk.

    When encountered in the pipeline, the current array is written to
    the specified path using the grdl.IO writer abstraction, then
    passed through unchanged to the next step.

    Attributes
    ----------
    path : str
        Output file path.
    format : Optional[str]
        Output format override.  If ``None``, auto-detected from extension.
    name : str
        Display name for logging.
    id : Optional[str]
        Unique step identifier for DAG wiring.
    depends_on : Optional[List[str]]
        Step IDs this step depends on.
    """

    path: str
    format: str | None = None
    name: str = "tap_out"
    id: str | None = None
    depends_on: list[str] | None = None


class BranchBuilder:
    """Builder for a single branch in a fan-out pattern.

    Created via ``Workflow.branch("name")`` and used with
    ``Workflow.branches()`` to define parallel branches.

    Parameters
    ----------
    name : str
        Branch name (used as prefix for step IDs).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._steps: list[WorkflowStep | DeferredStep | TapOutStep] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> list[WorkflowStep | DeferredStep | TapOutStep]:
        return list(self._steps)

    def step(
        self,
        callable_or_class: Callable | Any,
        *,
        name: str | None = None,
        id: str | None = None,
        condition: str | None = None,
        phase: str | None = None,
        retry: RetryPolicy | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> "BranchBuilder":
        """Add a step to this branch.  Same semantics as Workflow.step()."""
        obj = callable_or_class

        if isinstance(obj, type):
            step_name = name or obj.__name__
            self._steps.append(
                DeferredStep(
                    processor_cls=obj,
                    kwargs=kwargs,
                    name=step_name,
                    retry=retry,
                    timeout_seconds=timeout_seconds,
                    id=id,
                    condition=condition,
                    phase=phase,
                )
            )
            return self

        if kwargs:
            raise TypeError(
                "step() keyword arguments are only supported for class-type "
                "steps (deferred construction)."
            )

        if ImageTransform is not None and isinstance(obj, ImageTransform):
            fn = obj.apply
            step_name = name or type(obj).__name__
            gpu_ok = getattr(obj, "__gpu_compatible__", False)
        elif callable(obj):
            fn = obj
            step_name = name or _infer_step_name(obj)
            gpu_ok = _infer_gpu_compatible(obj)
        else:
            raise TypeError(
                f"step() requires a callable, ImageTransform instance, or "
                f"processor class, got {type(obj).__name__}"
            )

        self._steps.append(
            WorkflowStep(
                fn=fn,
                name=step_name,
                gpu_compatible=gpu_ok,
                retry=retry,
                timeout_seconds=timeout_seconds,
                id=id,
                condition=condition,
                phase=phase,
            )
        )
        return self


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
        modalities: list[str] | None = None,
        tags: WorkflowTags | None = None,
        band_expansion: str = "repeat",
        band_reduction: str = "first_n",
    ) -> None:
        self._name = name
        self._version = version
        self._description = description
        self._steps: list[WorkflowStep | DeferredStep | TapOutStep] = []
        self._source: Callable[[], np.ndarray] | None = None
        self._reader_cls: type | None = None
        self._chip_strategy: str | None = None
        self._chip_kwargs: dict[str, Any] = {}
        self._current_phase: str | None = None
        self._is_dag: bool = False
        self._branch_terminal_ids: list[str] = []
        self._step_counter: int = 0
        self._band_expansion = BandExpansion(band_expansion)
        self._band_reduction = BandReduction(band_reduction)

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
    def steps(self) -> list[WorkflowStep | DeferredStep | TapOutStep]:
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

    def reader(self, reader_cls: type) -> "Workflow":
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

    def chip(self, strategy: str = "center", **kwargs: Any) -> "Workflow":
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
    ) -> "Workflow":
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
        callable_or_class: Callable | Any,
        *,
        name: str | None = None,
        id: str | None = None,
        depends_on: list[str] | None = None,
        condition: str | None = None,
        phase: str | None = None,
        retry: RetryPolicy | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> "Workflow":
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
        id : str, optional
            Unique step identifier for DAG wiring.
        depends_on : List[str], optional
            Step IDs this step depends on.
        condition : str, optional
            Condition expression for conditional execution.
        phase : str, optional
            Execution phase annotation.  Overrides the current
            phase set by phase-scoped builders.
        retry : RetryPolicy, optional
            Per-step retry configuration with exponential backoff.
        timeout_seconds : float, optional
            Maximum wall-clock seconds for this step.
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
        step_phase = phase or self._current_phase
        if depends_on is not None or id is not None:
            self._is_dag = True

        obj = callable_or_class

        # Class → deferred construction
        if isinstance(obj, type):
            step_name = name or obj.__name__
            # Pop band adaptation kwargs before forwarding to constructor
            step_band_expansion = kwargs.pop("band_expansion", None)
            step_band_reduction = kwargs.pop("band_reduction", None)
            step_required_bands = kwargs.pop("required_bands", None)
            self._steps.append(
                DeferredStep(
                    processor_cls=obj,
                    kwargs=kwargs,
                    name=step_name,
                    retry=retry,
                    timeout_seconds=timeout_seconds,
                    id=id,
                    depends_on=depends_on,
                    condition=condition,
                    phase=step_phase,
                    required_bands=step_required_bands,
                    band_expansion=step_band_expansion,
                    band_reduction=step_band_reduction,
                )
            )
            return self

        # Non-class steps must not receive **kwargs
        if kwargs:
            raise TypeError(
                "step() keyword arguments are only supported for class-type "
                "steps (deferred construction).  For instances and callables, "
                "bind arguments before passing to step()."
            )

        # ImageProcessor instance → store instance for polymorphic dispatch
        if (
            ImageProcessor is not None
            and isinstance(obj, ImageProcessor)
            or ImageTransform is not None
            and isinstance(obj, ImageTransform)
        ):
            fn = obj
            step_name = name or type(obj).__name__
            gpu_ok = getattr(obj, "__gpu_compatible__", False)

        # Callable (bound method, function, lambda, etc.)
        elif callable(obj):
            fn = obj  # type: ignore[assignment]
            step_name = name or _infer_step_name(obj)
            gpu_ok = _infer_gpu_compatible(obj)

        else:
            raise TypeError(
                f"step() requires a callable, ImageTransform instance, or "
                f"processor class, got {type(obj).__name__}"
            )

        self._steps.append(
            WorkflowStep(
                fn=fn,
                name=step_name,
                gpu_compatible=gpu_ok,
                retry=retry,
                timeout_seconds=timeout_seconds,
                id=id,
                depends_on=depends_on,
                condition=condition,
                phase=step_phase,
            )
        )
        return self

    def tap_out(
        self,
        path: str | Path,
        format: str | OutputFormat | None = None,
    ) -> "Workflow":
        """Insert a tap-out point that writes current data to disk.

        The data is written to the specified path and then passed through
        unchanged to the next step.  This is useful for capturing intermediate
        results during pipeline development.

        Parameters
        ----------
        path : str or Path
            Output file path.
        format : str or OutputFormat, optional
            Writer format override.  Accepts ``OutputFormat`` enum values
            or raw strings (e.g., ``"geotiff"``, ``"numpy"``).
            If ``None``, auto-detected from the file extension.

        Returns
        -------
        Workflow
            Self, for fluent chaining.
        """
        fmt = format.value if isinstance(format, OutputFormat) else format
        self._steps.append(
            TapOutStep(
                path=str(path),
                format=fmt,
                name=f"tap_out({Path(path).name})",
            )
        )
        return self

    # ------------------------------------------------------------------
    # DAG Builder Methods
    # ------------------------------------------------------------------

    @staticmethod
    def branch(name: str) -> BranchBuilder:
        """Create a branch builder for fan-out patterns.

        Parameters
        ----------
        name : str
            Branch name (used as prefix for step IDs).

        Returns
        -------
        BranchBuilder
        """
        return BranchBuilder(name)

    def branches(self, *branch_builders: BranchBuilder) -> "Workflow":
        """Add parallel branches that fan out from the last step.

        All branches depend on the last step added before this call.
        After calling ``branches()``, use ``merge()`` to converge.

        Parameters
        ----------
        *branch_builders : BranchBuilder
            One or more branch builders created with ``Workflow.branch()``.

        Returns
        -------
        Workflow
            Self, for fluent chaining.
        """
        self._is_dag = True

        # The branch point is the last step added
        if self._steps:
            parent_id = self._steps[-1].id
            if parent_id is None:
                parent_id = f"_auto_{self._step_counter}"
                self._steps[-1].id = parent_id
                self._step_counter += 1
        else:
            parent_id = None

        self._branch_terminal_ids = []

        for bb in branch_builders:
            prev_id = parent_id
            for i, bstep in enumerate(bb.steps):
                # Assign ID if not set
                if bstep.id is None:
                    bstep.id = f"{bb.name}_{i}"
                # Wire dependency
                if bstep.depends_on is None and prev_id is not None:
                    bstep.depends_on = [prev_id]
                self._steps.append(bstep)
                prev_id = bstep.id
            # Track terminal step of each branch
            if prev_id is not None:
                self._branch_terminal_ids.append(prev_id)

        return self

    def merge(
        self,
        callable_or_class: Callable | Any,
        *,
        name: str | None = None,
        id: str | None = None,
        condition: str | None = None,
        phase: str | None = None,
        retry: RetryPolicy | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> "Workflow":
        """Add a merge step that consumes all preceding branch outputs.

        The merge step depends on the terminal step of every branch
        added by the preceding ``branches()`` call.

        Parameters
        ----------
        callable_or_class : callable, ImageTransform, or class
            The merge operation.  When called, it receives a dict
            of ``{branch_terminal_id: array}`` if multiple branches
            converge.
        name : str, optional
            Display name.
        id : str, optional
            Step ID.
        condition : str, optional
            Condition expression.
        phase : str, optional
            Execution phase.
        retry : RetryPolicy, optional
            Per-step retry config.
        timeout_seconds : float, optional
            Step timeout.
        **kwargs
            Constructor arguments for class-type steps.

        Returns
        -------
        Workflow
            Self, for fluent chaining.
        """
        merge_deps = list(self._branch_terminal_ids) if self._branch_terminal_ids else None
        self._branch_terminal_ids = []
        return self.step(
            callable_or_class,
            name=name,
            id=id,
            depends_on=merge_deps,
            condition=condition,
            phase=phase,
            retry=retry,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Phase-Scoped Builders
    # ------------------------------------------------------------------

    def io_phase(self) -> "Workflow":
        """Set the current phase to IO for subsequent steps."""
        self._current_phase = ExecutionPhase.IO.value
        return self

    def processing_phase(self) -> "Workflow":
        """Set the current phase to GLOBAL_PROCESSING."""
        self._current_phase = ExecutionPhase.GLOBAL_PROCESSING.value
        return self

    def data_prep_phase(self) -> "Workflow":
        """Set the current phase to DATA_PREP."""
        self._current_phase = ExecutionPhase.DATA_PREP.value
        return self

    def tiling_phase(self) -> "Workflow":
        """Set the current phase to TILING."""
        self._current_phase = ExecutionPhase.TILING.value
        return self

    def tile_processing_phase(self) -> "Workflow":
        """Set the current phase to TILE_PROCESSING."""
        self._current_phase = ExecutionPhase.TILE_PROCESSING.value
        return self

    def extraction_phase(self) -> "Workflow":
        """Set the current phase to EXTRACTION."""
        self._current_phase = ExecutionPhase.EXTRACTION.value
        return self

    def vector_phase(self) -> "Workflow":
        """Set the current phase to VECTOR_PROCESSING."""
        self._current_phase = ExecutionPhase.VECTOR_PROCESSING.value
        return self

    def finalization_phase(self) -> "Workflow":
        """Set the current phase to FINALIZATION."""
        self._current_phase = ExecutionPhase.FINALIZATION.value
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        source: np.ndarray | str | Path | None = None,
        *,
        prefer_gpu: bool = False,
        progress_callback: Callable[[float], None] | None = None,
        metadata: Any | None = None,
        auto_tap_out: bool = False,
        tap_out_format: str = "npy",
        tap_out_dir: str | Path | None = None,
    ) -> WorkflowResult:
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
        auto_tap_out : bool
            If ``True``, write all intermediates to a timestamped
            directory alongside workflow params and a manifest.
        tap_out_format : str
            Format for auto tap-out files (default ``'npy'``).
        tap_out_dir : str or Path, optional
            Override the auto tap-out output directory.

        Returns
        -------
        WorkflowResult
            Result containing the processed array and execution metrics.

        Raises
        ------
        ValueError
            If no source is available.
        RuntimeError
            If any step fails.
        """
        source_path: Path | None = None

        # File mode → framework orchestration
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            return self._execute_from_file(
                source_path,
                prefer_gpu=prefer_gpu,
                progress_callback=progress_callback,
                metadata_override=metadata,
                auto_tap_out=auto_tap_out,
                tap_out_format=tap_out_format,
                tap_out_dir=Path(tap_out_dir) if tap_out_dir else None,
            )

        # Array mode → direct pipeline
        if isinstance(source, np.ndarray):
            steps = self._resolve_steps(metadata)
            if self._is_dag:
                return self._run_dag_pipeline(
                    source,
                    steps,
                    prefer_gpu=prefer_gpu,
                    progress_callback=progress_callback,
                    metadata=metadata,
                )
            if auto_tap_out:
                return self._run_pipeline_with_auto_tap_out(
                    source,
                    steps,
                    prefer_gpu=prefer_gpu,
                    progress_callback=progress_callback,
                    source_path=None,
                    tap_out_format=tap_out_format,
                    tap_out_dir=Path(tap_out_dir) if tap_out_dir else None,
                )
            return self._run_pipeline(
                source,
                steps,
                prefer_gpu=prefer_gpu,
                progress_callback=progress_callback,
                metadata=metadata,
            )

        # No source → fall back to stored source or reader
        if source is None:
            if self._source is not None:
                array = self._source()
                steps = self._resolve_steps(metadata)
                if self._is_dag:
                    return self._run_dag_pipeline(
                        array,
                        steps,
                        prefer_gpu=prefer_gpu,
                        progress_callback=progress_callback,
                        metadata=metadata,
                    )
                if auto_tap_out:
                    return self._run_pipeline_with_auto_tap_out(
                        array,
                        steps,
                        prefer_gpu=prefer_gpu,
                        progress_callback=progress_callback,
                        source_path=None,
                        tap_out_format=tap_out_format,
                        tap_out_dir=Path(tap_out_dir) if tap_out_dir else None,
                    )
                return self._run_pipeline(
                    array,
                    steps,
                    prefer_gpu=prefer_gpu,
                    progress_callback=progress_callback,
                    metadata=metadata,
                )
            raise ValueError(
                "No source provided.  Pass a filepath, array, or "
                "configure a source with .reader() or .source()."
            )

        raise TypeError(
            f"execute() expects a filepath, np.ndarray, or None, " f"got {type(source).__name__}"
        )

    def execute_batch(
        self,
        sources: list[np.ndarray],
        *,
        prefer_gpu: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[WorkflowResult]:
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
        List[WorkflowResult]
            List of results, one per source.
        """
        n_sources = len(sources)
        n_steps = len(self._steps) or 1
        total_units = n_sources * n_steps
        results: list[WorkflowResult] = []

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
        progress_callback: Callable[[float], None] | None,
        metadata_override: Any | None,
        auto_tap_out: bool = False,
        tap_out_format: str = "npy",
        tap_out_dir: Path | None = None,
    ) -> WorkflowResult:
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
            if self._is_dag:
                return self._run_dag_pipeline(
                    chip,
                    resolved,
                    prefer_gpu=prefer_gpu,
                    progress_callback=progress_callback,
                    metadata=meta,
                )
            if auto_tap_out:
                return self._run_pipeline_with_auto_tap_out(
                    chip,
                    resolved,
                    prefer_gpu=prefer_gpu,
                    progress_callback=progress_callback,
                    source_path=filepath,
                    tap_out_format=tap_out_format,
                    tap_out_dir=tap_out_dir,
                )
            return self._run_pipeline(
                chip,
                resolved,
                prefer_gpu=prefer_gpu,
                progress_callback=progress_callback,
                metadata=meta,
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

        if self._chip_strategy is None or self._chip_strategy == "full":
            return reader.read_full()

        if self._chip_strategy == "center":
            if ChipExtractor is None:
                raise ImportError(
                    "grdl.data_prep.ChipExtractor is required for chip "
                    "strategies but grdl is not installed."
                )
            size = self._chip_kwargs.get("size", 5000)
            extractor = ChipExtractor(nrows=rows, ncols=cols)
            region = extractor.chip_at_point(
                rows // 2,
                cols // 2,
                row_width=size,
                col_width=size,
            )
            if isinstance(region, list):
                region = region[0]
            return reader.read_chip(
                region.row_start,
                region.row_end,
                region.col_start,
                region.col_end,
            )

        raise ValueError(
            f"Unknown chip strategy '{self._chip_strategy}'.  " f"Supported: 'center', 'full'."
        )

    def _resolve_steps(
        self,
        metadata: Any | None,
    ) -> list[WorkflowStep | TapOutStep]:
        """Resolve deferred steps into callable WorkflowSteps.

        ``TapOutStep`` instances pass through unchanged.

        Parameters
        ----------
        metadata : optional
            Reader metadata for constructor injection.

        Returns
        -------
        List[Union[WorkflowStep, TapOutStep]]
            Fully resolved step list.
        """
        resolved: list[WorkflowStep | TapOutStep] = []
        for step in self._steps:
            if isinstance(step, TapOutStep):
                resolved.append(step)
            elif isinstance(step, DeferredStep):
                resolved.append(self._resolve_deferred(step, metadata))
            else:
                resolved.append(step)
        return resolved

    def _resolve_deferred(
        self,
        ds: DeferredStep,
        metadata: Any | None,
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

        # Store the processor instance — execute_processor() handles dispatch.
        # Accepts: ImageProcessor subclass, callable instance, or object
        # with apply()/detect()/decompose().
        if (
            (ImageProcessor is not None and isinstance(instance, ImageProcessor))
            or hasattr(instance, "execute")
            or hasattr(instance, "apply")
            or callable(instance)
        ):
            fn = instance
        else:
            raise TypeError(
                f"Deferred step '{ds.name}' produced a {type(instance).__name__} "
                f"that has no execute(), apply(), or __call__ method."
            )

        gpu_ok = getattr(instance, "__gpu_compatible__", False)

        # Resolve required_bands: explicit step override > @processor_tags
        tags = getattr(ds.processor_cls, "__processor_tags__", {})
        required_bands = ds.required_bands or tags.get("required_bands")

        return WorkflowStep(
            fn=fn,
            name=ds.name,
            gpu_compatible=gpu_ok,
            retry=ds.retry,
            timeout_seconds=ds.timeout_seconds,
            id=ds.id,
            depends_on=ds.depends_on,
            condition=ds.condition,
            phase=ds.phase,
            required_bands=required_bands,
            band_expansion=ds.band_expansion,
            band_reduction=ds.band_reduction,
        )

    def _run_pipeline(
        self,
        source: np.ndarray,
        steps: list[WorkflowStep | TapOutStep],
        *,
        prefer_gpu: bool,
        progress_callback: Callable[[float], None] | None,
        metadata: Any | None = None,
    ) -> WorkflowResult:
        """Execute resolved steps sequentially on source data.

        Parameters
        ----------
        source : np.ndarray
        steps : List[Union[WorkflowStep, TapOutStep]]
        prefer_gpu : bool
        progress_callback : optional
        metadata : optional
            Image metadata threaded through the pipeline.

        Returns
        -------
        WorkflowResult
        """
        # Create execution context
        ctx = ExecutionContext(
            workflow_id=f"{self._name}:{self._version}",
            workflow_name=self._name,
        )
        cfg = get_runtime_config()

        # Start tracemalloc for peak memory tracking
        tracemalloc_was_running = tracemalloc.is_tracing()
        if not tracemalloc_was_running:
            tracemalloc.start()

        started_at = datetime.now(UTC).isoformat()
        pipeline_wall_t0 = time.perf_counter()
        pipeline_cpu_t0 = time.process_time()
        step_metrics_list: list[StepMetrics] = []
        overall_peak_rss = 0

        if not steps:
            pipeline_wall_elapsed = time.perf_counter() - pipeline_wall_t0
            pipeline_cpu_elapsed = time.process_time() - pipeline_cpu_t0
            completed_at = datetime.now(UTC).isoformat()

            wf_metrics = WorkflowMetrics(
                workflow_id=ctx.workflow_id,
                run_id=ctx.run_id,
                workflow_name=self._name,
                workflow_version=self._version,
                total_wall_time_s=pipeline_wall_elapsed,
                total_cpu_time_s=pipeline_cpu_elapsed,
                peak_rss_bytes=0,
                step_metrics=[],
                started_at=started_at,
                completed_at=completed_at,
            )

            if not tracemalloc_was_running:
                tracemalloc.stop()

            return WorkflowResult(result=source, metrics=wf_metrics)

        # Memory pre-flight
        processing_steps = [s for s in steps if isinstance(s, WorkflowStep)]
        if processing_steps:
            run_memory_preflight(
                source,
                n_steps=len(processing_steps),
                multiplier=cfg.memory.estimation_multiplier,
                warn_threshold=cfg.memory.warn_threshold,
                abort_threshold=cfg.memory.abort_threshold,
            )

        gpu = GpuBackend(prefer_gpu=prefer_gpu)
        n_steps = len(steps)
        current = source
        current_meta = metadata
        metadata_trace: list[dict[str, Any]] = []

        for i, ws in enumerate(steps):
            if isinstance(ws, TapOutStep):
                logger.debug(
                    "tap_out",
                    workflow_name=self._name,
                    step_index=i,
                    tap_out_path=ws.path,
                )
                self._execute_tap_out(
                    ws, gpu.to_cpu(current) if gpu.is_gpu_array(current) else current
                )
            else:
                logger.debug(
                    "step_start",
                    workflow_name=self._name,
                    step_index=i,
                    step_name=ws.name,
                )

                # Band adaptation (only for ndarray inputs)
                if ws.required_bands is not None and isinstance(current, np.ndarray):
                    exp = (
                        BandExpansion(ws.band_expansion)
                        if ws.band_expansion
                        else self._band_expansion
                    )
                    red = (
                        BandReduction(ws.band_reduction)
                        if ws.band_reduction
                        else self._band_reduction
                    )
                    current = adapt_bands(
                        current,
                        ws.required_bands,
                        expansion=exp,
                        reduction=red,
                    )

                # Capture step-level metrics
                snapshot_before = tracemalloc.take_snapshot()
                step_wall_t0 = time.perf_counter()
                step_cpu_t0 = time.process_time()

                current, current_meta, gpu_used = self._execute_step_gpu_aware(
                    ws,
                    current,
                    gpu,
                    i,
                    n_steps,
                    metadata=current_meta,
                )

                # Record metadata snapshot for trace
                if current_meta is not None:
                    metadata_trace.append(
                        {
                            "step_index": i,
                            "step_name": ws.name,
                            "metadata": current_meta,
                        }
                    )

                step_wall_elapsed = time.perf_counter() - step_wall_t0
                step_cpu_elapsed = time.process_time() - step_cpu_t0
                snapshot_after = tracemalloc.take_snapshot()

                # Compute peak memory delta
                stats = snapshot_after.compare_to(snapshot_before, "lineno")
                step_peak_rss = sum(s.size_diff for s in stats if s.size_diff > 0)
                if step_peak_rss > overall_peak_rss:
                    overall_peak_rss = step_peak_rss

                step_metrics_list.append(
                    StepMetrics(
                        step_index=i,
                        processor_name=ws.name,
                        wall_time_s=step_wall_elapsed,
                        cpu_time_s=step_cpu_elapsed,
                        peak_rss_bytes=step_peak_rss,
                        gpu_used=gpu_used,
                    )
                )

            if progress_callback is not None:
                progress_callback((i + 1) / n_steps)

        pipeline_wall_elapsed = time.perf_counter() - pipeline_wall_t0
        pipeline_cpu_elapsed = time.process_time() - pipeline_cpu_t0
        completed_at = datetime.now(UTC).isoformat()

        wf_metrics = WorkflowMetrics(
            workflow_id=ctx.workflow_id,
            run_id=ctx.run_id,
            workflow_name=self._name,
            workflow_version=self._version,
            total_wall_time_s=pipeline_wall_elapsed,
            total_cpu_time_s=pipeline_cpu_elapsed,
            peak_rss_bytes=overall_peak_rss,
            step_metrics=step_metrics_list,
            started_at=started_at,
            completed_at=completed_at,
        )

        if not tracemalloc_was_running:
            tracemalloc.stop()

        if gpu.is_gpu_array(current):
            current = gpu.to_cpu(current)

        return WorkflowResult(
            result=current,
            metrics=wf_metrics,
            metadata_trace=metadata_trace,
        )

    def _run_pipeline_with_auto_tap_out(
        self,
        source: np.ndarray,
        steps: list[WorkflowStep | TapOutStep],
        *,
        prefer_gpu: bool,
        progress_callback: Callable[[float], None] | None,
        source_path: Path | None,
        tap_out_format: str = "npy",
        tap_out_dir: Path | None = None,
    ) -> WorkflowResult:
        """Execute pipeline, writing all intermediates to a timestamped directory.

        Creates a directory structure::

            <tap_out_dir>/
                workflow_params.yaml
                manifest.json
                step_001_<name>.<ext>
                step_002_<name>.<ext>
                ...
                final_output.<ext>

        Parameters
        ----------
        source : np.ndarray
        steps : List[Union[WorkflowStep, TapOutStep]]
        prefer_gpu : bool
        progress_callback : optional
        source_path : Path, optional
            Original source file path (for default directory location).
        tap_out_format : str
            File extension for intermediate files (default ``'npy'``).
        tap_out_dir : Path, optional
            Override the auto tap-out output directory.

        Returns
        -------
        WorkflowResult
        """
        import json

        # Create execution context
        ctx = ExecutionContext(
            workflow_id=f"{self._name}:{self._version}",
            workflow_name=self._name,
        )

        # Start tracemalloc for peak memory tracking
        tracemalloc_was_running = tracemalloc.is_tracing()
        if not tracemalloc_was_running:
            tracemalloc.start()

        started_at = datetime.now(UTC).isoformat()
        pipeline_wall_t0 = time.perf_counter()
        pipeline_cpu_t0 = time.process_time()
        step_metrics_list: list[StepMetrics] = []
        overall_peak_rss = 0

        # Determine output directory
        if tap_out_dir is not None:
            run_dir = tap_out_dir
        else:
            base_dir = source_path.parent if source_path else Path.cwd()
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            run_dir = base_dir / f"grdl_run_{timestamp}"

        run_dir.mkdir(parents=True, exist_ok=True)

        ext = tap_out_format.lstrip(".")
        gpu = GpuBackend(prefer_gpu=prefer_gpu)
        n_steps = len(steps)
        current = source
        manifest_entries: list[dict[str, Any]] = []
        step_counter = 0

        for i, ws in enumerate(steps):
            if isinstance(ws, TapOutStep):
                # Explicit tap-out steps still write to their own path
                tap_wall_t0 = time.perf_counter()
                self._execute_tap_out(
                    ws, gpu.to_cpu(current) if gpu.is_gpu_array(current) else current
                )
                tap_elapsed = time.perf_counter() - tap_wall_t0
                manifest_entries.append(
                    {
                        "index": i,
                        "type": "tap_out",
                        "name": ws.name,
                        "path": ws.path,
                        "elapsed_s": round(tap_elapsed, 4),
                        "shape": list(current.shape),
                        "dtype": str(current.dtype),
                    }
                )
            else:
                logger.debug(
                    "step_start",
                    workflow_name=self._name,
                    step_index=i,
                    step_name=ws.name,
                )

                # Band adaptation
                if ws.required_bands is not None:
                    exp = (
                        BandExpansion(ws.band_expansion)
                        if ws.band_expansion
                        else self._band_expansion
                    )
                    red = (
                        BandReduction(ws.band_reduction)
                        if ws.band_reduction
                        else self._band_reduction
                    )
                    current = adapt_bands(
                        current,
                        ws.required_bands,
                        expansion=exp,
                        reduction=red,
                    )

                # Capture step-level metrics
                snapshot_before = tracemalloc.take_snapshot()
                step_wall_t0 = time.perf_counter()
                step_cpu_t0 = time.process_time()

                current, _meta, gpu_used = self._execute_step_gpu_aware(
                    ws,
                    current,
                    gpu,
                    i,
                    n_steps,
                )

                step_wall_elapsed = time.perf_counter() - step_wall_t0
                step_cpu_elapsed = time.process_time() - step_cpu_t0
                snapshot_after = tracemalloc.take_snapshot()

                # Compute peak memory delta
                stats = snapshot_after.compare_to(snapshot_before, "lineno")
                step_peak_rss = sum(s.size_diff for s in stats if s.size_diff > 0)
                if step_peak_rss > overall_peak_rss:
                    overall_peak_rss = step_peak_rss

                step_metrics_list.append(
                    StepMetrics(
                        step_index=i,
                        processor_name=ws.name,
                        wall_time_s=step_wall_elapsed,
                        cpu_time_s=step_cpu_elapsed,
                        peak_rss_bytes=step_peak_rss,
                        gpu_used=gpu_used,
                    )
                )

                step_counter += 1

                # Write intermediate
                safe_name = _sanitize_filename(ws.name)
                filename = f"step_{step_counter:03d}_{safe_name}.{ext}"
                out_path = run_dir / filename
                try:
                    from grdl.IO import write as io_write

                    io_write(
                        gpu.to_cpu(current) if gpu.is_gpu_array(current) else current,
                        out_path,
                    )
                except Exception as e:
                    logger.warning(
                        "Auto tap-out write failed for '%s': %s",
                        out_path,
                        e,
                    )

                manifest_entries.append(
                    {
                        "index": i,
                        "type": "processing",
                        "name": ws.name,
                        "file": filename,
                        "elapsed_s": round(step_wall_elapsed, 4),
                        "shape": list(current.shape),
                        "dtype": str(current.dtype),
                    }
                )

            if progress_callback is not None:
                progress_callback((i + 1) / n_steps)

        # Write final output
        final_filename = f"final_output.{ext}"
        try:
            from grdl.IO import write as io_write

            io_write(
                gpu.to_cpu(current) if gpu.is_gpu_array(current) else current,
                run_dir / final_filename,
            )
        except Exception as e:
            logger.warning("Auto tap-out final output write failed: %s", e)

        # Write workflow_params.yaml
        try:
            import yaml

            params = {
                "name": self._name,
                "version": self._version,
                "description": self._description,
                "steps": [
                    ws.name if not isinstance(ws, TapOutStep) else f"tap_out({ws.path})"
                    for ws in steps
                ],
            }
            yaml_path = run_dir / "workflow_params.yaml"
            yaml_path.write_text(
                yaml.dump(params, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Auto tap-out workflow_params.yaml write failed: %s", e)

        # Write manifest.json
        try:
            manifest = {
                "workflow_name": self._name,
                "source_path": str(source_path) if source_path else None,
                "tap_out_format": tap_out_format,
                "final_output": final_filename,
                "steps": manifest_entries,
            }
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Auto tap-out manifest.json write failed: %s", e)

        logger.info("Auto tap-out complete: %s", run_dir)

        pipeline_wall_elapsed = time.perf_counter() - pipeline_wall_t0
        pipeline_cpu_elapsed = time.process_time() - pipeline_cpu_t0
        completed_at = datetime.now(UTC).isoformat()

        wf_metrics = WorkflowMetrics(
            workflow_id=ctx.workflow_id,
            run_id=ctx.run_id,
            workflow_name=self._name,
            workflow_version=self._version,
            total_wall_time_s=pipeline_wall_elapsed,
            total_cpu_time_s=pipeline_cpu_elapsed,
            peak_rss_bytes=overall_peak_rss,
            step_metrics=step_metrics_list,
            started_at=started_at,
            completed_at=completed_at,
        )

        if not tracemalloc_was_running:
            tracemalloc.stop()

        if gpu.is_gpu_array(current):
            current = gpu.to_cpu(current)

        return WorkflowResult(result=current, metrics=wf_metrics)

    # ------------------------------------------------------------------
    # DAG Pipeline
    # ------------------------------------------------------------------

    def _run_dag_pipeline(
        self,
        source: np.ndarray,
        steps: list[WorkflowStep | TapOutStep],
        *,
        prefer_gpu: bool,
        progress_callback: Callable[[float], None] | None,
        metadata: Any | None = None,
    ) -> WorkflowResult:
        """Execute resolved steps as a DAG with readiness-based scheduling.

        Steps are dispatched to a thread pool the instant all of their
        specific dependencies complete.  Each step's input is routed from
        its ``depends_on`` entries (not the sequential ``current``),
        ensuring correct fan-out semantics for branches.

        .. note:: **Readiness-based scheduling**

           Each step starts as soon as its direct dependencies finish,
           regardless of unrelated branches.  Total execution time equals
           the duration of the longest critical path through the DAG —
           not the sum of the longest steps at each topological level.

        Parameters
        ----------
        source : np.ndarray
            Input data fed to root steps (those with no ``depends_on``).
        steps : List[Union[WorkflowStep, TapOutStep]]
            Fully resolved step list (no ``DeferredStep`` instances).
        prefer_gpu : bool
            Whether to attempt GPU acceleration.
        progress_callback : callable, optional
            Called with a float in ``[0.0, 1.0]`` after each step completes.
        metadata : optional
            Image metadata threaded through the pipeline.

        Returns
        -------
        WorkflowResult
            Result with ``step_results`` populated as ``{step_id: output}``.
        """
        ctx = ExecutionContext(
            workflow_id=f"{self._name}:{self._version}",
            workflow_name=self._name,
        )
        cfg = get_runtime_config()

        tracemalloc_was_running = tracemalloc.is_tracing()
        if not tracemalloc_was_running:
            tracemalloc.start()
        tracemalloc.reset_peak()

        started_at = datetime.now(UTC).isoformat()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        step_metrics_list: list[StepMetrics] = []

        # Build step lookup — every step must have an id for DAG routing
        step_by_id: dict[str, WorkflowStep | TapOutStep] = {}
        for s in steps:
            if s.id is None:
                raise ValueError(
                    "DAG pipeline requires all steps to have an id.  "
                    "Use .branches() or set id= on each .step()."
                )
            step_by_id[s.id] = s

        # Memory pre-flight (estimate based on widest parallel level)
        levels = _topological_sort_steps(steps)
        processing_steps = [s for s in steps if isinstance(s, WorkflowStep)]
        if processing_steps:
            max_width = max(len(level) for level in levels) if levels else 1
            run_memory_preflight(
                source,
                n_steps=max_width,
                multiplier=cfg.memory.estimation_multiplier,
                warn_threshold=cfg.memory.warn_threshold,
                abort_threshold=cfg.memory.abort_threshold,
            )

        results: dict[str, Any] = {}
        gpu = GpuBackend(prefer_gpu=prefer_gpu)

        # Pre-compute stable step indices from topological order so that
        # parallel steps get unique, deterministic indices regardless of
        # completion order.
        step_index_map: dict[str, int] = {}
        idx = 0
        for _lvl in levels:
            for _sid in _lvl:
                step_index_map[_sid] = idx
                idx += 1

        total = len(steps)

        # Build flat step list and dependency map for readiness dispatch
        id_set = {s.id for s in steps if s.id is not None}
        deps_map: dict[str, list[str]] = {}
        for _s in steps:
            if _s.id is not None:
                raw = getattr(_s, "depends_on", None) or []
                deps_map[_s.id] = [d for d in raw if d in id_set]
        all_step_ids = [sid for lvl in levels for sid in lvl]

        def _gather(sid: str) -> Any:
            ws = step_by_id[sid]
            return _gather_input(ws, source, results)  # called under lock

        def _exec_step(
            sid: str, step_input: Any, reset_mem_peak: bool = False
        ) -> tuple[StepMetrics, Any]:
            ws = step_by_id[sid]
            return self._execute_dag_step(
                ws,
                step_input,
                gpu,
                step_index_map[sid],
                total,
                step_id=sid,
                metadata=metadata,
                reset_mem_peak=reset_mem_peak,
            )

        step_metrics_list, overall_peak = run_dag_ready_dispatch(
            all_step_ids,
            deps_map,
            step_index_map,
            _exec_step,
            _gather,
            results,
            progress_callback=progress_callback,
        )

        # Determine terminal output
        terminal_ids = [
            s.id
            for s in steps
            if s.id is not None
            and not any(s.id in (getattr(o, "depends_on", None) or []) for o in steps)
        ]
        if len(terminal_ids) == 1:
            final = results[terminal_ids[0]]
        elif terminal_ids:
            final = results[levels[-1][-1]] if levels else source
        else:
            final = source

        wall = time.perf_counter() - t0_wall
        cpu = time.process_time() - t0_cpu
        completed_at = datetime.now(UTC).isoformat()

        wf_metrics = WorkflowMetrics(
            workflow_id=ctx.workflow_id,
            run_id=ctx.run_id,
            workflow_name=self._name,
            workflow_version=self._version,
            total_wall_time_s=wall,
            total_cpu_time_s=cpu,
            peak_rss_bytes=overall_peak,
            step_metrics=step_metrics_list,
            started_at=started_at,
            completed_at=completed_at,
        )

        if not tracemalloc_was_running:
            tracemalloc.stop()

        return WorkflowResult(
            result=final,
            metrics=wf_metrics,
            step_results=results,
        )

    def _execute_dag_step(
        self,
        ws: WorkflowStep | TapOutStep,
        step_input: Any,
        gpu: GpuBackend,
        step_index: int,
        n_steps: int,
        *,
        step_id: str | None = None,
        metadata: Any | None = None,
        reset_mem_peak: bool = False,
    ) -> tuple[StepMetrics, Any]:
        """Execute a single DAG step and return ``(metrics, output)``.

        Delegates to :meth:`_execute_step_gpu_aware` for the actual
        processing, preserving GPU dispatch, retry, and timeout logic.

        Parameters
        ----------
        ws : WorkflowStep or TapOutStep
        step_input : Any
            Routed input (array, dict, or source).
        gpu : GpuBackend
        step_index : int
        n_steps : int
        step_id : str, optional
            Stable step identifier for DAG metric grouping.
        metadata : optional

        Returns
        -------
        Tuple[StepMetrics, Any]
            ``(step_metrics, output)``
        """
        if isinstance(ws, TapOutStep):
            self._execute_tap_out(ws, step_input)
            sm = StepMetrics(
                step_index=step_index,
                processor_name=ws.name,
                wall_time_s=0.0,
                cpu_time_s=0.0,
                peak_rss_bytes=0,
                gpu_used=False,
                step_id=step_id,
            )
            return sm, step_input

        if reset_mem_peak:
            tracemalloc.reset_peak()

        t0_wall = time.perf_counter()
        t0_cpu = time.thread_time()

        result, _meta, gpu_used = self._execute_step_gpu_aware(
            ws,
            step_input,
            gpu,
            step_index,
            n_steps,
            metadata=metadata,
        )

        step_peak = tracemalloc.get_traced_memory()[1] if reset_mem_peak else 0

        sm = StepMetrics(
            step_index=step_index,
            processor_name=ws.name,
            wall_time_s=time.perf_counter() - t0_wall,
            cpu_time_s=time.thread_time() - t0_cpu,
            peak_rss_bytes=step_peak,  # 0 for concurrent steps, overwritten by run_dag_ready_dispatch()
            gpu_used=gpu_used,
            step_id=step_id,
        )
        return sm, result

    def _execute_tap_out(
        self,
        tap: TapOutStep,
        data: np.ndarray,
    ) -> None:
        """Write current pipeline data to disk without modifying it."""
        try:
            from grdl.IO import write as io_write

            io_write(data, tap.path, format=tap.format)
        except Exception as e:
            logger.warning(
                "Tap-out to '%s' failed: %s (continuing pipeline)",
                tap.path,
                e,
            )

    def _execute_step_gpu_aware(
        self,
        ws: WorkflowStep,
        source: Any,
        gpu: GpuBackend,
        step_index: int,
        n_steps: int,
        *,
        metadata: Any | None = None,
    ) -> tuple[Any, Any, bool]:
        """Execute a single step with optional GPU acceleration, retry, and timeout.

        Uses ``execute_processor()`` for polymorphic dispatch — works with
        ``ImageTransform``, ``ImageDetector``, ``PolarimetricDecomposition``,
        ``WorkflowOperator``, and raw callables.

        Parameters
        ----------
        ws : WorkflowStep
        source : Any
        gpu : GpuBackend
        step_index : int
        n_steps : int
        metadata : optional
            Current pipeline metadata.

        Returns
        -------
        Tuple[Any, Any, bool]
            ``(result, updated_metadata, gpu_used)``
        """
        from grdl_rt.execution.dispatch import (
            _minimal_metadata,
            execute_processor,
            supports_gpu_transfer,
        )

        step_meta = metadata
        if step_meta is None and isinstance(source, np.ndarray):
            step_meta = _minimal_metadata(source)

        def _do_raw_step() -> tuple[Any, Any, bool]:
            try:
                is_gpu_source = gpu.is_gpu_array(source)

                if (
                    gpu.cupy_available
                    and ws.gpu_compatible
                    and supports_gpu_transfer(ws.fn)
                ):
                    try:
                        # Re-use the GPU array if previous step already left
                        # it on device; otherwise upload from CPU.
                        gpu_source = source if is_gpu_source else gpu.to_gpu(source)
                        result, out_meta = execute_processor(
                            ws.fn,
                            step_meta,  # type: ignore[arg-type]
                            gpu_source,
                        )
                        return result, out_meta, True
                    except Exception as gpu_err:
                        logger.warning(
                            "GPU execution failed for step '%s', " "falling back to CPU: %s",
                            ws.name,
                            gpu_err,
                        )

                # CPU path — pull off GPU first if the previous step left it there.
                cpu_source = gpu.to_cpu(source) if is_gpu_source else source
                result, out_meta = execute_processor(
                    ws.fn,
                    step_meta,  # type: ignore[arg-type]
                    cpu_source,
                )
                return result, out_meta, False

            except Exception as e:
                if GrdlError is not None and isinstance(e, GrdlError):
                    logger.error(
                        "Workflow '%s' step %d/%d '%s' GRDL error (%s): %s",
                        self._name,
                        step_index + 1,
                        n_steps,
                        ws.name,
                        type(e).__name__,
                        e,
                    )
                else:
                    logger.error(
                        "Workflow '%s' step %d/%d '%s' failed: %s",
                        self._name,
                        step_index + 1,
                        n_steps,
                        ws.name,
                        e,
                    )
                raise RuntimeError(
                    f"Workflow '{self._name}' step {step_index + 1}/{n_steps} "
                    f"'{ws.name}' failed: {e}"
                ) from e

        timeout = ws.timeout_seconds
        retry = ws.retry

        def _step_with_timeout() -> tuple[Any, Any, bool]:
            if timeout is not None:
                return execute_with_timeout(
                    _do_raw_step,
                    timeout,
                    ws.name,
                )
            return _do_raw_step()

        if retry is not None and retry.max_retries > 0:
            return execute_with_retry(
                _step_with_timeout,
                retry,
                ws.name,
            )

        return _step_with_timeout()


# ======================================================================
# Module-level helpers
# ======================================================================


def _construct_processor(
    cls: type,
    kwargs: dict[str, Any],
    metadata: Any | None,
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
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    params = sig.parameters

    if "metadata" in params and "metadata" not in kwargs:
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

    qualname = getattr(obj, "__qualname__", None)
    if qualname:
        return qualname

    name = getattr(obj, "__name__", None)
    if name:
        return name

    return repr(obj)


def _infer_gpu_compatible(obj: Any) -> bool:
    """Check whether a callable's owning instance is GPU-compatible."""
    owner = getattr(obj, "__self__", None)
    if owner is not None:
        return bool(getattr(owner, "__gpu_compatible__", False))

    return False


def _sanitize_filename(name: str) -> str:
    """Convert a step name into a safe filename component.

    Replaces spaces, dots, and path separators with underscores,
    and strips non-alphanumeric characters.
    """
    import re

    safe = re.sub(r"[^\w\-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "step"


def _gather_input(
    ws: WorkflowStep | DeferredStep | TapOutStep,
    source: np.ndarray,
    results: dict[str, Any],
) -> Any:
    """Route the correct input to a step based on ``depends_on``.

    Parameters
    ----------
    ws : WorkflowStep, DeferredStep, or TapOutStep
        The step whose input to resolve.
    source : np.ndarray
        Original pipeline source (for root steps with no deps).
    results : Dict[str, Any]
        Completed step outputs keyed by step ID.

    Returns
    -------
    Any
        Single array for single-dependency, dict for multi-dependency,
        or *source* for root steps.
    """
    deps = getattr(ws, "depends_on", None)
    if not deps:
        return source
    if len(deps) == 1:
        return results[deps[0]]
    return {dep_id: results[dep_id] for dep_id in deps}


def _topological_sort_steps(
    steps: list[WorkflowStep | TapOutStep],
) -> list[list[str]]:
    """Sort resolved steps into parallel execution levels.

    Steps with no unmet dependencies form a level.  All steps within
    a level can execute concurrently.

    Parameters
    ----------
    steps : list
        Resolved steps, each with an ``id`` attribute.

    Returns
    -------
    List[List[str]]
        Ordered list of levels, each level is a list of step IDs.

    Raises
    ------
    ValueError
        If a cycle is detected in the dependency graph.
    """
    id_set = {s.id for s in steps if s.id is not None}
    deps_map: dict[str, list[str]] = {}
    for s in steps:
        sid = s.id
        if sid is None:
            continue
        raw_deps = getattr(s, "depends_on", None) or []
        deps_map[sid] = [d for d in raw_deps if d in id_set]

    levels: list[list[str]] = []
    placed: set[str] = set()
    remaining = set(deps_map.keys())

    while remaining:
        level = [sid for sid in remaining if all(d in placed for d in deps_map[sid])]
        if not level:
            raise ValueError("Cycle detected in workflow DAG")
        levels.append(sorted(level))
        placed.update(level)
        remaining -= set(level)

    return levels
