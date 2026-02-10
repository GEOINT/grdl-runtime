# -*- coding: utf-8 -*-
"""
Workflow Executor - Headless and interactive workflow execution.

Executes a compiled WorkflowDefinition by instantiating GRDL processors
and running them in sequence over input imagery. Supports both single-image
and batch execution with optional GPU acceleration.

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
import logging
from typing import Any, Callable, List, Optional

# Third-party
import numpy as np

logger = logging.getLogger(__name__)

# grdl-runtime internal
from grdl_rt.execution.discovery import resolve_processor_class
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

# GRDL exceptions (optional — graceful fallback if GRDL is old)
try:
    from grdl.exceptions import GrdlError
except ImportError:
    GrdlError = None  # type: ignore[misc,assignment]


class WorkflowExecutor:
    """Execute a compiled workflow headlessly or interactively.

    Instantiates each processor in the workflow and runs them in
    sequence on input imagery.

    Parameters
    ----------
    workflow : WorkflowDefinition
        Compiled workflow to execute.
    gpu : Optional[GpuBackend]
        GPU backend for acceleration. If None, CPU only.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        gpu: Optional[GpuBackend] = None,
    ) -> None:
        self._workflow = workflow
        self._gpu = gpu or GpuBackend(prefer_gpu=False)

    def execute(
        self,
        source: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the full pipeline on a single image.

        Parameters
        ----------
        source : np.ndarray
            Input image array.
        progress_callback : Optional[Callable[[float], None]]
            Called with progress fraction in [0.0, 1.0] as each step
            completes. Also forwarded into individual processor calls
            via the GRDL progress_callback protocol.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        np.ndarray
            Result after all processing steps.
        """
        n_steps = len(self._workflow.steps)
        current = source
        for i, step in enumerate(self._workflow.steps):
            # Build a rescaled callback for this step's internal progress
            step_kwargs = dict(kwargs)
            if progress_callback is not None and n_steps > 0:
                base = i / n_steps
                scale = 1.0 / n_steps
                step_kwargs['progress_callback'] = (
                    lambda f, _b=base, _s=scale: progress_callback(_b + f * _s)
                )
            current = self._execute_step(step, current, **step_kwargs)
            if progress_callback is not None and n_steps > 0:
                progress_callback((i + 1) / n_steps)
        return current

    def execute_batch(
        self,
        sources: List[np.ndarray],
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Run the pipeline on multiple images.

        Parameters
        ----------
        sources : List[np.ndarray]
            List of input image arrays.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        List[np.ndarray]
            List of results.
        """
        return [self.execute(src, **kwargs) for src in sources]

    def execute_step(
        self,
        step_index: int,
        source: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a single step from the workflow.

        Useful for interactive preview of individual steps.

        Parameters
        ----------
        step_index : int
            Index of the step to execute.
        source : np.ndarray
            Input image array.
        **kwargs
            Additional arguments.

        Returns
        -------
        np.ndarray
            Result of this step.
        """
        step = self._workflow.steps[step_index]
        return self._execute_step(step, source, **kwargs)

    def _execute_step(
        self,
        step: ProcessingStep,
        source: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a single processing step.

        Parameters
        ----------
        step : ProcessingStep
        source : np.ndarray
        **kwargs

        Returns
        -------
        np.ndarray
        """
        logger.debug("Executing step: %s", step.processor_name)
        try:
            processor_cls = resolve_processor_class(step.processor_name)
            processor = processor_cls()
        except (ImportError, Exception) as e:
            raise ImportError(
                f"Failed to resolve processor '{step.processor_name}': {e}"
            ) from e

        # Merge step params with kwargs (step params take precedence)
        merged_kwargs = {**kwargs, **step.params}

        try:
            result = self._gpu.apply_transform(processor, source, **merged_kwargs)
        except Exception as e:
            # Distinguish GRDL library errors from general Python errors
            if GrdlError is not None and isinstance(e, GrdlError):
                logger.error(
                    "Step '%s' GRDL error (%s): %s",
                    step.processor_name, type(e).__name__, e,
                )
            else:
                logger.error(
                    "Step '%s' failed: %s", step.processor_name, e,
                )
            raise RuntimeError(
                f"Pipeline step '{step.processor_name}' failed: {e}"
            ) from e

        return result
