# -*- coding: utf-8 -*-
"""
API — High-level convenience functions for grdl-runtime.

Provides ``load_workflow`` and ``execute_workflow`` as the primary
entry-points for consumers who want to load a workflow definition
from various sources and execute it on imagery in a single call.

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
2026-02-09
"""

# Standard library
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.dsl import DslCompiler
from grdl_rt.execution.executor import WorkflowExecutor
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.workflow import WorkflowDefinition

__all__ = ["load_workflow", "execute_workflow"]


def load_workflow(
    source: Union[str, Path, Dict[str, Any]],
) -> WorkflowDefinition:
    """Load a workflow definition from a YAML file, YAML string, or dict.

    Auto-detects the input type:

    * **dict** — passed directly to ``WorkflowDefinition.from_dict``.
    * **Path** — read from disk via ``DslCompiler.compile_yaml``.
    * **str** — if the string points to an existing file it is loaded as
      a YAML file; otherwise it is parsed as inline YAML content.

    Parameters
    ----------
    source : str | Path | dict
        Workflow definition in one of the supported formats.

    Returns
    -------
    WorkflowDefinition

    Raises
    ------
    TypeError
        If *source* is not ``str``, ``Path``, or ``dict``.
    """
    compiler = DslCompiler()

    if isinstance(source, dict):
        return WorkflowDefinition.from_dict(source)

    if isinstance(source, Path):
        return compiler.compile_yaml(source)

    if isinstance(source, str):
        if os.path.exists(source):
            return compiler.compile_yaml(Path(source))
        return compiler.compile_yaml_string(source)

    raise TypeError(
        f"source must be str, Path, or dict, got {type(source).__name__}"
    )


def execute_workflow(
    workflow: WorkflowDefinition,
    source: np.ndarray,
    *,
    prefer_gpu: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Execute a workflow on a single image.

    Convenience wrapper that creates a :class:`GpuBackend` and
    :class:`WorkflowExecutor`, runs the pipeline, and returns the result.

    Parameters
    ----------
    workflow : WorkflowDefinition
        Compiled workflow to execute.
    source : np.ndarray
        Input image array.
    prefer_gpu : bool
        Whether to attempt GPU acceleration (default ``True``).
    progress_callback : callable, optional
        Called with a float in ``[0.0, 1.0]`` as each step completes.
    **kwargs
        Extra arguments forwarded to each processor.

    Returns
    -------
    np.ndarray
        Processed image.
    """
    gpu = GpuBackend(prefer_gpu=prefer_gpu)
    executor = WorkflowExecutor(workflow, gpu=gpu)
    return executor.execute(source, progress_callback=progress_callback, **kwargs)
