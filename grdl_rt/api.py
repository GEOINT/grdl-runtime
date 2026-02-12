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
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.execution.dsl import DslCompiler
from grdl_rt.execution.executor import WorkflowExecutor
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.hardware import HardwareContext, LocalHardwareContext
from grdl_rt.execution.plan import ResolvedExecutionPlan
from grdl_rt.execution.resolver import Resolver
from grdl_rt.execution.result import WorkflowResult
from grdl_rt.execution.workflow import WorkflowDefinition

__all__ = ["load_workflow", "execute_workflow", "resolve_workflow"]


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
) -> WorkflowResult:
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
    WorkflowResult
        Result array and execution metrics.
    """
    gpu = GpuBackend(prefer_gpu=prefer_gpu)
    executor = WorkflowExecutor(workflow, gpu=gpu)
    return executor.execute(source, progress_callback=progress_callback, **kwargs)


def resolve_workflow(
    workflow: WorkflowDefinition,
    hardware: Optional[HardwareContext] = None,
    catalog: Optional[ArtifactCatalogBase] = None,
) -> ResolvedExecutionPlan:
    """Resolve a workflow against available hardware.

    Convenience wrapper that creates a :class:`Resolver`, optionally
    auto-detects the local hardware context, and returns a
    :class:`ResolvedExecutionPlan` describing the optimal execution path.

    Parameters
    ----------
    workflow : WorkflowDefinition
        Compiled workflow to resolve.
    hardware : HardwareContext, optional
        Hardware to resolve against.  Defaults to
        :class:`LocalHardwareContext` (auto-detect).
    catalog : ArtifactCatalogBase, optional
        Catalog for processor lookup and alternatives.  Defaults to
        :class:`SqliteArtifactCatalog`.

    Returns
    -------
    ResolvedExecutionPlan
        The resolved execution plan.
    """
    if hardware is None:
        hardware = LocalHardwareContext()
    if catalog is None:
        from grdl_rt.catalog.database import SqliteArtifactCatalog
        catalog = SqliteArtifactCatalog()
    return Resolver(catalog).resolve(workflow, hardware)
