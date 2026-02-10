# -*- coding: utf-8 -*-
"""grdl-runtime execution subpackage — workflow engine and processor orchestration."""

from grdl_rt.execution.tags import (
    ImageModality,
    DetectionType,
    SegmentationType,
    ProjectTags,
    WorkflowTags,
)
from grdl_rt.execution.chip import (
    ChipLabel,
    PolygonRegion,
    Chip,
    ChipSet,
)
from grdl_rt.execution.config import (
    GrdkConfig,
    load_config,
)
from grdl_rt.execution.gpu import (
    GpuBackend,
)
from grdl_rt.execution.discovery import (
    discover_processors,
    resolve_processor_class,
    get_processor_tags,
    get_all_modalities,
    get_all_categories,
    filter_processors,
)
from grdl_rt.execution.workflow import (
    WorkflowState,
    ProcessingStep,
    WorkflowDefinition,
)
from grdl_rt.execution.dsl import (
    step,
    workflow,
    DslCompiler,
)
from grdl_rt.execution.project import (
    GrdkProject,
)
from grdl_rt.execution.executor import (
    WorkflowExecutor,
)

__all__ = [
    # tags
    "ImageModality",
    "DetectionType",
    "SegmentationType",
    "ProjectTags",
    "WorkflowTags",
    # chip
    "ChipLabel",
    "PolygonRegion",
    "Chip",
    "ChipSet",
    # config
    "GrdkConfig",
    "load_config",
    # gpu
    "GpuBackend",
    # discovery
    "discover_processors",
    "resolve_processor_class",
    "get_processor_tags",
    "get_all_modalities",
    "get_all_categories",
    "filter_processors",
    # workflow
    "WorkflowState",
    "ProcessingStep",
    "WorkflowDefinition",
    # dsl
    "step",
    "workflow",
    "DslCompiler",
    # project
    "GrdkProject",
    # executor
    "WorkflowExecutor",
]
