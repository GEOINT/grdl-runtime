# -*- coding: utf-8 -*-
"""
grdl-runtime — Headless execution engine for GRDL workflows.

Sits between grdl (processing library) and grdk (GUI toolkit), providing
workflow execution, artifact catalog management, and GPU orchestration
without any GUI framework dependencies.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-09
"""

__version__ = "0.1.0"
__author__ = "Claude Code (Anthropic)"

# ── API convenience functions ────────────────────────────────────────
from grdl_rt.api import load_workflow, execute_workflow

# ── Execution subpackage ─────────────────────────────────────────────
from grdl_rt.execution import (
    # tags
    ImageModality,
    DetectionType,
    SegmentationType,
    ProjectTags,
    WorkflowTags,
    # chip
    ChipLabel,
    PolygonRegion,
    Chip,
    ChipSet,
    # config
    GrdkConfig,
    load_config,
    # gpu
    GpuBackend,
    # discovery
    discover_processors,
    resolve_processor_class,
    get_processor_tags,
    get_all_modalities,
    get_all_categories,
    filter_processors,
    # workflow
    WorkflowState,
    ProcessingStep,
    WorkflowDefinition,
    # dsl
    step,
    workflow,
    DslCompiler,
    # project
    GrdkProject,
    # executor
    WorkflowExecutor,
)

# ── Catalog subpackage ───────────────────────────────────────────────
from grdl_rt.catalog import (
    Artifact,
    UpdateResult,
    resolve_catalog_path,
    ensure_config_dir,
    ArtifactCatalogBase,
    SqliteArtifactCatalog,
    ArtifactCatalog,
    YamlArtifactCatalog,
    FederatedArtifactCatalog,
    ArtifactUpdateWorker,
    ThreadExecutorPool,
)

__all__ = [
    # api
    "load_workflow",
    "execute_workflow",
    # execution: tags
    "ImageModality",
    "DetectionType",
    "SegmentationType",
    "ProjectTags",
    "WorkflowTags",
    # execution: chip
    "ChipLabel",
    "PolygonRegion",
    "Chip",
    "ChipSet",
    # execution: config
    "GrdkConfig",
    "load_config",
    # execution: gpu
    "GpuBackend",
    # execution: discovery
    "discover_processors",
    "resolve_processor_class",
    "get_processor_tags",
    "get_all_modalities",
    "get_all_categories",
    "filter_processors",
    # execution: workflow
    "WorkflowState",
    "ProcessingStep",
    "WorkflowDefinition",
    # execution: dsl
    "step",
    "workflow",
    "DslCompiler",
    # execution: project
    "GrdkProject",
    # execution: executor
    "WorkflowExecutor",
    # catalog
    "Artifact",
    "UpdateResult",
    "resolve_catalog_path",
    "ensure_config_dir",
    "ArtifactCatalogBase",
    "SqliteArtifactCatalog",
    "ArtifactCatalog",
    "YamlArtifactCatalog",
    "FederatedArtifactCatalog",
    "ArtifactUpdateWorker",
    "ThreadExecutorPool",
]
