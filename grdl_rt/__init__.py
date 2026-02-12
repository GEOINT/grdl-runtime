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
from grdl_rt.api import load_workflow, execute_workflow, resolve_workflow

# ── Execution subpackage ─────────────────────────────────────────────
from grdl_rt.execution import (
    # tags
    ImageModality,
    DetectionType,
    GpuCapability,
    SegmentationType,
    ProjectTags,
    WorkflowTags,
    # chip
    ChipLabel,
    PolygonRegion,
    Chip,
    ChipSet,
    # config
    RuntimeConfig,
    LogConfig,
    RetryDefaults,
    MemoryConfig,
    GpuConfig,
    TapOutConfig,
    QuotaConfig,
    PrometheusConfig,
    OtelConfig,
    load_runtime_config,
    get_runtime_config,
    reset_runtime_config,
    # gpu
    GpuBackend,
    # discovery
    discover_processors,
    resolve_processor_class,
    get_processor_tags,
    get_gpu_capability,
    get_all_modalities,
    get_all_categories,
    filter_processors,
    # workflow
    WorkflowState,
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
    # dsl
    step,
    tap_out,
    workflow,
    DslCompiler,
    # project
    GrdkProject,
    # builder
    Workflow,
    WorkflowStep,
    TapOutStep,
    # executor
    WorkflowExecutor,
    # context
    ExecutionContext,
    configure_logging,
    get_logger,
    # metrics
    StepMetrics,
    WorkflowMetrics,
    # result
    WorkflowResult,
    # validation
    ValidationError,
    validate_workflow,
    # errors
    StepRetryExhaustedError,
    StepTimeoutError,
    MemoryThresholdError,
    CheckpointError,
    ResumeError,
    ResolutionError,
    FallbackExhaustedError,
    QuotaExceededError,
    # hardware
    GpuDeviceInfo,
    HardwareContext,
    LocalHardwareContext,
    # plan
    ResolvedStep,
    ParallelGroup,
    Substitution,
    ResolvedExecutionPlan,
    ExecutedStepRecord,
    AsExecutedManifest,
    # resolver
    Resolver,
    # resilience
    RetryPolicy,
    CircuitBreaker,
    ShutdownCoordinator,
    TilingStrategy,
    # checkpoint
    CheckpointState,
    CheckpointManager,
    compute_workflow_hash,
    CHECKPOINT_SCHEMA_VERSION,
    # history
    ExecutionRecord,
    ExecutionHistoryDB,
    # quota
    ResourceQuota,
    QuotaEnforcer,
    # instrumentation
    ExecutionHook,
    # lineage
    DataLineage,
    LineageTransform,
    compute_array_hash,
    build_lineage,
    embed_lineage_geotiff,
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
    "resolve_workflow",
    # execution: tags
    "ImageModality",
    "DetectionType",
    "GpuCapability",
    "SegmentationType",
    "ProjectTags",
    "WorkflowTags",
    # execution: chip
    "ChipLabel",
    "PolygonRegion",
    "Chip",
    "ChipSet",
    # execution: config
    "RuntimeConfig",
    "LogConfig",
    "RetryDefaults",
    "MemoryConfig",
    "GpuConfig",
    "TapOutConfig",
    "QuotaConfig",
    "PrometheusConfig",
    "OtelConfig",
    "load_runtime_config",
    "get_runtime_config",
    "reset_runtime_config",
    # execution: gpu
    "GpuBackend",
    # execution: discovery
    "discover_processors",
    "resolve_processor_class",
    "get_processor_tags",
    "get_gpu_capability",
    "get_all_modalities",
    "get_all_categories",
    "filter_processors",
    # execution: workflow
    "WorkflowState",
    "ProcessingStep",
    "TapOutStepDef",
    "WorkflowDefinition",
    # execution: dsl
    "step",
    "tap_out",
    "workflow",
    "DslCompiler",
    # execution: project
    "GrdkProject",
    # execution: builder
    "Workflow",
    "WorkflowStep",
    "TapOutStep",
    # execution: executor
    "WorkflowExecutor",
    # execution: context
    "ExecutionContext",
    "configure_logging",
    "get_logger",
    # execution: metrics
    "StepMetrics",
    "WorkflowMetrics",
    # execution: result
    "WorkflowResult",
    # execution: validation
    "ValidationError",
    "validate_workflow",
    # execution: errors
    "StepRetryExhaustedError",
    "StepTimeoutError",
    "MemoryThresholdError",
    "CheckpointError",
    "ResumeError",
    "ResolutionError",
    "FallbackExhaustedError",
    "QuotaExceededError",
    # execution: hardware
    "GpuDeviceInfo",
    "HardwareContext",
    "LocalHardwareContext",
    # execution: plan
    "ResolvedStep",
    "ParallelGroup",
    "Substitution",
    "ResolvedExecutionPlan",
    "ExecutedStepRecord",
    "AsExecutedManifest",
    # execution: resolver
    "Resolver",
    # execution: resilience
    "RetryPolicy",
    "CircuitBreaker",
    "ShutdownCoordinator",
    "TilingStrategy",
    # execution: checkpoint
    "CheckpointState",
    "CheckpointManager",
    "compute_workflow_hash",
    "CHECKPOINT_SCHEMA_VERSION",
    # execution: history
    "ExecutionRecord",
    "ExecutionHistoryDB",
    # execution: quota
    "ResourceQuota",
    "QuotaEnforcer",
    # execution: instrumentation
    "ExecutionHook",
    # execution: lineage
    "DataLineage",
    "LineageTransform",
    "compute_array_hash",
    "build_lineage",
    "embed_lineage_geotiff",
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
