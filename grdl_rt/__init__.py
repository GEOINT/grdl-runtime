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

__version__ = "0.1.2"
__author__ = "Claude Code (Anthropic)"

# ── API convenience functions ────────────────────────────────────────
from grdl_rt.api import execute_workflow, load_workflow, resolve_workflow

# ── Catalog subpackage ───────────────────────────────────────────────
from grdl_rt.catalog import (
    Artifact,
    ArtifactCatalog,
    ArtifactCatalogBase,
    ArtifactUpdateWorker,
    FederatedArtifactCatalog,
    SqliteArtifactCatalog,
    ThreadExecutorPool,
    UpdateResult,
    YamlArtifactCatalog,
    ensure_config_dir,
    resolve_catalog_path,
)

# ── Execution subpackage ─────────────────────────────────────────────
from grdl_rt.execution import (
    CHECKPOINT_SCHEMA_VERSION,
    AsExecutedManifest,
    CheckpointError,
    CheckpointManager,
    # checkpoint
    CheckpointState,
    Chip,
    # chip
    ChipLabel,
    ChipSet,
    CircuitBreaker,
    # lineage
    DataLineage,
    DetectionType,
    DslCompiler,
    ExecutedStepRecord,
    # context
    ExecutionContext,
    ExecutionHistoryDB,
    # instrumentation
    ExecutionHook,
    ExecutionPhase,
    # history
    ExecutionRecord,
    FallbackExhaustedError,
    # gpu
    GpuBackend,
    GpuCapability,
    GpuConfig,
    # hardware
    GpuDeviceInfo,
    # project
    GrdkProject,
    HardwareContext,
    # tags
    ImageModality,
    LineageTransform,
    LocalHardwareContext,
    LogConfig,
    MemoryConfig,
    MemoryThresholdError,
    OtelConfig,
    OutputFormat,
    ParallelGroup,
    PolygonRegion,
    ProcessingStep,
    ProjectTags,
    PrometheusConfig,
    QuotaConfig,
    QuotaEnforcer,
    QuotaExceededError,
    ResolutionError,
    ResolvedExecutionPlan,
    # plan
    ResolvedStep,
    # resolver
    Resolver,
    # quota
    ResourceQuota,
    ResumeError,
    RetryDefaults,
    # resilience
    RetryPolicy,
    # config
    RuntimeConfig,
    SegmentationType,
    ShutdownCoordinator,
    # metrics
    StepMetrics,
    # errors
    StepRetryExhaustedError,
    StepTimeoutError,
    Substitution,
    TapOutConfig,
    TapOutStep,
    TapOutStepDef,
    TilingStrategy,
    # validation
    ValidationError,
    # builder
    Workflow,
    WorkflowDefinition,
    # executor
    WorkflowExecutor,
    WorkflowMetrics,
    # result
    WorkflowResult,
    # workflow
    WorkflowState,
    WorkflowStep,
    WorkflowTags,
    build_lineage,
    compute_array_hash,
    compute_workflow_hash,
    configure_logging,
    # discovery
    discover_processors,
    embed_lineage_geotiff,
    filter_processors,
    get_all_categories,
    get_all_modalities,
    get_gpu_capability,
    get_logger,
    get_processor_tags,
    get_runtime_config,
    load_runtime_config,
    reset_runtime_config,
    resolve_processor_class,
    # dsl
    step,
    tap_out,
    validate_workflow,
    workflow,
)

__all__ = [
    # api
    "load_workflow",
    "execute_workflow",
    "resolve_workflow",
    # execution: tags
    "ImageModality",
    "DetectionType",
    "ExecutionPhase",
    "GpuCapability",
    "OutputFormat",
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
