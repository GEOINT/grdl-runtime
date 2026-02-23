"""grdl-runtime execution subpackage — workflow engine and processor orchestration."""

from grdl_rt.execution.builder import (
    BranchBuilder,
    DeferredStep,
    TapOutStep,
    Workflow,
    WorkflowStep,
)
from grdl_rt.execution.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    CheckpointState,
    compute_workflow_hash,
)
from grdl_rt.execution.chip import (
    Chip,
    ChipLabel,
    ChipSet,
    PolygonRegion,
)
from grdl_rt.execution.config import (
    GpuConfig,
    LogConfig,
    MemoryConfig,
    OtelConfig,
    PrometheusConfig,
    QuotaConfig,
    RetryDefaults,
    RuntimeConfig,
    TapOutConfig,
    get_runtime_config,
    load_runtime_config,
    reset_runtime_config,
)
from grdl_rt.execution.context import (
    ExecutionContext,
    configure_logging,
    get_logger,
)
from grdl_rt.execution.dag import (
    evaluate_condition,
)
from grdl_rt.execution.dag_executor import (
    DAGExecutor,
)
from grdl_rt.execution.discovery import (
    discover_processors,
    filter_processors,
    get_all_categories,
    get_all_modalities,
    get_gpu_capability,
    get_processor_tags,
    resolve_processor_class,
)
from grdl_rt.execution.dsl import (
    DslCompiler,
    step,
    tap_out,
    workflow,
)
from grdl_rt.execution.errors import (
    CheckpointError,
    ConditionError,
    DAGCycleError,
    FallbackExhaustedError,
    MemoryThresholdError,
    QuotaExceededError,
    ResolutionError,
    ResumeError,
    StepRetryExhaustedError,
    StepTimeoutError,
)
from grdl_rt.execution.executor import (
    WorkflowExecutor,
)
from grdl_rt.execution.gpu import (
    GpuBackend,
)
from grdl_rt.execution.hardware import (
    GpuDeviceInfo,
    HardwareContext,
    LocalHardwareContext,
)
from grdl_rt.execution.history import (
    ExecutionHistoryDB,
    ExecutionRecord,
)
from grdl_rt.execution.instrumentation import (
    ExecutionHook,
)
from grdl_rt.execution.lineage import (
    DataLineage,
    LineageTransform,
    build_lineage,
    compute_array_hash,
    embed_lineage_geotiff,
)
from grdl_rt.execution.metrics import (
    StepMetrics,
    WorkflowMetrics,
)
from grdl_rt.execution.plan import (
    AsExecutedManifest,
    ExecutedStepRecord,
    ParallelGroup,
    ResolvedExecutionPlan,
    ResolvedStep,
    Substitution,
)
from grdl_rt.execution.project import (
    GrdkProject,
)
from grdl_rt.execution.quota import (
    QuotaEnforcer,
    ResourceQuota,
)
from grdl_rt.execution.resilience import (
    CircuitBreaker,
    RetryPolicy,
    ShutdownCoordinator,
    TilingStrategy,
)
from grdl_rt.execution.resolver import (
    Resolver,
)
from grdl_rt.execution.result import (
    WorkflowResult,
)
from grdl_rt.execution.tags import (
    DetectionType,
    ExecutionPhase,
    GpuCapability,
    ImageModality,
    OutputFormat,
    ProjectTags,
    SegmentationType,
    WorkflowTags,
)
from grdl_rt.execution.validation import (
    ValidationError,
    validate_workflow,
)
from grdl_rt.execution.workflow import (
    SCHEMA_VERSION,
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
    WorkflowState,
)

__all__ = [
    # tags
    "ImageModality",
    "DetectionType",
    "ExecutionPhase",
    "GpuCapability",
    "OutputFormat",
    "SegmentationType",
    "ProjectTags",
    "WorkflowTags",
    # chip
    "ChipLabel",
    "PolygonRegion",
    "Chip",
    "ChipSet",
    # config
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
    # gpu
    "GpuBackend",
    # discovery
    "discover_processors",
    "resolve_processor_class",
    "get_processor_tags",
    "get_gpu_capability",
    "get_all_modalities",
    "get_all_categories",
    "filter_processors",
    # workflow
    "SCHEMA_VERSION",
    "WorkflowState",
    "ExecutionPhase",
    "ProcessingStep",
    "TapOutStepDef",
    "WorkflowDefinition",
    # dsl
    "step",
    "tap_out",
    "workflow",
    "DslCompiler",
    # project
    "GrdkProject",
    # builder
    "Workflow",
    "WorkflowStep",
    "DeferredStep",
    "TapOutStep",
    "BranchBuilder",
    # executor
    "WorkflowExecutor",
    # dag
    "evaluate_condition",
    "DAGExecutor",
    # context
    "ExecutionContext",
    "configure_logging",
    "get_logger",
    # metrics
    "StepMetrics",
    "WorkflowMetrics",
    # result
    "WorkflowResult",
    # validation
    "ValidationError",
    "validate_workflow",
    # errors
    "StepRetryExhaustedError",
    "StepTimeoutError",
    "MemoryThresholdError",
    "DAGCycleError",
    "ConditionError",
    # resilience
    "RetryPolicy",
    "CircuitBreaker",
    "ShutdownCoordinator",
    "TilingStrategy",
    # checkpoint
    "CheckpointState",
    "CheckpointManager",
    "compute_workflow_hash",
    "CHECKPOINT_SCHEMA_VERSION",
    # history
    "ExecutionRecord",
    "ExecutionHistoryDB",
    # errors (TG4)
    "CheckpointError",
    "ResumeError",
    # errors (TG7)
    "ResolutionError",
    "FallbackExhaustedError",
    # hardware
    "GpuDeviceInfo",
    "HardwareContext",
    "LocalHardwareContext",
    # plan
    "ResolvedStep",
    "ParallelGroup",
    "Substitution",
    "ResolvedExecutionPlan",
    "ExecutedStepRecord",
    "AsExecutedManifest",
    # resolver
    "Resolver",
    # errors (TG9)
    "QuotaExceededError",
    # quota
    "ResourceQuota",
    "QuotaEnforcer",
    # instrumentation
    "ExecutionHook",
    # lineage
    "DataLineage",
    "LineageTransform",
    "compute_array_hash",
    "build_lineage",
    "embed_lineage_geotiff",
]
