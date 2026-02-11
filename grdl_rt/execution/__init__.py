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
    RuntimeConfig,
    LogConfig,
    RetryDefaults,
    MemoryConfig,
    GpuConfig,
    TapOutConfig,
    load_runtime_config,
    get_runtime_config,
    reset_runtime_config,
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
    SCHEMA_VERSION,
    WorkflowState,
    ExecutionPhase,
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
)
from grdl_rt.execution.dsl import (
    step,
    tap_out,
    workflow,
    DslCompiler,
)
from grdl_rt.execution.project import (
    GrdkProject,
)
from grdl_rt.execution.builder import (
    Workflow,
    WorkflowStep,
    DeferredStep,
    TapOutStep,
    BranchBuilder,
)
from grdl_rt.execution.executor import (
    WorkflowExecutor,
)
from grdl_rt.execution.dag import (
    evaluate_condition,
)
from grdl_rt.execution.dag_executor import (
    DAGExecutor,
)
from grdl_rt.execution.context import (
    ExecutionContext,
    configure_logging,
    get_logger,
)
from grdl_rt.execution.metrics import (
    StepMetrics,
    WorkflowMetrics,
)
from grdl_rt.execution.result import (
    WorkflowResult,
)
from grdl_rt.execution.validation import (
    ValidationError,
    validate_workflow,
)
from grdl_rt.execution.errors import (
    StepRetryExhaustedError,
    StepTimeoutError,
    MemoryThresholdError,
    CheckpointError,
    ResumeError,
    DAGCycleError,
    ConditionError,
)
from grdl_rt.execution.resilience import (
    RetryPolicy,
    CircuitBreaker,
    ShutdownCoordinator,
    TilingStrategy,
)
from grdl_rt.execution.checkpoint import (
    CheckpointState,
    CheckpointManager,
    compute_workflow_hash,
    CHECKPOINT_SCHEMA_VERSION,
)
from grdl_rt.execution.history import (
    ExecutionRecord,
    ExecutionHistoryDB,
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
    "RuntimeConfig",
    "LogConfig",
    "RetryDefaults",
    "MemoryConfig",
    "GpuConfig",
    "TapOutConfig",
    "load_runtime_config",
    "get_runtime_config",
    "reset_runtime_config",
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
]
