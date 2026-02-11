# -*- coding: utf-8 -*-
"""
Workflow Validation — Pre-execution validation of workflow definitions.

Checks that all processors can be resolved, required parameters are present,
and parameter values satisfy type and constraint checks.  Validation runs
before any step executes, providing fast feedback on malformed workflows.

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
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from grdl_rt.catalog.base import ArtifactCatalogBase
    from grdl_rt.execution.workflow import WorkflowDefinition

# grdl-runtime internal
from grdl_rt.execution.workflow import ProcessingStep, TapOutStepDef


@dataclass
class ValidationError:
    """A single validation problem found in a WorkflowDefinition.

    Attributes
    ----------
    step_index : Optional[int]
        Index of the step with the problem, or ``None`` for workflow-level
        issues.
    processor_name : Optional[str]
        Processor name involved, if applicable.
    code : str
        Machine-readable error code.  One of:

        - ``EMPTY_WORKFLOW`` — no processing steps
        - ``PROCESSOR_NOT_FOUND`` — processor cannot be resolved
        - ``MISSING_REQUIRED_PARAM`` — required parameter not provided
        - ``INVALID_PARAM_TYPE`` — parameter value has wrong type
        - ``INVALID_PARAM_VALUE`` — parameter value violates constraints
    message : str
        Human-readable description of the problem.
    """

    step_index: Optional[int]
    processor_name: Optional[str]
    code: str
    message: str


def validate_workflow(
    workflow: 'WorkflowDefinition',
    catalog: Optional['ArtifactCatalogBase'] = None,
) -> List[ValidationError]:
    """Validate a WorkflowDefinition against the catalog and processor signatures.

    Checks performed:

    1. Workflow has at least one processing step (non-tap-out).
    2. Each ``ProcessingStep``'s ``processor_name`` resolves to a class via
       :func:`resolve_processor_class`.
    3. Required ``__init__`` parameters (no default value) are present in
       ``step.params``.  Parameters named ``self`` and ``metadata`` are
       excluded (``metadata`` is injected by the framework).
    4. If the processor class has ``__param_specs__`` (Annotated metadata),
       parameter types and constraint values are checked.

    Parameters
    ----------
    workflow : WorkflowDefinition
        The workflow to validate.
    catalog : ArtifactCatalogBase, optional
        Catalog for processor resolution.  If ``None``, uses the default
        discovery catalog.

    Returns
    -------
    List[ValidationError]
        Empty list if valid; otherwise, list of problems found.
    """
    from grdl_rt.execution.discovery import resolve_processor_class

    if catalog is not None:
        from grdl_rt.execution.discovery import init_discovery
        init_discovery(catalog)

    errors: List[ValidationError] = []

    # Check: workflow has at least one processing step
    proc_steps = [s for s in workflow.steps if isinstance(s, ProcessingStep)]
    if not proc_steps:
        errors.append(ValidationError(
            step_index=None,
            processor_name=None,
            code="EMPTY_WORKFLOW",
            message="Workflow has no processing steps.",
        ))

    # DAG structure validation
    dag_errors = workflow.validate_dag()
    for msg in dag_errors:
        code = "DAG_CYCLE" if "cycle" in msg.lower() else "UNRESOLVED_DEPENDENCY"
        if "Duplicate" in msg:
            code = "DUPLICATE_STEP_ID"
        errors.append(ValidationError(
            step_index=None,
            processor_name=None,
            code=code,
            message=msg,
        ))

    # Phase ordering validation
    _validate_phase_ordering(workflow, errors)

    # Condition expression validation
    _validate_conditions(workflow, errors)

    for i, step in enumerate(workflow.steps):
        if isinstance(step, TapOutStepDef):
            continue

        if not isinstance(step, ProcessingStep):
            continue

        # Check 1: processor resolves
        try:
            cls = resolve_processor_class(step.processor_name)
        except (ImportError, Exception) as e:
            errors.append(ValidationError(
                step_index=i,
                processor_name=step.processor_name,
                code="PROCESSOR_NOT_FOUND",
                message=f"Cannot resolve processor '{step.processor_name}': {e}",
            ))
            continue

        # Check 2: required parameters via __param_specs__ or signature
        param_specs = getattr(cls, '__param_specs__', None)
        if param_specs is not None:
            _validate_via_param_specs(i, step, param_specs, errors)
        else:
            _validate_via_signature(i, step, cls, errors)

    return errors


def _validate_via_param_specs(
    step_index: int,
    step: ProcessingStep,
    param_specs: list,
    errors: List[ValidationError],
) -> None:
    """Validate step params against the processor's ``__param_specs__``."""
    for spec in param_specs:
        name = getattr(spec, 'name', None)
        if name is None:
            continue

        required = getattr(spec, 'required', False)
        if required and name not in step.params:
            errors.append(ValidationError(
                step_index=step_index,
                processor_name=step.processor_name,
                code="MISSING_REQUIRED_PARAM",
                message=(
                    f"Processor '{step.processor_name}' requires parameter "
                    f"'{name}' but it is not provided."
                ),
            ))
        elif name in step.params:
            value = step.params[name]
            validate_fn = getattr(spec, 'validate', None)
            if validate_fn is not None:
                try:
                    validate_fn(value)
                except TypeError as e:
                    errors.append(ValidationError(
                        step_index=step_index,
                        processor_name=step.processor_name,
                        code="INVALID_PARAM_TYPE",
                        message=str(e),
                    ))
                except ValueError as e:
                    errors.append(ValidationError(
                        step_index=step_index,
                        processor_name=step.processor_name,
                        code="INVALID_PARAM_VALUE",
                        message=str(e),
                    ))


def _validate_via_signature(
    step_index: int,
    step: ProcessingStep,
    cls: type,
    errors: List[ValidationError],
) -> None:
    """Fallback: validate step params against ``__init__`` signature."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'metadata'):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if (
            param.default is inspect.Parameter.empty
            and param_name not in step.params
        ):
            errors.append(ValidationError(
                step_index=step_index,
                processor_name=step.processor_name,
                code="MISSING_REQUIRED_PARAM",
                message=(
                    f"Processor '{step.processor_name}' requires parameter "
                    f"'{param_name}' but it is not provided."
                ),
            ))


def _validate_phase_ordering(
    workflow: 'WorkflowDefinition',
    errors: List[ValidationError],
) -> None:
    """Check that step phases respect canonical ordering.

    A step should not be in an earlier phase than its dependencies.
    Violations produce warnings (PHASE_ORDER_VIOLATION), not blocking errors.
    """
    import ast as _ast  # noqa: F811
    from grdl_rt.execution.workflow import ExecutionPhase, PHASE_ORDER

    # Build step_id -> phase mapping
    step_phases = {}
    for step in workflow.steps:
        if hasattr(step, 'phase') and step.phase is not None:
            try:
                phase_enum = ExecutionPhase(step.phase)
                step_phases[step.id] = phase_enum
            except ValueError:
                pass

    # Check each step's phase against its dependencies' phases
    for step in workflow.steps:
        if step.id not in step_phases:
            continue
        step_phase = step_phases[step.id]
        step_order = PHASE_ORDER[step_phase]

        for dep_id in step.depends_on:
            if dep_id not in step_phases:
                continue
            dep_phase = step_phases[dep_id]
            dep_order = PHASE_ORDER[dep_phase]

            if step_order < dep_order:
                errors.append(ValidationError(
                    step_index=None,
                    processor_name=getattr(step, 'processor_name', None),
                    code="PHASE_ORDER_VIOLATION",
                    message=(
                        f"Step '{step.id}' (phase {step_phase.value}) "
                        f"depends on '{dep_id}' (phase {dep_phase.value}) "
                        f"which is a later phase"
                    ),
                ))


def _validate_conditions(
    workflow: 'WorkflowDefinition',
    errors: List[ValidationError],
) -> None:
    """Check that condition expressions parse without errors."""
    import ast

    for i, step in enumerate(workflow.steps):
        if not isinstance(step, ProcessingStep):
            continue
        if step.condition is None:
            continue
        try:
            ast.parse(step.condition, mode='eval')
        except SyntaxError as e:
            errors.append(ValidationError(
                step_index=i,
                processor_name=step.processor_name,
                code="INVALID_CONDITION",
                message=f"Invalid condition expression: {e}",
            ))
