"""
DSL Compiler - Bidirectional conversion between Python DSL, YAML, and runtime.

Provides the DslCompiler for converting between:
- Python DSL (@workflow decorator + step() calls)
- YAML workflow definitions
- ProcessingPipeline runtime objects (WorkflowDefinition)

Also provides the @workflow and step() decorators for the Python DSL.

Dependencies
------------
pyyaml

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
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Third-party
import yaml

# grdl-runtime internal
from grdl_rt.execution.tags import WorkflowTags
from grdl_rt.execution.workflow import (
    ProcessingStep,
    TapOutStepDef,
    WorkflowDefinition,
    WorkflowState,
)

# ---------------------------------------------------------------------------
# Python DSL decorators and functions
# ---------------------------------------------------------------------------

# Thread-local accumulator for steps during DSL function execution
_local = threading.local()


def _get_current_steps() -> list:
    """Return the thread-local step accumulator list."""
    if not hasattr(_local, "steps"):
        _local.steps = []
    return _local.steps


def step(
    processor: str,
    version: str = "",
    *,
    id: str | None = None,
    depends_on: list[str] | None = None,
    condition: str | None = None,
    phase: str | None = None,
    **params: Any,
) -> None:
    """Declare a processing step in a Python DSL workflow function.

    Must be called inside a function decorated with @workflow.

    Parameters
    ----------
    processor : str
        Processor class name.
    version : str
        Processor version string.
    id : str, optional
        Unique step identifier for DAG wiring.
    depends_on : List[str], optional
        Step IDs this step depends on.
    condition : str, optional
        Condition expression for conditional execution.
    phase : str, optional
        Execution phase annotation.
    **params
        Tunable parameter values.
    """
    _get_current_steps().append(
        ProcessingStep(
            processor_name=processor,
            processor_version=version,
            params=params if params else {},
            id=id,
            depends_on=depends_on,
            condition=condition,
            phase=phase,
        )
    )


def tap_out(
    path: str,
    format: str | None = None,
) -> None:
    """Declare a tap-out point in a Python DSL workflow function.

    Must be called inside a function decorated with @workflow.
    Writes the current intermediate to disk without modifying it.

    Parameters
    ----------
    path : str
        Output file path.
    format : str, optional
        Writer format override.
    """
    _get_current_steps().append(TapOutStepDef(path=path, format=format))


def workflow(
    name: str,
    version: str = "0.1.0",
    description: str = "",
    modalities: list[str] | None = None,
    niirs_range: tuple | None = None,
    day_capable: bool = True,
    night_capable: bool = False,
    detection_types: list[str] | None = None,
    segmentation_types: list[str] | None = None,
) -> Callable:
    """Decorator for Python DSL workflow definitions.

    Decorates a function that calls step() to define processing steps.
    The function is executed at decoration time to capture the steps.

    Parameters
    ----------
    name : str
        Workflow name.
    version : str
        Workflow version.
    description : str
        Workflow description.
    modalities : Optional[List[str]]
        Image modality tags.
    niirs_range : Optional[tuple]
        (min_niirs, max_niirs) quality range.
    day_capable : bool
    night_capable : bool
    detection_types : Optional[List[str]]
    segmentation_types : Optional[List[str]]

    Returns
    -------
    Callable
        Decorator that captures the workflow definition.
    """
    from grdl_rt.execution.tags import (
        DetectionType,
        ImageModality,
        SegmentationType,
    )

    def decorator(func: Callable) -> Callable:
        _local.steps = []

        # Execute the function to capture step() calls
        func()

        tags = WorkflowTags(
            modalities=[ImageModality(m) for m in (modalities or [])],
            niirs_range=tuple(niirs_range) if niirs_range else (0.0, 9.0),
            day_capable=day_capable,
            night_capable=night_capable,
            detection_types=[DetectionType(d) for d in (detection_types or [])],
            segmentation_types=[SegmentationType(s) for s in (segmentation_types or [])],
        )

        wf = WorkflowDefinition(
            name=name,
            version=version,
            description=description,
            steps=list(_get_current_steps()),
            tags=tags,
            state=WorkflowState.PUBLISHED,
        )

        # Attach the workflow definition to the function
        func._workflow_definition = wf  # type: ignore[attr-defined]
        _local.steps = []
        return func

    return decorator


# ---------------------------------------------------------------------------
# DslCompiler
# ---------------------------------------------------------------------------


class DslCompiler:
    """Bidirectional conversion between Python DSL, YAML, and runtime.

    Compiles workflow definitions from YAML files or Python DSL source
    into WorkflowDefinition objects, and generates YAML or Python DSL
    source from runtime WorkflowDefinition objects.
    """

    def compile_yaml(self, yaml_path: Path) -> WorkflowDefinition:
        """Compile a YAML workflow definition file.

        Parameters
        ----------
        yaml_path : Path
            Path to the YAML file.

        Returns
        -------
        WorkflowDefinition
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return WorkflowDefinition.from_dict(data)

    def compile_yaml_string(self, yaml_string: str) -> WorkflowDefinition:
        """Compile a YAML workflow definition from a string.

        Parameters
        ----------
        yaml_string : str
            YAML content.

        Returns
        -------
        WorkflowDefinition
        """
        data = yaml.safe_load(yaml_string)
        return WorkflowDefinition.from_dict(data)

    def to_yaml(self, workflow_def: WorkflowDefinition) -> str:
        """Generate YAML from a WorkflowDefinition.

        Parameters
        ----------
        workflow_def : WorkflowDefinition

        Returns
        -------
        str
            YAML string.
        """
        return yaml.dump(
            workflow_def.to_dict(),
            default_flow_style=False,
            sort_keys=False,
        )

    def to_python(self, workflow_def: WorkflowDefinition) -> str:
        """Generate Python DSL source from a WorkflowDefinition.

        Parameters
        ----------
        workflow_def : WorkflowDefinition

        Returns
        -------
        str
            Python source code.
        """
        tags = workflow_def.tags
        has_tap_out = any(isinstance(s, TapOutStepDef) for s in workflow_def.steps)
        if has_tap_out:
            import_line = "from grdl_rt.execution.dsl import workflow, step, tap_out"
        else:
            import_line = "from grdl_rt.execution.dsl import workflow, step"
        lines = [
            import_line,
            "",
            "",
        ]

        # Build decorator arguments
        decorator_args = [
            f'name="{workflow_def.name}"',
            f'version="{workflow_def.version}"',
        ]
        if workflow_def.description:
            decorator_args.append(f'description="{workflow_def.description}"')
        if tags.modalities:
            modalities_str = ", ".join(f'"{m.value}"' for m in tags.modalities)
            decorator_args.append(f"modalities=[{modalities_str}]")
        if tags.niirs_range != (0.0, 9.0):
            decorator_args.append(f"niirs_range=({tags.niirs_range[0]}, {tags.niirs_range[1]})")
        if tags.day_capable:
            decorator_args.append("day_capable=True")
        if tags.night_capable:
            decorator_args.append("night_capable=True")
        if tags.detection_types:
            dt_str = ", ".join(f'"{d.value}"' for d in tags.detection_types)
            decorator_args.append(f"detection_types=[{dt_str}]")
        if tags.segmentation_types:
            st_str = ", ".join(f'"{s.value}"' for s in tags.segmentation_types)
            decorator_args.append(f"segmentation_types=[{st_str}]")

        decorator = "@workflow(\n" + textwrap.indent(",\n".join(decorator_args), "    ") + ",\n)"
        lines.append(decorator)

        # Function name from workflow name
        func_name = workflow_def.name.lower().replace(" ", "_").replace("-", "_")
        lines.append(f"def {func_name}():")

        if not workflow_def.steps:
            lines.append("    pass")
        else:
            for s in workflow_def.steps:
                if isinstance(s, TapOutStepDef):
                    tap_args = [f'"{s.path}"']
                    if s.format:
                        tap_args.append(f'format="{s.format}"')
                    lines.append(f'    tap_out({", ".join(tap_args)})')
                else:
                    args = [f'"{s.processor_name}"']
                    if s.processor_version:
                        args.append(f'version="{s.processor_version}"')
                    if s.id is not None:
                        args.append(f'id="{s.id}"')
                    if s.depends_on:
                        deps_str = ", ".join(f'"{d}"' for d in s.depends_on)
                        args.append(f"depends_on=[{deps_str}]")
                    if s.condition is not None:
                        args.append(f'condition="{s.condition}"')
                    if s.phase is not None:
                        args.append(f'phase="{s.phase}"')
                    for k, v in s.params.items():
                        if isinstance(v, str):
                            args.append(f'{k}="{v}"')
                        else:
                            args.append(f"{k}={v!r}")
                    step_call = f'    step({", ".join(args)})'
                    lines.append(step_call)

        lines.append("")
        return "\n".join(lines)
