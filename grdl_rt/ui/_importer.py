"""
Dynamic Importer — Discover processors and workflows in Python files.

Loads ``.py`` files at runtime via ``importlib`` and scans for GRDL
processor classes, ``@workflow``-decorated functions, or ``Workflow``
builder instances.  Also extracts tunable parameter metadata for UI
form generation.

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
2026-02-13
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# grdl base classes — optional graceful fallback
try:
    from grdl.image_processing.base import ImageProcessor, ImageTransform
except ImportError:
    ImageProcessor = None  # type: ignore[misc,assignment]
    ImageTransform = None  # type: ignore[misc,assignment]

# grdl-runtime types used for workflow detection
from grdl_rt.execution.builder import Workflow
from grdl_rt.execution.workflow import WorkflowDefinition

# ── ParamInfo ────────────────────────────────────────────────────────


@dataclass
class ParamInfo:
    """Metadata for a single tunable parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    param_type : str
        Type as a string (``"int"``, ``"float"``, ``"str"``, ``"bool"``).
    default : Any
        Default value, or ``inspect.Parameter.empty`` if required.
    required : bool
        Whether the parameter must be supplied.
    description : str
        Human-readable description.
    choices : list or None
        Enumeration of allowed values.
    min_value : float or None
        Lower bound (inclusive).
    max_value : float or None
        Upper bound (inclusive).
    """

    name: str
    param_type: str = "str"
    default: Any = None
    required: bool = False
    description: str = ""
    choices: list | None = None
    min_value: float | None = None
    max_value: float | None = None


# ── Module loading ───────────────────────────────────────────────────


def _load_module(path: Path):
    """Import a ``.py`` file as a module and return it.

    The module is inserted into ``sys.modules`` under a synthetic name
    to allow subsequent introspection.  If the module was previously
    loaded (same path), the cached version is returned.
    """
    abs_path = path.resolve()
    module_name = f"_grdl_ui_import_.{abs_path.stem}_{id(abs_path)}"

    # Reuse if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(abs_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {abs_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


# ── Processor discovery ──────────────────────────────────────────────


def _is_processor_class(obj: Any) -> bool:
    """Heuristic: does *obj* look like a GRDL processor class?"""
    if not isinstance(obj, type):
        return False
    # Explicit GRDL base classes
    if ImageProcessor is not None and issubclass(obj, ImageProcessor):
        return obj is not ImageProcessor
    if ImageTransform is not None and issubclass(obj, ImageTransform):
        return obj is not ImageTransform
    # Duck-typing: has apply() method
    if hasattr(obj, "apply") and callable(obj.apply):
        return True
    # Has param specs (decorated processor)
    return bool(hasattr(obj, "__param_specs__"))


def discover_processors_in_module(
    path: Path,
) -> list[tuple[str, type]]:
    """Import a ``.py`` file and find processor classes within it.

    Parameters
    ----------
    path : Path
        Path to a Python source file.

    Returns
    -------
    list of (name, class)
        Each discovered processor class with its name.
    """
    module = _load_module(path)
    processors: list[tuple[str, type]] = []
    for name in dir(module):
        obj = getattr(module, name)
        if _is_processor_class(obj) and getattr(obj, "__module__", None) == module.__name__:
            processors.append((name, obj))
    return processors


# ── Workflow discovery ───────────────────────────────────────────────


def discover_workflow_in_module(
    path: Path,
) -> WorkflowDefinition | Workflow | None:
    """Import a ``.py`` file and check if it defines a workflow.

    Detection priority:

    1. Module-level object with ``_workflow_definition`` attribute
       (from the ``@workflow`` decorator).
    2. Module-level ``Workflow`` builder instance.
    3. A ``main()`` or ``build_workflow()`` callable that returns
       a ``Workflow`` or ``WorkflowDefinition``.

    Parameters
    ----------
    path : Path
        Path to a Python source file.

    Returns
    -------
    WorkflowDefinition, Workflow, or None
        The discovered workflow, or ``None`` if the file is not a
        workflow definition.
    """
    module = _load_module(path)

    # 1. @workflow-decorated function → has _workflow_definition
    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, "_workflow_definition"):
            wf_def = obj._workflow_definition
            if isinstance(wf_def, WorkflowDefinition):
                return wf_def

    # 2. Module-level Workflow builder instance
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Workflow):
            return obj

    # 3. Callable that returns a workflow
    for func_name in ("main", "build_workflow", "create_workflow"):
        func = getattr(module, func_name, None)
        if func is not None and callable(func):
            try:
                result = func()
                if isinstance(result, (Workflow, WorkflowDefinition)):
                    return result
            except Exception:
                continue

    return None


def classify_py_file(path: Path) -> Literal["workflow", "component"]:
    """Determine whether a ``.py`` file is a workflow or a bare component.

    Parameters
    ----------
    path : Path
        Path to a Python source file.

    Returns
    -------
    ``"workflow"`` or ``"component"``
    """
    wf = discover_workflow_in_module(path)
    if wf is not None:
        return "workflow"
    return "component"


# ── Parameter extraction ─────────────────────────────────────────────

# Python type → display string
_TYPE_NAMES = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
}


def extract_tunable_params(cls: type) -> dict[str, ParamInfo]:
    """Extract tunable parameters from a processor class.

    Tries ``extract_param_schema()`` from the catalog first (uses
    ``__param_specs__``).  Falls back to ``inspect.signature()``
    on ``__init__``.

    Parameters
    ----------
    cls : type
        A processor class.

    Returns
    -------
    Dict[str, ParamInfo]
        Ordered mapping of parameter name to info.
    """
    # Try catalog schema extraction first
    try:
        from grdl_rt.catalog.schema import extract_param_schema

        schema = extract_param_schema(cls)
        props = schema.get("properties", {})
        required_names = set(schema.get("required", []))
        if props:
            return _params_from_schema(props, required_names)
    except Exception:
        pass

    # Fallback: inspect __init__ signature
    return _params_from_signature(cls)


def _params_from_schema(
    properties: dict[str, Any],
    required_names: set,
) -> dict[str, ParamInfo]:
    """Convert JSON Schema properties to ParamInfo dict."""
    result: dict[str, ParamInfo] = {}
    schema_type_map = {
        "integer": "int",
        "number": "float",
        "string": "str",
        "boolean": "bool",
    }
    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        result[name] = ParamInfo(
            name=name,
            param_type=schema_type_map.get(json_type, "str"),
            default=prop.get("default"),
            required=name in required_names,
            description=prop.get("description", ""),
            choices=prop.get("enum"),
            min_value=prop.get("minimum"),
            max_value=prop.get("maximum"),
        )
    return result


def _params_from_signature(cls: type) -> dict[str, ParamInfo]:
    """Extract parameter info from ``__init__`` signature."""
    result: dict[str, ParamInfo] = {}
    try:
        sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    except (ValueError, TypeError):
        return result

    skip = {"self", "metadata", "args", "kwargs"}
    for pname, param in sig.parameters.items():
        if pname in skip:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        has_default = param.default is not inspect.Parameter.empty
        annotation = param.annotation

        # Resolve type name
        if annotation is not inspect.Parameter.empty:
            ptype = _TYPE_NAMES.get(annotation, "str")
        elif has_default and param.default is not None:
            ptype = _TYPE_NAMES.get(type(param.default), "str")
        else:
            ptype = "str"

        result[pname] = ParamInfo(
            name=pname,
            param_type=ptype,
            default=param.default if has_default else None,
            required=not has_default,
        )

    return result
