"""
Parameter Schema Extraction — JSON Schema generation from processor metadata.

Extracts ``__param_specs__`` and ``Annotated`` type metadata from GRDL
processor classes and produces JSON Schema (draft 2020-12) documents.
The resulting schemas are stored in the catalog so that tools like grdk
can generate UI controls without importing processor code.

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
"""

# Standard library
from typing import Any

# Type map: Python type → JSON Schema type string
_TYPE_MAP: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def extract_param_schema(processor_class: type) -> dict[str, Any]:
    """Produce a JSON Schema from a processor's ``__param_specs__``.

    Introspects the ``__param_specs__`` tuple (built by
    ``grdl.image_processing.params.collect_param_specs``) and emits a
    JSON Schema draft 2020-12 ``object`` schema with one property per
    tunable parameter.

    Handles:

    * ``int`` / ``float`` / ``str`` / ``bool`` base types
    * ``Range(min, max)`` → ``minimum`` / ``maximum``
    * ``Options(...)`` → ``enum``
    * ``Desc(...)`` → ``description``
    * Optional parameters with defaults → ``default``
    * Required parameters → ``required`` array

    Parameters
    ----------
    processor_class : type
        A processor class that has a ``__param_specs__`` attribute
        (typically an ``ImageProcessor`` subclass).

    Returns
    -------
    dict
        A valid JSON Schema document.  Returns an empty ``object``
        schema when the class has no ``__param_specs__``.
    """
    param_specs: tuple | None = getattr(processor_class, "__param_specs__", None)

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    if not param_specs:
        return schema

    required: list = []

    for spec in param_specs:
        prop = _spec_to_property(spec)
        schema["properties"][spec.name] = prop

        if spec.required:
            required.append(spec.name)

    if required:
        schema["required"] = required

    return schema


def _spec_to_property(spec: Any) -> dict[str, Any]:
    """Convert a single ``ParamSpec`` to a JSON Schema property dict."""
    prop: dict[str, Any] = {}

    # Base type mapping
    json_type = _TYPE_MAP.get(spec.param_type)
    if json_type:
        prop["type"] = json_type
    # object / unknown types → no type constraint in schema

    # Description
    if spec.description:
        prop["description"] = spec.description

    # Range constraints
    if spec.min_value is not None:
        prop["minimum"] = spec.min_value
    if spec.max_value is not None:
        prop["maximum"] = spec.max_value

    # Enum choices
    if spec.choices is not None:
        prop["enum"] = list(spec.choices)

    # Default value
    if spec._has_default:
        prop["default"] = spec.default

    return prop
