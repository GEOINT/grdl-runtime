# -*- coding: utf-8 -*-
"""
Catalog Models - Data models for artifact metadata.

Defines the Artifact and UpdateResult data models used by the
artifact catalog database and update checker.

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
from typing import Any, Dict, List, Optional


class Artifact:
    """Metadata for a GRDL processor or GRDK workflow artifact.

    Parameters
    ----------
    name : str
        Artifact name.
    version : str
        Semantic version string.
    artifact_type : str
        One of 'grdl_processor' or 'grdk_workflow'.
    description : str
        Human-readable description.
    author : str
        Author name.
    license : str
        License identifier.
    pypi_package : Optional[str]
        PyPI package name.
    conda_package : Optional[str]
        Conda package name.
    conda_channel : Optional[str]
        Conda channel (e.g., 'conda-forge').
    processor_class : Optional[str]
        Fully-qualified Python class path (grdl_processor only).
    processor_version : Optional[str]
        @processor_version value (grdl_processor only).
    processor_type : Optional[str]
        One of 'transform', 'detector', 'decomposition' (grdl_processor only).
    yaml_definition : Optional[str]
        Full YAML workflow definition (grdk_workflow only).
    python_dsl : Optional[str]
        Full Python DSL source (grdk_workflow only).
    tags : Optional[Dict[str, List[str]]]
        Tag key-value pairs (grdk_workflow only).
    id : Optional[int]
        Database row ID (set after insertion).
    """

    def __init__(
        self,
        name: str,
        version: str,
        artifact_type: str,
        description: str = "",
        author: str = "",
        license: str = "MIT",
        pypi_package: Optional[str] = None,
        conda_package: Optional[str] = None,
        conda_channel: Optional[str] = None,
        processor_class: Optional[str] = None,
        processor_version: Optional[str] = None,
        processor_type: Optional[str] = None,
        yaml_definition: Optional[str] = None,
        python_dsl: Optional[str] = None,
        tags: Optional[Dict[str, List[str]]] = None,
        id: Optional[int] = None,
    ) -> None:
        if artifact_type not in ('grdl_processor', 'grdk_workflow'):
            raise ValueError(
                f"artifact_type must be 'grdl_processor' or 'grdk_workflow', "
                f"got {artifact_type!r}"
            )
        self.id = id
        self.name = name
        self.version = version
        self.artifact_type = artifact_type
        self.description = description
        self.author = author
        self.license = license
        self.pypi_package = pypi_package
        self.conda_package = conda_package
        self.conda_channel = conda_channel
        self.processor_class = processor_class
        self.processor_version = processor_version
        self.processor_type = processor_type
        self.yaml_definition = yaml_definition
        self.python_dsl = python_dsl
        self.tags = tags or {}

    def __repr__(self) -> str:
        return (
            f"Artifact(name={self.name!r}, version={self.version!r}, "
            f"type={self.artifact_type!r})"
        )


class UpdateResult:
    """Result of checking a single artifact for updates.

    Parameters
    ----------
    artifact : Artifact
        The artifact that was checked.
    source : str
        Update source ('pypi' or 'conda').
    current_version : str
        Currently installed version.
    latest_version : Optional[str]
        Latest available version, or None if check failed.
    update_available : bool
        Whether a newer version is available.
    error : Optional[str]
        Error message if the check failed.
    """

    def __init__(
        self,
        artifact: Artifact,
        source: str,
        current_version: str,
        latest_version: Optional[str] = None,
        update_available: bool = False,
        error: Optional[str] = None,
    ) -> None:
        self.artifact = artifact
        self.source = source
        self.current_version = current_version
        self.latest_version = latest_version
        self.update_available = update_available
        self.error = error

    def __repr__(self) -> str:
        if self.update_available:
            return (
                f"UpdateResult({self.artifact.name!r}: "
                f"{self.current_version} → {self.latest_version})"
            )
        return f"UpdateResult({self.artifact.name!r}: up to date)"
