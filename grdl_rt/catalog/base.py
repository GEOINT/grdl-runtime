# -*- coding: utf-8 -*-
"""
Catalog Base - Abstract interface for artifact catalog storage backends.

Defines the ArtifactCatalogBase ABC that all catalog storage
implementations must satisfy. Concrete backends include SQLite
(full-featured) and YAML (lightweight, human-readable).

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
2026-02-10
"""

# Standard library
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# grdl-runtime internal
from grdl_rt.catalog.models import Artifact


class ArtifactCatalogBase(ABC):
    """Abstract interface for artifact catalog storage backends.

    Defines the contract for storing, retrieving, searching, and
    managing GRDL processor and GRDK workflow artifact metadata.

    Concrete implementations must provide all abstract methods.
    Context manager support (``__enter__`` / ``__exit__``) is
    provided by default and delegates to :meth:`close`.
    """

    @abstractmethod
    def add_artifact(self, artifact: Artifact) -> int:
        """Add or replace an artifact in the catalog.

        Parameters
        ----------
        artifact : Artifact

        Returns
        -------
        int
            Unique ID of the inserted/replaced artifact.
        """
        ...

    @abstractmethod
    def remove_artifact(self, name: str, version: str) -> bool:
        """Remove an artifact by name and version.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        bool
            True if an artifact was removed.
        """
        ...

    @abstractmethod
    def get_artifact(self, name: str, version: str) -> Optional[Artifact]:
        """Retrieve a specific artifact by name and version.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        Optional[Artifact]
            The artifact, or None if not found.
        """
        ...

    @abstractmethod
    def list_artifacts(
        self, artifact_type: Optional[str] = None,
    ) -> List[Artifact]:
        """List all artifacts, optionally filtered by type.

        Parameters
        ----------
        artifact_type : Optional[str]
            Filter by 'grdl_processor' or 'grdk_workflow'.

        Returns
        -------
        List[Artifact]
        """
        ...

    @abstractmethod
    def search(self, query: str) -> List[Artifact]:
        """Search artifacts by text query.

        The search mechanism is backend-specific. SQLite uses FTS5
        full-text search; YAML uses case-insensitive substring
        matching on name and description.

        Parameters
        ----------
        query : str

        Returns
        -------
        List[Artifact]
        """
        ...

    @abstractmethod
    def search_by_tags(self, tags: Dict[str, str]) -> List[Artifact]:
        """Search artifacts by tag key-value pairs (AND logic).

        All specified tag key-value pairs must match for an artifact
        to be included in the results.

        Parameters
        ----------
        tags : Dict[str, str]
            Tag filters. All must match.

        Returns
        -------
        List[Artifact]
        """
        ...

    @abstractmethod
    def update_remote_version(
        self, artifact_id: int, source: str, latest_version: str,
    ) -> None:
        """Record the latest remote version for an artifact.

        Parameters
        ----------
        artifact_id : int
        source : str
            Update source (e.g. 'pypi' or 'conda').
        latest_version : str
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the catalog backend."""
        ...

    # --- Default implementations ---

    def __enter__(self) -> 'ArtifactCatalogBase':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
