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
from typing import Any, Dict, List, Optional

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

    def get_alternatives(
        self, name: str, version: str,
    ) -> List[Dict[str, Any]]:
        """Return alternative processor entries for the given artifact.

        Default implementation returns an empty list.  Backends that
        support alternatives override this method.

        Parameters
        ----------
        name : str
            Artifact name.
        version : str
            Artifact version.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict has ``processor_name``, ``priority``, and
            ``compatibility_notes`` keys.
        """
        return []

    def set_alternatives(
        self, name: str, version: str,
        alternatives: List[Dict[str, Any]],
    ) -> None:
        """Set alternative processor entries for the given artifact.

        Default implementation raises ``NotImplementedError``.
        Backends that support alternatives override this method.

        Parameters
        ----------
        name : str
            Artifact name.
        version : str
            Artifact version.
        alternatives : List[Dict[str, Any]]
            Each dict must have ``processor_name``, ``priority``,
            and ``compatibility_notes`` keys.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_alternatives"
        )

    def get_param_schema(
        self, name: str, version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the JSON Schema for a processor's tunable parameters.

        Default implementation finds the artifact and returns its
        ``param_schema`` field.  If *version* is ``None``, returns the
        schema from the first artifact matching *name*.

        Parameters
        ----------
        name : str
            Processor (artifact) name.
        version : Optional[str]
            Artifact version.  If ``None``, picks the first match.

        Returns
        -------
        Optional[Dict[str, Any]]
            JSON Schema dict, or ``None`` if not found or no schema
            stored.
        """
        if version is not None:
            artifact = self.get_artifact(name, version)
            if artifact is not None:
                return artifact.param_schema
            return None

        # No version specified — pick first matching processor
        for artifact in self.list_artifacts(artifact_type="grdl_processor"):
            if artifact.name == name:
                return artifact.param_schema
        return None

    def __enter__(self) -> 'ArtifactCatalogBase':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
