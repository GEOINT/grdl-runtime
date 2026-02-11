# -*- coding: utf-8 -*-
"""
Federated Catalog - Composite catalog that broadcasts queries to multiple backends.

Implements ArtifactCatalogBase by federating read queries across one or
more backend catalogs, deduplicating results by (name, version) with
first-catalog-wins priority.  Write operations are not supported;
callers must write directly to a specific backend catalog.

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
from typing import Dict, List, Optional, Sequence

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.models import Artifact
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)


class FederatedArtifactCatalog(ArtifactCatalogBase):
    """Composite catalog that federates read queries across multiple backends.

    Queries are broadcast to all backend catalogs and results are
    deduplicated by ``(name, version)`` with first-catalog-wins priority.
    Write operations raise ``NotImplementedError``; callers must write
    directly to a specific backend.

    Parameters
    ----------
    catalogs : Sequence[ArtifactCatalogBase]
        Ordered list of backend catalogs.  Must contain at least one.
        The first catalog has highest priority for deduplication.
    """

    def __init__(self, catalogs: Sequence[ArtifactCatalogBase]) -> None:
        if not catalogs:
            raise ValueError(
                "FederatedArtifactCatalog requires at least one backend catalog."
            )
        self._catalogs: List[ArtifactCatalogBase] = list(catalogs)

    # -- Read operations (federated) --------------------------------------

    def get_artifact(self, name: str, version: str) -> Optional[Artifact]:
        """Return the first matching artifact across all backends."""
        for catalog in self._catalogs:
            result = catalog.get_artifact(name, version)
            if result is not None:
                return result
        return None

    def list_artifacts(
        self, artifact_type: Optional[str] = None,
    ) -> List[Artifact]:
        """List artifacts from all backends, deduplicated."""
        return self._federate_query("list_artifacts", artifact_type=artifact_type)

    def search(self, query: str) -> List[Artifact]:
        """Search all backends, deduplicated."""
        return self._federate_query("search", query)

    def search_by_tags(self, tags: Dict[str, str]) -> List[Artifact]:
        """Search by tags across all backends, deduplicated."""
        return self._federate_query("search_by_tags", tags)

    # -- Write operations (not supported) ----------------------------------

    def add_artifact(self, artifact: Artifact) -> int:
        """Not supported on federated catalogs.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "FederatedArtifactCatalog is read-only. "
            "Write directly to a specific backend catalog."
        )

    def remove_artifact(self, name: str, version: str) -> bool:
        """Not supported on federated catalogs.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "FederatedArtifactCatalog is read-only. "
            "Write directly to a specific backend catalog."
        )

    def update_remote_version(
        self, artifact_id: int, source: str, latest_version: str,
    ) -> None:
        """Not supported on federated catalogs.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "FederatedArtifactCatalog is read-only. "
            "Write directly to a specific backend catalog."
        )

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close all backend catalogs."""
        for catalog in self._catalogs:
            catalog.close()

    # -- Introspection -----------------------------------------------------

    @property
    def catalogs(self) -> List[ArtifactCatalogBase]:
        """Ordered list of backend catalogs (read-only copy)."""
        return list(self._catalogs)

    # -- Private helpers ---------------------------------------------------

    def _federate_query(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> List[Artifact]:
        """Broadcast a read query to all backends and deduplicate results.

        Deduplication uses ``(name, version)`` as the identity key;
        the first occurrence (from the highest-priority catalog) wins.
        """
        seen: set = set()
        results: List[Artifact] = []
        for catalog in self._catalogs:
            method = getattr(catalog, method_name)
            for artifact in method(*args, **kwargs):
                key = (artifact.name, artifact.version)
                if key not in seen:
                    seen.add(key)
                    results.append(artifact)
        return results
