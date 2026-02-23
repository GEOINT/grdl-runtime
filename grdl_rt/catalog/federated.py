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
from collections.abc import Sequence
from typing import Any

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
            raise ValueError("FederatedArtifactCatalog requires at least one backend catalog.")
        self._catalogs: list[ArtifactCatalogBase] = list(catalogs)

    # -- Read operations (federated) --------------------------------------

    def get_artifact(self, name: str, version: str) -> Artifact | None:
        """Return the first matching artifact across all backends."""
        for catalog in self._catalogs:
            result = catalog.get_artifact(name, version)
            if result is not None:
                return result
        return None

    def list_artifacts(
        self,
        artifact_type: str | None = None,
    ) -> list[Artifact]:
        """List artifacts from all backends, deduplicated."""
        return self._federate_query("list_artifacts", artifact_type=artifact_type)

    def search(self, query: str) -> list[Artifact]:
        """Search all backends, deduplicated."""
        return self._federate_query("search", query)

    def search_by_tags(self, tags: dict[str, str]) -> list[Artifact]:
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
        self,
        artifact_id: int,
        source: str,
        latest_version: str,
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

    # -- Alternative processors -------------------------------------------

    def get_alternatives(
        self,
        name: str,
        version: str,
    ) -> list[dict[str, Any]]:
        """Return alternatives from the first backend with the artifact."""
        for catalog in self._catalogs:
            result = catalog.get_artifact(name, version)
            if result is not None:
                return catalog.get_alternatives(name, version)
        return []

    def set_alternatives(
        self,
        name: str,
        version: str,
        alternatives: list[dict[str, Any]],
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

    # -- Parameter schema --------------------------------------------------

    def get_param_schema(
        self,
        name: str,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """Return param schema from the first backend that has it."""
        for catalog in self._catalogs:
            schema = catalog.get_param_schema(name, version)
            if schema is not None:
                return schema
        return None

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close all backend catalogs."""
        for catalog in self._catalogs:
            catalog.close()

    # -- Introspection -----------------------------------------------------

    @property
    def catalogs(self) -> list[ArtifactCatalogBase]:
        """Ordered list of backend catalogs (read-only copy)."""
        return list(self._catalogs)

    # -- Private helpers ---------------------------------------------------

    def _federate_query(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> list[Artifact]:
        """Broadcast a read query to all backends and deduplicate results.

        Deduplication uses ``(name, version)`` as the identity key;
        the first occurrence (from the highest-priority catalog) wins.
        """
        seen: set = set()
        results: list[Artifact] = []
        for catalog in self._catalogs:
            method = getattr(catalog, method_name)
            for artifact in method(*args, **kwargs):
                key = (artifact.name, artifact.version)
                if key not in seen:
                    seen.add(key)
                    results.append(artifact)
        return results
