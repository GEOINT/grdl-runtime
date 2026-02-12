# -*- coding: utf-8 -*-
"""
YAML Catalog - Lightweight YAML-file-backed artifact catalog.

Provides a human-readable, file-based alternative to the SQLite
catalog backend. Suited for small catalogs (dozens to low hundreds
of artifacts) where editability and portability are preferred over
query performance.

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
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import yaml

# grdl-runtime internal
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.models import Artifact

_CONFIG_DIR = ".grdl"
_DEFAULT_YAML = "catalog.yaml"


class YamlArtifactCatalog(ArtifactCatalogBase):
    """YAML-file-backed catalog for GRDL and GRDK artifacts.

    Stores all artifact metadata in a single YAML file. The entire
    file is loaded into memory on open and rewritten on every
    mutation, making this backend best suited for small catalogs.

    Parameters
    ----------
    file_path : Optional[Path]
        Path to the YAML catalog file. If None, defaults to
        ``~/.grdl/catalog.yaml``.
    """

    def __init__(self, file_path: Optional[Path] = None) -> None:
        if file_path is None:
            config_dir = Path.home() / _CONFIG_DIR
            config_dir.mkdir(parents=True, exist_ok=True)
            file_path = config_dir / _DEFAULT_YAML

        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        self._meta: Dict[str, Any] = {"format_version": 1, "next_id": 1}
        self._artifacts: List[Dict[str, Any]] = []
        self._remote_versions: List[Dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        """Load catalog state from the YAML file, or initialize empty."""
        if not self._file_path.exists():
            return

        with open(self._file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        self._meta = data.get("catalog_meta", self._meta)
        self._artifacts = data.get("artifacts", [])
        self._remote_versions = data.get("remote_versions", [])

    def _save(self) -> None:
        """Write the full catalog state to the YAML file."""
        data = {
            "catalog_meta": self._meta,
            "artifacts": self._artifacts,
            "remote_versions": self._remote_versions,
        }
        with open(self._file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

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
        # Check for existing entry with same (name, version)
        for i, entry in enumerate(self._artifacts):
            if entry["name"] == artifact.name and entry["version"] == artifact.version:
                artifact_id = entry["id"]
                self._artifacts[i] = self._artifact_to_dict(artifact, artifact_id)
                artifact.id = artifact_id
                self._save()
                return artifact_id

        # New entry — assign next ID
        artifact_id = self._meta["next_id"]
        self._meta["next_id"] = artifact_id + 1
        self._artifacts.append(self._artifact_to_dict(artifact, artifact_id))
        artifact.id = artifact_id
        self._save()
        return artifact_id

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
        original_len = len(self._artifacts)
        removed_ids = {
            entry["id"]
            for entry in self._artifacts
            if entry["name"] == name and entry["version"] == version
        }
        self._artifacts = [
            entry for entry in self._artifacts
            if entry["id"] not in removed_ids
        ]

        if removed_ids:
            self._remote_versions = [
                rv for rv in self._remote_versions
                if rv["artifact_id"] not in removed_ids
            ]

        removed = len(self._artifacts) < original_len
        if removed:
            self._save()
        return removed

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
        for entry in self._artifacts:
            if entry["name"] == name and entry["version"] == version:
                return self._dict_to_artifact(entry)
        return None

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
        entries = self._artifacts
        if artifact_type:
            entries = [
                e for e in entries if e["artifact_type"] == artifact_type
            ]

        entries = sorted(entries, key=lambda e: (e["name"], e["version"]))
        return [self._dict_to_artifact(e) for e in entries]

    def search(self, query: str) -> List[Artifact]:
        """Search artifacts by case-insensitive substring matching.

        Matches against artifact name and description fields.

        Parameters
        ----------
        query : str

        Returns
        -------
        List[Artifact]
        """
        query_lower = query.lower()
        results = []
        for entry in self._artifacts:
            name = (entry.get("name") or "").lower()
            desc = (entry.get("description") or "").lower()
            if query_lower in name or query_lower in desc:
                results.append(self._dict_to_artifact(entry))
        return results

    def search_by_tags(self, tags: Dict[str, str]) -> List[Artifact]:
        """Search artifacts by tag key-value pairs (AND logic).

        Parameters
        ----------
        tags : Dict[str, str]
            Tag filters. All must match.

        Returns
        -------
        List[Artifact]
        """
        if not tags:
            return self.list_artifacts()

        results = []
        for entry in self._artifacts:
            entry_tags = entry.get("tags") or {}
            match = True
            for key, value in tags.items():
                tag_values = entry_tags.get(key, [])
                if value not in tag_values:
                    match = False
                    break
            if match:
                results.append(self._dict_to_artifact(entry))
        return results

    def update_remote_version(
        self, artifact_id: int, source: str, latest_version: str,
    ) -> None:
        """Record the latest remote version for an artifact.

        Parameters
        ----------
        artifact_id : int
        source : str
            'pypi' or 'conda'.
        latest_version : str
        """
        for rv in self._remote_versions:
            if rv["artifact_id"] == artifact_id and rv["source"] == source:
                rv["latest_version"] = latest_version
                self._save()
                return

        self._remote_versions.append({
            "artifact_id": artifact_id,
            "source": source,
            "latest_version": latest_version,
        })
        self._save()

    def get_alternatives(
        self, name: str, version: str,
    ) -> List[Dict[str, Any]]:
        """Return alternative processor entries for the given artifact.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        List[Dict[str, Any]]
        """
        for entry in self._artifacts:
            if entry["name"] == name and entry["version"] == version:
                return list(entry.get("alternatives") or [])
        return []

    def set_alternatives(
        self, name: str, version: str,
        alternatives: List[Dict[str, Any]],
    ) -> None:
        """Set alternative processor entries for the given artifact.

        Parameters
        ----------
        name : str
        version : str
        alternatives : List[Dict[str, Any]]
        """
        for entry in self._artifacts:
            if entry["name"] == name and entry["version"] == version:
                entry["alternatives"] = list(alternatives)
                self._save()
                return
        raise KeyError(f"No artifact with name={name!r}, version={version!r}")

    def close(self) -> None:
        """Flush any pending state and release resources."""
        self._save()

    # --- Internal helpers ---

    @staticmethod
    def _artifact_to_dict(artifact: Artifact, artifact_id: int) -> Dict[str, Any]:
        """Convert an Artifact instance to a plain dict for YAML storage."""
        return {
            "id": artifact_id,
            "name": artifact.name,
            "version": artifact.version,
            "artifact_type": artifact.artifact_type,
            "description": artifact.description,
            "author": artifact.author,
            "license": artifact.license,
            "pypi_package": artifact.pypi_package,
            "conda_package": artifact.conda_package,
            "conda_channel": artifact.conda_channel,
            "processor_class": artifact.processor_class,
            "processor_version": artifact.processor_version,
            "processor_type": artifact.processor_type,
            "yaml_definition": artifact.yaml_definition,
            "python_dsl": artifact.python_dsl,
            "tags": dict(artifact.tags) if artifact.tags else {},
            "requires_global_pass": artifact.requires_global_pass,
            "alternatives": list(artifact.alternatives),
            "param_schema": artifact.param_schema,
        }

    @staticmethod
    def _dict_to_artifact(entry: Dict[str, Any]) -> Artifact:
        """Convert a YAML dict entry to an Artifact instance."""
        return Artifact(
            id=entry.get("id"),
            name=entry["name"],
            version=entry["version"],
            artifact_type=entry["artifact_type"],
            description=entry.get("description") or "",
            author=entry.get("author") or "",
            license=entry.get("license") or "MIT",
            pypi_package=entry.get("pypi_package"),
            conda_package=entry.get("conda_package"),
            conda_channel=entry.get("conda_channel"),
            processor_class=entry.get("processor_class"),
            processor_version=entry.get("processor_version"),
            processor_type=entry.get("processor_type"),
            yaml_definition=entry.get("yaml_definition"),
            python_dsl=entry.get("python_dsl"),
            tags=entry.get("tags") or {},
            requires_global_pass=bool(entry.get("requires_global_pass", False)),
            alternatives=entry.get("alternatives") or [],
            param_schema=entry.get("param_schema"),
        )
