# -*- coding: utf-8 -*-
"""
Catalog Database - SQLite-backed artifact catalog for GRDL and GRDK.

Provides the SqliteArtifactCatalog class for storing and querying metadata
about GRDL image processing components and GRDK orchestrated workflows.

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
import logging
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.models import Artifact
from grdl_rt.catalog.resolver import resolve_catalog_path, ensure_config_dir


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK(artifact_type IN ('grdl_processor', 'grdk_workflow')),
    description TEXT DEFAULT '',
    author TEXT DEFAULT '',
    license TEXT DEFAULT 'MIT',

    pypi_package TEXT,
    conda_package TEXT,
    conda_channel TEXT,

    processor_class TEXT,
    processor_version TEXT,
    processor_type TEXT,

    yaml_definition TEXT,
    python_dsl TEXT,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS workflow_tags (
    artifact_id INTEGER REFERENCES artifacts(id) ON DELETE CASCADE,
    tag_key TEXT NOT NULL,
    tag_value TEXT NOT NULL,
    PRIMARY KEY (artifact_id, tag_key, tag_value)
);

CREATE TABLE IF NOT EXISTS remote_versions (
    artifact_id INTEGER REFERENCES artifacts(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK(source IN ('pypi', 'conda')),
    latest_version TEXT NOT NULL,
    checked_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (artifact_id, source)
);
"""

_FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    name, description, content='artifacts', content_rowid='id'
);
"""

_FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
    INSERT INTO artifacts_fts(rowid, name, description)
    VALUES (new.id, new.name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
    INSERT INTO artifacts_fts(artifacts_fts, rowid, name, description)
    VALUES ('delete', old.id, old.name, old.description);
END;

CREATE TRIGGER IF NOT EXISTS artifacts_au AFTER UPDATE ON artifacts BEGIN
    INSERT INTO artifacts_fts(artifacts_fts, rowid, name, description)
    VALUES ('delete', old.id, old.name, old.description);
    INSERT INTO artifacts_fts(rowid, name, description)
    VALUES (new.id, new.name, new.description);
END;
"""


_SCHEMA_VERSION_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""

_CURRENT_SCHEMA_VERSION = 1

# Migration functions: (target_version, callable)
# Add new migrations here as schema evolves.
_MIGRATIONS: List[tuple] = [
    # (1, lambda conn: None),  # Version 1 is the initial schema
]


class SqliteArtifactCatalog(ArtifactCatalogBase):
    """SQLite-backed catalog for GRDL and GRDK artifacts.

    Parameters
    ----------
    db_path : Optional[Path]
        Path to the SQLite database file. If None, resolved via
        the catalog path priority chain (env var > config > default).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            ensure_config_dir()
            db_path = resolve_catalog_path()

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._run_migrations()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.executescript(_FTS_SCHEMA_SQL)
        self._conn.executescript(_FTS_TRIGGERS_SQL)
        self._conn.executescript(_SCHEMA_VERSION_SQL)
        self._conn.commit()

        # Set initial schema version if not present
        row = self._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_CURRENT_SCHEMA_VERSION,),
            )
            self._conn.commit()

    def _run_migrations(self) -> None:
        """Run any pending schema migrations."""
        row = self._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        current = row['version'] if row else 0

        for target_version, migrate_fn in _MIGRATIONS:
            if target_version > current:
                logger.info(
                    "Running migration to schema version %d", target_version
                )
                migrate_fn(self._conn)
                self._conn.execute(
                    "UPDATE schema_version SET version = ?",
                    (target_version,),
                )
                self._conn.commit()
                current = target_version

    @property
    def schema_version(self) -> int:
        """Current schema version."""
        row = self._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        return row['version'] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def add_artifact(self, artifact: Artifact) -> int:
        """Add or replace an artifact in the catalog.

        Parameters
        ----------
        artifact : Artifact

        Returns
        -------
        int
            Row ID of the inserted artifact.
        """
        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO artifacts
            (name, version, artifact_type, description, author, license,
             pypi_package, conda_package, conda_channel,
             processor_class, processor_version, processor_type,
             yaml_definition, python_dsl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact.name, artifact.version, artifact.artifact_type,
                artifact.description, artifact.author, artifact.license,
                artifact.pypi_package, artifact.conda_package,
                artifact.conda_channel,
                artifact.processor_class, artifact.processor_version,
                artifact.processor_type,
                artifact.yaml_definition, artifact.python_dsl,
            ),
        )
        artifact_id = cursor.lastrowid

        # Insert tags
        if artifact.tags:
            self._conn.execute(
                "DELETE FROM workflow_tags WHERE artifact_id = ?",
                (artifact_id,),
            )
            for key, values in artifact.tags.items():
                for value in values:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO workflow_tags "
                        "(artifact_id, tag_key, tag_value) VALUES (?, ?, ?)",
                        (artifact_id, key, value),
                    )

        self._conn.commit()
        artifact.id = artifact_id
        return artifact_id

    def remove_artifact(self, name: str, version: str) -> bool:
        """Remove an artifact from the catalog.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        bool
            True if an artifact was removed.
        """
        cursor = self._conn.execute(
            "DELETE FROM artifacts WHERE name = ? AND version = ?",
            (name, version),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_artifact(self, name: str, version: str) -> Optional[Artifact]:
        """Get a specific artifact by name and version.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        Optional[Artifact]
            The artifact, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_artifact(row)

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
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
        if artifact_type:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_type = ? "
                "ORDER BY name, version",
                (artifact_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM artifacts ORDER BY name, version"
            ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def search(self, query: str) -> List[Artifact]:
        """Full-text search over artifact names and descriptions.

        Parameters
        ----------
        query : str
            Search query.

        Returns
        -------
        List[Artifact]
        """
        rows = self._conn.execute(
            """SELECT a.* FROM artifacts a
            INNER JOIN artifacts_fts f ON a.id = f.rowid
            WHERE artifacts_fts MATCH ?
            ORDER BY rank""",
            (query,),
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

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

        conditions = []
        params: list = []
        for key, value in tags.items():
            conditions.append(
                "a.id IN (SELECT artifact_id FROM workflow_tags "
                "WHERE tag_key = ? AND tag_value = ?)"
            )
            params.extend([key, value])

        sql = f"SELECT a.* FROM artifacts a WHERE {' AND '.join(conditions)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def update_remote_version(
        self,
        artifact_id: int,
        source: str,
        latest_version: str,
    ) -> None:
        """Record the latest remote version for an artifact.

        Parameters
        ----------
        artifact_id : int
        source : str
            'pypi' or 'conda'.
        latest_version : str
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO remote_versions
            (artifact_id, source, latest_version, checked_at)
            VALUES (?, ?, ?, datetime('now'))""",
            (artifact_id, source, latest_version),
        )
        self._conn.commit()

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        """Convert a database row to an Artifact instance."""
        artifact_id = row['id']

        # Load tags
        tag_rows = self._conn.execute(
            "SELECT tag_key, tag_value FROM workflow_tags "
            "WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchall()

        tags: Dict[str, List[str]] = {}
        for tr in tag_rows:
            key = tr['tag_key']
            if key not in tags:
                tags[key] = []
            tags[key].append(tr['tag_value'])

        return Artifact(
            id=artifact_id,
            name=row['name'],
            version=row['version'],
            artifact_type=row['artifact_type'],
            description=row['description'] or '',
            author=row['author'] or '',
            license=row['license'] or 'MIT',
            pypi_package=row['pypi_package'],
            conda_package=row['conda_package'],
            conda_channel=row['conda_channel'],
            processor_class=row['processor_class'],
            processor_version=row['processor_version'],
            processor_type=row['processor_type'],
            yaml_definition=row['yaml_definition'],
            python_dsl=row['python_dsl'],
            tags=tags,
        )


# Backward-compatibility alias
ArtifactCatalog = SqliteArtifactCatalog
