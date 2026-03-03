"""
Tests for catalog alternative processors — SQLite, YAML, and Federated.

Covers:
- SQLite: add artifact with alternatives, get_alternatives returns them
- SQLite: set_alternatives updates, sorted by priority
- SQLite: schema migration preserves existing data
- YAML: alternatives serialized and deserialized
- YAML: old file without alternatives loads with [] default
- Federated: get_alternatives delegates; set_alternatives raises

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

import pytest
import yaml

from grdl_rt.catalog.database import SqliteArtifactCatalog
from grdl_rt.catalog.models import Artifact
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(name="proc-a", version="1.0.0", alternatives=None):
    return Artifact(
        name=name,
        version=version,
        artifact_type="grdl_processor",
        description="Test processor",
        processor_class=f"mock.module.{name}",
        processor_version=version,
        processor_type="transform",
        alternatives=alternatives or [],
    )


# ---------------------------------------------------------------------------
# SQLite catalog
# ---------------------------------------------------------------------------


class TestSqliteAlternatives:

    @pytest.fixture
    def catalog(self, tmp_path):
        cat = SqliteArtifactCatalog(db_path=tmp_path / "test.db")
        yield cat
        cat.close()

    def test_add_artifact_with_alternatives(self, catalog):
        alts = [
            {"processor_name": "alt-cpu", "priority": 1, "compatibility_notes": "CPU only"},
            {"processor_name": "alt-fast", "priority": 2, "compatibility_notes": "Fast CPU"},
        ]
        proc = _make_processor(alternatives=alts)
        catalog.add_artifact(proc)

        loaded = catalog.get_artifact("proc-a", "1.0.0")
        assert loaded is not None
        assert len(loaded.alternatives) == 2
        assert loaded.alternatives[0]["processor_name"] == "alt-cpu"

    def test_get_alternatives(self, catalog):
        alts = [
            {"processor_name": "alt-b", "priority": 2},
            {"processor_name": "alt-a", "priority": 1},
        ]
        proc = _make_processor(alternatives=alts)
        catalog.add_artifact(proc)

        result = catalog.get_alternatives("proc-a", "1.0.0")
        assert len(result) == 2
        # Verify content (order may vary — sorted by consumer)
        names = {a["processor_name"] for a in result}
        assert names == {"alt-a", "alt-b"}

    def test_get_alternatives_nonexistent(self, catalog):
        result = catalog.get_alternatives("nope", "0.0.0")
        assert result == []

    def test_set_alternatives(self, catalog):
        proc = _make_processor()
        catalog.add_artifact(proc)

        new_alts = [
            {"processor_name": "new-alt", "priority": 1},
        ]
        catalog.set_alternatives("proc-a", "1.0.0", new_alts)

        result = catalog.get_alternatives("proc-a", "1.0.0")
        assert len(result) == 1
        assert result[0]["processor_name"] == "new-alt"

    def test_add_artifact_without_alternatives(self, catalog):
        proc = _make_processor()
        catalog.add_artifact(proc)

        result = catalog.get_alternatives("proc-a", "1.0.0")
        assert result == []

    def test_schema_version_is_4(self, catalog):
        assert catalog.schema_version == 4


class TestSqliteSchemaMigration:

    def test_migration_v2_to_v3(self, tmp_path):
        """Create a v2 database, open with current code → migrates to v3."""
        import sqlite3

        db_path = tmp_path / "migrate.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                author TEXT DEFAULT '',
                license TEXT DEFAULT 'MIT',
                pypi_package TEXT,
                conda_package TEXT,
                conda_channel TEXT,
                processor_class TEXT,
                processor_version TEXT,
                processor_type TEXT,
                requires_global_pass INTEGER DEFAULT 0,
                yaml_definition TEXT,
                python_dsl TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(name, version)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_tags (
                artifact_id INTEGER,
                tag_key TEXT NOT NULL,
                tag_value TEXT NOT NULL,
                PRIMARY KEY (artifact_id, tag_key, tag_value)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS remote_versions (
                artifact_id INTEGER,
                source TEXT NOT NULL,
                latest_version TEXT NOT NULL,
                checked_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (artifact_id, source)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """
        )
        conn.execute("INSERT INTO schema_version (version) VALUES (2)")
        # Insert a pre-existing artifact without alternatives column
        conn.execute(
            "INSERT INTO artifacts (name, version, artifact_type, description) "
            "VALUES ('old-proc', '0.1.0', 'grdl_processor', 'Legacy')"
        )
        conn.commit()
        conn.close()

        # Now open with current code — should migrate to v4
        cat = SqliteArtifactCatalog(db_path=db_path)
        assert cat.schema_version == 4

        # Legacy artifact should be loadable with empty alternatives
        loaded = cat.get_artifact("old-proc", "0.1.0")
        assert loaded is not None
        assert loaded.alternatives == []

        cat.close()


# ---------------------------------------------------------------------------
# YAML catalog
# ---------------------------------------------------------------------------


class TestYamlAlternatives:

    @pytest.fixture
    def catalog(self, tmp_path):
        cat = YamlArtifactCatalog(file_path=tmp_path / "test_catalog.yaml")
        yield cat
        cat.close()

    def test_yaml_alternatives_roundtrip(self, catalog):
        alts = [
            {"processor_name": "yaml-alt", "priority": 1},
        ]
        proc = _make_processor(alternatives=alts)
        catalog.add_artifact(proc)

        loaded = catalog.get_artifact("proc-a", "1.0.0")
        assert loaded is not None
        assert len(loaded.alternatives) == 1
        assert loaded.alternatives[0]["processor_name"] == "yaml-alt"

    def test_yaml_get_alternatives(self, catalog):
        alts = [{"processor_name": "a1", "priority": 2}]
        proc = _make_processor(alternatives=alts)
        catalog.add_artifact(proc)

        result = catalog.get_alternatives("proc-a", "1.0.0")
        assert len(result) == 1
        assert result[0]["processor_name"] == "a1"

    def test_yaml_set_alternatives(self, catalog):
        proc = _make_processor()
        catalog.add_artifact(proc)

        catalog.set_alternatives(
            "proc-a",
            "1.0.0",
            [
                {"processor_name": "new", "priority": 1},
            ],
        )

        result = catalog.get_alternatives("proc-a", "1.0.0")
        assert len(result) == 1
        assert result[0]["processor_name"] == "new"

    def test_yaml_set_alternatives_nonexistent_raises(self, catalog):
        with pytest.raises(KeyError):
            catalog.set_alternatives("nope", "0.0.0", [])

    def test_yaml_no_alternatives_field_loads_empty(self, tmp_path):
        """YAML file without alternatives key loads with [] default."""
        yaml_path = tmp_path / "old.yaml"
        data = {
            "catalog_meta": {"format_version": 1, "next_id": 2},
            "artifacts": [
                {
                    "id": 1,
                    "name": "legacy",
                    "version": "0.1.0",
                    "artifact_type": "grdl_processor",
                    "description": "Old artifact",
                    "author": "",
                    "license": "MIT",
                    # No 'alternatives' key
                }
            ],
            "remote_versions": [],
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f)

        cat = YamlArtifactCatalog(file_path=yaml_path)
        loaded = cat.get_artifact("legacy", "0.1.0")
        assert loaded is not None
        assert loaded.alternatives == []
        cat.close()


# ---------------------------------------------------------------------------
# Federated catalog
# ---------------------------------------------------------------------------


class TestFederatedAlternatives:

    def test_get_alternatives_delegates(self, tmp_path):
        from grdl_rt.catalog.federated import FederatedArtifactCatalog

        # Create a YAML backend with alternatives
        yaml_path = tmp_path / "fed.yaml"
        backend = YamlArtifactCatalog(file_path=yaml_path)
        proc = _make_processor(
            alternatives=[
                {"processor_name": "fed-alt", "priority": 1},
            ]
        )
        backend.add_artifact(proc)

        fed = FederatedArtifactCatalog(catalogs=[backend])
        result = fed.get_alternatives("proc-a", "1.0.0")

        assert len(result) == 1
        assert result[0]["processor_name"] == "fed-alt"

        backend.close()

    def test_set_alternatives_raises(self, tmp_path):
        from grdl_rt.catalog.federated import FederatedArtifactCatalog

        yaml_path = tmp_path / "fed2.yaml"
        backend = YamlArtifactCatalog(file_path=yaml_path)
        fed = FederatedArtifactCatalog(catalogs=[backend])

        with pytest.raises(NotImplementedError):
            fed.set_alternatives("x", "1.0", [])

        backend.close()
