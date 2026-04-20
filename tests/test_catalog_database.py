"""
Tests for grdl_rt.catalog.database — SqliteArtifactCatalog.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06
"""

from unittest import mock

import pytest

from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.database import ArtifactCatalog, SqliteArtifactCatalog
from grdl_rt.catalog.models import Artifact


@pytest.fixture
def catalog(tmp_path):
    """Create a temporary catalog database."""
    db_path = tmp_path / "test_catalog.db"
    cat = SqliteArtifactCatalog(db_path=db_path)
    yield cat
    cat.close()


class TestArtifactCatalogBasic:

    def test_add_and_get(self, catalog, sample_processor):
        artifact_id = catalog.add_artifact(sample_processor)
        assert artifact_id > 0

        loaded = catalog.get_artifact("lee-filter", "1.0.0")
        assert loaded is not None
        assert loaded.name == "lee-filter"
        assert loaded.processor_class == "grdl.image_processing.filters.LeeFilter"

    def test_get_nonexistent_returns_none(self, catalog):
        assert catalog.get_artifact("nonexistent", "0.0.0") is None

    def test_remove_artifact(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        assert catalog.remove_artifact("lee-filter", "1.0.0") is True
        assert catalog.get_artifact("lee-filter", "1.0.0") is None

    def test_remove_nonexistent_returns_false(self, catalog):
        assert catalog.remove_artifact("nope", "0.0.0") is False


class TestArtifactCatalogListing:

    def test_list_all(self, catalog, sample_processor, sample_workflow):
        catalog.add_artifact(sample_processor)
        catalog.add_artifact(sample_workflow)
        all_artifacts = catalog.list_artifacts()
        assert len(all_artifacts) == 2

    def test_list_by_type(self, catalog, sample_processor, sample_workflow):
        catalog.add_artifact(sample_processor)
        catalog.add_artifact(sample_workflow)
        processors = catalog.list_artifacts(artifact_type="grdl_processor")
        assert len(processors) == 1
        assert processors[0].name == "lee-filter"

        workflows = catalog.list_artifacts(artifact_type="grdk_workflow")
        assert len(workflows) == 1
        assert workflows[0].name == "sar-vehicle-detection"


class TestArtifactCatalogSearch:

    def test_fts_search(self, catalog, sample_processor, sample_workflow):
        catalog.add_artifact(sample_processor)
        catalog.add_artifact(sample_workflow)

        results = catalog.search("speckle")
        assert len(results) == 1
        assert results[0].name == "lee-filter"

        results = catalog.search("vehicle")
        assert len(results) == 1
        assert results[0].name == "sar-vehicle-detection"

    def test_search_no_results(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search("nonexistent_term")
        assert len(results) == 0


class TestArtifactCatalogTags:

    def test_workflow_tags_stored(self, catalog, sample_workflow):
        catalog.add_artifact(sample_workflow)
        loaded = catalog.get_artifact("sar-vehicle-detection", "2.0.0")
        assert loaded is not None
        assert "modality" in loaded.tags
        assert "SAR" in loaded.tags["modality"]
        assert "detection_type" in loaded.tags
        assert "classification" in loaded.tags["detection_type"]

    def test_search_by_tags(self, catalog, sample_workflow):
        catalog.add_artifact(sample_workflow)
        results = catalog.search_by_tags({"modality": "SAR"})
        assert len(results) == 1
        assert results[0].name == "sar-vehicle-detection"

    def test_search_by_tags_no_match(self, catalog, sample_workflow):
        catalog.add_artifact(sample_workflow)
        results = catalog.search_by_tags({"modality": "PAN"})
        assert len(results) == 0


class TestArtifactCatalogRemoteVersions:

    def test_update_remote_version(self, catalog, sample_processor):
        artifact_id = catalog.add_artifact(sample_processor)
        catalog.update_remote_version(artifact_id, "pypi", "1.1.0")
        # Verify it was stored (no crash)
        catalog.update_remote_version(artifact_id, "pypi", "1.2.0")


class TestArtifactCatalogContextManager:

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx_test.db"
        with SqliteArtifactCatalog(db_path=db_path) as cat:
            cat.add_artifact(
                Artifact(
                    name="test",
                    version="1.0.0",
                    artifact_type="grdl_processor",
                )
            )
        # Connection should be closed, but we can open a new one
        with SqliteArtifactCatalog(db_path=db_path) as cat:
            assert cat.get_artifact("test", "1.0.0") is not None


class TestArtifactModels:

    def test_artifact_repr(self):
        a = Artifact(name="x", version="1.0", artifact_type="grdl_processor")
        assert "x" in repr(a)
        assert "1.0" in repr(a)

    def test_artifact_invalid_type_raises(self):
        with pytest.raises(ValueError, match="artifact_type must be"):
            Artifact(name="x", version="1.0", artifact_type="bad_type")

    def test_update_result_repr_update_available(self):
        from grdl_rt.catalog.models import UpdateResult

        a = Artifact(name="x", version="1.0", artifact_type="grdl_processor")
        r = UpdateResult(
            a, source="pypi", current_version="1.0", latest_version="2.0", update_available=True
        )
        assert "2.0" in repr(r)

    def test_update_result_repr_up_to_date(self):
        from grdl_rt.catalog.models import UpdateResult

        a = Artifact(name="x", version="1.0", artifact_type="grdl_processor")
        r = UpdateResult(a, source="pypi", current_version="1.0", update_available=False)
        assert "up to date" in repr(r)


class TestArtifactCatalogSchemaVersion:

    def test_schema_version_property(self, catalog):
        assert catalog.schema_version == 5

    def test_search_by_tags_empty_returns_all(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search_by_tags({})
        assert len(results) == 1


class TestArtifactCatalogDefaultPath:

    def test_default_path_resolution(self, tmp_path):
        with (
            mock.patch(
                "grdl_rt.catalog.database.resolve_catalog_path",
                return_value=tmp_path / "resolved.db",
            ),
            mock.patch(
                "grdl_rt.catalog.database.ensure_config_dir",
            ),
        ):
            cat = SqliteArtifactCatalog(db_path=None)
            try:
                assert cat._db_path == tmp_path / "resolved.db"
            finally:
                cat.close()


class TestArtifactCatalogMigrations:

    def test_migration_runs(self, tmp_path):
        """A pre-existing DB at version 1 triggers the v2 migration."""
        import sqlite3

        db_path = tmp_path / "migrate_test.db"

        # Seed a v1 database without the requires_global_pass column
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
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
                source TEXT NOT NULL,
                latest_version TEXT NOT NULL,
                checked_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (artifact_id, source)
            );
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
            INSERT INTO schema_version (version) VALUES (1);
        """)
        conn.commit()
        conn.close()

        # Opening the catalog should run all pending migrations (v1→v5)
        cat = SqliteArtifactCatalog(db_path=db_path)
        try:
            assert cat.schema_version == 5
        finally:
            cat.close()


class TestBackwardCompatAlias:

    def test_alias_is_sqlite(self):
        assert ArtifactCatalog is SqliteArtifactCatalog


class TestSqliteSubclassesABC:

    def test_is_subclass(self):
        assert issubclass(SqliteArtifactCatalog, ArtifactCatalogBase)

    def test_instance_check(self, catalog):
        assert isinstance(catalog, ArtifactCatalogBase)
