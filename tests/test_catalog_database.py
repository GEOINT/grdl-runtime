# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.catalog.database — ArtifactCatalog SQLite database.

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

import pytest
from unittest import mock

from grdl_rt.catalog.database import ArtifactCatalog, _MIGRATIONS
from grdl_rt.catalog.models import Artifact


@pytest.fixture
def catalog(tmp_path):
    """Create a temporary catalog database."""
    db_path = tmp_path / "test_catalog.db"
    cat = ArtifactCatalog(db_path=db_path)
    yield cat
    cat.close()


@pytest.fixture
def sample_processor():
    return Artifact(
        name="lee-filter",
        version="1.0.0",
        artifact_type="grdl_processor",
        description="Lee speckle filter for SAR imagery",
        author="Steven Siebert",
        pypi_package="grdl-lee-filter",
        conda_package="grdl-lee-filter",
        conda_channel="conda-forge",
        processor_class="grdl.image_processing.filters.LeeFilter",
        processor_version="1.0.0",
        processor_type="transform",
    )


@pytest.fixture
def sample_workflow():
    return Artifact(
        name="sar-vehicle-detection",
        version="2.0.0",
        artifact_type="grdk_workflow",
        description="SAR vehicle detection and classification",
        yaml_definition="name: SAR Vehicle Detection\nsteps: []",
        python_dsl="@workflow(name='SAR')\ndef f(): pass",
        tags={
            'modality': ['SAR'],
            'detection_type': ['classification'],
        },
    )


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
        assert 'modality' in loaded.tags
        assert 'SAR' in loaded.tags['modality']
        assert 'detection_type' in loaded.tags
        assert 'classification' in loaded.tags['detection_type']

    def test_search_by_tags(self, catalog, sample_workflow):
        catalog.add_artifact(sample_workflow)
        results = catalog.search_by_tags({'modality': 'SAR'})
        assert len(results) == 1
        assert results[0].name == "sar-vehicle-detection"

    def test_search_by_tags_no_match(self, catalog, sample_workflow):
        catalog.add_artifact(sample_workflow)
        results = catalog.search_by_tags({'modality': 'PAN'})
        assert len(results) == 0


class TestArtifactCatalogRemoteVersions:

    def test_update_remote_version(self, catalog, sample_processor):
        artifact_id = catalog.add_artifact(sample_processor)
        catalog.update_remote_version(artifact_id, 'pypi', '1.1.0')
        # Verify it was stored (no crash)
        catalog.update_remote_version(artifact_id, 'pypi', '1.2.0')


class TestArtifactCatalogContextManager:

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx_test.db"
        with ArtifactCatalog(db_path=db_path) as cat:
            cat.add_artifact(Artifact(
                name="test", version="1.0.0",
                artifact_type="grdl_processor",
            ))
        # Connection should be closed, but we can open a new one
        with ArtifactCatalog(db_path=db_path) as cat:
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
        r = UpdateResult(a, source="pypi", current_version="1.0",
                         latest_version="2.0", update_available=True)
        assert "2.0" in repr(r)

    def test_update_result_repr_up_to_date(self):
        from grdl_rt.catalog.models import UpdateResult
        a = Artifact(name="x", version="1.0", artifact_type="grdl_processor")
        r = UpdateResult(a, source="pypi", current_version="1.0",
                         update_available=False)
        assert "up to date" in repr(r)


class TestArtifactCatalogSchemaVersion:

    def test_schema_version_property(self, catalog):
        assert catalog.schema_version == 1

    def test_search_by_tags_empty_returns_all(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search_by_tags({})
        assert len(results) == 1


class TestArtifactCatalogDefaultPath:

    def test_default_path_resolution(self, tmp_path):
        with mock.patch(
            'grdl_rt.catalog.database.resolve_catalog_path',
            return_value=tmp_path / "resolved.db",
        ), mock.patch(
            'grdl_rt.catalog.database.ensure_config_dir',
        ):
            cat = ArtifactCatalog(db_path=None)
            try:
                assert cat._db_path == tmp_path / "resolved.db"
            finally:
                cat.close()


class TestArtifactCatalogMigrations:

    def test_migration_runs(self, tmp_path):
        db_path = tmp_path / "migrate_test.db"
        migrate_fn = mock.MagicMock()
        with mock.patch(
            'grdl_rt.catalog.database._MIGRATIONS',
            [(2, migrate_fn)],
        ):
            cat = ArtifactCatalog(db_path=db_path)
            try:
                migrate_fn.assert_called_once()
                assert cat.schema_version == 2
            finally:
                cat.close()
