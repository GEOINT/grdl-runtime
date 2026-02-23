# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.catalog.yaml_catalog — YamlArtifactCatalog.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-10
"""

import pytest

from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog
from grdl_rt.catalog.models import Artifact


@pytest.fixture
def catalog(tmp_path):
    """Create a temporary YAML catalog."""
    file_path = tmp_path / "test_catalog.yaml"
    cat = YamlArtifactCatalog(file_path=file_path)
    yield cat
    cat.close()


class TestYamlCatalogBasic:

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

    def test_add_replace_preserves_id(self, catalog, sample_processor):
        first_id = catalog.add_artifact(sample_processor)
        # Create a new artifact with same name+version but different description
        updated = Artifact(
            name="lee-filter",
            version="1.0.0",
            artifact_type="grdl_processor",
            description="Updated description",
        )
        second_id = catalog.add_artifact(updated)
        assert second_id == first_id

        loaded = catalog.get_artifact("lee-filter", "1.0.0")
        assert loaded.description == "Updated description"

    def test_ids_never_reused(self, catalog):
        a1 = Artifact(name="a", version="1.0", artifact_type="grdl_processor")
        a2 = Artifact(name="b", version="1.0", artifact_type="grdl_processor")
        id1 = catalog.add_artifact(a1)
        id2 = catalog.add_artifact(a2)
        catalog.remove_artifact("a", "1.0")

        a3 = Artifact(name="c", version="1.0", artifact_type="grdl_processor")
        id3 = catalog.add_artifact(a3)
        assert id3 > id2  # ID should never be reused


class TestYamlCatalogListing:

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


class TestYamlCatalogSearch:

    def test_substring_search(self, catalog, sample_processor, sample_workflow):
        catalog.add_artifact(sample_processor)
        catalog.add_artifact(sample_workflow)

        results = catalog.search("speckle")
        assert len(results) == 1
        assert results[0].name == "lee-filter"

        results = catalog.search("vehicle")
        assert len(results) == 1
        assert results[0].name == "sar-vehicle-detection"

    def test_search_case_insensitive(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search("SPECKLE")
        assert len(results) == 1
        assert results[0].name == "lee-filter"

    def test_search_by_name(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search("lee")
        assert len(results) == 1
        assert results[0].name == "lee-filter"

    def test_search_no_results(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search("nonexistent_term")
        assert len(results) == 0


class TestYamlCatalogTags:

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

    def test_search_by_tags_empty_returns_all(self, catalog, sample_processor):
        catalog.add_artifact(sample_processor)
        results = catalog.search_by_tags({})
        assert len(results) == 1


class TestYamlCatalogRemoteVersions:

    def test_update_remote_version(self, catalog, sample_processor):
        artifact_id = catalog.add_artifact(sample_processor)
        catalog.update_remote_version(artifact_id, "pypi", "1.1.0")
        # Verify it was stored (no crash)
        catalog.update_remote_version(artifact_id, "pypi", "1.2.0")

    def test_remote_versions_cleaned_on_remove(self, catalog, sample_processor):
        artifact_id = catalog.add_artifact(sample_processor)
        catalog.update_remote_version(artifact_id, "pypi", "1.1.0")
        catalog.remove_artifact("lee-filter", "1.0.0")
        # Remote versions should have been cleaned up
        assert len(catalog._remote_versions) == 0


class TestYamlCatalogContextManager:

    def test_context_manager(self, tmp_path):
        file_path = tmp_path / "ctx_test.yaml"
        with YamlArtifactCatalog(file_path=file_path) as cat:
            cat.add_artifact(
                Artifact(
                    name="test",
                    version="1.0.0",
                    artifact_type="grdl_processor",
                )
            )
        # File should persist — reopen and verify
        with YamlArtifactCatalog(file_path=file_path) as cat:
            assert cat.get_artifact("test", "1.0.0") is not None


class TestYamlCatalogPersistence:

    def test_data_persists_after_close(self, tmp_path):
        file_path = tmp_path / "persist_test.yaml"
        cat = YamlArtifactCatalog(file_path=file_path)
        cat.add_artifact(
            Artifact(
                name="persist",
                version="1.0.0",
                artifact_type="grdl_processor",
            )
        )
        cat.close()

        cat2 = YamlArtifactCatalog(file_path=file_path)
        loaded = cat2.get_artifact("persist", "1.0.0")
        assert loaded is not None
        assert loaded.name == "persist"
        cat2.close()

    def test_empty_file_initializes(self, tmp_path):
        file_path = tmp_path / "new_catalog.yaml"
        assert not file_path.exists()

        cat = YamlArtifactCatalog(file_path=file_path)
        assert cat.list_artifacts() == []
        cat.close()

    def test_auto_save_on_add(self, tmp_path):
        file_path = tmp_path / "autosave_test.yaml"
        cat = YamlArtifactCatalog(file_path=file_path)
        cat.add_artifact(
            Artifact(
                name="saved",
                version="1.0.0",
                artifact_type="grdl_processor",
            )
        )

        # Read back without closing — file should already be written
        cat2 = YamlArtifactCatalog(file_path=file_path)
        assert cat2.get_artifact("saved", "1.0.0") is not None
        cat.close()
        cat2.close()


class TestYamlSubclassesABC:

    def test_is_subclass(self):
        assert issubclass(YamlArtifactCatalog, ArtifactCatalogBase)

    def test_instance_check(self, catalog):
        assert isinstance(catalog, ArtifactCatalogBase)
