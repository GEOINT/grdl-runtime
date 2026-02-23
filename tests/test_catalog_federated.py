# -*- coding: utf-8 -*-
"""
Tests for FederatedArtifactCatalog.

Verifies federated read queries, deduplication by (name, version),
write rejection, context manager lifecycle, and ABC compliance.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-10
"""

import pytest

from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.federated import FederatedArtifactCatalog
from grdl_rt.catalog.models import Artifact
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def primary(tmp_path):
    return YamlArtifactCatalog(file_path=tmp_path / "primary.yaml")


@pytest.fixture
def secondary(tmp_path):
    return YamlArtifactCatalog(file_path=tmp_path / "secondary.yaml")


@pytest.fixture
def federated(primary, secondary):
    return FederatedArtifactCatalog([primary, secondary])


def _make_processor(name, version="1.0.0", description=""):
    return Artifact(
        name=name,
        version=version,
        artifact_type="grdl_processor",
        description=description,
    )


def _make_workflow(name, version="1.0.0", description="", tags=None):
    return Artifact(
        name=name,
        version=version,
        artifact_type="grdk_workflow",
        description=description,
        tags=tags or {},
    )


# ── Init tests ───────────────────────────────────────────────────────


class TestFederatedInit:
    def test_requires_at_least_one_catalog(self):
        with pytest.raises(ValueError, match="at least one"):
            FederatedArtifactCatalog([])

    def test_accepts_single_catalog(self, primary):
        fed = FederatedArtifactCatalog([primary])
        assert len(fed.catalogs) == 1

    def test_accepts_multiple_catalogs(self, primary, secondary):
        fed = FederatedArtifactCatalog([primary, secondary])
        assert len(fed.catalogs) == 2

    def test_catalogs_property_returns_copy(self, federated):
        cats = federated.catalogs
        cats.append(None)
        assert len(federated.catalogs) == 2


class TestFederatedIsABC:
    def test_is_subclass(self):
        assert issubclass(FederatedArtifactCatalog, ArtifactCatalogBase)

    def test_instance_check(self, federated):
        assert isinstance(federated, ArtifactCatalogBase)


# ── Write rejection tests ────────────────────────────────────────────


class TestFederatedWrites:
    def test_add_artifact_raises(self, federated):
        a = _make_processor("test")
        with pytest.raises(NotImplementedError, match="read-only"):
            federated.add_artifact(a)

    def test_remove_artifact_raises(self, federated):
        with pytest.raises(NotImplementedError, match="read-only"):
            federated.remove_artifact("test", "1.0.0")

    def test_update_remote_version_raises(self, federated):
        with pytest.raises(NotImplementedError, match="read-only"):
            federated.update_remote_version(1, "pypi", "2.0.0")


# ── Read tests ───────────────────────────────────────────────────────


class TestFederatedGetArtifact:
    def test_found_in_primary(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("alpha"))
        result = federated.get_artifact("alpha", "1.0.0")
        assert result is not None
        assert result.name == "alpha"

    def test_falls_through_to_secondary(self, primary, secondary, federated):
        secondary.add_artifact(_make_processor("beta"))
        result = federated.get_artifact("beta", "1.0.0")
        assert result is not None
        assert result.name == "beta"

    def test_returns_none_when_absent(self, federated):
        assert federated.get_artifact("missing", "1.0.0") is None

    def test_primary_wins_over_secondary(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("shared", description="primary"))
        secondary.add_artifact(_make_processor("shared", description="secondary"))
        result = federated.get_artifact("shared", "1.0.0")
        assert result.description == "primary"


class TestFederatedListArtifacts:
    def test_empty_catalogs(self, federated):
        assert federated.list_artifacts() == []

    def test_merges_from_both(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("a"))
        secondary.add_artifact(_make_processor("b"))
        results = federated.list_artifacts()
        names = {a.name for a in results}
        assert names == {"a", "b"}

    def test_deduplicates_by_name_version(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("dup", description="primary"))
        secondary.add_artifact(_make_processor("dup", description="secondary"))
        results = federated.list_artifacts()
        assert len(results) == 1
        assert results[0].description == "primary"

    def test_different_versions_not_deduplicated(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("pkg", version="1.0.0"))
        secondary.add_artifact(_make_processor("pkg", version="2.0.0"))
        results = federated.list_artifacts()
        assert len(results) == 2

    def test_type_filter(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("proc"))
        secondary.add_artifact(_make_workflow("wf"))
        procs = federated.list_artifacts(artifact_type="grdl_processor")
        assert len(procs) == 1
        assert procs[0].name == "proc"


class TestFederatedSearch:
    def test_search_across_backends(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("lee-filter", description="Lee speckle"))
        secondary.add_artifact(_make_processor("frost-filter", description="Frost speckle"))
        results = federated.search("speckle")
        assert len(results) == 2

    def test_search_deduplicates(self, primary, secondary, federated):
        primary.add_artifact(_make_processor("lee-filter", description="Lee speckle"))
        secondary.add_artifact(_make_processor("lee-filter", description="Lee speckle dup"))
        results = federated.search("speckle")
        assert len(results) == 1


class TestFederatedSearchByTags:
    def test_search_by_tags_across_backends(self, primary, secondary, federated):
        primary.add_artifact(_make_workflow("wf1", tags={"modality": ["SAR"]}))
        secondary.add_artifact(_make_workflow("wf2", tags={"modality": ["SAR"]}))
        results = federated.search_by_tags({"modality": "SAR"})
        assert len(results) == 2

    def test_search_by_tags_deduplicates(self, primary, secondary, federated):
        primary.add_artifact(_make_workflow("wf1", tags={"modality": ["SAR"]}))
        secondary.add_artifact(_make_workflow("wf1", tags={"modality": ["SAR"]}))
        results = federated.search_by_tags({"modality": "SAR"})
        assert len(results) == 1


# ── Lifecycle tests ──────────────────────────────────────────────────


class TestFederatedLifecycle:
    def test_close_closes_all(self):
        from unittest.mock import MagicMock

        m1 = MagicMock(spec=ArtifactCatalogBase)
        m2 = MagicMock(spec=ArtifactCatalogBase)
        fed = FederatedArtifactCatalog([m1, m2])
        fed.close()
        m1.close.assert_called_once()
        m2.close.assert_called_once()

    def test_context_manager(self):
        from unittest.mock import MagicMock

        m1 = MagicMock(spec=ArtifactCatalogBase)
        m2 = MagicMock(spec=ArtifactCatalogBase)
        with FederatedArtifactCatalog([m1, m2]) as fed:
            assert isinstance(fed, FederatedArtifactCatalog)
        m1.close.assert_called_once()
        m2.close.assert_called_once()
