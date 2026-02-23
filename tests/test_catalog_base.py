"""
Tests for grdl_rt.catalog.base — ArtifactCatalogBase ABC contract.

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


class TestArtifactCatalogBaseABC:

    def test_cannot_instantiate_directly(self):
        """ArtifactCatalogBase is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ArtifactCatalogBase()

    def test_abstract_methods_defined(self):
        """Verify all expected abstract methods exist."""
        expected = {
            "add_artifact",
            "remove_artifact",
            "get_artifact",
            "list_artifacts",
            "search",
            "search_by_tags",
            "update_remote_version",
            "close",
        }
        assert expected <= ArtifactCatalogBase.__abstractmethods__

    def test_context_manager_not_abstract(self):
        """__enter__ and __exit__ have default implementations."""
        assert "__enter__" not in ArtifactCatalogBase.__abstractmethods__
        assert "__exit__" not in ArtifactCatalogBase.__abstractmethods__

    def test_schema_version_not_on_abc(self):
        """schema_version is SQLite-specific, not on the ABC."""
        assert not hasattr(ArtifactCatalogBase, "schema_version")
