# -*- coding: utf-8 -*-
"""
Tests for the processor discovery module.

Verifies processor resolution (short name and FQN), catalog-backed
lookup, discover/filter operations, and init_discovery lifecycle.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-10
"""

import pytest

from grdl_rt.catalog.models import Artifact
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog
from grdl_rt.execution.discovery import (
    _import_class,
    discover_processors,
    filter_processors,
    get_all_categories,
    get_all_modalities,
    get_processor_tags,
    init_discovery,
    resolve_processor_class,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_catalog(tmp_path, artifacts=None):
    """Create a YamlArtifactCatalog seeded with artifacts."""
    cat = YamlArtifactCatalog(file_path=tmp_path / "discovery_test.yaml")
    for a in artifacts or []:
        cat.add_artifact(a)
    return cat


def _processor_artifact(name, processor_class, processor_type="transform"):
    return Artifact(
        name=name,
        version="1.0.0",
        artifact_type="grdl_processor",
        processor_class=processor_class,
        processor_type=processor_type,
    )


# ── _import_class tests ──────────────────────────────────────────────


class TestImportClass:
    def test_import_stdlib_class(self):
        cls = _import_class("collections.OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_import_nested_module(self):
        cls = _import_class("os.path.join")
        import os.path

        assert cls is os.path.join

    def test_invalid_fqn_no_module(self):
        with pytest.raises(ImportError, match="no module path"):
            _import_class("NoModule")

    def test_nonexistent_module(self):
        with pytest.raises(ImportError):
            _import_class("nonexistent.module.Class")

    def test_nonexistent_attribute(self):
        with pytest.raises(AttributeError):
            _import_class("collections.NonExistentClass")


# ── init_discovery tests ─────────────────────────────────────────────


class TestInitDiscovery:
    def test_init_sets_catalog(self, tmp_path):
        cat = _make_catalog(tmp_path)
        init_discovery(cat)
        from grdl_rt.execution import discovery

        assert discovery._catalog is cat


# ── resolve_processor_class tests ────────────────────────────────────


class TestResolveProcessorClass:
    def test_fqn_import(self, tmp_path):
        """FQN import works without catalog lookup."""
        init_discovery(_make_catalog(tmp_path))
        cls = resolve_processor_class("collections.OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_catalog_short_name(self, tmp_path):
        """Short name is resolved via catalog processor_class FQN."""
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("ordered-dict", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        cls = resolve_processor_class("OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_catalog_artifact_name(self, tmp_path):
        """Artifact name (not just class name) can be used for lookup."""
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("ordered-dict", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        cls = resolve_processor_class("ordered-dict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_unknown_short_name_raises(self, tmp_path):
        init_discovery(_make_catalog(tmp_path))
        with pytest.raises(ImportError, match="not found in catalog"):
            resolve_processor_class("CompletelyUnknownProcessor")

    def test_bad_fqn_falls_through_to_catalog(self, tmp_path):
        """A dotted name that fails direct import still checks the catalog."""
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("bad.dotted.name", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        # "bad.dotted.name" fails direct import, but matches artifact name
        cls = resolve_processor_class("bad.dotted.name")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_catalog_entry_broken_import_raises(self, tmp_path):
        """Catalog entry found but the processor_class can't be imported."""
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("broken", "nonexistent.module.BrokenClass"),
            ],
        )
        init_discovery(cat)
        with pytest.raises(ImportError, match="import failed"):
            resolve_processor_class("broken")


# ── discover_processors tests ────────────────────────────────────────


class TestDiscoverProcessors:
    def test_empty_catalog(self, tmp_path):
        init_discovery(_make_catalog(tmp_path))
        result = discover_processors()
        assert result == {}

    def test_discovers_importable(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("odict", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        result = discover_processors()
        assert "OrderedDict" in result
        from collections import OrderedDict

        assert result["OrderedDict"] is OrderedDict

    def test_skips_without_processor_class(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                Artifact(name="no-class", version="1.0.0", artifact_type="grdl_processor"),
            ],
        )
        init_discovery(cat)
        assert discover_processors() == {}

    def test_skips_unimportable(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("bad", "nonexistent.module.Bad"),
            ],
        )
        init_discovery(cat)
        assert discover_processors() == {}

    def test_skips_workflows(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                Artifact(name="wf", version="1.0.0", artifact_type="grdk_workflow"),
            ],
        )
        init_discovery(cat)
        assert discover_processors() == {}


# ── get_processor_tags tests ─────────────────────────────────────────


class TestGetProcessorTags:
    def test_returns_tags_from_attribute(self):
        class Tagged:
            __processor_tags__ = {"modalities": ["SAR"], "category": "filter"}

        assert get_processor_tags(Tagged) == {
            "modalities": ["SAR"],
            "category": "filter",
        }

    def test_returns_empty_for_untagged(self):
        class Untagged:
            pass

        assert get_processor_tags(Untagged) == {}


# ── get_all_modalities / get_all_categories tests ────────────────────


class TestGetAllModalities:
    def test_empty_catalog(self, tmp_path):
        init_discovery(_make_catalog(tmp_path))
        assert get_all_modalities() == set()

    def test_collects_modalities(self, tmp_path):
        # Create a class with tags, register it via a known importable path
        # We'll use a catalog artifact pointing to collections.OrderedDict
        # which won't have tags, so result is empty — this tests the plumbing
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("odict", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        # OrderedDict has no __processor_tags__, so modalities should be empty
        assert get_all_modalities() == set()


class TestGetAllCategories:
    def test_empty_catalog(self, tmp_path):
        init_discovery(_make_catalog(tmp_path))
        assert get_all_categories() == set()


# ── filter_processors tests ──────────────────────────────────────────


class TestFilterProcessors:
    def test_no_filters_returns_all(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("odict", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        result = filter_processors()
        assert "OrderedDict" in result

    def test_filter_by_processor_type(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("a", "collections.OrderedDict", processor_type="transform"),
                _processor_artifact("b", "collections.Counter", processor_type="detector"),
            ],
        )
        init_discovery(cat)

        transforms = filter_processors(processor_type="transform")
        assert "OrderedDict" in transforms
        assert "Counter" not in transforms

        detectors = filter_processors(processor_type="detector")
        assert "Counter" in detectors
        assert "OrderedDict" not in detectors

    def test_filter_by_nonexistent_type_returns_empty(self, tmp_path):
        cat = _make_catalog(
            tmp_path,
            [
                _processor_artifact("a", "collections.OrderedDict"),
            ],
        )
        init_discovery(cat)
        assert filter_processors(processor_type="imaginary") == {}
