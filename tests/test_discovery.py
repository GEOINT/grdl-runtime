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
    filter_processors_for_connection,
    filter_processors_for_modality,
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


# ---------------------------------------------------------------------------
# TestFilterProcessors — implicit ANY modality fix
# ---------------------------------------------------------------------------

# Custom test processor classes live in a module-level dict so they can be
# resolved by _import_class via the catalog lookup path.  We use a helper
# that patches the discovery module's catalog, not stdlib immutable types.


def _make_processor_class(name: str, processor_tags: dict) -> type:
    """Create a fresh processor class with __processor_tags__ set."""
    return type(name, (), {"__processor_tags__": processor_tags})


def _fqn_in_test_module(cls: type) -> str:
    """Return a fake FQN that resolves back to this class via tests module."""
    # We'll register the class in __main__ globals for import resolution.
    import tests.test_discovery as _self_module

    setattr(_self_module, cls.__name__, cls)
    return f"tests.test_discovery.{cls.__name__}"


class TestFilterProcessorsImplicitAny:
    """Tests for the implicit-ANY modality rule in filter_processors().

    A processor with an empty modalities list must be included when
    filtering by any specific modality (implicit ANY).  Previously this
    was a bug: the processor was excluded.
    """

    def _make_catalog_with(self, tmp_path, tags: dict):
        cls = _make_processor_class("_TestProcImplicit", tags)
        fqn = _fqn_in_test_module(cls)
        cat = _make_catalog(tmp_path, [_processor_artifact("p", fqn)])
        init_discovery(cat)
        return cls.__name__

    def test_empty_modalities_included_for_sar(self, tmp_path):
        """Processor with no modality declaration appears in SAR palette."""
        short = self._make_catalog_with(tmp_path, {"modalities": []})
        result = filter_processors(modality="SAR")
        assert (
            short in result
        ), "Processor with no modalities (implicit ANY) must appear in SAR palette"

    def test_wrong_modality_excluded(self, tmp_path):
        """Processor that declares EO-only must not appear in SAR palette."""
        try:
            from grdl.vocabulary import ImageModality
        except ImportError:
            pytest.skip("grdl not installed")
        short = self._make_catalog_with(tmp_path, {"modalities": [ImageModality.EO]})
        result = filter_processors(modality="SAR")
        assert short not in result, "Processor that declares EO-only must not appear in SAR palette"

    def test_correct_modality_included(self, tmp_path):
        """Processor that declares SAR appears in SAR palette."""
        try:
            from grdl.vocabulary import ImageModality
        except ImportError:
            pytest.skip("grdl not installed")
        short = self._make_catalog_with(tmp_path, {"modalities": [ImageModality.SAR]})
        result = filter_processors(modality="SAR")
        assert short in result


# ---------------------------------------------------------------------------
# TestFilterProcessorsForModality
# ---------------------------------------------------------------------------


class TestFilterProcessorsForModality:
    def _make_catalog_with(self, tmp_path, tags: dict):
        cls = _make_processor_class("_TestProcModality", tags)
        fqn = _fqn_in_test_module(cls)
        cat = _make_catalog(tmp_path, [_processor_artifact("p", fqn)])
        init_discovery(cat)
        return cls.__name__

    def test_none_modality_returns_all(self, tmp_path):
        short = self._make_catalog_with(tmp_path, {"modalities": []})
        result = filter_processors_for_modality(None)
        assert short in result

    def test_implicit_any_included(self, tmp_path):
        """Processor with empty modalities list appears for any modality."""
        short = self._make_catalog_with(tmp_path, {"modalities": []})
        for modality in ("SAR", "EO", "MSI"):
            result = filter_processors_for_modality(modality)
            assert short in result, f"Implicit-ANY processor must appear for modality {modality}"

    def test_exclude_categories(self, tmp_path):
        """exclude_categories removes processors in the specified category."""
        short = self._make_catalog_with(tmp_path, {"modalities": [], "category": "filters"})

        included = filter_processors_for_modality("SAR")
        assert short in included

        excluded = filter_processors_for_modality("SAR", exclude_categories={"filters"})
        assert short not in excluded


# ---------------------------------------------------------------------------
# TestFilterProcessorsForConnection
# ---------------------------------------------------------------------------


class TestFilterProcessorsForConnection:
    def _artifact_with_input_type(self, name, fqn, input_type, processor_type="transform"):
        return Artifact(
            name=name,
            version="1.0.0",
            artifact_type="grdl_processor",
            processor_class=fqn,
            processor_type=processor_type,
            input_type=input_type,
        )

    def _make_tagged_proc(self, name, tags, input_type=None):
        cls = _make_processor_class(name, tags)
        fqn = _fqn_in_test_module(cls)
        return fqn, cls.__name__

    def test_none_upstream_returns_all(self, tmp_path):
        fqn, short = self._make_tagged_proc("_Proc1", {"modalities": []})
        cat = _make_catalog(
            tmp_path,
            [self._artifact_with_input_type("p1", fqn, "raster")],
        )
        init_discovery(cat)
        result = filter_processors_for_connection(None)
        assert short in result

    def test_raster_upstream_includes_raster_processor(self, tmp_path):
        fqn, short = self._make_tagged_proc("_Proc2", {"modalities": []})
        cat = _make_catalog(
            tmp_path,
            [self._artifact_with_input_type("p2", fqn, "raster")],
        )
        init_discovery(cat)
        result = filter_processors_for_connection("raster")
        assert short in result

    def test_binary_mask_upstream_includes_raster_processor(self, tmp_path):
        """binary_mask is a raster subtype — raster processors accept it."""
        fqn, short = self._make_tagged_proc("_Proc3", {"modalities": []})
        cat = _make_catalog(
            tmp_path,
            [self._artifact_with_input_type("p3", fqn, "raster")],
        )
        init_discovery(cat)
        result = filter_processors_for_connection("binary_mask")
        assert short in result

    def test_raster_upstream_excludes_detection_processor(self, tmp_path):
        """A raster step's output cannot feed a detection_set processor.

        This is how detection post-processors are kept out of the
        raster-stage palette without any special-case logic.
        """
        fqn, short = self._make_tagged_proc("_Proc4", {"modalities": []})
        cat = _make_catalog(
            tmp_path,
            [self._artifact_with_input_type("p4", fqn, "detection_set", "postprocess")],
        )
        init_discovery(cat)
        result = filter_processors_for_connection("raster")
        assert short not in result

    def test_detection_set_upstream_includes_feature_set_processor(self, tmp_path):
        """detection_set is a feature_set subtype."""
        fqn, short = self._make_tagged_proc("_Proc5", {"modalities": []})
        cat = _make_catalog(
            tmp_path,
            [self._artifact_with_input_type("p5", fqn, "feature_set")],
        )
        init_discovery(cat)
        result = filter_processors_for_connection("detection_set")
        assert short in result

    def test_modality_filter_combined_with_type(self, tmp_path):
        """Modality and type filtering are ANDed together."""
        try:
            from grdl.vocabulary import ImageModality
        except ImportError:
            pytest.skip("grdl not installed")
        fqn, short = self._make_tagged_proc("_Proc6", {"modalities": [ImageModality.EO]})
        cat = _make_catalog(
            tmp_path,
            [self._artifact_with_input_type("p6", fqn, "raster")],
        )
        init_discovery(cat)
        result = filter_processors_for_connection("raster", modality="SAR")
        assert short not in result
