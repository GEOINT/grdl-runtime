"""Tests for entry-point catalog auto-discovery across grdl, grdl-sar, and grdl-eo."""

from grdl_rt.execution.discovery import (
    build_default_catalog,
    discover_entry_point_catalogs,
    discover_processors,
    filter_processors,
    init_discovery,
    resolve_processor_class,
)


def test_entry_points_load_core_and_modality_catalogs():
    catalogs = discover_entry_point_catalogs()
    assert len(catalogs) >= 1
    names = {a.name for cat in catalogs for a in cat.list_artifacts("grdl_processor")}
    assert "MeanFilter" in names


def test_default_federated_discovery_resolves_across_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("GRDK_CATALOG_PATH", str(tmp_path / "empty.db"))
    init_discovery(build_default_catalog())
    procs = discover_processors()
    assert "MeanFilter" in procs
    assert resolve_processor_class("MeanFilter").__module__.startswith("grdl.")

    if "PauliDecomposition" in procs:
        assert resolve_processor_class("PauliDecomposition").__module__.startswith("grdl_sar.")
        assert "PauliDecomposition" in filter_processors(modality="SAR")

    if "SpectralIndexTransform" in procs:
        assert resolve_processor_class("SpectralIndexTransform").__module__.startswith("grdl_eo.")
        assert "SpectralIndexTransform" in filter_processors(modality="MSI")

