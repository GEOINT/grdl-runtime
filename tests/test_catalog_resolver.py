# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.catalog.resolver — Catalog path resolution.

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

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from grdl_rt.catalog.resolver import resolve_catalog_path, ensure_config_dir


class TestResolveCatalogPath:

    def test_env_var_highest_priority(self):
        with mock.patch.dict(os.environ, {"GRDK_CATALOG_PATH": "/custom/path/db.sqlite"}):
            path = resolve_catalog_path()
            assert path == Path("/custom/path/db.sqlite")

    def test_config_file_second_priority(self, tmp_path):
        config_dir = tmp_path / ".grdl"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"catalog_path": "/from/config/catalog.db"}))

        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove env var if set
            os.environ.pop("GRDK_CATALOG_PATH", None)
            with mock.patch("grdl_rt.catalog.resolver.Path.home", return_value=tmp_path):
                path = resolve_catalog_path()
                assert path == Path("/from/config/catalog.db")

    def test_default_fallback(self, tmp_path):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRDK_CATALOG_PATH", None)
            with mock.patch("grdl_rt.catalog.resolver.Path.home", return_value=tmp_path):
                path = resolve_catalog_path()
                assert path == tmp_path / ".grdl" / "catalog.db"

    def test_malformed_config_falls_through(self, tmp_path):
        config_dir = tmp_path / ".grdl"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text("not valid json!!!")

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRDK_CATALOG_PATH", None)
            with mock.patch("grdl_rt.catalog.resolver.Path.home", return_value=tmp_path):
                path = resolve_catalog_path()
                assert path == tmp_path / ".grdl" / "catalog.db"


class TestEnsureConfigDir:

    def test_creates_directory(self, tmp_path):
        with mock.patch("grdl_rt.catalog.resolver.Path.home", return_value=tmp_path):
            config_dir = ensure_config_dir()
            assert config_dir.is_dir()
            assert config_dir == tmp_path / ".grdl"

    def test_idempotent(self, tmp_path):
        with mock.patch("grdl_rt.catalog.resolver.Path.home", return_value=tmp_path):
            ensure_config_dir()
            ensure_config_dir()  # Should not raise
