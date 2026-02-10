# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.config — GrdkConfig and load_config.

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
from pathlib import Path

import pytest

from grdl_rt.execution.config import GrdkConfig, load_config


class TestGrdkConfig:
    def test_defaults(self):
        cfg = GrdkConfig()
        assert cfg.thumb_size == 128
        assert cfg.preview_thumb == 160
        assert cfg.debounce_ms == 50
        assert cfg.update_timeout == 10.0
        assert cfg.max_workers == 4

    def test_custom_values(self):
        cfg = GrdkConfig(thumb_size=256, max_workers=8)
        assert cfg.thumb_size == 256
        assert cfg.max_workers == 8

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "config.json"
        cfg = GrdkConfig(thumb_size=64, debounce_ms=100)
        cfg.save(path)

        loaded = load_config(path)
        assert loaded.thumb_size == 64
        assert loaded.debounce_ms == 100
        # Other fields should be default
        assert loaded.preview_thumb == 160

    def test_load_missing_file_returns_defaults(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cfg = load_config(path)
        assert cfg.thumb_size == 128

    def test_load_corrupted_file_returns_defaults(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        cfg = load_config(path)
        assert cfg.thumb_size == 128

    def test_load_ignores_unknown_fields(self, tmp_path):
        path = tmp_path / "config.json"
        data = {"thumb_size": 200, "unknown_field": 42}
        with open(path, 'w') as f:
            json.dump(data, f)
        cfg = load_config(path)
        assert cfg.thumb_size == 200
