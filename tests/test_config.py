# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.config — RuntimeConfig, env var loading, file loading.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06

Modified
--------
2026-02-11
"""

import os
from pathlib import Path

import pytest

from grdl_rt.execution.config import (
    LogConfig,
    MemoryConfig,
    RetryDefaults,
    RuntimeConfig,
    GpuConfig,
    TapOutConfig,
    _overlay_env_vars,
    load_runtime_config,
    get_runtime_config,
    reset_runtime_config,
)

# ======================================================================
# RuntimeConfig defaults
# ======================================================================


class TestRuntimeConfigDefaults:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.log.level == "INFO"
        assert cfg.log.format == "json"
        assert cfg.retry.max_retries == 0
        assert cfg.retry.backoff_base == 1.0
        assert cfg.retry.backoff_max == 60.0
        assert cfg.memory.warn_threshold == 0.80
        assert cfg.memory.abort_threshold == 0.95
        assert cfg.memory.estimation_multiplier == 1.5
        assert cfg.gpu.prefer_gpu is False
        assert cfg.catalog_path is None
        assert cfg.tap_out.default_format == "npy"


# ======================================================================
# Validation
# ======================================================================


class TestConfigValidation:
    def test_log_level_case_insensitive(self):
        cfg = LogConfig(level="debug")
        assert cfg.level == "DEBUG"

    def test_log_level_invalid_raises(self):
        with pytest.raises(ValueError, match="log level"):
            LogConfig(level="TRACE")

    def test_log_format_normalized(self):
        cfg = LogConfig(format="JSON")
        assert cfg.format == "json"

    def test_log_format_invalid_raises(self):
        with pytest.raises(ValueError, match="log format"):
            LogConfig(format="xml")

    def test_memory_thresholds_valid_range(self):
        cfg = MemoryConfig(warn_threshold=0.0, abort_threshold=1.0)
        assert cfg.warn_threshold == 0.0
        assert cfg.abort_threshold == 1.0

    def test_memory_warn_threshold_invalid(self):
        with pytest.raises(ValueError):
            MemoryConfig(warn_threshold=1.5)

    def test_retry_negative_raises(self):
        with pytest.raises(ValueError):
            RetryDefaults(max_retries=-1)

    def test_retry_zero_backoff_raises(self):
        with pytest.raises(ValueError):
            RetryDefaults(backoff_base=0)


# ======================================================================
# Env var overlay
# ======================================================================


class TestEnvVarOverlay:
    def test_no_env_vars_set(self):
        data = _overlay_env_vars({})
        # When no GRDL_RT_* env vars are set, data should be unchanged
        # (it may have keys from other env vars, but log/memory etc. shouldn't be there)
        # Just verify function doesn't crash
        assert isinstance(data, dict)

    def test_overlay_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("GRDL_RT_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GRDL_RT_MEMORY_WARN_THRESHOLD", "0.70")
        monkeypatch.setenv("GRDL_RT_GPU_PREFER_GPU", "true")
        monkeypatch.setenv("GRDL_RT_RETRY_MAX_RETRIES", "5")
        monkeypatch.setenv("GRDL_RT_CATALOG_PATH", "/data/catalog.db")

        data = _overlay_env_vars({})
        assert data["log"]["level"] == "DEBUG"
        assert data["memory"]["warn_threshold"] == 0.70
        assert data["gpu"]["prefer_gpu"] is True
        assert data["retry"]["max_retries"] == 5
        assert data["catalog_path"] == "/data/catalog.db"

    def test_env_vars_override_file(self, monkeypatch):
        monkeypatch.setenv("GRDL_RT_LOG_LEVEL", "ERROR")
        data = {"log": {"level": "DEBUG"}}
        data = _overlay_env_vars(data)
        assert data["log"]["level"] == "ERROR"

    def test_gpu_prefer_false(self, monkeypatch):
        monkeypatch.setenv("GRDL_RT_GPU_PREFER_GPU", "false")
        data = _overlay_env_vars({})
        assert data["gpu"]["prefer_gpu"] is False


# ======================================================================
# File loading
# ======================================================================


class TestFileLoading:
    def test_load_yaml_config(self, tmp_path):
        import yaml

        config_data = {
            "log": {"level": "DEBUG", "format": "console"},
            "memory": {"warn_threshold": 0.70},
        }
        yaml_path = tmp_path / "runtime.yaml"
        yaml_path.write_text(yaml.dump(config_data), encoding="utf-8")

        cfg = load_runtime_config(config_path=yaml_path)
        assert cfg.log.level == "DEBUG"
        assert cfg.log.format == "console"
        assert cfg.memory.warn_threshold == 0.70
        # Unset fields use defaults
        assert cfg.retry.max_retries == 0

    def test_load_toml_config(self, tmp_path):
        toml_content = """\
[log]
level = "WARNING"
format = "json"

[memory]
abort_threshold = 0.90
"""
        toml_path = tmp_path / "runtime.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        cfg = load_runtime_config(config_path=toml_path)
        assert cfg.log.level == "WARNING"
        assert cfg.memory.abort_threshold == 0.90

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_runtime_config(config_path=tmp_path / "nonexistent.yaml")
        assert cfg.log.level == "INFO"

    def test_load_corrupted_file_returns_defaults(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("not: [valid: yaml: {{{", encoding="utf-8")
        cfg = load_runtime_config(config_path=path)
        assert cfg.log.level == "INFO"


# ======================================================================
# Env vars take precedence over file
# ======================================================================


class TestEnvPrecedence:
    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        import yaml

        config_data = {"log": {"level": "DEBUG"}}
        yaml_path = tmp_path / "runtime.yaml"
        yaml_path.write_text(yaml.dump(config_data), encoding="utf-8")

        monkeypatch.setenv("GRDL_RT_LOG_LEVEL", "ERROR")
        cfg = load_runtime_config(config_path=yaml_path)
        assert cfg.log.level == "ERROR"


# ======================================================================
# Singleton
# ======================================================================


class TestSingleton:
    def test_get_runtime_config_returns_same(self):
        reset_runtime_config()
        c1 = get_runtime_config()
        c2 = get_runtime_config()
        assert c1 is c2

    def test_reset_clears_cache(self):
        reset_runtime_config()
        c1 = get_runtime_config()
        reset_runtime_config()
        c2 = get_runtime_config()
        assert c1 is not c2


# ======================================================================
# Full config via GRDL_RT_* env vars
# ======================================================================


class TestFullEnvConfig:
    def test_all_config_via_env(self, monkeypatch):
        monkeypatch.setenv("GRDL_RT_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("GRDL_RT_LOG_FORMAT", "console")
        monkeypatch.setenv("GRDL_RT_RETRY_MAX_RETRIES", "3")
        monkeypatch.setenv("GRDL_RT_RETRY_BACKOFF_BASE", "2.0")
        monkeypatch.setenv("GRDL_RT_RETRY_BACKOFF_MAX", "120.0")
        monkeypatch.setenv("GRDL_RT_MEMORY_WARN_THRESHOLD", "0.75")
        monkeypatch.setenv("GRDL_RT_MEMORY_ABORT_THRESHOLD", "0.90")
        monkeypatch.setenv("GRDL_RT_MEMORY_ESTIMATION_MULTIPLIER", "2.0")
        monkeypatch.setenv("GRDL_RT_GPU_PREFER_GPU", "true")
        monkeypatch.setenv("GRDL_RT_GPU_MEMORY_FRACTION", "0.8")
        monkeypatch.setenv("GRDL_RT_CATALOG_PATH", "/data/cat.db")
        monkeypatch.setenv("GRDL_RT_TAP_OUT_FORMAT", "tif")
        monkeypatch.setenv("GRDL_RT_TAP_OUT_DIR", "/tmp/tapout")

        reset_runtime_config()
        cfg = load_runtime_config()

        assert cfg.log.level == "WARNING"
        assert cfg.log.format == "console"
        assert cfg.retry.max_retries == 3
        assert cfg.retry.backoff_base == 2.0
        assert cfg.retry.backoff_max == 120.0
        assert cfg.memory.warn_threshold == 0.75
        assert cfg.memory.abort_threshold == 0.90
        assert cfg.memory.estimation_multiplier == 2.0
        assert cfg.gpu.prefer_gpu is True
        assert cfg.gpu.memory_fraction == 0.8
        assert cfg.catalog_path == "/data/cat.db"
        assert cfg.tap_out.default_format == "tif"
        assert cfg.tap_out.default_dir == "/tmp/tapout"
