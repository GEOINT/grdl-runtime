# -*- coding: utf-8 -*-
"""
Configuration Module — 12-factor, environment-variable-driven runtime config.

Provides ``RuntimeConfig``, a pydantic-validated configuration model loaded
from environment variables (``GRDL_RT_*``), TOML files, YAML files, or
programmatic defaults.  Env vars always take highest precedence.

Source priority (highest → lowest):
    1. Environment variables (``GRDL_RT_*``)
    2. TOML config file
    3. YAML config file
    4. Built-in defaults

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-11
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

_CONFIG_DIR = Path.home() / ".grdl"


# ======================================================================
# Config sub-models
# ======================================================================


class LogConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(default="json", description="Log format: 'json' or 'console'")

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log level must be one of {allowed}, got '{v}'")
        return v

    @field_validator("format")
    @classmethod
    def _validate_format(cls, v: str) -> str:
        v = v.lower()
        if v not in ("json", "console"):
            raise ValueError(f"log format must be 'json' or 'console', got '{v}'")
        return v


class RetryDefaults(BaseModel):
    """Default retry policy applied when a step has no explicit retry config."""

    max_retries: int = Field(default=0, ge=0)
    backoff_base: float = Field(default=1.0, gt=0)
    backoff_max: float = Field(default=60.0, gt=0)
    retryable_exceptions: List[str] = Field(
        default=["RuntimeError", "OSError", "MemoryError"],
    )


class MemoryConfig(BaseModel):
    """Memory estimation and threshold configuration."""

    warn_threshold: float = Field(
        default=0.80, ge=0, le=1.0,
        description="Warn when estimated usage exceeds this fraction of available RAM",
    )
    abort_threshold: float = Field(
        default=0.95, ge=0, le=1.0,
        description="Abort when estimated usage exceeds this fraction of available RAM",
    )
    estimation_multiplier: float = Field(
        default=1.5, gt=0,
        description="Safety multiplier applied to input_array.nbytes * n_steps",
    )


class GpuConfig(BaseModel):
    """GPU preferences."""

    prefer_gpu: bool = Field(default=False, description="Prefer GPU execution when available")
    memory_fraction: float = Field(
        default=0.9, gt=0, le=1.0,
        description="Maximum fraction of GPU memory to use",
    )


class TapOutConfig(BaseModel):
    """Tap-out defaults."""

    default_format: str = Field(default="npy", description="Default intermediate output format")
    default_dir: Optional[str] = Field(
        default=None, description="Default directory for tap-out files",
    )


class QuotaConfig(BaseModel):
    """Per-workflow resource quota defaults.

    These are absolute limits (not fractions of available RAM).
    Set to ``None`` to disable a particular quota.
    """

    max_memory_bytes: Optional[int] = Field(
        default=None, ge=1, description="Maximum RSS memory in bytes",
    )
    max_cpu_percent: Optional[float] = Field(
        default=None, gt=0, description="Maximum CPU usage percentage",
    )
    max_gpu_memory_bytes: Optional[int] = Field(
        default=None, ge=1, description="Maximum GPU memory in bytes",
    )
    max_wall_clock_seconds: Optional[float] = Field(
        default=None, gt=0, description="Maximum wall-clock seconds for the workflow",
    )


class PrometheusConfig(BaseModel):
    """Prometheus metrics configuration (opt-in)."""

    enabled: bool = Field(default=False, description="Enable Prometheus metrics collection")
    prefix: str = Field(default="grdl_rt", description="Metric name prefix")


class OtelConfig(BaseModel):
    """OpenTelemetry tracing configuration (opt-in)."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    exporter: str = Field(
        default="none", description="Trace exporter: 'otlp', 'console', or 'none'",
    )
    service_name: str = Field(
        default="grdl-runtime", description="Service name for OTel resource",
    )

    @field_validator("exporter")
    @classmethod
    def _validate_exporter(cls, v: str) -> str:
        v = v.lower()
        if v not in ("otlp", "console", "none"):
            raise ValueError(f"exporter must be 'otlp', 'console', or 'none', got '{v}'")
        return v


# ======================================================================
# Top-level RuntimeConfig
# ======================================================================


class RuntimeConfig(BaseModel):
    """12-factor runtime configuration for grdl-runtime.

    All fields can be overridden by ``GRDL_RT_*`` environment variables.
    Nested fields use double-underscore separators:

    - ``GRDL_RT_LOG__LEVEL=DEBUG``
    - ``GRDL_RT_MEMORY__WARN_THRESHOLD=0.70``
    - ``GRDL_RT_RETRY__MAX_RETRIES=3``
    - ``GRDL_RT_GPU__PREFER_GPU=true``

    Flat fields use single prefix:

    - ``GRDL_RT_CATALOG_PATH=/data/catalog.db``
    """

    log: LogConfig = Field(default_factory=LogConfig)
    retry: RetryDefaults = Field(default_factory=RetryDefaults)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    catalog_path: Optional[str] = Field(
        default=None, description="Path to artifact catalog database",
    )
    tap_out: TapOutConfig = Field(default_factory=TapOutConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    otel: OtelConfig = Field(default_factory=OtelConfig)


# ======================================================================
# Env-var overlay
# ======================================================================

# Mapping from GRDL_RT_* env var name → (section, key) or (None, key) for top-level.
_ENV_MAP: Dict[str, tuple] = {
    "GRDL_RT_LOG_LEVEL": ("log", "level"),
    "GRDL_RT_LOG_FORMAT": ("log", "format"),
    "GRDL_RT_RETRY_MAX_RETRIES": ("retry", "max_retries"),
    "GRDL_RT_RETRY_BACKOFF_BASE": ("retry", "backoff_base"),
    "GRDL_RT_RETRY_BACKOFF_MAX": ("retry", "backoff_max"),
    "GRDL_RT_MEMORY_WARN_THRESHOLD": ("memory", "warn_threshold"),
    "GRDL_RT_MEMORY_ABORT_THRESHOLD": ("memory", "abort_threshold"),
    "GRDL_RT_MEMORY_ESTIMATION_MULTIPLIER": ("memory", "estimation_multiplier"),
    "GRDL_RT_GPU_PREFER_GPU": ("gpu", "prefer_gpu"),
    "GRDL_RT_GPU_MEMORY_FRACTION": ("gpu", "memory_fraction"),
    "GRDL_RT_CATALOG_PATH": (None, "catalog_path"),
    "GRDL_RT_TAP_OUT_FORMAT": ("tap_out", "default_format"),
    "GRDL_RT_TAP_OUT_DIR": ("tap_out", "default_dir"),
    # Quota
    "GRDL_RT_QUOTA_MAX_MEMORY_BYTES": ("quota", "max_memory_bytes"),
    "GRDL_RT_QUOTA_MAX_CPU_PERCENT": ("quota", "max_cpu_percent"),
    "GRDL_RT_QUOTA_MAX_GPU_MEMORY_BYTES": ("quota", "max_gpu_memory_bytes"),
    "GRDL_RT_QUOTA_MAX_WALL_CLOCK_SECONDS": ("quota", "max_wall_clock_seconds"),
    # Prometheus
    "GRDL_RT_PROMETHEUS_ENABLED": ("prometheus", "enabled"),
    "GRDL_RT_PROMETHEUS_PREFIX": ("prometheus", "prefix"),
    # OpenTelemetry
    "GRDL_RT_OTEL_ENABLED": ("otel", "enabled"),
    "GRDL_RT_OTEL_EXPORTER": ("otel", "exporter"),
    "GRDL_RT_OTEL_SERVICE_NAME": ("otel", "service_name"),
}

_BOOL_TRUTHY = {"1", "true", "yes", "on"}
_BOOL_FALSY = {"0", "false", "no", "off", ""}


def _coerce_env_value(raw: str, section: Optional[str], key: str) -> Any:
    """Coerce a string env var value to the appropriate Python type."""
    # Boolean fields
    if key in ("prefer_gpu", "enabled"):
        return raw.lower() in _BOOL_TRUTHY

    # Float fields
    if key in (
        "backoff_base", "backoff_max", "warn_threshold",
        "abort_threshold", "estimation_multiplier", "memory_fraction",
        "max_cpu_percent", "max_wall_clock_seconds",
    ):
        return float(raw)

    # Int fields
    if key in ("max_retries", "max_memory_bytes", "max_gpu_memory_bytes"):
        return int(raw)

    return raw


def _overlay_env_vars(data: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay GRDL_RT_* environment variables onto a config dict."""
    for env_name, (section, key) in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue

        value = _coerce_env_value(raw, section, key)

        if section is None:
            data[key] = value
        else:
            if section not in data:
                data[section] = {}
            data[section][key] = value

    return data


# ======================================================================
# File loaders
# ======================================================================


def _load_toml(path: Path) -> Dict[str, Any]:
    """Load a TOML file, returning a dict.  Returns {} on failure."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.debug("toml_load_skipped", reason="no toml library available")
            return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning("toml_load_failed", path=str(path), error=str(e))
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning a dict.  Returns {} on failure."""
    try:
        import yaml
    except ImportError:
        logger.debug("yaml_load_skipped", reason="pyyaml not installed")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            result = yaml.safe_load(f)
            return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("yaml_load_failed", path=str(path), error=str(e))
        return {}


def _load_file(path: Path) -> Dict[str, Any]:
    """Auto-dispatch to TOML or YAML loader based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return _load_toml(path)
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    logger.warning("config_unknown_format", path=str(path), suffix=suffix)
    return {}


# ======================================================================
# Public API
# ======================================================================


def load_runtime_config(
    config_path: Optional[Path] = None,
) -> RuntimeConfig:
    """Load runtime configuration with 12-factor precedence.

    Resolution order:

    1. Built-in defaults
    2. Config file (TOML or YAML) — auto-discovered from
       ``~/.grdl/runtime.toml`` or ``~/.grdl/runtime.yaml``
       if *config_path* is not given.
    3. ``GRDL_RT_*`` environment variables (highest priority).

    Parameters
    ----------
    config_path : Path, optional
        Explicit path to a TOML or YAML config file.

    Returns
    -------
    RuntimeConfig
        Validated runtime configuration.
    """
    data: Dict[str, Any] = {}

    # 1. Load from file
    if config_path is not None and config_path.exists():
        data = _load_file(config_path)
    else:
        # Auto-discover
        for candidate in (
            _CONFIG_DIR / "runtime.toml",
            _CONFIG_DIR / "runtime.yaml",
            _CONFIG_DIR / "runtime.yml",
        ):
            if candidate.exists():
                data = _load_file(candidate)
                logger.debug("config_loaded", path=str(candidate))
                break

    # 2. Overlay env vars
    data = _overlay_env_vars(data)

    # 3. Validate
    try:
        return RuntimeConfig(**data)
    except Exception as e:
        logger.warning("config_validation_failed", error=str(e))
        return RuntimeConfig()


# Module-level singleton (lazy)
_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """Return the cached runtime configuration singleton.

    Calls :func:`load_runtime_config` on first access.
    """
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = load_runtime_config()
    return _runtime_config


def reset_runtime_config() -> None:
    """Clear the cached config singleton (useful for testing)."""
    global _runtime_config
    _runtime_config = None
