# -*- coding: utf-8 -*-
"""
Configuration Module - Configurable defaults for GRDK.

Provides a GrdkConfig dataclass with default values for thumbnail
sizes, debounce intervals, timeouts, and worker counts. Loads from
~/.grdl/grdk_config.json if it exists, otherwise uses sensible
defaults.

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
2026-02-06
"""

# Standard library
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".grdl"
_CONFIG_FILE = _CONFIG_DIR / "grdk_config.json"


@dataclass
class GrdkConfig:
    """Global GRDK configuration with defaults.

    Attributes
    ----------
    thumb_size : int
        Chip gallery thumbnail size in pixels.
    preview_thumb : int
        Preview panel thumbnail size in pixels.
    debounce_ms : int
        UI debounce interval in milliseconds.
    update_timeout : float
        HTTP timeout for update checks in seconds.
    max_workers : int
        Maximum worker threads for background operations.
    """

    thumb_size: int = 128
    preview_thumb: int = 160
    debounce_ms: int = 50
    update_timeout: float = 10.0
    max_workers: int = 4

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to JSON file."""
        path = path or _CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)


def load_config(path: Optional[Path] = None) -> GrdkConfig:
    """Load configuration from file, or return defaults.

    Parameters
    ----------
    path : Optional[Path]
        Config file path. Defaults to ~/.grdl/grdk_config.json.

    Returns
    -------
    GrdkConfig
        Loaded or default configuration.
    """
    path = path or _CONFIG_FILE
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return GrdkConfig(**{
                k: v for k, v in data.items()
                if k in GrdkConfig.__dataclass_fields__
            })
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", path, e)

    return GrdkConfig()
