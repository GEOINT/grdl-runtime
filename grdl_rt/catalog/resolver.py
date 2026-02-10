# -*- coding: utf-8 -*-
"""
Catalog Path Resolver - Locate the GRDK artifact catalog database.

Resolves the catalog database path using a priority chain:
1. GRDK_CATALOG_PATH environment variable (highest priority)
2. ~/.grdl/config.json "catalog_path" field
3. ~/.grdl/catalog.db (default fallback)

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
import os
from pathlib import Path
from typing import Optional


_ENV_VAR = "GRDK_CATALOG_PATH"
_CONFIG_DIR = ".grdl"
_CONFIG_FILE = "config.json"
_DEFAULT_DB = "catalog.db"


def resolve_catalog_path() -> Path:
    """Resolve the catalog database path.

    Priority:
    1. ``GRDK_CATALOG_PATH`` environment variable
    2. ``~/.grdl/config.json`` → ``catalog_path`` field
    3. ``~/.grdl/catalog.db`` (default)

    Returns
    -------
    Path
        Resolved path to the catalog database file.
    """
    # Priority 1: Environment variable
    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        return Path(env_path)

    home = Path.home()
    config_dir = home / _CONFIG_DIR

    # Priority 2: Config file
    config_path = config_dir / _CONFIG_FILE
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            catalog_path = config.get('catalog_path')
            if catalog_path:
                return Path(catalog_path)
        except (json.JSONDecodeError, OSError):
            pass

    # Priority 3: Default location
    return config_dir / _DEFAULT_DB


def ensure_config_dir() -> Path:
    """Ensure the ~/.grdl/ configuration directory exists.

    Returns
    -------
    Path
        Path to the configuration directory.
    """
    config_dir = Path.home() / _CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
