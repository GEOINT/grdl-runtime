# -*- coding: utf-8 -*-
"""
Shared test fixtures for grdl-runtime.

Provides synthetic images, temporary directories, mock processors,
and catalog fixtures used across multiple test modules.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-09
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Synthetic image fixtures ──────────────────────────────────────────


@pytest.fixture
def grayscale_8x8():
    """8x8 float32 grayscale ramp."""
    return np.arange(64, dtype=np.float32).reshape(8, 8)


@pytest.fixture
def rgb_4x4():
    """4x4 uint8 RGB image (3 channels)."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 0] = 100  # R
    arr[:, :, 1] = 150  # G
    arr[:, :, 2] = 200  # B
    return arr


@pytest.fixture
def complex_8x8():
    """8x8 complex64 array (SAR-like)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))).astype(
        np.complex64
    )


# ── Temporary directory / database fixtures ───────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite database path."""
    return tmp_path / "test_catalog.db"


# ── Mock fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_catalog():
    """Mock ArtifactCatalog with empty default returns."""
    cat = MagicMock()
    cat.list_artifacts.return_value = []
    cat.get_artifact.return_value = None
    cat.search.return_value = []
    cat.search_by_tags.return_value = []
    return cat


@pytest.fixture
def mock_gpu_backend():
    """Mock GpuBackend with GPU disabled."""
    backend = MagicMock()
    backend.gpu_available = False
    backend.cupy_available = False
    backend.torch_available = False
    backend.to_gpu.side_effect = lambda x: x
    backend.to_cpu.side_effect = lambda x: x
    return backend


# ── Mock processor helpers ────────────────────────────────────────────


class ScaleTransform:
    """Trivial processor that multiplies by a scale factor."""

    def apply(self, source, **kwargs):
        kwargs.pop("progress_callback", None)
        return source * kwargs.get("scale", 1.0)


class IdentityTransform:
    """No-op processor that returns input unchanged."""

    def apply(self, source, **kwargs):
        kwargs.pop("progress_callback", None)
        return source
