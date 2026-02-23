"""
Band Adaptation - Automatic band count adjustment between pipeline steps.

Detects the band axis and count of an input array, then expands or reduces
bands to match a processor's declared ``required_bands``.  Strategies are
configurable per-workflow and per-step via :class:`BandExpansion` and
:class:`BandReduction` enums from the GRDL vocabulary.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-12
"""

from __future__ import annotations

import numpy as np

from grdl_rt.execution.context import get_logger

# GRDL vocabulary enums (optional — graceful fallback if grdl is old)
try:
    from grdl.vocabulary import BandExpansion, BandReduction
except ImportError:
    from enum import Enum

    class BandExpansion(Enum):  # type: ignore[no-redef]
        REPEAT = "repeat"
        ZERO_PAD = "zero_pad"

    class BandReduction(Enum):  # type: ignore[no-redef]
        FIRST_N = "first_n"
        MEAN = "mean"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        LUMINANCE = "luminance"


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Band axis detection
# ------------------------------------------------------------------


def detect_band_axis(source: np.ndarray) -> int | None:
    """Auto-detect the band axis of an image array.

    Parameters
    ----------
    source : np.ndarray
        2-D or 3-D image array.

    Returns
    -------
    Optional[int]
        Band axis index (0 for CHW, 2 for HWC), or ``None`` for 2-D arrays.

    Notes
    -----
    Heuristic for 3-D arrays:

    - ``shape[0] <= shape[2]`` → CHW (bands-first, e.g. ``(3, 512, 512)``)
    - ``shape[2] < shape[0]`` → HWC (bands-last, e.g. ``(512, 512, 3)``)
    """
    if source.ndim == 2:
        return None
    if source.ndim != 3:
        return None
    if source.shape[0] <= source.shape[2]:
        return 0  # CHW
    return 2  # HWC


def get_band_count(source: np.ndarray) -> int:
    """Return the number of spectral bands in an image array.

    Parameters
    ----------
    source : np.ndarray
        2-D or 3-D image array.

    Returns
    -------
    int
        Number of bands (1 for 2-D arrays).
    """
    axis = detect_band_axis(source)
    if axis is None:
        return 1
    return source.shape[axis]


# ------------------------------------------------------------------
# Band adaptation
# ------------------------------------------------------------------


def adapt_bands(
    source: np.ndarray,
    required_bands: int,
    expansion: BandExpansion = BandExpansion.REPEAT,
    reduction: BandReduction = BandReduction.FIRST_N,
) -> np.ndarray:
    """Adapt *source* to have exactly *required_bands* bands.

    No-op when the current band count already matches.

    Parameters
    ----------
    source : np.ndarray
        2-D or 3-D image array.
    required_bands : int
        Target number of bands.
    expansion : BandExpansion
        Strategy when the source has fewer bands than required.
    reduction : BandReduction
        Strategy when the source has more bands than required.

    Returns
    -------
    np.ndarray
        Array with exactly *required_bands* bands.

    Raises
    ------
    ValueError
        If LUMINANCE reduction is requested with a non-3-band input.
    """
    current_bands = get_band_count(source)

    if current_bands == required_bands:
        return source

    axis = detect_band_axis(source)

    logger.warning(
        "band_adaptation",
        current_bands=current_bands,
        required_bands=required_bands,
        expansion=expansion.value if current_bands < required_bands else None,
        reduction=reduction.value if current_bands > required_bands else None,
    )

    if current_bands < required_bands:
        return _expand_bands(source, current_bands, required_bands, axis, expansion)
    return _reduce_bands(source, current_bands, required_bands, axis, reduction)


# ------------------------------------------------------------------
# Expansion strategies
# ------------------------------------------------------------------


def _expand_bands(
    source: np.ndarray,
    current: int,
    target: int,
    axis: int | None,
    strategy: BandExpansion,
) -> np.ndarray:
    """Expand *source* from *current* to *target* bands."""
    if axis is None:
        # 2-D → 3-D (HWC convention for expanded output)
        source = source[:, :, np.newaxis]
        axis = 2
        current = 1

    if strategy is BandExpansion.REPEAT:
        return _expand_repeat(source, current, target, axis)
    if strategy is BandExpansion.ZERO_PAD:
        return _expand_zero_pad(source, current, target, axis)

    raise ValueError(f"Unknown expansion strategy: {strategy!r}")


def _expand_repeat(
    source: np.ndarray,
    current: int,
    target: int,
    axis: int,
) -> np.ndarray:
    """Tile bands cyclically to reach *target*."""
    repeats = target // current
    remainder = target % current

    parts = [source] * repeats
    if remainder > 0:
        slices = [slice(None)] * source.ndim
        slices[axis] = slice(0, remainder)
        parts.append(source[tuple(slices)])

    return np.concatenate(parts, axis=axis)


def _expand_zero_pad(
    source: np.ndarray,
    current: int,
    target: int,
    axis: int,
) -> np.ndarray:
    """Pad with zero bands to reach *target*."""
    pad_count = target - current
    pad_shape = list(source.shape)
    pad_shape[axis] = pad_count
    zeros = np.zeros(pad_shape, dtype=source.dtype)
    return np.concatenate([source, zeros], axis=axis)


# ------------------------------------------------------------------
# Reduction strategies
# ------------------------------------------------------------------


def _reduce_bands(
    source: np.ndarray,
    current: int,
    target: int,
    axis: int | None,
    strategy: BandReduction,
) -> np.ndarray:
    """Reduce *source* from *current* to *target* bands."""
    if axis is None:
        # 2-D with 1 band — nothing to reduce
        return source

    if strategy is BandReduction.FIRST_N:
        return _reduce_first_n(source, target, axis)
    if strategy is BandReduction.MEAN:
        return _reduce_stat(source, target, axis, np.mean)
    if strategy is BandReduction.MEDIAN:
        return _reduce_stat(source, target, axis, np.median)
    if strategy is BandReduction.MAX:
        return _reduce_stat(source, target, axis, np.max)
    if strategy is BandReduction.MIN:
        return _reduce_stat(source, target, axis, np.min)
    if strategy is BandReduction.LUMINANCE:
        return _reduce_luminance(source, current, target, axis)

    raise ValueError(f"Unknown reduction strategy: {strategy!r}")


def _reduce_first_n(source: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Take the first *target* bands."""
    slices = [slice(None)] * source.ndim
    slices[axis] = slice(0, target)
    return source[tuple(slices)]


def _reduce_stat(
    source: np.ndarray,
    target: int,
    axis: int,
    stat_fn,
) -> np.ndarray:
    """Reduce via a statistical function across all bands.

    The stat is computed across the band axis, yielding a single band.
    If *target* > 1, the result is tiled to *target* bands.
    """
    reduced = stat_fn(source, axis=axis, keepdims=True)
    if target > 1:
        repeats = [1] * source.ndim
        repeats[axis] = target
        reduced = np.tile(reduced, repeats)
    return reduced


def _reduce_luminance(
    source: np.ndarray,
    current: int,
    target: int,
    axis: int,
) -> np.ndarray:
    """Weighted luminance: 0.2126*R + 0.7152*G + 0.0722*B.

    Only valid for 3-band → 1-band reduction.
    """
    if current != 3:
        raise ValueError(f"LUMINANCE reduction requires exactly 3 input bands, got {current}")
    if target != 1:
        raise ValueError(f"LUMINANCE reduction produces 1 band, but {target} were requested")

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

    # Build the weighted sum along the band axis
    slices_r: list[int | slice] = [slice(None)] * source.ndim
    slices_g: list[int | slice] = [slice(None)] * source.ndim
    slices_b: list[int | slice] = [slice(None)] * source.ndim
    slices_r[axis] = 0
    slices_g[axis] = 1
    slices_b[axis] = 2

    luminance = (
        weights[0] * source[tuple(slices_r)].astype(np.float64)
        + weights[1] * source[tuple(slices_g)].astype(np.float64)
        + weights[2] * source[tuple(slices_b)].astype(np.float64)
    )
    return np.expand_dims(luminance, axis=axis)
