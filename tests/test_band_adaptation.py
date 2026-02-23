"""
Band Adaptation Tests — Unit tests for band axis detection and adaptation.

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
import pytest

from grdl_rt.execution.band_adaptation import (
    BandExpansion,
    BandReduction,
    adapt_bands,
    detect_band_axis,
    get_band_count,
)

# ------------------------------------------------------------------
# detect_band_axis
# ------------------------------------------------------------------


class TestDetectBandAxis:
    def test_hwc(self):
        """(100, 100, 3) → axis 2 (HWC)."""
        assert detect_band_axis(np.zeros((100, 100, 3))) == 2

    def test_chw(self):
        """(3, 100, 100) → axis 0 (CHW)."""
        assert detect_band_axis(np.zeros((3, 100, 100))) == 0

    def test_2d(self):
        """(100, 100) → None."""
        assert detect_band_axis(np.zeros((100, 100))) is None

    def test_square_hwc(self):
        """(100, 100, 4) → axis 2 (HWC)."""
        assert detect_band_axis(np.zeros((100, 100, 4))) == 2

    def test_equal_dims(self):
        """(3, 100, 3) — shape[0] <= shape[2] → axis 0 (CHW)."""
        assert detect_band_axis(np.zeros((3, 100, 3))) == 0

    def test_single_band_chw(self):
        """(1, 100, 100) → axis 0 (CHW) since 1 <= 100."""
        assert detect_band_axis(np.zeros((1, 100, 100))) == 0

    def test_single_band_hwc(self):
        """(100, 100, 1) → axis 2 (HWC) since 100 > 1."""
        assert detect_band_axis(np.zeros((100, 100, 1))) == 2


# ------------------------------------------------------------------
# get_band_count
# ------------------------------------------------------------------


class TestGetBandCount:
    def test_2d(self):
        assert get_band_count(np.zeros((100, 100))) == 1

    def test_hwc_3(self):
        assert get_band_count(np.zeros((100, 100, 3))) == 3

    def test_chw_3(self):
        assert get_band_count(np.zeros((3, 100, 100))) == 3

    def test_hwc_4(self):
        assert get_band_count(np.zeros((100, 100, 4))) == 4


# ------------------------------------------------------------------
# adapt_bands — no-op
# ------------------------------------------------------------------


class TestAdaptBandsNoop:
    def test_matching_hwc(self):
        """3 bands, requires 3 → unchanged."""
        source = np.random.rand(100, 100, 3)
        result = adapt_bands(source, 3)
        assert result is source

    def test_matching_chw(self):
        """3 bands CHW, requires 3 → unchanged."""
        source = np.random.rand(3, 100, 100)
        result = adapt_bands(source, 3)
        assert result is source

    def test_matching_2d_requires_1(self):
        """2D (1 band), requires 1 → unchanged."""
        source = np.random.rand(100, 100)
        result = adapt_bands(source, 1)
        assert result is source


# ------------------------------------------------------------------
# adapt_bands — expansion
# ------------------------------------------------------------------


class TestExpandBands:
    def test_repeat_1_to_3_hwc(self):
        """1-band HWC → 3 bands, all channels identical."""
        source = np.random.rand(50, 50, 1)
        result = adapt_bands(source, 3, expansion=BandExpansion.REPEAT)
        assert result.shape == (50, 50, 3)
        np.testing.assert_array_equal(result[:, :, 0], source[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], source[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 2], source[:, :, 0])

    def test_repeat_2d_to_3(self):
        """2D grayscale → 3 bands via REPEAT."""
        source = np.random.rand(50, 50)
        result = adapt_bands(source, 3, expansion=BandExpansion.REPEAT)
        assert result.shape == (50, 50, 3)
        np.testing.assert_array_equal(result[:, :, 0], source)
        np.testing.assert_array_equal(result[:, :, 1], source)
        np.testing.assert_array_equal(result[:, :, 2], source)

    def test_repeat_1_to_4_chw(self):
        """1-band CHW → 4 bands (SN7 case)."""
        source = np.random.rand(1, 100, 100)
        result = adapt_bands(source, 4, expansion=BandExpansion.REPEAT)
        assert result.shape == (4, 100, 100)
        for b in range(4):
            np.testing.assert_array_equal(result[b], source[0])

    def test_repeat_2_to_3_hwc(self):
        """2-band → 3 bands via cyclic repeat."""
        source = np.random.rand(50, 50, 2)
        result = adapt_bands(source, 3, expansion=BandExpansion.REPEAT)
        assert result.shape == (50, 50, 3)
        np.testing.assert_array_equal(result[:, :, 0], source[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], source[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], source[:, :, 0])

    def test_zero_pad_1_to_3_hwc(self):
        """1-band HWC → 3 bands via ZERO_PAD: bands 2,3 are zeros."""
        source = np.random.rand(50, 50, 1)
        result = adapt_bands(source, 3, expansion=BandExpansion.ZERO_PAD)
        assert result.shape == (50, 50, 3)
        np.testing.assert_array_equal(result[:, :, 0], source[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], np.zeros((50, 50)))
        np.testing.assert_array_equal(result[:, :, 2], np.zeros((50, 50)))

    def test_zero_pad_2d_to_3(self):
        """2D grayscale → 3 bands via ZERO_PAD."""
        source = np.random.rand(50, 50)
        result = adapt_bands(source, 3, expansion=BandExpansion.ZERO_PAD)
        assert result.shape == (50, 50, 3)
        np.testing.assert_array_equal(result[:, :, 0], source)
        np.testing.assert_array_equal(result[:, :, 1], np.zeros((50, 50)))


# ------------------------------------------------------------------
# adapt_bands — reduction
# ------------------------------------------------------------------


class TestReduceBands:
    def test_first_n_4_to_3_hwc(self):
        """Takes first 3 of 4 bands."""
        source = np.random.rand(50, 50, 4)
        result = adapt_bands(source, 3, reduction=BandReduction.FIRST_N)
        assert result.shape == (50, 50, 3)
        np.testing.assert_array_equal(result, source[:, :, :3])

    def test_first_n_4_to_3_chw(self):
        """Takes first 3 of 4 bands in CHW layout."""
        source = np.random.rand(4, 100, 100)
        result = adapt_bands(source, 3, reduction=BandReduction.FIRST_N)
        assert result.shape == (3, 100, 100)
        np.testing.assert_array_equal(result, source[:3])

    def test_mean_3_to_1_hwc(self):
        """Mean reduction across 3 bands → 1."""
        source = np.full((10, 10, 3), [1.0, 2.0, 3.0])
        result = adapt_bands(source, 1, reduction=BandReduction.MEAN)
        assert result.shape == (10, 10, 1)
        assert result[0, 0, 0] == pytest.approx(2.0)

    def test_median_3_to_1(self):
        """Median across 3 bands."""
        source = np.full((10, 10, 3), [1.0, 5.0, 3.0])
        result = adapt_bands(source, 1, reduction=BandReduction.MEDIAN)
        assert result.shape == (10, 10, 1)
        assert result[0, 0, 0] == pytest.approx(3.0)

    def test_max_3_to_1(self):
        """Max across 3 bands."""
        source = np.full((10, 10, 3), [1.0, 5.0, 3.0])
        result = adapt_bands(source, 1, reduction=BandReduction.MAX)
        assert result.shape == (10, 10, 1)
        assert result[0, 0, 0] == pytest.approx(5.0)

    def test_min_3_to_1(self):
        """Min across 3 bands."""
        source = np.full((10, 10, 3), [1.0, 5.0, 3.0])
        result = adapt_bands(source, 1, reduction=BandReduction.MIN)
        assert result.shape == (10, 10, 1)
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_luminance_3_to_1(self):
        """Weighted luminance sum matches expected."""
        source = np.full((10, 10, 3), [100.0, 150.0, 200.0])
        result = adapt_bands(source, 1, reduction=BandReduction.LUMINANCE)
        expected = 0.2126 * 100.0 + 0.7152 * 150.0 + 0.0722 * 200.0
        assert result.shape == (10, 10, 1)
        assert result[0, 0, 0] == pytest.approx(expected)

    def test_luminance_wrong_bands_raises(self):
        """LUMINANCE with non-3-band input → ValueError."""
        source = np.random.rand(50, 50, 4)
        with pytest.raises(ValueError, match="3 input bands"):
            adapt_bands(source, 1, reduction=BandReduction.LUMINANCE)

    def test_luminance_wrong_target_raises(self):
        """LUMINANCE requesting non-1 target → ValueError."""
        source = np.random.rand(50, 50, 3)
        with pytest.raises(ValueError, match="1 band"):
            adapt_bands(source, 2, reduction=BandReduction.LUMINANCE)


# ------------------------------------------------------------------
# dtype preservation
# ------------------------------------------------------------------


class TestDtypePreservation:
    def test_expand_preserves_dtype(self):
        """uint8 input stays uint8 after repeat expansion."""
        source = np.zeros((50, 50, 1), dtype=np.uint8)
        result = adapt_bands(source, 3, expansion=BandExpansion.REPEAT)
        assert result.dtype == np.uint8

    def test_zero_pad_preserves_dtype(self):
        """float32 input stays float32 after zero pad."""
        source = np.zeros((50, 50, 1), dtype=np.float32)
        result = adapt_bands(source, 3, expansion=BandExpansion.ZERO_PAD)
        assert result.dtype == np.float32

    def test_first_n_preserves_dtype(self):
        source = np.zeros((50, 50, 4), dtype=np.uint16)
        result = adapt_bands(source, 3, reduction=BandReduction.FIRST_N)
        assert result.dtype == np.uint16
