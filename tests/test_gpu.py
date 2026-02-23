# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.gpu — GpuBackend CPU fallback and transform dispatch.

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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from grdl_rt.execution.gpu import GpuBackend, _check_cupy, _check_torch

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


class TestAvailabilityChecks:
    def test_check_cupy_returns_bool(self):
        result = _check_cupy()
        assert isinstance(result, bool)

    def test_check_torch_returns_bool(self):
        result = _check_torch()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# GpuBackend construction
# ---------------------------------------------------------------------------


class TestGpuBackendConstruction:
    def test_cpu_only_mode(self):
        backend = GpuBackend(prefer_gpu=False)
        assert backend.cupy_available is False
        assert backend.torch_available is False
        assert backend.gpu_available is False

    def test_default_construction(self):
        backend = GpuBackend()
        # May or may not have GPU — just verify no crash
        assert isinstance(backend.gpu_available, bool)


# ---------------------------------------------------------------------------
# to_gpu / to_cpu (CPU fallback path)
# ---------------------------------------------------------------------------


class TestCpuFallback:
    def test_to_gpu_returns_same_array_without_cupy(self):
        backend = GpuBackend(prefer_gpu=False)
        arr = np.array([1.0, 2.0, 3.0])
        result = backend.to_gpu(arr)
        assert result is arr

    def test_to_cpu_returns_numpy(self):
        backend = GpuBackend(prefer_gpu=False)
        arr = np.array([1.0, 2.0, 3.0])
        result = backend.to_cpu(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_cpu_roundtrip(self):
        backend = GpuBackend(prefer_gpu=False)
        arr = np.random.rand(4, 4)
        gpu_arr = backend.to_gpu(arr)
        cpu_arr = backend.to_cpu(gpu_arr)
        np.testing.assert_array_equal(cpu_arr, arr)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------


class TestApplyTransform:
    def test_cpu_transform(self):
        backend = GpuBackend(prefer_gpu=False)
        source = np.ones((8, 8), dtype=np.float32) * 2.0

        mock_transform = MagicMock()
        mock_transform.apply.return_value = source * 3.0

        result = backend.apply_transform(mock_transform, source)
        mock_transform.apply.assert_called_once_with(source)
        np.testing.assert_array_almost_equal(result, np.ones((8, 8)) * 6.0)

    def test_transform_with_kwargs(self):
        backend = GpuBackend(prefer_gpu=False)
        source = np.zeros((4, 4), dtype=np.float32)

        mock_transform = MagicMock()
        mock_transform.apply.return_value = source

        backend.apply_transform(mock_transform, source, threshold=0.5, mode="test")
        mock_transform.apply.assert_called_once_with(source, threshold=0.5, mode="test")

    def test_transform_exception_propagates_on_cpu(self):
        backend = GpuBackend(prefer_gpu=False)
        source = np.zeros((4, 4))

        mock_transform = MagicMock()
        mock_transform.apply.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            backend.apply_transform(mock_transform, source)


# ---------------------------------------------------------------------------
# device_info
# ---------------------------------------------------------------------------


class TestDeviceInfo:
    def test_device_info_cpu_only(self):
        backend = GpuBackend(prefer_gpu=False)
        info = backend.device_info
        assert info["cupy_available"] is False
        assert info["torch_available"] is False
        assert "cupy_device" not in info
        assert "torch_device" not in info


# ---------------------------------------------------------------------------
# apply_torch_model
# ---------------------------------------------------------------------------


class TestApplyTorchModel:
    def test_raises_without_torch(self):
        backend = GpuBackend(prefer_gpu=False)
        source = np.zeros((8, 8))
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch"):
                backend.apply_torch_model("model.pt", source)
