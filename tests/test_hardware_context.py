# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.hardware — HardwareContext and LocalHardwareContext.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

from unittest.mock import patch, MagicMock

import pytest

from grdl_rt.execution.hardware import (
    GpuDeviceInfo,
    HardwareContext,
    LocalHardwareContext,
)


class TestGpuDeviceInfo:

    def test_basic_construction(self):
        info = GpuDeviceInfo(name="RTX 4090", memory_bytes=24_000_000_000)
        assert info.name == "RTX 4090"
        assert info.memory_bytes == 24_000_000_000
        assert info.device_index == 0

    def test_frozen(self):
        info = GpuDeviceInfo(name="A100", memory_bytes=80_000_000_000, device_index=1)
        with pytest.raises(AttributeError):
            info.name = "V100"


class TestHardwareContextABC:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            HardwareContext()

    def test_gpu_memory_bytes_sums_devices(self):
        """Concrete gpu_memory_bytes sums device memories."""

        class _StubHW(HardwareContext):
            @property
            def cpu_count(self):
                return 4

            @property
            def total_memory_bytes(self):
                return 16_000_000_000

            @property
            def available_memory_bytes(self):
                return 8_000_000_000

            @property
            def gpu_available(self):
                return True

            @property
            def gpu_devices(self):
                return [
                    GpuDeviceInfo("A", 10_000),
                    GpuDeviceInfo("B", 20_000, device_index=1),
                ]

            def to_dict(self):
                return {}

        hw = _StubHW()
        assert hw.gpu_memory_bytes == 30_000

    def test_gpu_memory_bytes_no_devices(self):
        """gpu_memory_bytes is 0 when no GPU devices."""

        class _NoGpuHW(HardwareContext):
            @property
            def cpu_count(self):
                return 1

            @property
            def total_memory_bytes(self):
                return 1_000_000

            @property
            def available_memory_bytes(self):
                return 500_000

            @property
            def gpu_available(self):
                return False

            @property
            def gpu_devices(self):
                return []

            def to_dict(self):
                return {}

        hw = _NoGpuHW()
        assert hw.gpu_memory_bytes == 0


class TestLocalHardwareContext:

    @patch("grdl_rt.execution.hardware.os.cpu_count", return_value=8)
    def test_cpu_count(self, mock_cpu):
        hw = LocalHardwareContext()
        assert hw.cpu_count == 8

    @patch("grdl_rt.execution.hardware.os.cpu_count", return_value=None)
    def test_cpu_count_none_fallback(self, mock_cpu):
        """Falls back to 1 if os.cpu_count() returns None."""
        hw = LocalHardwareContext()
        assert hw.cpu_count == 1

    def test_positive_memory(self):
        hw = LocalHardwareContext()
        assert hw.total_memory_bytes > 0
        assert hw.available_memory_bytes > 0
        assert hw.available_memory_bytes <= hw.total_memory_bytes

    def test_to_dict_has_expected_keys(self):
        hw = LocalHardwareContext()
        d = hw.to_dict()
        assert "cpu_count" in d
        assert "total_memory_bytes" in d
        assert "available_memory_bytes" in d
        assert "gpu_available" in d
        assert "gpu_devices" in d
        assert "gpu_memory_bytes" in d
        assert isinstance(d["cpu_count"], int)
        assert isinstance(d["total_memory_bytes"], int)

    def test_gpu_devices_is_list(self):
        hw = LocalHardwareContext()
        assert isinstance(hw.gpu_devices, list)
        assert isinstance(hw.gpu_available, bool)

    def test_gpu_memory_consistency(self):
        hw = LocalHardwareContext()
        expected = sum(d.memory_bytes for d in hw.gpu_devices)
        assert hw.gpu_memory_bytes == expected
