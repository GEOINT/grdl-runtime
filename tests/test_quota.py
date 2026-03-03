"""
Tests for grdl_rt.execution.quota — ResourceQuota and QuotaEnforcer.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

from unittest.mock import patch

import pytest

from grdl_rt.execution.errors import QuotaExceededError
from grdl_rt.execution.quota import QuotaEnforcer, ResourceQuota

# ======================================================================
# ResourceQuota — defaults, validation, serialization
# ======================================================================


class TestResourceQuota:
    def test_resource_quota_defaults(self):
        """All fields default to None (no limits)."""
        q = ResourceQuota()
        assert q.max_memory_bytes is None
        assert q.max_cpu_percent is None
        assert q.max_gpu_memory_bytes is None
        assert q.max_wall_clock_seconds is None

    def test_resource_quota_validation_negative_memory(self):
        """Negative max_memory_bytes raises ValueError."""
        with pytest.raises(ValueError, match="max_memory_bytes must be positive"):
            ResourceQuota(max_memory_bytes=-1)

    def test_resource_quota_validation_zero_memory(self):
        """Zero max_memory_bytes raises ValueError (must be strictly positive)."""
        with pytest.raises(ValueError, match="max_memory_bytes must be positive"):
            ResourceQuota(max_memory_bytes=0)

    def test_resource_quota_validation_negative_wall_clock(self):
        """Negative max_wall_clock_seconds raises ValueError."""
        with pytest.raises(ValueError, match="max_wall_clock_seconds must be positive"):
            ResourceQuota(max_wall_clock_seconds=-10.0)

    def test_resource_quota_validation_negative_cpu(self):
        """Negative max_cpu_percent raises ValueError."""
        with pytest.raises(ValueError, match="max_cpu_percent must be positive"):
            ResourceQuota(max_cpu_percent=-50.0)

    def test_resource_quota_validation_negative_gpu(self):
        """Negative max_gpu_memory_bytes raises ValueError."""
        with pytest.raises(ValueError, match="max_gpu_memory_bytes must be positive"):
            ResourceQuota(max_gpu_memory_bytes=-1024)

    def test_resource_quota_serialization(self):
        """to_dict / from_dict round-trips preserve all fields."""
        original = ResourceQuota(
            max_memory_bytes=1_000_000,
            max_cpu_percent=80.0,
            max_gpu_memory_bytes=2_000_000,
            max_wall_clock_seconds=300.0,
        )
        d = original.to_dict()
        restored = ResourceQuota.from_dict(d)

        assert restored.max_memory_bytes == 1_000_000
        assert restored.max_cpu_percent == 80.0
        assert restored.max_gpu_memory_bytes == 2_000_000
        assert restored.max_wall_clock_seconds == 300.0

    def test_resource_quota_serialization_none_fields(self):
        """Round-trip with all-None fields."""
        original = ResourceQuota()
        d = original.to_dict()
        restored = ResourceQuota.from_dict(d)

        assert restored.max_memory_bytes is None
        assert restored.max_cpu_percent is None
        assert restored.max_gpu_memory_bytes is None
        assert restored.max_wall_clock_seconds is None

    def test_resource_quota_frozen(self):
        """ResourceQuota is frozen; assignment raises."""
        q = ResourceQuota(max_memory_bytes=1024)
        with pytest.raises(AttributeError):
            q.max_memory_bytes = 2048  # type: ignore[misc]


# ======================================================================
# QuotaEnforcer — memory pre-flight
# ======================================================================


class TestQuotaEnforcerMemoryPreflight:
    def test_quota_enforcer_memory_preflight_ok(self, grayscale_8x8):
        """No error when estimated memory fits within the quota.

        grayscale_8x8 is 8x8 float32 = 256 bytes.
        estimate_memory(arr, n_steps=2, multiplier=1.5) = 256 * 1.5 * 2 = 768.
        A limit of 1024 is sufficient.
        """
        quota = ResourceQuota(max_memory_bytes=1024)
        enforcer = QuotaEnforcer(quota)
        # Should not raise
        enforcer.check_before_execution(grayscale_8x8, n_steps=2, multiplier=1.5)

    def test_quota_enforcer_memory_preflight_exceeds(self, grayscale_8x8):
        """Raises QuotaExceededError when estimated memory exceeds the limit.

        grayscale_8x8 is 8x8 float32 = 256 bytes.
        estimate_memory(arr, n_steps=4, multiplier=2.0) = 256 * 2.0 * 4 = 2048.
        A limit of 500 is too small.
        """
        quota = ResourceQuota(max_memory_bytes=500)
        enforcer = QuotaEnforcer(quota)

        with pytest.raises(QuotaExceededError) as exc_info:
            enforcer.check_before_execution(
                grayscale_8x8,
                n_steps=4,
                multiplier=2.0,
            )

        assert exc_info.value.quota_type == "memory_preflight"
        assert exc_info.value.limit == 500
        assert exc_info.value.actual == 2048

    def test_quota_enforcer_memory_preflight_exact_boundary(self, grayscale_8x8):
        """Exactly at the limit does not raise (estimated <= limit)."""
        # 256 * 1.5 * 2 = 768
        quota = ResourceQuota(max_memory_bytes=768)
        enforcer = QuotaEnforcer(quota)
        # Should not raise — 768 is not > 768
        enforcer.check_before_execution(grayscale_8x8, n_steps=2, multiplier=1.5)


# ======================================================================
# QuotaEnforcer — wall-clock enforcement
# ======================================================================


class TestQuotaEnforcerWallClock:
    def test_quota_enforcer_wall_clock_ok(self):
        """No error when elapsed time is within the quota."""
        quota = ResourceQuota(max_wall_clock_seconds=10.0)
        enforcer = QuotaEnforcer(quota)
        enforcer.mark_start()
        # Immediately check — elapsed ~0s, well within 10s
        enforcer.check_wall_clock()

    def test_quota_enforcer_wall_clock_exceeds(self):
        """Raises QuotaExceededError when wall-clock quota is exceeded.

        Uses mock to simulate time passing beyond the limit.
        """
        quota = ResourceQuota(max_wall_clock_seconds=5.0)
        enforcer = QuotaEnforcer(quota)
        enforcer.mark_start()

        # Patch time.monotonic so that the next call returns start + 10s
        original_start = enforcer._start_time
        with patch("grdl_rt.execution.quota.time.monotonic", return_value=original_start + 10.0):
            with pytest.raises(QuotaExceededError) as exc_info:
                enforcer.check_wall_clock()

            assert exc_info.value.quota_type == "wall_clock"
            assert exc_info.value.limit == 5.0
            assert exc_info.value.actual == 10.0


# ======================================================================
# QuotaEnforcer — no-limit skip behavior
# ======================================================================


class TestQuotaEnforcerNoLimit:
    def test_quota_enforcer_no_limit_skips_memory(self, grayscale_8x8):
        """No memory limit means check_before_execution is a no-op."""
        quota = ResourceQuota()  # all None
        enforcer = QuotaEnforcer(quota)
        # Should not raise regardless of input size
        enforcer.check_before_execution(grayscale_8x8, n_steps=1000)

    def test_quota_enforcer_no_limit_skips_wall_clock(self):
        """No wall-clock limit means check_wall_clock is a no-op."""
        quota = ResourceQuota()  # all None
        enforcer = QuotaEnforcer(quota)
        enforcer.mark_start()
        # Should not raise even if we pretend a lot of time has passed
        with patch("grdl_rt.execution.quota.time.monotonic", return_value=1e9):
            enforcer.check_wall_clock()

    def test_quota_enforcer_no_limit_skips_monitoring(self):
        """No memory limit means start_monitoring does not launch a thread."""
        quota = ResourceQuota()  # all None
        enforcer = QuotaEnforcer(quota)
        enforcer.start_monitoring()
        assert enforcer._monitor_thread is None
        enforcer.stop_monitoring()


# ======================================================================
# QuotaEnforcer — memory violation flag
# ======================================================================


class TestQuotaEnforcerMemoryViolation:
    def test_quota_enforcer_memory_violation_flag(self):
        """Setting the violation event directly causes check_memory_violation
        to raise QuotaExceededError with the recorded detail.
        """
        quota = ResourceQuota(max_memory_bytes=1_000_000)
        enforcer = QuotaEnforcer(quota)

        # Simulate what the background monitor thread would do
        enforcer._violation_detail = str(2_000_000)
        enforcer._violation_event.set()

        with pytest.raises(QuotaExceededError) as exc_info:
            enforcer.check_memory_violation()

        assert exc_info.value.quota_type == "memory_runtime"
        assert exc_info.value.limit == 1_000_000
        assert exc_info.value.actual == str(2_000_000)

    def test_quota_enforcer_memory_violation_not_set(self):
        """No violation event means check_memory_violation does not raise."""
        quota = ResourceQuota(max_memory_bytes=1_000_000)
        enforcer = QuotaEnforcer(quota)
        # Event is not set — should pass silently
        enforcer.check_memory_violation()


# ======================================================================
# QuotaEnforcer — background memory monitoring thread
# ======================================================================


class TestQuotaMonitoring:
    def test_start_monitoring_psutil_unavailable(self):
        """start_monitoring() is a no-op when psutil is not installed."""
        with patch("grdl_rt.execution.quota._psutil", None):
            enforcer = QuotaEnforcer(ResourceQuota(max_memory_bytes=10 * 1024**3))
            enforcer.start_monitoring()
            assert enforcer._monitor_thread is None

    def test_start_and_stop_monitoring(self):
        """start_monitoring() spawns a daemon thread; stop_monitoring() joins it."""
        enforcer = QuotaEnforcer(ResourceQuota(max_memory_bytes=10 * 1024**3))
        enforcer.start_monitoring()
        assert enforcer._monitor_thread is not None
        enforcer.stop_monitoring()
        assert enforcer._monitor_thread is None

    def test_monitor_detects_violation(self):
        """_monitor_loop sets the violation event when RSS exceeds the limit."""
        # A limit of 1 byte is always exceeded by any real process
        enforcer = QuotaEnforcer(ResourceQuota(max_memory_bytes=1))
        enforcer.start_monitoring()
        triggered = enforcer._violation_event.wait(timeout=5.0)
        enforcer.stop_monitoring()
        assert triggered, "violation event was not set within timeout"
        assert enforcer._violation_event.is_set()
        assert enforcer._violation_detail is not None

    def test_quota_property(self):
        """quota property returns the ResourceQuota passed at construction."""
        q = ResourceQuota(max_memory_bytes=1_000_000)
        enforcer = QuotaEnforcer(q)
        assert enforcer.quota is q
