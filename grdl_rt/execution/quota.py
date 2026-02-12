# -*- coding: utf-8 -*-
"""
Resource Quota System — per-workflow memory, CPU, GPU, and wall-clock limits.

Provides ``ResourceQuota`` (immutable quota specification) and
``QuotaEnforcer`` (runtime enforcement with pre-flight checks,
inter-step wall-clock checks, and background memory monitoring).

The enforcer reuses ``resilience.estimate_memory()`` for memory
estimation, sharing code with the existing memory pre-flight system.

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
2026-02-11
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from grdl_rt.execution.context import get_logger
from grdl_rt.execution.errors import QuotaExceededError
from grdl_rt.execution.resilience import estimate_memory

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResourceQuota:
    """Per-workflow resource limits.

    All fields are optional.  ``None`` means "no limit" for that
    resource.

    Attributes
    ----------
    max_memory_bytes : int, optional
        Maximum RSS memory in bytes the workflow may use.
    max_cpu_percent : float, optional
        Maximum CPU usage percentage (informational — not enforced
        at the OS level).
    max_gpu_memory_bytes : int, optional
        Maximum GPU memory in bytes.
    max_wall_clock_seconds : float, optional
        Maximum wall-clock seconds for the entire workflow execution.
    """

    max_memory_bytes: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_gpu_memory_bytes: Optional[int] = None
    max_wall_clock_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_memory_bytes is not None and self.max_memory_bytes <= 0:
            raise ValueError(
                f"max_memory_bytes must be positive, got {self.max_memory_bytes}"
            )
        if self.max_cpu_percent is not None and self.max_cpu_percent <= 0:
            raise ValueError(
                f"max_cpu_percent must be positive, got {self.max_cpu_percent}"
            )
        if self.max_gpu_memory_bytes is not None and self.max_gpu_memory_bytes <= 0:
            raise ValueError(
                f"max_gpu_memory_bytes must be positive, got {self.max_gpu_memory_bytes}"
            )
        if self.max_wall_clock_seconds is not None and self.max_wall_clock_seconds <= 0:
            raise ValueError(
                f"max_wall_clock_seconds must be positive, got {self.max_wall_clock_seconds}"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "max_memory_bytes": self.max_memory_bytes,
            "max_cpu_percent": self.max_cpu_percent,
            "max_gpu_memory_bytes": self.max_gpu_memory_bytes,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResourceQuota:
        """Deserialize from dictionary."""
        return cls(
            max_memory_bytes=data.get("max_memory_bytes"),
            max_cpu_percent=data.get("max_cpu_percent"),
            max_gpu_memory_bytes=data.get("max_gpu_memory_bytes"),
            max_wall_clock_seconds=data.get("max_wall_clock_seconds"),
        )


class QuotaEnforcer:
    """Enforces a ``ResourceQuota`` during workflow execution.

    Usage::

        enforcer = QuotaEnforcer(quota)
        enforcer.check_before_execution(source_array, n_steps)
        enforcer.start_monitoring()
        try:
            for step in steps:
                enforcer.check_wall_clock()
                enforcer.check_memory_violation()
                # ... execute step ...
        finally:
            enforcer.stop_monitoring()

    Parameters
    ----------
    quota : ResourceQuota
        The quota to enforce.
    """

    def __init__(self, quota: ResourceQuota) -> None:
        self._quota = quota
        self._start_time: float = 0.0
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._violation_event = threading.Event()
        self._violation_detail: Optional[str] = None

    @property
    def quota(self) -> ResourceQuota:
        """Return the quota being enforced."""
        return self._quota

    def check_before_execution(
        self,
        input_array: np.ndarray,
        n_steps: int,
        multiplier: float = 1.5,
    ) -> None:
        """Pre-flight check: will estimated memory exceed the quota?

        Reuses ``resilience.estimate_memory()`` for the estimate.

        Raises
        ------
        QuotaExceededError
            If the estimated memory exceeds ``max_memory_bytes``.
        """
        if self._quota.max_memory_bytes is None:
            return

        estimated = estimate_memory(input_array, n_steps, multiplier)
        if estimated > self._quota.max_memory_bytes:
            raise QuotaExceededError(
                "memory_preflight",
                limit=self._quota.max_memory_bytes,
                actual=estimated,
            )
        logger.debug(
            "quota_memory_preflight_ok",
            estimated_bytes=estimated,
            limit_bytes=self._quota.max_memory_bytes,
        )

    def mark_start(self) -> None:
        """Record the execution start time for wall-clock checks."""
        self._start_time = time.monotonic()

    def check_wall_clock(self) -> None:
        """Check whether the wall-clock quota has been exceeded.

        Called between steps by the executor.

        Raises
        ------
        QuotaExceededError
            If elapsed time exceeds ``max_wall_clock_seconds``.
        """
        if self._quota.max_wall_clock_seconds is None:
            return
        elapsed = time.monotonic() - self._start_time
        if elapsed > self._quota.max_wall_clock_seconds:
            raise QuotaExceededError(
                "wall_clock",
                limit=self._quota.max_wall_clock_seconds,
                actual=round(elapsed, 2),
            )

    def check_memory_violation(self) -> None:
        """Check whether the background memory monitor has detected a violation.

        Raises
        ------
        QuotaExceededError
            If the monitor detected RSS exceeding ``max_memory_bytes``.
        """
        if self._violation_event.is_set():
            raise QuotaExceededError(
                "memory_runtime",
                limit=self._quota.max_memory_bytes,
                actual=self._violation_detail,
            )

    def start_monitoring(self) -> None:
        """Start background memory monitoring thread.

        Polls ``psutil.Process().memory_info().rss`` every second.
        Sets an event if ``max_memory_bytes`` is exceeded.
        """
        if self._quota.max_memory_bytes is None:
            return
        if _psutil is None:
            logger.debug("quota_monitor_skipped", reason="psutil not available")
            return

        self._stop_event.clear()
        self._violation_event.clear()
        self._violation_detail = None

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="grdl_quota_monitor",
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop the background memory monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        """Background loop: poll RSS every second."""
        process = _psutil.Process()
        limit = self._quota.max_memory_bytes

        while not self._stop_event.is_set():
            try:
                rss = process.memory_info().rss
                if rss > limit:
                    self._violation_detail = str(rss)
                    self._violation_event.set()
                    logger.warning(
                        "quota_memory_exceeded",
                        rss_bytes=rss,
                        limit_bytes=limit,
                    )
                    return
            except Exception:
                pass
            self._stop_event.wait(timeout=1.0)
