"""
Memory Sampler — Background-thread tracemalloc time-series collector.

Provides ``MemorySampler``, a lightweight background thread that polls
``tracemalloc.get_traced_memory()`` at a configurable interval and stores
a corrected time-series of ``(timestamp, memory_bytes)`` samples.

The sampler subtracts its own internal storage overhead from every
reading so that only the *pipeline's* memory is reflected in the
recorded values.

Typical usage::

    sampler = MemorySampler(interval_s=0.001)
    sampler.start()

    # … run pipeline …

    timeline = sampler.stop()
    peak = sampler.peak_between(t_start, t_end)

Author
------
Claude Code (Anthropic)

Contributor
-----------
Ava Courtney

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-19

Modified
--------
2026-03-19
"""

# Standard library
import array
import sys
import threading
import time
import tracemalloc
from typing import Any


class MemoryTimeline:
    """Immutable snapshot of a completed memory sampling session.

    Attributes
    ----------
    timestamps : list[float]
        ``time.perf_counter()`` values for each sample, relative to the
        pipeline reference time (``t0``).
    values : list[int]
        Corrected memory in bytes at each sample point.
    t0 : float
        Absolute ``perf_counter`` value at sampling start.  Step
        wall-clock timestamps can be converted to timeline-relative
        offsets by subtracting this value.
    """

    __slots__ = ("timestamps", "values", "t0")

    def __init__(
        self,
        timestamps: list[float],
        values: list[int],
        t0: float,
    ) -> None:
        self.timestamps = timestamps
        self.values = values
        self.t0 = t0

    # -- Queries -----------------------------------------------------------

    def peak(self) -> int:
        """Return the overall peak memory across the entire timeline."""
        return max(self.values) if self.values else 0

    def peak_between(self, t_start: float, t_end: float) -> int:
        """Return peak memory between two *absolute* ``perf_counter`` times.

        Parameters
        ----------
        t_start, t_end : float
            Absolute ``time.perf_counter()`` boundaries.

        Returns
        -------
        int
            Maximum corrected memory observed in ``[t_start, t_end]``.
            Returns 0 if no samples fall in the window.
        """
        rel_start = t_start - self.t0
        rel_end = t_end - self.t0
        peak = 0
        for i, t in enumerate(self.timestamps):
            if rel_start <= t <= rel_end and self.values[i] > peak:
                peak = self.values[i]
        return peak

    def value_at(self, t: float) -> int:
        """Return the memory value of the sample closest to *t*.

        Parameters
        ----------
        t : float
            Absolute ``time.perf_counter()`` value.

        Returns
        -------
        int
            Memory in bytes at the nearest sample.  Returns 0 if the
            timeline is empty.
        """
        if not self.timestamps:
            return 0
        rel = t - self.t0
        best_idx = 0
        best_dist = abs(self.timestamps[0] - rel)
        for i in range(1, len(self.timestamps)):
            dist = abs(self.timestamps[i] - rel)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return self.values[best_idx]

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        The timeline is stored as two parallel lists to keep the payload
        compact (no per-sample dict overhead).
        """
        return {
            "t0": self.t0,
            "timestamps": self.timestamps,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryTimeline":
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            timestamps=data["timestamps"],
            values=data["values"],
            t0=data["t0"],
        )

    def __len__(self) -> int:
        return len(self.timestamps)


class MemorySampler:
    """Background-thread memory sampler backed by ``tracemalloc``.

    The sampler owns the ``tracemalloc.start()`` / ``stop()`` lifecycle
    for the duration of a pipeline execution.  Internal bookkeeping
    arrays are allocated *after* ``tracemalloc.start()`` and their
    footprint is subtracted from every reading so that the recorded
    timeline reflects only the pipeline's memory.

    Parameters
    ----------
    interval_s : float
        Sampling interval in seconds.  Default ``0.001`` (1 ms).
    """

    def __init__(self, interval_s: float = 0.001) -> None:
        self.interval_s = interval_s
        self._timestamps: array.array[float] | None = None
        self._mem_values: array.array[float] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._header_overhead: int = 0
        self._t0: float = 0.0
        self._tracemalloc_was_running: bool = False

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start ``tracemalloc``, allocate buffers, and begin sampling."""
        self._tracemalloc_was_running = tracemalloc.is_tracing()
        if not self._tracemalloc_was_running:
            tracemalloc.start()

        # Allocate sampling buffers while tracemalloc is active so we
        # can measure (and later subtract) their exact footprint.
        before = tracemalloc.get_traced_memory()[0]
        self._timestamps = array.array("d")
        self._mem_values = array.array("d")
        after = tracemalloc.get_traced_memory()[0]

        # The gap between tracemalloc's view and sys.getsizeof is a
        # constant per-object header cost.  Measured once, applied on
        # every sample.
        self._header_overhead = (after - before) - (
            sys.getsizeof(self._timestamps) + sys.getsizeof(self._mem_values)
        )

        self._t0 = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="grdl-mem-sampler"
        )
        self._thread.start()

    def stop(self) -> MemoryTimeline:
        """Stop sampling and return the collected timeline.

        Also stops ``tracemalloc`` if it was not running before
        :meth:`start` was called.

        Returns
        -------
        MemoryTimeline
            Immutable timeline of corrected memory samples.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Build the immutable result before tearing down buffers.
        ts = self._timestamps
        mv = self._mem_values
        timeline = MemoryTimeline(
            timestamps=list(ts) if ts else [],
            values=[int(v) for v in mv] if mv else [],
            t0=self._t0,
        )

        # Clean up
        self._timestamps = None
        self._mem_values = None

        if not self._tracemalloc_was_running:
            tracemalloc.stop()

        return timeline

    # -- Internal ----------------------------------------------------------

    def _own_footprint(self) -> int:
        """Actual bytes consumed by the sampling arrays (tracemalloc's view)."""
        ts = self._timestamps
        mv = self._mem_values
        if ts is None or mv is None:
            return 0
        return (
            sys.getsizeof(ts)
            + sys.getsizeof(mv)
            + self._header_overhead
        )

    def _sample_loop(self) -> None:
        """Polling loop executed in the background thread."""
        stop = self._stop_event
        interval = self.interval_s
        t0 = self._t0
        ts = self._timestamps
        mv = self._mem_values

        assert ts is not None and mv is not None  # guaranteed by start()

        while not stop.is_set():
            # 1. Measure our own footprint BEFORE appending.
            overhead = self._own_footprint()
            # 2. Read process-wide traced memory.
            raw = tracemalloc.get_traced_memory()[0]
            # 3. Corrected = raw minus our arrays' footprint.
            corrected = raw - overhead
            # 4. Append (may trigger realloc — next iteration accounts
            #    for the new buffer size via _own_footprint).
            ts.append(time.perf_counter() - t0)
            mv.append(float(max(0, corrected)))

            stop.wait(interval)
