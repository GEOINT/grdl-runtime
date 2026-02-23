"""
Resilience primitives — retry, timeout, circuit breaker, memory estimation,
graceful shutdown, and tiling strategy stub.

These building blocks are used by both ``WorkflowExecutor`` (string-based)
and ``Workflow`` (fluent builder) execution paths to make step execution
production-hardened.

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

import abc
import json
import os
import random
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]

import contextlib

from grdl_rt.execution.context import get_logger
from grdl_rt.execution.errors import (
    MemoryThresholdError,
    StepRetryExhaustedError,
    StepTimeoutError,
)

logger = get_logger(__name__)


# ======================================================================
# Retry Policy
# ======================================================================


@dataclass(frozen=True)
class RetryPolicy:
    """Per-step retry configuration.

    Attributes
    ----------
    max_retries : int
        Maximum number of retry attempts (0 = fail immediately).
    backoff_base : float
        Base delay in seconds for exponential backoff.
    backoff_max : float
        Maximum delay in seconds between retries.
    retryable_exceptions : Tuple[str, ...]
        Exception class names that are eligible for retry.
    """

    max_retries: int = 0
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    retryable_exceptions: tuple[str, ...] = ("RuntimeError", "OSError", "MemoryError")

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.backoff_base <= 0:
            raise ValueError(f"backoff_base must be > 0, got {self.backoff_base}")
        if self.backoff_max <= 0:
            raise ValueError(f"backoff_max must be > 0, got {self.backoff_max}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for YAML/JSON storage."""
        return {
            "max_retries": self.max_retries,
            "backoff_base": self.backoff_base,
            "backoff_max": self.backoff_max,
            "retryable_exceptions": list(self.retryable_exceptions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryPolicy:
        """Deserialize from dictionary."""
        exceptions = data.get("retryable_exceptions", cls.retryable_exceptions)
        if isinstance(exceptions, list):
            exceptions = tuple(exceptions)
        return cls(
            max_retries=data.get("max_retries", 0),
            backoff_base=data.get("backoff_base", 1.0),
            backoff_max=data.get("backoff_max", 60.0),
            retryable_exceptions=exceptions,
        )


def _is_retryable(exc: Exception, retryable_names: tuple[str, ...]) -> bool:
    """Check whether *exc* matches any of the retryable exception names."""
    return any(cls.__name__ in retryable_names for cls in type(exc).__mro__)


def _backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """Compute exponential backoff with jitter for the given attempt."""
    raw = policy.backoff_base * (2**attempt)
    capped = min(raw, policy.backoff_max)
    jitter = random.uniform(0.5, 1.5)
    return capped * jitter


def execute_with_retry(
    fn: Callable[[], Any],
    policy: RetryPolicy,
    step_name: str,
    log: Any = None,
) -> Any:
    """Execute *fn* with retry logic according to *policy*.

    Parameters
    ----------
    fn : callable
        Zero-argument callable that performs the step work.
    policy : RetryPolicy
        Retry configuration.
    step_name : str
        Human-readable step name for logging.
    log : structlog logger, optional
        Logger instance.  Falls back to module logger.

    Returns
    -------
    Any
        The return value of *fn* on success.

    Raises
    ------
    StepRetryExhaustedError
        If all attempts (1 initial + max_retries) are exhausted.
    """
    log = log or logger
    attempts: list[Exception] = []
    total_attempts = 1 + policy.max_retries

    for attempt in range(total_attempts):
        try:
            return fn()
        except Exception as exc:
            attempts.append(exc)

            if attempt >= policy.max_retries:
                break

            if not _is_retryable(exc, policy.retryable_exceptions):
                log.warning(
                    "step_not_retryable",
                    step_name=step_name,
                    attempt=attempt + 1,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                break

            delay = _backoff_delay(attempt, policy)
            log.warning(
                "step_retry",
                step_name=step_name,
                attempt=attempt + 1,
                max_retries=policy.max_retries,
                wait_seconds=round(delay, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            time.sleep(delay)

    raise StepRetryExhaustedError(step_name, attempts)


# ======================================================================
# Step Timeout
# ======================================================================


def execute_with_timeout(
    fn: Callable[[], Any],
    timeout_seconds: float,
    step_name: str,
) -> Any:
    """Run *fn* in a worker thread with a timeout.

    Parameters
    ----------
    fn : callable
        Zero-argument callable.
    timeout_seconds : float
        Maximum wall-clock seconds to wait.
    step_name : str
        Human-readable step name for error messages.

    Returns
    -------
    Any
        Return value of *fn*.

    Raises
    ------
    StepTimeoutError
        If *fn* does not complete within *timeout_seconds*.

    Notes
    -----
    On timeout the worker thread is **not** killed (Python cannot
    forcibly terminate threads).  GPU memory cleanup should be
    performed by the caller's finally block.  Each call uses a fresh
    thread pool to avoid blocking from prior timed-out tasks.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grdl_timeout")
    try:
        future: Future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            future.cancel()
            _release_gpu_memory()
            raise StepTimeoutError(step_name, timeout_seconds) from None
    finally:
        pool.shutdown(wait=False)


def _release_gpu_memory() -> None:
    """Best-effort GPU memory cleanup after timeout or failure."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
    except ImportError:
        pass


# ======================================================================
# Circuit Breaker
# ======================================================================


class CircuitBreaker:
    """Track consecutive failures per processor and refuse execution
    during a cooldown period.

    Disabled by default (``failure_threshold=0``).  When enabled, after
    *failure_threshold* consecutive failures for the same processor name,
    the breaker opens and remains open for *cooldown_seconds*.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before opening.  0 disables.
    cooldown_seconds : float
        Seconds the circuit stays open before allowing attempts.
    """

    def __init__(
        self,
        failure_threshold: int = 0,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.failure_threshold > 0

    def is_open(self, processor_name: str) -> bool:
        """Return True if the circuit is open (execution should be refused)."""
        if not self.enabled:
            return False
        with self._lock:
            opened = self._opened_at.get(processor_name)
            if opened is None:
                return False
            if time.monotonic() - opened >= self.cooldown_seconds:
                # Cooldown elapsed — half-open, allow a probe
                del self._opened_at[processor_name]
                self._consecutive_failures[processor_name] = 0
                return False
            return True

    def record_success(self, processor_name: str) -> None:
        """Reset failure counter on success."""
        if not self.enabled:
            return
        with self._lock:
            self._consecutive_failures.pop(processor_name, None)
            self._opened_at.pop(processor_name, None)

    def record_failure(self, processor_name: str) -> None:
        """Increment failure counter; open the circuit if threshold is reached."""
        if not self.enabled:
            return
        with self._lock:
            count = self._consecutive_failures.get(processor_name, 0) + 1
            self._consecutive_failures[processor_name] = count
            if count >= self.failure_threshold:
                self._opened_at[processor_name] = time.monotonic()
                logger.warning(
                    "circuit_breaker_open",
                    processor_name=processor_name,
                    consecutive_failures=count,
                    cooldown_seconds=self.cooldown_seconds,
                )


# ======================================================================
# Memory Estimation
# ======================================================================


def estimate_memory(
    input_array: np.ndarray,
    n_steps: int,
    multiplier: float = 1.5,
) -> int:
    """Estimate peak memory needed to execute a workflow.

    Parameters
    ----------
    input_array : np.ndarray
        The input image array.
    n_steps : int
        Number of processing steps.
    multiplier : float
        Configurable safety multiplier.

    Returns
    -------
    int
        Estimated memory in bytes.
    """
    return int(input_array.nbytes * multiplier * max(n_steps, 1))


def check_memory(
    estimated_bytes: int,
    warn_threshold: float = 0.80,
    abort_threshold: float = 0.95,
) -> tuple[bool, bool]:
    """Compare estimated memory against available system RAM.

    Parameters
    ----------
    estimated_bytes : int
        Estimated memory requirement.
    warn_threshold : float
        Fraction of available memory at which to warn (default 0.80).
    abort_threshold : float
        Fraction of available memory at which to abort (default 0.95).

    Returns
    -------
    Tuple[bool, bool]
        ``(should_warn, should_abort)``
    """
    if _psutil is None:
        logger.debug("psutil not available; skipping memory check")
        return False, False

    available = _psutil.virtual_memory().available

    if available <= 0:
        return False, False

    ratio = estimated_bytes / available
    return ratio > warn_threshold, ratio > abort_threshold


def run_memory_preflight(
    input_array: np.ndarray,
    n_steps: int,
    multiplier: float = 1.5,
    warn_threshold: float = 0.80,
    abort_threshold: float = 0.95,
    log: Any = None,
) -> int:
    """Estimate memory and warn/abort before execution.

    Parameters
    ----------
    input_array : np.ndarray
    n_steps : int
    multiplier : float
    warn_threshold : float
    abort_threshold : float
    log : optional logger

    Returns
    -------
    int
        Estimated bytes.

    Raises
    ------
    MemoryThresholdError
        If estimated usage exceeds *abort_threshold*.
    """
    log = log or logger
    est = estimate_memory(input_array, n_steps, multiplier)
    should_warn, should_abort = check_memory(est, warn_threshold, abort_threshold)

    available = _psutil.virtual_memory().available if _psutil is not None else 0

    if should_abort:
        log.error(
            "memory_abort",
            estimated_bytes=est,
            available_bytes=available,
            abort_threshold=abort_threshold,
        )
        raise MemoryThresholdError(est, available, abort_threshold)

    if should_warn:
        log.warning(
            "memory_warning",
            estimated_bytes=est,
            available_bytes=available,
            warn_threshold=warn_threshold,
        )

    return est


# ======================================================================
# Graceful Shutdown
# ======================================================================


class ShutdownCoordinator:
    """Signal handler that coordinates graceful shutdown.

    On the first SIGTERM/SIGINT, sets a flag so the executor can finish
    the current step and write a checkpoint.  On the second signal,
    forces immediate exit.

    Parameters
    ----------
    checkpoint_dir : Path, optional
        Directory for checkpoint files.  Defaults to a temp directory.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._shutdown_requested = False
        self._signal_count = 0
        self._lock = threading.Lock()
        self._checkpoint_dir = checkpoint_dir or Path.cwd()
        self._original_handlers: dict[int, Any] = {}
        self._registered = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def register(self) -> None:
        """Install signal handlers for SIGTERM and SIGINT.

        Safe to call from non-main threads (silently skips registration).
        """
        if self._registered:
            return
        try:
            self._original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_signal)

            self._original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle_signal)

            # Windows: also handle SIGBREAK if available
            sigbreak = getattr(signal, "SIGBREAK", None)
            if sigbreak is not None:
                self._original_handlers[sigbreak] = signal.getsignal(sigbreak)
                signal.signal(sigbreak, self._handle_signal)

            self._registered = True
        except ValueError:
            # signal.signal() must be called from the main thread
            logger.debug("shutdown_coordinator_skipped_non_main_thread")

    def unregister(self) -> None:
        """Restore original signal handlers."""
        if not self._registered:
            return
        for sig, handler in self._original_handlers.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)
        self._original_handlers.clear()
        self._registered = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle incoming signal."""
        with self._lock:
            self._signal_count += 1
            count = self._signal_count

        if count == 1:
            self._shutdown_requested = True
            logger.info(
                "shutdown_requested",
                signal=signum,
                message="Finishing current step before exit",
            )
        else:
            logger.warning(
                "shutdown_forced",
                signal=signum,
                signal_count=count,
            )
            os._exit(130)

    def write_checkpoint(
        self,
        workflow_dict: dict[str, Any],
        current_step_index: int,
        current_array: np.ndarray,
        run_id: str,
        step_outputs: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Write a checkpoint file for later resume.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow definition (``WorkflowDefinition.to_dict()``).
        current_step_index : int
            Index of the last completed step.
        current_array : np.ndarray
            Intermediate array after the last completed step.
        run_id : str
            Unique run identifier.
        step_outputs : list, optional
            Metadata about completed steps.

        Returns
        -------
        Path
            Path to the checkpoint directory.

        Notes
        -----
        Checkpoint format (``checkpoint.json``)::

            {
                "format_version": "1.0",
                "run_id": "...",
                "created_at": "...",
                "completed_step_index": N,
                "intermediate_path": "intermediate.npy",
                "workflow": { ... },
                "step_outputs": [ ... ]
            }

        TG4 will implement resume from this checkpoint.
        """
        ckpt_dir = self._checkpoint_dir / f"grdl_checkpoint_{run_id}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save intermediate array
        intermediate_path = ckpt_dir / "intermediate.npy"
        np.save(intermediate_path, current_array)

        # Write checkpoint metadata
        checkpoint = {
            "format_version": "1.0",
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "completed_step_index": current_step_index,
            "intermediate_path": "intermediate.npy",
            "workflow": workflow_dict,
            "step_outputs": step_outputs or [],
        }
        ckpt_path = ckpt_dir / "checkpoint.json"
        ckpt_path.write_text(
            json.dumps(checkpoint, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "checkpoint_written",
            checkpoint_dir=str(ckpt_dir),
            completed_step_index=current_step_index,
        )
        return ckpt_dir

    def cleanup_gpu(self) -> None:
        """Release GPU memory before exit."""
        _release_gpu_memory()


# ======================================================================
# Tiling Strategy (stub interface for future implementation)
# ======================================================================


class TilingStrategy(abc.ABC):
    """Abstract base for memory-constrained tiling strategies.

    When an input image is too large to process in one pass, a
    ``TilingStrategy`` breaks it into chunks that fit within a memory
    budget.  Concrete implementations will be provided in a later
    task group.

    Subclasses must implement :meth:`chunk`.
    """

    @abc.abstractmethod
    def chunk(
        self,
        input_array: np.ndarray,
        max_chunk_bytes: int,
    ) -> list[np.ndarray]:
        """Split *input_array* into chunks that each fit within
        *max_chunk_bytes*.

        Parameters
        ----------
        input_array : np.ndarray
            Full input image.
        max_chunk_bytes : int
            Maximum memory budget per chunk in bytes.

        Returns
        -------
        List[np.ndarray]
            Ordered list of non-overlapping chunks.
        """
        ...
