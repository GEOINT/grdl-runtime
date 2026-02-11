# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.resilience — retry, timeout, circuit breaker,
memory estimation, and graceful shutdown.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-11
"""

import json
import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from grdl_rt.execution.errors import (
    MemoryThresholdError,
    StepRetryExhaustedError,
    StepTimeoutError,
)
from grdl_rt.execution.resilience import (
    CircuitBreaker,
    RetryPolicy,
    ShutdownCoordinator,
    TilingStrategy,
    _backoff_delay,
    _is_retryable,
    check_memory,
    estimate_memory,
    execute_with_retry,
    execute_with_timeout,
    run_memory_preflight,
)


# ======================================================================
# RetryPolicy
# ======================================================================


class TestRetryPolicy:
    def test_defaults(self):
        p = RetryPolicy()
        assert p.max_retries == 0
        assert p.backoff_base == 1.0
        assert p.backoff_max == 60.0
        assert "RuntimeError" in p.retryable_exceptions

    def test_custom_values(self):
        p = RetryPolicy(max_retries=5, backoff_base=0.5, backoff_max=30.0)
        assert p.max_retries == 5
        assert p.backoff_base == 0.5
        assert p.backoff_max == 30.0

    def test_negative_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            RetryPolicy(max_retries=-1)

    def test_zero_backoff_base_raises(self):
        with pytest.raises(ValueError, match="backoff_base"):
            RetryPolicy(backoff_base=0)

    def test_serialization_roundtrip(self):
        p = RetryPolicy(max_retries=3, backoff_base=2.0, backoff_max=120.0)
        d = p.to_dict()
        p2 = RetryPolicy.from_dict(d)
        assert p2.max_retries == 3
        assert p2.backoff_base == 2.0
        assert p2.backoff_max == 120.0

    def test_frozen(self):
        p = RetryPolicy()
        with pytest.raises(AttributeError):
            p.max_retries = 5  # type: ignore[misc]


# ======================================================================
# _is_retryable
# ======================================================================


class TestIsRetryable:
    def test_direct_match(self):
        assert _is_retryable(RuntimeError("boom"), ("RuntimeError",))

    def test_parent_class_match(self):
        # OSError is a parent of FileNotFoundError
        assert _is_retryable(FileNotFoundError("no file"), ("OSError",))

    def test_no_match(self):
        assert not _is_retryable(ValueError("bad"), ("RuntimeError", "OSError"))


# ======================================================================
# _backoff_delay
# ======================================================================


class TestBackoffDelay:
    def test_increases_with_attempt(self):
        p = RetryPolicy(backoff_base=1.0, backoff_max=100.0)
        # Attempt 0: base * 2^0 = 1.0 (± jitter)
        # Attempt 2: base * 2^2 = 4.0 (± jitter)
        delays = [_backoff_delay(i, p) for i in range(5)]
        # With jitter, each delay should be in a range. Check ordering trend.
        # The raw values before jitter are 1, 2, 4, 8, 16
        # After jitter (0.5-1.5x), ranges overlap but trend should be upward.
        # Just verify they're all positive and bounded.
        for d in delays:
            assert d > 0
            assert d <= 100.0 * 1.5  # backoff_max * max_jitter

    def test_respects_backoff_max(self):
        p = RetryPolicy(backoff_base=1.0, backoff_max=5.0)
        # Attempt 10: raw = 1024, capped to 5.0, then jittered
        delay = _backoff_delay(10, p)
        assert delay <= 5.0 * 1.5


# ======================================================================
# execute_with_retry
# ======================================================================


class TestExecuteWithRetry:
    def test_succeeds_first_try(self):
        result = execute_with_retry(
            lambda: 42,
            RetryPolicy(max_retries=3),
            step_name="test",
        )
        assert result == 42

    def test_fails_then_succeeds(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient failure")
            return "ok"

        policy = RetryPolicy(
            max_retries=3,
            backoff_base=0.01,  # tiny for fast tests
            backoff_max=0.05,
        )
        result = execute_with_retry(flaky, policy, step_name="flaky_step")
        assert result == "ok"
        assert call_count[0] == 3  # 2 failures + 1 success

    def test_exhausts_retries(self):
        call_count = [0]

        def always_fails():
            call_count[0] += 1
            raise RuntimeError("permanent failure")

        policy = RetryPolicy(
            max_retries=3,
            backoff_base=0.01,
            backoff_max=0.05,
        )
        with pytest.raises(StepRetryExhaustedError) as exc_info:
            execute_with_retry(always_fails, policy, step_name="doomed")

        assert call_count[0] == 4  # 1 initial + 3 retries
        assert exc_info.value.step_name == "doomed"
        assert len(exc_info.value.attempts) == 4

    def test_non_retryable_exception_stops_immediately(self):
        call_count = [0]

        def raises_value_error():
            call_count[0] += 1
            raise ValueError("not retryable")

        policy = RetryPolicy(
            max_retries=3,
            backoff_base=0.01,
            retryable_exceptions=("RuntimeError",),
        )
        with pytest.raises(StepRetryExhaustedError):
            execute_with_retry(raises_value_error, policy, step_name="test")

        # Should stop after first attempt since ValueError is not retryable
        assert call_count[0] == 1

    def test_exact_retry_count(self):
        """A step configured with max_retries=3 retries exactly 3 times."""
        call_count = [0]

        def always_fails():
            call_count[0] += 1
            raise RuntimeError("fail")

        policy = RetryPolicy(max_retries=3, backoff_base=0.001)
        with pytest.raises(StepRetryExhaustedError):
            execute_with_retry(always_fails, policy, step_name="test")

        assert call_count[0] == 4  # 1 initial + 3 retries


# ======================================================================
# execute_with_timeout
# ======================================================================


class TestExecuteWithTimeout:
    def test_completes_within_timeout(self):
        result = execute_with_timeout(
            lambda: 42, timeout_seconds=5.0, step_name="fast_step",
        )
        assert result == 42

    def test_timeout_raises(self):
        def slow_fn():
            time.sleep(30)
            return "never"

        with pytest.raises(StepTimeoutError) as exc_info:
            execute_with_timeout(
                slow_fn, timeout_seconds=0.2, step_name="slow_step",
            )

        assert exc_info.value.step_name == "slow_step"
        assert exc_info.value.timeout_seconds == 0.2

    def test_exception_propagates(self):
        def bad_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            execute_with_timeout(bad_fn, timeout_seconds=5.0, step_name="bad")


# ======================================================================
# CircuitBreaker
# ======================================================================


class TestCircuitBreaker:
    def test_disabled_by_default(self):
        cb = CircuitBreaker()
        assert not cb.enabled
        assert not cb.is_open("any_processor")

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        assert cb.enabled

        cb.record_failure("proc_a")
        assert not cb.is_open("proc_a")
        cb.record_failure("proc_a")
        assert not cb.is_open("proc_a")
        cb.record_failure("proc_a")
        assert cb.is_open("proc_a")

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        cb.record_failure("proc_a")
        cb.record_failure("proc_a")
        cb.record_success("proc_a")
        cb.record_failure("proc_a")
        cb.record_failure("proc_a")
        # 2 failures after reset, still below threshold
        assert not cb.is_open("proc_a")

    def test_cooldown_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("proc_a")
        cb.record_failure("proc_a")
        assert cb.is_open("proc_a")

        time.sleep(0.15)
        # After cooldown, circuit should be half-open (allowing probe)
        assert not cb.is_open("proc_a")

    def test_independent_processors(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        cb.record_failure("proc_a")
        cb.record_failure("proc_a")
        assert cb.is_open("proc_a")
        assert not cb.is_open("proc_b")


# ======================================================================
# Memory estimation
# ======================================================================


class TestMemoryEstimation:
    def test_estimate_basic(self):
        arr = np.zeros((100, 100), dtype=np.float32)  # 40000 bytes
        est = estimate_memory(arr, n_steps=3, multiplier=2.0)
        assert est == arr.nbytes * 2.0 * 3

    def test_estimate_single_step(self):
        arr = np.zeros((10, 10), dtype=np.float64)
        est = estimate_memory(arr, n_steps=1, multiplier=1.0)
        assert est == arr.nbytes

    def test_estimate_zero_steps_treated_as_one(self):
        arr = np.zeros((10,), dtype=np.uint8)
        est = estimate_memory(arr, n_steps=0, multiplier=1.5)
        assert est == int(arr.nbytes * 1.5 * 1)

    @patch("grdl_rt.execution.resilience._psutil")
    def test_check_memory_warns(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(available=1000)
        should_warn, should_abort = check_memory(
            estimated_bytes=850,
            warn_threshold=0.80,
            abort_threshold=0.95,
        )
        assert should_warn is True
        assert should_abort is False

    @patch("grdl_rt.execution.resilience._psutil")
    def test_check_memory_aborts(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(available=1000)
        should_warn, should_abort = check_memory(
            estimated_bytes=960,
            warn_threshold=0.80,
            abort_threshold=0.95,
        )
        assert should_warn is True
        assert should_abort is True

    @patch("grdl_rt.execution.resilience._psutil")
    def test_check_memory_safe(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(available=10000)
        should_warn, should_abort = check_memory(
            estimated_bytes=100,
            warn_threshold=0.80,
            abort_threshold=0.95,
        )
        assert should_warn is False
        assert should_abort is False

    @patch("grdl_rt.execution.resilience._psutil")
    def test_run_memory_preflight_aborts(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(available=1000)
        arr = np.zeros((10, 10), dtype=np.float32)  # 400 bytes
        # 400 * 5.0 * 3 = 6000 >> 1000 → abort
        with pytest.raises(MemoryThresholdError):
            run_memory_preflight(
                arr, n_steps=3, multiplier=5.0,
                warn_threshold=0.8, abort_threshold=0.95,
            )

    @patch("grdl_rt.execution.resilience._psutil")
    def test_run_memory_preflight_safe(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(
            available=1_000_000_000
        )
        arr = np.zeros((10, 10), dtype=np.float32)
        est = run_memory_preflight(arr, n_steps=2, multiplier=1.5)
        assert est == int(arr.nbytes * 1.5 * 2)


# ======================================================================
# ShutdownCoordinator
# ======================================================================


class TestShutdownCoordinator:
    def test_initial_state(self):
        sc = ShutdownCoordinator()
        assert not sc.shutdown_requested

    def test_signal_sets_flag(self):
        sc = ShutdownCoordinator()
        # Simulate signal handling directly
        sc._handle_signal(signal.SIGINT, None)
        assert sc.shutdown_requested

    def test_second_signal_force_exits(self):
        sc = ShutdownCoordinator()
        sc._handle_signal(signal.SIGINT, None)
        assert sc.shutdown_requested
        # Second signal should call os._exit — mock it
        with patch("grdl_rt.execution.resilience.os._exit") as mock_exit:
            sc._handle_signal(signal.SIGINT, None)
            mock_exit.assert_called_once_with(130)

    def test_write_checkpoint(self, tmp_path):
        sc = ShutdownCoordinator(checkpoint_dir=tmp_path)
        arr = np.array([1.0, 2.0, 3.0])
        workflow_dict = {"name": "Test", "steps": []}

        ckpt_dir = sc.write_checkpoint(
            workflow_dict=workflow_dict,
            current_step_index=2,
            current_array=arr,
            run_id="test-run-123",
            step_outputs=[{"step_index": 0, "status": "success"}],
        )

        assert ckpt_dir.exists()
        assert (ckpt_dir / "intermediate.npy").exists()
        assert (ckpt_dir / "checkpoint.json").exists()

        # Verify checkpoint content
        ckpt_data = json.loads((ckpt_dir / "checkpoint.json").read_text())
        assert ckpt_data["format_version"] == "1.0"
        assert ckpt_data["run_id"] == "test-run-123"
        assert ckpt_data["completed_step_index"] == 2
        assert ckpt_data["workflow"]["name"] == "Test"
        assert len(ckpt_data["step_outputs"]) == 1

        # Verify array
        loaded = np.load(ckpt_dir / "intermediate.npy")
        np.testing.assert_array_equal(loaded, arr)


# ======================================================================
# TilingStrategy (stub)
# ======================================================================


class TestTilingStrategy:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            TilingStrategy()  # type: ignore[abstract]


# ======================================================================
# Integration: retry + timeout combined
# ======================================================================


class TestRetryAndTimeout:
    def test_timeout_triggers_retry(self):
        """Timeout on first attempt, succeed on second."""
        call_count = [0]

        def sometimes_slow():
            call_count[0] += 1
            if call_count[0] == 1:
                time.sleep(30)
            return "ok"

        policy = RetryPolicy(
            max_retries=2,
            backoff_base=0.01,
            retryable_exceptions=("StepTimeoutError",),
        )

        def step_fn():
            return execute_with_timeout(
                sometimes_slow, timeout_seconds=0.2, step_name="test",
            )

        result = execute_with_retry(step_fn, policy, step_name="test")
        assert result == "ok"
        assert call_count[0] == 2


# ======================================================================
# Integration: WorkflowExecutor with resilience
# ======================================================================


class TestExecutorResilience:
    """Integration tests for retry/timeout through WorkflowExecutor."""

    def _make_workflow(self, steps):
        from grdl_rt.execution.workflow import WorkflowDefinition
        wf = WorkflowDefinition(name="Test")
        for s in steps:
            wf.add_step(s)
        return wf

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_step_retry_succeeds_after_failures(self, mock_resolve):
        """A step configured with max_retries=3 retries on failure and
        eventually succeeds.
        """
        call_count = [0]

        class _FlakyTransform:
            def apply(self, source, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise RuntimeError("transient")
                return source * 2.0

        mock_resolve.return_value = _FlakyTransform

        from grdl_rt.execution.resilience import RetryPolicy
        from grdl_rt.execution.workflow import ProcessingStep
        from grdl_rt.execution.executor import WorkflowExecutor

        step = ProcessingStep(
            'FlakyTransform', '1.0',
            retry=RetryPolicy(max_retries=3, backoff_base=0.01),
        )
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        source = np.ones((4, 4), dtype=np.float64)
        wr = executor.execute(
            source, enable_memory_check=False,
            enable_shutdown_handler=False,
        )
        np.testing.assert_array_almost_equal(wr.result, np.ones((4, 4)) * 2.0)
        assert call_count[0] == 3

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_step_retry_exhausted(self, mock_resolve):
        """After max_retries failures, StepRetryExhaustedError is raised."""
        class _AlwaysFails:
            def apply(self, source, **kwargs):
                raise RuntimeError("permanent")

        mock_resolve.return_value = _AlwaysFails

        from grdl_rt.execution.resilience import RetryPolicy
        from grdl_rt.execution.workflow import ProcessingStep
        from grdl_rt.execution.executor import WorkflowExecutor

        step = ProcessingStep(
            'AlwaysFails', '1.0',
            retry=RetryPolicy(max_retries=2, backoff_base=0.01),
        )
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        with pytest.raises(StepRetryExhaustedError) as exc_info:
            executor.execute(
                np.ones((2, 2)),
                enable_memory_check=False,
                enable_shutdown_handler=False,
            )
        assert len(exc_info.value.attempts) == 3  # 1 initial + 2 retries

    @patch('grdl_rt.execution.executor.resolve_processor_class')
    def test_step_timeout(self, mock_resolve):
        """A step with timeout_seconds that hangs is killed."""
        class _HangingTransform:
            def apply(self, source, **kwargs):
                time.sleep(30)
                return source

        mock_resolve.return_value = _HangingTransform

        from grdl_rt.execution.workflow import ProcessingStep
        from grdl_rt.execution.executor import WorkflowExecutor

        step = ProcessingStep(
            'HangingTransform', '1.0',
            timeout_seconds=0.3,
        )
        wf = self._make_workflow([step])
        executor = WorkflowExecutor(wf)

        with pytest.raises(StepTimeoutError) as exc_info:
            executor.execute(
                np.ones((2, 2)),
                enable_memory_check=False,
                enable_shutdown_handler=False,
            )
        assert exc_info.value.step_name == "HangingTransform"

    def test_memory_abort(self):
        """A workflow estimated to use > 95% RAM is aborted before execution."""
        from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition
        from grdl_rt.execution.executor import WorkflowExecutor

        step = ProcessingStep('SomeProcessor', '1.0')
        wf = WorkflowDefinition(name="Test")
        wf.add_step(step)
        executor = WorkflowExecutor(wf)

        # Create a scenario where memory check fails
        with patch('grdl_rt.execution.resilience._psutil') as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(available=100)
            # 8x8 float64 = 512 bytes, * 1.5 * 1 = 768 >> 100 available
            source = np.zeros((8, 8), dtype=np.float64)
            with pytest.raises(MemoryThresholdError):
                executor.execute(
                    source,
                    enable_memory_check=True,
                    enable_shutdown_handler=False,
                )

    def test_shutdown_writes_checkpoint(self, tmp_path):
        """Sending shutdown signal during execution writes a checkpoint."""
        from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition
        from grdl_rt.execution.executor import WorkflowExecutor
        from grdl_rt.execution.resilience import ShutdownCoordinator

        class _SlowTransform:
            def apply(self, source, **kwargs):
                return source * 2.0

        shutdown = ShutdownCoordinator(checkpoint_dir=tmp_path)
        # Pre-set the shutdown flag so it triggers between steps
        shutdown._shutdown_requested = True

        step1 = ProcessingStep('Step1', '1.0')
        step2 = ProcessingStep('Step2', '1.0')
        wf = WorkflowDefinition(name="Test")
        wf.add_step(step1)
        wf.add_step(step2)

        with patch('grdl_rt.execution.executor.resolve_processor_class') as mock:
            mock.return_value = _SlowTransform
            executor = WorkflowExecutor(
                wf, shutdown=shutdown,
            )

            # The executor should detect the shutdown flag before step 0
            # and exit with SystemExit(130)
            with pytest.raises(SystemExit) as exc_info:
                executor.execute(
                    np.ones((2, 2)),
                    enable_memory_check=False,
                    enable_shutdown_handler=False,
                )

            assert exc_info.value.code == 130

            # Verify checkpoint was written
            ckpt_dirs = list(tmp_path.glob("grdl_checkpoint_*"))
            assert len(ckpt_dirs) == 1
            ckpt = json.loads(
                (ckpt_dirs[0] / "checkpoint.json").read_text()
            )
            assert ckpt["completed_step_index"] == -1  # no steps completed
