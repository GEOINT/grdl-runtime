# -*- coding: utf-8 -*-
"""
Runner — Background execution engine for the grdl-rt GUI.

Wraps workflow loading, image reading, execution, and accuracy
computation in a daemon thread.  Communicates back to the UI thread
via a thread-safe queue polled by ``tkinter.after()``.

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
2026-02-13
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from grdl_rt.ui._accuracy import AccuracyReport, compute_accuracy
from grdl_rt.ui._importer import (
    classify_py_file,
    discover_processors_in_module,
    discover_workflow_in_module,
    extract_tunable_params,
)

# Lazy imports for grdl-runtime APIs (imported inside methods to keep
# module-level imports light and to avoid circular-import issues).

# ── Message types ────────────────────────────────────────────────────


@dataclass
class RunRequest:
    """Specification for a single GUI run."""

    input_paths: List[Path]
    workflow_source: Path
    is_component: bool = False
    component_class: Optional[str] = None
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prefer_gpu: bool = False
    max_workers: int = 1
    output_dir: Optional[Path] = None
    ground_truth_path: Optional[Path] = None
    enable_checkpointing: bool = False


@dataclass
class RunResult:
    """Aggregated outcome of a run."""

    workflow_results: list = field(default_factory=list)
    input_paths: List[Path] = field(default_factory=list)
    accuracy: Optional[AccuracyReport] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


# Queue message tags
_MSG_PROGRESS = "progress"
_MSG_LOG = "log"
_MSG_COMPLETE = "complete"


# ── Image I/O ────────────────────────────────────────────────────────


def _read_input(input_path: Path) -> np.ndarray:
    """Read an image file into a numpy array.

    Mirrors the pattern used in ``grdl_rt.cli._read_input``.
    """
    ext = input_path.suffix.lower()
    if ext == ".npy":
        return np.load(str(input_path))

    try:
        from grdl.IO import open_image
        with open_image(input_path) as reader:
            return reader.read_full()
    except ImportError:
        raise ImportError(
            f"grdl.IO is required to read '{ext}' files.  "
            "Install grdl or convert to .npy first."
        )


def _save_output(
    result: Any,
    output_path: Path,
) -> None:
    """Write a result array to disk."""
    try:
        from grdl.IO import write as io_write
        io_write(result, output_path)
    except (ImportError, Exception):
        # Fallback to numpy
        np.save(str(output_path), result)


# ── WorkflowRunner ───────────────────────────────────────────────────


class WorkflowRunner:
    """Manages background workflow execution for the GUI.

    Parameters
    ----------
    on_progress : callable(fraction: float, message: str)
        Called on the main thread with progress updates.
    on_log : callable(message: str, level: str)
        Called on the main thread for log messages.
    on_complete : callable(result: RunResult)
        Called on the main thread when the run finishes.
    """

    def __init__(
        self,
        on_progress: Callable[[float, str], None],
        on_log: Callable[[str, str], None],
        on_complete: Callable[[RunResult], None],
    ) -> None:
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_complete = on_complete
        self._queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._polling = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── public control ───────────────────────────────────────────

    def start(self, request: RunRequest, root) -> None:
        """Launch background execution.

        Parameters
        ----------
        request : RunRequest
            The run specification.
        root : tk.Tk
            The Tk root widget (for ``after()`` polling).
        """
        if self.is_running:
            return
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(request,), daemon=True,
        )
        self._thread.start()
        self._start_polling(root)

    def cancel(self) -> None:
        """Request cancellation (checked between images)."""
        self._cancel_event.set()

    # ── queue polling ────────────────────────────────────────────

    def _start_polling(self, root) -> None:
        if self._polling:
            return
        self._polling = True
        self._poll(root)

    def _poll(self, root) -> None:
        try:
            while True:
                tag, payload = self._queue.get_nowait()
                if tag == _MSG_PROGRESS:
                    self._on_progress(*payload)
                elif tag == _MSG_LOG:
                    self._on_log(*payload)
                elif tag == _MSG_COMPLETE:
                    self._on_complete(payload)
                    self._polling = False
                    return
        except queue.Empty:
            pass
        root.after(100, self._poll, root)

    # ── queue helpers ────────────────────────────────────────────

    def _emit_progress(self, fraction: float, message: str) -> None:
        self._queue.put((_MSG_PROGRESS, (fraction, message)))

    def _emit_log(self, message: str, level: str = "info") -> None:
        self._queue.put((_MSG_LOG, (message, level)))

    def _emit_complete(self, result: RunResult) -> None:
        self._queue.put((_MSG_COMPLETE, result))

    # ── background thread entry point ────────────────────────────

    def _run(self, request: RunRequest) -> None:
        t0 = time.perf_counter()
        try:
            result = self._execute(request)
            result.elapsed_seconds = time.perf_counter() - t0
            self._emit_complete(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self._emit_log(f"Fatal error: {exc}\n{tb}", "error")
            self._emit_complete(RunResult(
                error=str(exc),
                elapsed_seconds=time.perf_counter() - t0,
            ))

    def _execute(self, req: RunRequest) -> RunResult:
        from grdl_rt.api import load_workflow
        from grdl_rt.execution.builder import Workflow
        from grdl_rt.execution.executor import WorkflowExecutor
        from grdl_rt.execution.gpu import GpuBackend

        suffix = req.workflow_source.suffix.lower()
        workflow_results: list = []

        # ── Load workflow / component ────────────────────────────
        if suffix in (".yaml", ".yml"):
            self._emit_log(f"Loading YAML workflow: {req.workflow_source.name}")
            wf_def = load_workflow(req.workflow_source)

            # Validate
            errors = wf_def.validate()
            if errors:
                for err in errors:
                    self._emit_log(
                        f"Validation: [{err.code}] {err.message}", "warn",
                    )

            # Apply param overrides
            for step in wf_def.steps:
                step_key = getattr(step, "processor_name", None) or getattr(step, "id", None)
                if step_key and step_key in req.params:
                    step.params.update(req.params[step_key])

            gpu = GpuBackend(prefer_gpu=req.prefer_gpu)
            executor = WorkflowExecutor(wf_def, gpu=gpu)

            workflow_results = self._run_images(
                req, executor=executor,
            )

        elif suffix == ".py":
            file_kind = classify_py_file(req.workflow_source)

            if file_kind == "workflow":
                self._emit_log(f"Loading Python workflow: {req.workflow_source.name}")
                wf = discover_workflow_in_module(req.workflow_source)

                if isinstance(wf, Workflow):
                    workflow_results = self._run_images(
                        req, builder=wf,
                    )
                else:
                    # WorkflowDefinition from @workflow decorator
                    gpu = GpuBackend(prefer_gpu=req.prefer_gpu)
                    executor = WorkflowExecutor(wf, gpu=gpu)
                    workflow_results = self._run_images(
                        req, executor=executor,
                    )
            else:
                # Bare component
                self._emit_log(f"Loading component: {req.workflow_source.name}")
                processors = discover_processors_in_module(req.workflow_source)
                if not processors:
                    raise ValueError(
                        f"No processor classes found in {req.workflow_source.name}"
                    )

                # Find the selected component class
                cls = None
                if req.component_class:
                    for name, c in processors:
                        if name == req.component_class:
                            cls = c
                            break
                if cls is None:
                    cls = processors[0][1]

                cls_name = cls.__name__
                params = req.params.get(cls_name, {})
                self._emit_log(f"Using processor: {cls_name}")

                # Build single-step workflow
                wf = Workflow(f"UI-{cls_name}").step(cls, **params)
                workflow_results = self._run_images(
                    req, builder=wf,
                )
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # ── Accuracy comparison ──────────────────────────────────
        accuracy = None
        if req.ground_truth_path and req.ground_truth_path.exists():
            self._emit_log("Computing accuracy against ground truth\u2026")
            try:
                accuracy = compute_accuracy(
                    workflow_results,
                    req.ground_truth_path,
                )
                self._emit_log(
                    f"Accuracy: P={accuracy.precision:.3f}  "
                    f"R={accuracy.recall:.3f}  F1={accuracy.f1:.3f}  "
                    f"mIoU={accuracy.mean_iou:.3f}",
                    "success",
                )
            except Exception as exc:
                self._emit_log(f"Accuracy computation failed: {exc}", "warn")

        return RunResult(
            workflow_results=workflow_results,
            input_paths=req.input_paths,
            accuracy=accuracy,
        )

    # ── Image iteration ──────────────────────────────────────────

    def _run_images(
        self,
        req: RunRequest,
        *,
        executor=None,
        builder=None,
    ) -> list:
        """Run the workflow on each input image.

        Uses ``ThreadPoolExecutor`` for concurrent processing when
        ``req.max_workers > 1``.
        """
        n = len(req.input_paths)
        if n == 0:
            self._emit_log("No input images selected.", "warn")
            return []

        self._emit_log(f"Processing {n} image(s)\u2026")
        results: list = [None] * n
        completed = 0

        def _process_one(idx: int, path: Path):
            nonlocal completed
            if self._cancel_event.is_set():
                return None

            self._emit_log(f"[{idx + 1}/{n}] Reading {path.name}\u2026")

            # Progress callback for this image
            def _step_progress(frac: float) -> None:
                img_frac = (completed + frac) / n
                self._emit_progress(img_frac, f"{path.name} ({frac:.0%})")

            try:
                if builder is not None:
                    # Workflow builder: can accept filepath directly
                    wr = builder.execute(
                        str(path),
                        prefer_gpu=req.prefer_gpu,
                        progress_callback=_step_progress,
                    )
                else:
                    # WorkflowExecutor: needs ndarray
                    source = _read_input(path)
                    self._emit_log(
                        f"  Loaded: shape={source.shape} dtype={source.dtype}",
                    )
                    wr = executor.execute(
                        source,
                        progress_callback=_step_progress,
                    )

                # Save output if requested
                if req.output_dir:
                    out_path = req.output_dir / f"{path.stem}_output.npy"
                    _save_output(wr.result, out_path)
                    self._emit_log(f"  Output saved: {out_path.name}")

                self._emit_log(
                    f"  Done in {wr.metrics.total_wall_time_s:.2f}s",
                    "success",
                )
                return wr

            except Exception as exc:
                self._emit_log(f"  Error: {exc}", "error")
                return None

        if req.max_workers > 1 and n > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=req.max_workers) as pool:
                futures = {
                    pool.submit(_process_one, i, p): i
                    for i, p in enumerate(req.input_paths)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        pass
                    completed += 1
                    self._emit_progress(
                        completed / n,
                        f"Completed {completed}/{n}",
                    )
        else:
            # Sequential execution
            for i, path in enumerate(req.input_paths):
                if self._cancel_event.is_set():
                    self._emit_log("Run cancelled by user.", "warn")
                    break
                results[i] = _process_one(i, path)
                completed += 1
                self._emit_progress(completed / n, f"Completed {completed}/{n}")

        # Filter out None (failed / cancelled)
        return [r for r in results if r is not None]
