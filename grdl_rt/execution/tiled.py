"""
Tiled Execution - Run tile-safe workflow steps in parallel on halo tiles.

A linear workflow is split into *segments*.  Consecutive steps whose
processors declare a ``tile_halo`` (see ``grdl.image_processing.tiling``)
form a tiled segment: each tile is read with the summed halo, passed
through every step of the segment, cropped to its core and stitched.
Steps that need the whole image (``tile_halo`` is ``None`` or undeclared)
form global segments that run on the assembled array, so a workflow can
tile up to a global step (for example IDAN filtering) and tile again
afterwards.

Tiled steps do not get per-step retry or timeout handling, and steps
that declare ``required_bands`` always run globally.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-07
"""

from __future__ import annotations

import dataclasses
import errno
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from grdl_rt.execution.context import get_logger
from grdl_rt.execution.dispatch import execute_processor

try:
    from grdl.data_prep import Stitcher, Tiler
    from grdl.data_prep.tiler import HaloTile, total_halo
    from grdl.image_processing.tiling import get_tile_halo
except ImportError:  # grdl missing or too old: nothing can be tiled
    Stitcher = Tiler = HaloTile = total_halo = None  # type: ignore[assignment,misc]

    def get_tile_halo(processor: Any) -> None:  # type: ignore[misc]
        return None


logger = get_logger(__name__)

BACKENDS = ("process", "thread", "serial")


class TilingError(RuntimeError):
    """Raised when a workflow cannot be executed tile by tile."""


# ----------------------------------------------------------------------
# Configuration and planning
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TilingConfig:
    """Settings for tiled execution.

    Attributes
    ----------
    tile_size : int or tuple of int
        Core tile size in pixels (rows, cols); an int is square.
    stride : int or tuple of int, optional
        Tile step.  ``None`` (default) uses non-overlapping cores, so each
        pixel is computed once and the halo alone gives exact results.  A
        stride smaller than ``tile_size`` overlaps cores and blends them
        with ``overlap_method``.
    workers : int, optional
        Pool size; ``None`` uses the CPU count, capped by the tile count.
    backend : str
        ``'process'`` (default), ``'thread'`` or ``'serial'``.
    overlap_method : str
        Overlap resolution passed to ``grdl.data_prep.Stitcher``; the
        default is a weighted average.
    scratch_dir : str or Path, optional
        Directory for memmap-backed stitched outputs.  When set, every
        stitched array, including the final one, lives there and is kept.
        When ``None``, intermediates use a temporary directory for the
        process backend (removed afterwards) or memory otherwise, and the
        final output is in memory.
    mp_context : str
        Multiprocessing start method for the process backend.
    """

    tile_size: int | tuple[int, int] = 1024
    stride: int | tuple[int, int] | None = None
    workers: int | None = None
    backend: str = "process"
    overlap_method: str = "weighted"
    scratch_dir: str | Path | None = None
    mp_context: str = "spawn"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers must be >= 1")


@dataclass
class Segment:
    """A run of consecutive steps executed the same way.

    Attributes
    ----------
    kind : str
        ``'tiled'``, ``'global'`` or ``'tap_out'``.
    steps : list
        Workflow steps in the segment.
    halo : tuple of int
        Summed ``(row, col)`` halo of a tiled segment; ``(0, 0)`` otherwise.
    """

    kind: str
    steps: list[Any] = field(default_factory=list)
    halo: tuple[int, int] = (0, 0)


def step_halo(step: Any) -> tuple[int, int] | None:
    """Return a step's tile halo, or None when it must run globally."""
    fn = getattr(step, "fn", None)
    if fn is None or getattr(step, "required_bands", None) is not None:
        return None
    return get_tile_halo(fn)


def plan_segments(steps: Sequence[Any]) -> list[Segment]:
    """Group steps into tiled, global and tap-out segments.

    Parameters
    ----------
    steps : sequence
        Resolved workflow steps (``WorkflowStep`` or tap-out steps).

    Returns
    -------
    list of Segment
    """
    segments: list[Segment] = []
    for step in steps:
        if not hasattr(step, "fn"):
            segments.append(Segment("tap_out", [step]))
            continue
        halo = step_halo(step)
        kind = "global" if halo is None else "tiled"
        if segments and segments[-1].kind == kind:
            segments[-1].steps.append(step)
        else:
            segments.append(Segment(kind, [step]))
        if halo is not None:
            segments[-1].halo = total_halo(segments[-1].halo, halo)
    return segments


# ----------------------------------------------------------------------
# Tile sources
# ----------------------------------------------------------------------


class ReaderSource:
    """Read tiles from a reader that each thread or process opens itself.

    Parameters
    ----------
    factory : callable
        Picklable zero-argument callable returning an opened reader
        (for example ``functools.partial(SICDCollectionReader, paths)``).
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._init_state()

    def _init_state(self) -> None:
        self._local = threading.local()
        self._opened: list[Any] = []
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        return {"_factory": self._factory}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._factory = state["_factory"]
        self._init_state()

    def read(self, region: Any) -> np.ndarray:
        reader = getattr(self._local, "reader", None)
        if reader is None:
            reader = self._factory()
            self._local.reader = reader
            with self._lock:
                self._opened.append(reader)
        return reader.read_chip(
            region.row_start,
            region.row_end,
            region.col_start,
            region.col_end,
        )

    def close(self) -> None:
        with self._lock:
            opened, self._opened = self._opened, []
        for reader in opened:
            try:
                reader.close()
            except Exception:
                logger.warning("tile_reader_close_failed")
        self._local = threading.local()


class ArraySource:
    """Read tiles from an array, or from a memmap reopened per process.

    Parameters
    ----------
    array : np.ndarray, optional
        In-memory array ``(..., rows, cols)``; cannot be sent to a
        process pool.
    spec : tuple, optional
        ``(path, dtype, shape)`` of a raw memmap file; pickles cheaply.
    """

    def __init__(
        self,
        array: np.ndarray | None = None,
        spec: tuple[str, str, tuple[int, ...]] | None = None,
    ) -> None:
        if (array is None) == (spec is None):
            raise ValueError("give exactly one of array or spec")
        self._array = array
        self._spec = spec

    def __getstate__(self) -> dict[str, Any]:
        if self._spec is None:
            raise TilingError("in-memory array sources cannot be sent to a process pool")
        return {"_array": None, "_spec": self._spec}

    def _get(self) -> np.ndarray:
        if self._array is None:
            path, dtype, shape = self._spec  # type: ignore[misc]
            self._array = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        return self._array

    def read(self, region: Any) -> np.ndarray:
        arr = self._get()
        return np.array(
            arr[..., region.row_start : region.row_end, region.col_start : region.col_end]
        )

    def close(self) -> None:
        if self._spec is not None:
            self._array = None


# ----------------------------------------------------------------------
# Tile worker
# ----------------------------------------------------------------------


def _tile_metadata(meta: Any, block: np.ndarray) -> Any:
    """Copy ``meta`` with the tile's spatial size."""
    if meta is None or not dataclasses.is_dataclass(meta):
        return meta
    names = {f.name for f in dataclasses.fields(meta)}
    if not {"rows", "cols"} <= names:
        return meta
    return dataclasses.replace(meta, rows=block.shape[-2], cols=block.shape[-1])


def _spatial_shapes(data: Any) -> list[tuple[int, ...]]:
    if isinstance(data, dict):
        return [np.shape(v)[-2:] for v in data.values()]
    return [np.shape(data)[-2:]]


@dataclass
class _StepTiming:
    wall_start: float = math.inf
    wall_end: float = 0.0
    cpu_time_s: float = 0.0

    @property
    def wall_time_s(self) -> float:
        return self.wall_end - self.wall_start

    def merge(self, other: _StepTiming) -> None:
        self.wall_start = min(self.wall_start, other.wall_start)
        self.wall_end = max(self.wall_end, other.wall_end)
        self.cpu_time_s += other.cpu_time_s


class _TileJob:
    """Everything a worker needs to process one tile through a segment."""

    def __init__(
        self,
        source: Any,
        steps: list[Any],
        meta: Any,
        first_tile: Any,
        *,
        thread_cpu: bool = False,
    ) -> None:
        self.source = source
        self.steps = steps
        self.meta = meta
        self.first_tile = first_tile
        self.thread_cpu = thread_cpu

    def __call__(self, tile: Any) -> tuple[Any, list[Any] | None, Any, list[_StepTiming], float]:
        """Return tile output, step timings, and actual worker CPU time.

        Metadata is returned for the first tile only, to keep results small.
        """
        cpu_clock = time.thread_time if self.thread_cpu else time.process_time
        job_cpu_start = cpu_clock()
        block = self.source.read(tile.read)
        meta = _tile_metadata(self.meta, block)
        want = (tile.read.row_end - tile.read.row_start, tile.read.col_end - tile.read.col_start)
        timings: list[_StepTiming] = []
        metas: list[Any] = []
        for step in self.steps:
            t0 = time.perf_counter()
            cpu_t0 = cpu_clock()
            block, meta = execute_processor(step.fn, meta, block)
            timings.append(_StepTiming(t0, time.perf_counter(), cpu_clock() - cpu_t0))
            metas.append(meta)
            for shape in _spatial_shapes(block):
                if tuple(shape) != want:
                    raise TilingError(
                        f"step '{step.name}' changed the spatial size "
                        f"{want} -> {tuple(shape)}; it is not tile-safe"
                    )
        return (
            tile,
            (metas if tile == self.first_tile else None),
            block,
            timings,
            cpu_clock() - job_cpu_start,
        )


_WORKER_JOB: _TileJob | None = None


def _init_worker(job: _TileJob) -> None:
    global _WORKER_JOB
    _WORKER_JOB = job


def _run_in_worker(tile: Any) -> Any:
    assert _WORKER_JOB is not None
    return _WORKER_JOB(tile)


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------


@dataclass
class TiledRunResult:
    """Outcome of :meth:`TiledExecutor.run`.

    Attributes
    ----------
    result : Any
        Final array or dict of arrays.
    metadata : Any
        Metadata after the last step.
    metadata_trace : list of dict
        Per-step metadata snapshots.
    step_times : list of tuple
        ``(step, timing, tiled, gpu_used)``. Tiled wall time spans the
        first invocation's start to the last invocation's end; CPU time
        sums measured CPU work across tiles, not elapsed time.
    n_tiles : int
        Tiles processed across all tiled segments.
    """

    result: Any
    metadata: Any
    metadata_trace: list[dict[str, Any]]
    step_times: list[tuple[Any, _StepTiming, bool, bool]]
    n_tiles: int = 0
    worker_cpu_time_s: float = 0.0


GlobalRunner = Callable[[Any, Any, Any, int], "tuple[Any, Any, bool]"]
TapOutRunner = Callable[[Any, Any], None]


class TiledExecutor:
    """Run a linear step list as alternating tiled and global segments.

    Parameters
    ----------
    config : TilingConfig
        Tiling settings.
    """

    def __init__(self, config: TilingConfig) -> None:
        if Tiler is None:
            raise ImportError("Tiled execution requires grdl with data_prep.Stitcher")
        self._cfg = config

    def run(
        self,
        source: Any,
        shape: tuple[int, int],
        steps: Sequence[Any],
        metadata: Any,
        run_global_step: GlobalRunner,
        run_tap_out: TapOutRunner,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TiledRunResult:
        """Execute ``steps`` over the full image.

        Parameters
        ----------
        source : ReaderSource or ArraySource
            Where the first segment reads tiles from.
        shape : tuple of int
            Full image ``(rows, cols)``.
        steps : sequence
            Resolved workflow steps.
        metadata : Any
            Reader metadata for the full image.
        run_global_step : callable
            ``(step, array, metadata, index) -> (array, metadata, gpu_used)``.
        run_tap_out : callable
            ``(tap_out_step, array) -> None``.
        progress_callback : callable, optional
            Called with a fraction in ``[0, 1]``.

        Returns
        -------
        TiledRunResult
        """
        axis_order = getattr(metadata, "axis_order", None)
        if axis_order == "YXC":
            raise TilingError("tiling needs channel-first (CYX) or 2-D data, got YXC")

        segments = plan_segments(steps)
        if not segments:
            raise TilingError("no steps to execute")
        self._log_plan(segments)

        scratch = self._scratch_root()
        total_units = sum(len(s.steps) if s.kind != "tiled" else 1 for s in segments)
        done_units = 0
        trace: list[dict[str, Any]] = []
        times: list[tuple[Any, _StepTiming, bool, bool]] = []
        worker_cpu_time_s = 0.0
        n_tiles = 0
        current: Any = None  # array/dict once a segment has produced output
        meta = metadata
        step_index = 0
        try:
            for seg_no, seg in enumerate(segments):
                is_last = seg_no == len(segments) - 1
                if seg.kind == "tiled":
                    src = source if current is None else self._as_source(current, scratch)
                    current, seg_metas, seg_times, count, worker_cpu = self._run_tiled(
                        seg,
                        src,
                        shape,
                        meta,
                        scratch,
                        is_last,
                        progress=_segment_progress(
                            progress_callback,
                            done_units,
                            total_units,
                        ),
                    )
                    n_tiles += count
                    if self._cfg.backend == "process":
                        worker_cpu_time_s += worker_cpu
                    for ws, timing, step_meta in zip(seg.steps, seg_times, seg_metas, strict=True):
                        meta = step_meta
                        times.append((ws, timing, True, False))
                        trace.append(
                            {"step_index": step_index, "step_name": ws.name, "metadata": meta}
                        )
                        step_index += 1
                    done_units += 1
                else:
                    if current is None:
                        current = self._materialize(source, shape)
                    for ws in seg.steps:
                        t0 = time.perf_counter()
                        cpu_t0 = time.process_time()
                        if seg.kind == "tap_out":
                            run_tap_out(ws, current)
                        else:
                            current, meta, gpu_used = run_global_step(ws, current, meta, step_index)
                            timing = _StepTiming(
                                t0,
                                time.perf_counter(),
                                time.process_time() - cpu_t0,
                            )
                            trace.append(
                                {"step_index": step_index, "step_name": ws.name, "metadata": meta}
                            )
                            times.append((ws, timing, False, gpu_used))
                        step_index += 1
                        done_units += 1
                        if progress_callback is not None:
                            progress_callback(done_units / total_units)
            if isinstance(current, np.memmap) or _holds_memmap(current):
                current = _detach(current) if scratch.temporary else current
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
            scratch.cleanup()
        return TiledRunResult(current, meta, trace, times, n_tiles, worker_cpu_time_s)

    # ------------------------------------------------------------------

    def _log_plan(self, segments: list[Segment]) -> None:
        logger.info(
            "tiling_plan",
            segments=[
                (s.kind, [getattr(x, "name", "?") for x in s.steps], s.halo) for s in segments
            ],
        )

    def _materialize(self, source: Any, shape: tuple[int, int]) -> np.ndarray:
        """Read the whole image for a leading global segment."""
        region = _Region(0, shape[0], 0, shape[1])
        return source.read(region)

    def _scratch_root(self) -> _Scratch:
        cfg = self._cfg
        if cfg.scratch_dir is not None:
            return _Scratch(Path(cfg.scratch_dir), temporary=False, persistent_output=True)
        if cfg.backend == "process":
            return _Scratch(
                Path(tempfile.mkdtemp(prefix="grdl_rt_tiles_")),
                temporary=True,
                persistent_output=False,
            )
        return _Scratch(None, temporary=False, persistent_output=False)

    def _as_source(self, data: Any, scratch: _Scratch) -> Any:
        """Wrap a previous segment's output as a tile source."""
        if isinstance(data, dict):
            raise TilingError(
                "a tiled step follows a step that returned a dict of arrays; "
                "dict outputs cannot be tiled further"
            )
        if self._cfg.backend != "process":
            return ArraySource(array=np.asarray(data))
        if isinstance(data, np.memmap) and getattr(data, "filename", None):
            return ArraySource(spec=(str(data.filename), str(data.dtype), data.shape))
        path = scratch.new_path("global_out")
        mm = _scratch_memmap(path, data.shape, data.dtype)
        mm[...] = data
        mm.flush()
        return ArraySource(spec=(str(path), str(data.dtype), tuple(data.shape)))

    def _run_tiled(
        self,
        seg: Segment,
        source: Any,
        shape: tuple[int, int],
        meta: Any,
        scratch: _Scratch,
        is_last: bool,
        progress: Callable[[float], Any],
    ) -> tuple[Any, list[Any], list[_StepTiming], int, float]:
        cfg = self._cfg
        rows, cols = shape
        overlapping = cfg.stride is not None
        tiler = Tiler(rows, cols, tile_size=cfg.tile_size, stride=cfg.stride)
        tiles = tiler.halo_tiles(seg.halo, overlapping=overlapping)

        method = cfg.overlap_method if overlapping else "last"
        sink = None
        if scratch.root is not None and (cfg.scratch_dir is not None or not is_last):
            sink = _numbered_sink(scratch)
        stitcher = Stitcher(rows, cols, method=method, sink=sink)

        job = _TileJob(
            source,
            seg.steps,
            meta,
            tiles[0],
            thread_cpu=cfg.backend == "thread",
        )
        timings = [_StepTiming() for _ in seg.steps]
        worker_cpu = 0.0
        step_metas: list[Any] = []
        count = 0
        for tile, tmetas, block, tile_timings, cpu in self._map(job, tiles):
            if tmetas is not None:
                step_metas = [_full_metadata(m, shape) for m in tmetas]
            stitcher.add_halo_tile(tile, block)
            for accumulated, timing in zip(timings, tile_timings, strict=True):
                accumulated.merge(timing)
            worker_cpu += cpu
            count += 1
            progress(count / len(tiles))
        out = stitcher.finalize()
        return out, step_metas, timings, count, worker_cpu

    def _map(self, job: _TileJob, tiles: list[Any]):
        cfg = self._cfg
        workers = cfg.workers or min(_cpu_count(), len(tiles))
        workers = max(1, min(workers, len(tiles)))
        if cfg.backend == "serial" or workers == 1 and cfg.backend != "process":
            yield from (job(t) for t in tiles)
            return
        if cfg.backend == "thread":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                yield from pool.map(job, tiles)
            return
        ctx = get_context(cfg.mp_context)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(job,),
        ) as pool:
            yield from pool.map(_run_in_worker, tiles)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Region:
    row_start: int
    row_end: int
    col_start: int
    col_end: int


class _Scratch:
    """Directory for memmap files, optionally removed on cleanup."""

    def __init__(self, root: Path | None, temporary: bool, persistent_output: bool) -> None:
        self.root = root
        self.temporary = temporary
        self.persistent_output = persistent_output
        self._n = 0
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def new_path(self, stem: str) -> Path:
        assert self.root is not None
        self._n += 1
        return self.root / f"{self._n:03d}_{stem}.dat"

    def cleanup(self) -> None:
        if self.temporary and self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)


def _numbered_sink(scratch: _Scratch):
    """Sink writing each output array to a fresh memmap in ``scratch``."""
    prefix = f"{scratch._n + 1:03d}"
    scratch._n += 1
    root = scratch.root
    assert root is not None

    def sink(key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return _scratch_memmap(root / f"{prefix}_{key}.dat", shape, dtype)

    return sink


def _scratch_memmap(path: Path, shape: tuple[int, ...], dtype: Any) -> np.memmap:
    """Check and reserve backing storage before any mapped pages are written."""
    required = math.prod(shape) * np.dtype(dtype).itemsize
    available = shutil.disk_usage(path.parent).free
    if required > available:
        raise TilingError(
            f"Insufficient scratch space in {path.parent}: array {shape} ({np.dtype(dtype)}) "
            f"requires {required / 2**30:.2f} GiB, only {available / 2**30:.2f} GiB available. "
            "Set scratch_dir (execution.scratch_dir in the PolSAR YAML) or TMPDIR "
            "to a filesystem with enough free space."
        )
    # Truncation alone creates a sparse file: mmap writes can later SIGBUS
    # instead of raising ENOSPC. Reserve blocks before returning the mapping.
    with path.open("w+b") as file:
        file.truncate(required)
        if hasattr(os, "posix_fallocate"):
            try:
                os.posix_fallocate(file.fileno(), 0, required)
            except OSError as exc:
                if exc.errno not in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL):
                    raise TilingError(
                        f"Cannot reserve {required / 2**30:.2f} GiB of scratch storage "
                        f"for {path}: {exc}. Choose another scratch_dir or free disk space."
                    ) from exc
                logger.warning(
                    "scratch_reservation_unsupported",
                    path=str(path),
                    reason=str(exc),
                    required_bytes=required,
                )
        else:
            logger.warning("scratch_reservation_unavailable", path=str(path))
    return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def _holds_memmap(data: Any) -> bool:
    return isinstance(data, dict) and any(isinstance(v, np.memmap) for v in data.values())


def _detach(data: Any) -> Any:
    """Copy memmap-backed output into memory before its directory is deleted."""
    if isinstance(data, dict):
        return {k: np.array(v) for k, v in data.items()}
    return np.array(data)


def _full_metadata(meta: Any, shape: tuple[int, int]) -> Any:
    if meta is None or not dataclasses.is_dataclass(meta):
        return meta
    names = {f.name for f in dataclasses.fields(meta)}
    if not {"rows", "cols"} <= names:
        return meta
    return dataclasses.replace(meta, rows=shape[0], cols=shape[1])


def _segment_progress(
    callback: Callable[[float], None] | None,
    done: int,
    total: int,
) -> Callable[[float], None]:
    """Map a segment's own 0..1 progress onto the whole run."""

    def report(fraction: float) -> None:
        if callback is not None:
            callback((done + fraction) / total)

    return report


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1
