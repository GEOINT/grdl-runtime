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
    from grdl.data_prep import Stitcher, Tiler, memmap_sink
    from grdl.data_prep.tiler import HaloTile, total_halo
    from grdl.image_processing.tiling import get_tile_halo
except ImportError:  # grdl missing or too old: nothing can be tiled
    Stitcher = Tiler = memmap_sink = HaloTile = total_halo = None  # type: ignore[assignment,misc]

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


class _TileJob:
    """Everything a worker needs to process one tile through a segment."""

    def __init__(self, source: Any, steps: list[Any], meta: Any, first_tile: Any) -> None:
        self.source = source
        self.steps = steps
        self.meta = meta
        self.first_tile = first_tile

    def __call__(self, tile: Any) -> tuple[Any, list[Any] | None, Any, list[float]]:
        """Return ``(tile, per_step_metadata, block, per_step_seconds)``.

        Metadata is returned for the first tile only, to keep results small.
        """
        block = self.source.read(tile.read)
        meta = _tile_metadata(self.meta, block)
        want = (tile.read.row_end - tile.read.row_start, tile.read.col_end - tile.read.col_start)
        durations: list[float] = []
        metas: list[Any] = []
        for step in self.steps:
            t0 = time.perf_counter()
            block, meta = execute_processor(step.fn, meta, block)
            durations.append(time.perf_counter() - t0)
            metas.append(meta)
            for shape in _spatial_shapes(block):
                if tuple(shape) != want:
                    raise TilingError(
                        f"step '{step.name}' changed the spatial size "
                        f"{want} -> {tuple(shape)}; it is not tile-safe"
                    )
        return tile, (metas if tile == self.first_tile else None), block, durations


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
        ``(step, seconds, tiled)``; seconds for tiled steps is the sum
        over tiles (compute time, not elapsed wall time).
    n_tiles : int
        Tiles processed across all tiled segments.
    """

    result: Any
    metadata: Any
    metadata_trace: list[dict[str, Any]]
    step_times: list[tuple[Any, float, bool]]
    n_tiles: int = 0


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
        times: list[tuple[Any, float, bool]] = []
        n_tiles = 0
        current: Any = None  # array/dict once a segment has produced output
        meta = metadata
        step_index = 0
        try:
            for seg_no, seg in enumerate(segments):
                is_last = seg_no == len(segments) - 1
                if seg.kind == "tiled":
                    src = source if current is None else self._as_source(current, scratch)
                    current, seg_metas, seg_times, count = self._run_tiled(
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
                    for ws, seconds, step_meta in zip(seg.steps, seg_times, seg_metas, strict=True):
                        meta = step_meta
                        times.append((ws, seconds, True))
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
                        if seg.kind == "tap_out":
                            run_tap_out(ws, current)
                        else:
                            current, meta, _ = run_global_step(ws, current, meta, step_index)
                            trace.append(
                                {"step_index": step_index, "step_name": ws.name, "metadata": meta}
                            )
                            times.append((ws, time.perf_counter() - t0, False))
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
        return TiledRunResult(current, meta, trace, times, n_tiles)

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
        mm = np.memmap(path, dtype=data.dtype, mode="w+", shape=data.shape)
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
    ) -> tuple[Any, list[Any], list[float], int]:
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

        job = _TileJob(source, seg.steps, meta, tiles[0])
        sums = [0.0] * len(seg.steps)
        step_metas: list[Any] = []
        count = 0
        for tile, tmetas, block, durations in self._map(job, tiles):
            if tmetas is not None:
                step_metas = [_full_metadata(m, shape) for m in tmetas]
            stitcher.add_halo_tile(tile, block)
            for i, d in enumerate(durations):
                sums[i] += d
            count += 1
            progress(count / len(tiles))
        out = stitcher.finalize()
        return out, step_metas, sums, count

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
    inner = memmap_sink(scratch.root)  # type: ignore[arg-type]

    def sink(key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return inner(f"{prefix}_{key}", shape, dtype)

    return sink


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
