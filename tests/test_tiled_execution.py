"""
Tiled Execution Tests - segment planning, halo tiling and Workflow integration.

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

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("grdl.data_prep.stitcher")

from grdl.image_processing.decomposition import CovarianceMatrix
from grdl.image_processing.filters import IDANFilter, RefinedLeeFilter
from grdl.IO.base import ImageReader
from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.IO.sar import CollectionValidationError, ReciprocityError

from grdl_rt.execution.builder import Workflow
from grdl_rt.execution.tiled import (
    ArraySource,
    TilingConfig,
    TilingError,
    plan_segments,
)

ROWS, COLS = 83, 101
QUAD = ["HH", "HV", "VH", "VV"]


class FakeQuadReader(ImageReader):
    """Deterministic in-memory quad-pol reader (picklable for spawn)."""

    def __init__(self, filepath, pols=None, reciprocal=True, axis_order="CYX"):
        self._pols = list(pols or QUAD)
        self._reciprocal = reciprocal
        self._axis_order = axis_order
        first = filepath[0] if isinstance(filepath, (list, tuple)) else filepath
        self.received = filepath
        super().__init__(first)

    def _load_metadata(self):
        rng = np.random.default_rng(11)
        shape = (len(self._pols), ROWS, COLS)
        cube = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        if len(self._pols) == 4:
            cube[[0, 3]] *= 3.0
            if self._reciprocal:
                cube[2] = cube[1]
        self._cube = cube.astype(np.complex64)
        self.metadata = ImageMetadata(
            format="fake",
            rows=ROWS,
            cols=COLS,
            dtype="complex64",
            bands=len(self._pols),
            axis_order=self._axis_order,
            channel_metadata=[
                ChannelMetadata(index=i, name=p, polarization=p) for i, p in enumerate(self._pols)
            ],
        )

    def read_chip(self, row_start, row_end, col_start, col_end, bands=None):
        return self._cube[:, row_start:row_end, col_start:col_end].copy()

    def get_shape(self):
        return (ROWS, COLS)

    def get_dtype(self):
        return np.dtype("complex64")

    def close(self):
        pass


class _Plain:
    """Callable processor; Workflow.step() needs a callable or ImageProcessor."""

    def __call__(self, source, **kwargs):
        return self.apply(source, **kwargs)


class Scale(_Plain):
    """Pixelwise processor that declares itself tile-safe."""

    tile_halo = 0

    def __init__(self, k):
        self.k = k

    def apply(self, source, **kwargs):
        return source * self.k


class GlobalMaxNorm(_Plain):
    """Needs the whole image; declares no halo."""

    def apply(self, source, **kwargs):
        return source / np.abs(source).max()


class Crop(_Plain):
    """Claims to be tile-safe but changes the spatial size."""

    tile_halo = 0

    def apply(self, source, **kwargs):
        return source[..., 1:, 1:]


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "scene.bin"
    p.write_bytes(b"x")
    return p


def _workflow(*steps, tiling=None, **reader_kwargs):
    wf = Workflow("tiled-test").reader(FakeQuadReader, **reader_kwargs)
    for step in steps:
        if isinstance(step, tuple):
            wf.step(step[0], **step[1])
        else:
            wf.step(step)
    if tiling is not None:
        wf.tiling(**tiling)
    return wf


THREADS = {"tile_size": 32, "backend": "thread", "workers": 3}


def _ws(fn, name="s", required_bands=None):
    return SimpleNamespace(fn=fn, name=name, required_bands=required_bands)


class TestPlanSegments:
    def test_groups_and_sums_halo(self):
        steps = [
            _ws(CovarianceMatrix(window_size=7)),
            _ws(RefinedLeeFilter(kernel_size=5)),
            _ws(GlobalMaxNorm()),
            _ws(Scale(2)),
        ]
        segs = plan_segments(steps)
        assert [(s.kind, len(s.steps)) for s in segs] == [
            ("tiled", 2),
            ("global", 1),
            ("tiled", 1),
        ]
        assert segs[0].halo == (5, 5)
        assert segs[2].halo == (0, 0)

    def test_idan_and_auto_enl_are_global(self):
        segs = plan_segments([_ws(IDANFilter(kernel_size=7, enl=2.0)), _ws(Scale(1))])
        assert [s.kind for s in segs] == ["global", "tiled"]

    def test_required_bands_forces_global(self):
        segs = plan_segments([_ws(Scale(1), required_bands=3)])
        assert segs[0].kind == "global"

    def test_tap_out_is_a_boundary(self):
        tap = SimpleNamespace(name="tap", path="x.npy")
        segs = plan_segments([_ws(Scale(1)), tap, _ws(Scale(2))])
        assert [s.kind for s in segs] == ["tiled", "tap_out", "tiled"]

    def test_consecutive_global_steps_merge(self):
        segs = plan_segments([_ws(GlobalMaxNorm()), _ws(GlobalMaxNorm())])
        assert [(s.kind, len(s.steps)) for s in segs] == [("global", 2)]


class TestTiledEquivalence:
    def test_covariance_and_refined_lee(self, image_path):
        steps = [
            (CovarianceMatrix, {"window_size": 7}),
            (RefinedLeeFilter, {"kernel_size": 7}),
        ]
        full = _workflow(*steps).execute(image_path).result
        tiled = _workflow(*steps, tiling=THREADS).execute(image_path).result
        assert tiled.shape == full.shape == (4, 4, ROWS, COLS) or tiled.shape == full.shape
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_tile_then_global_then_tile(self, image_path):
        steps = [
            (CovarianceMatrix, {"window_size": 5}),
            GlobalMaxNorm(),
            Scale(2.0),
        ]
        full = _workflow(*steps).execute(image_path).result
        res = _workflow(*steps, tiling=THREADS).execute(image_path)
        np.testing.assert_allclose(res.result, full, rtol=1e-5, atol=1e-6)
        assert len(res.metrics.step_metrics) == 3
        assert [m["step_index"] for m in res.metadata_trace] == [0, 1, 2]
        assert res.metadata_trace[0]["step_name"] == "CovarianceMatrix"

    def test_global_step_first(self, image_path):
        steps = [GlobalMaxNorm(), Scale(3.0)]
        full = _workflow(*steps).execute(image_path).result
        tiled = _workflow(*steps, tiling=THREADS).execute(image_path).result
        np.testing.assert_allclose(tiled, full, rtol=1e-5)

    def test_serial_backend(self, image_path):
        steps = [(CovarianceMatrix, {"window_size": 3})]
        full = _workflow(*steps).execute(image_path).result
        tiled = (
            _workflow(
                *steps,
                tiling={"tile_size": 40, "backend": "serial"},
            )
            .execute(image_path)
            .result
        )
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_overlapping_tiles_blend(self, image_path):
        steps = [(CovarianceMatrix, {"window_size": 3})]
        full = _workflow(*steps).execute(image_path).result
        tiled = (
            _workflow(
                *steps,
                tiling={**THREADS, "stride": 16},
            )
            .execute(image_path)
            .result
        )
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_process_backend(self, image_path):
        steps = [(CovarianceMatrix, {"window_size": 3}), Scale(2.0)]
        full = _workflow(*steps).execute(image_path).result
        tiled = (
            _workflow(
                *steps,
                tiling={"tile_size": 48, "backend": "process", "workers": 2},
            )
            .execute(image_path)
            .result
        )
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_process_backend_global_between_tiled(self, image_path):
        steps = [(CovarianceMatrix, {"window_size": 3}), GlobalMaxNorm(), Scale(2.0)]
        full = _workflow(*steps).execute(image_path).result
        tiled = (
            _workflow(
                *steps,
                tiling={"tile_size": 48, "backend": "process", "workers": 2},
            )
            .execute(image_path)
            .result
        )
        assert not isinstance(tiled, np.memmap)
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-5)

    def test_progress_reaches_one(self, image_path):
        seen = []
        _workflow(
            (CovarianceMatrix, {"window_size": 3}),
            GlobalMaxNorm(),
            tiling=THREADS,
        ).execute(image_path, progress_callback=seen.append)
        assert seen[-1] == pytest.approx(1.0)
        assert seen == sorted(seen)


class TestScratchOutput:
    def test_scratch_dir_keeps_memmap_output(self, image_path, tmp_path):
        out = tmp_path / "out"
        res = _workflow(
            (CovarianceMatrix, {"window_size": 3}),
            tiling={"tile_size": 40, "backend": "serial", "scratch_dir": out},
        ).execute(image_path)
        assert isinstance(res.result, np.memmap)
        assert list(out.glob("*.dat"))

    def test_default_output_in_memory(self, image_path):
        res = _workflow(
            (CovarianceMatrix, {"window_size": 3}),
            tiling={"tile_size": 40, "backend": "serial"},
        ).execute(image_path)
        assert not isinstance(res.result, np.memmap)


class TestErrors:
    def test_step_changing_size_is_rejected(self, image_path):
        wf = _workflow(Crop(), tiling=THREADS)
        with pytest.raises(TilingError, match="not tile-safe"):
            wf.execute(image_path)

    def test_chip_center_conflict(self, image_path):
        wf = _workflow(Scale(1), tiling=THREADS)
        wf.chip("center", size=10)
        with pytest.raises(ValueError, match="chip"):
            wf.execute(image_path)

    def test_yxc_rejected(self, image_path):
        wf = _workflow(Scale(1), tiling=THREADS, axis_order="YXC")
        with pytest.raises(TilingError, match="CYX"):
            wf.execute(image_path)

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            TilingConfig(backend="gpu")
        with pytest.raises(ValueError):
            TilingConfig(workers=0)

    def test_array_source_rejects_pickling_of_in_memory(self):
        import pickle

        with pytest.raises(TilingError):
            pickle.dumps(ArraySource(array=np.zeros((2, 2))))


class TestCollectionIntegration:
    def test_list_of_paths_reaches_reader(self, image_path, tmp_path):
        other = tmp_path / "b.bin"
        other.write_bytes(b"y")
        seen = {}

        class Spy(FakeQuadReader):
            def __init__(self, filepath, **kw):
                super().__init__(filepath, **kw)
                seen["paths"] = self.received

        wf = Workflow("c").reader(Spy).step(Scale(1.0))
        wf.execute([image_path, other])
        assert seen["paths"] == [image_path, other]

    def test_validation_error_stops_run(self, image_path):
        wf = _workflow(Scale(1.0), pols=["HH", "HH", "VH", "VV"])
        wf.validate_collection()
        with pytest.raises(CollectionValidationError, match="duplicate"):
            wf.execute(image_path)

    def test_non_strict_validation_continues(self, image_path):
        wf = _workflow(Scale(1.0), pols=["HH", "HH", "VH", "VV"])
        wf.validate_collection(strict=False)
        assert wf.execute(image_path).result.shape[-2:] == (ROWS, COLS)

    def test_axis_order_error(self, image_path):
        wf = _workflow(Scale(1.0), axis_order="YX")
        wf.validate_collection()
        with pytest.raises(CollectionValidationError, match="axis_order"):
            wf.execute(image_path)

    def test_reciprocity_strict(self, image_path):
        ok = _workflow(Scale(1.0), reciprocal=True)
        ok.validate_collection(reciprocity="strict")
        ok.execute(image_path)

        bad = _workflow(Scale(1.0), reciprocal=False)
        bad.validate_collection(reciprocity="strict")
        with pytest.raises(ReciprocityError):
            bad.execute(image_path)

    def test_invalid_reciprocity_mode(self):
        with pytest.raises(ValueError):
            Workflow("x").validate_collection(reciprocity="sometimes")
