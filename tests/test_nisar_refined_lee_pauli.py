"""Verify the NISAR example against the tutorial's explicit matrix APIs."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("grdl.data_prep.stitcher")
h5py = pytest.importorskip("h5py")

from grdl.image_processing.decomposition import CoherencyMatrix, PauliDecomposition
from grdl.image_processing.filters import RefinedLeeFilter
from grdl.IO.models.base import ImageMetadata
from grdl.IO.sar import NISARReader

from examples.nisar_refined_lee_pauli import (
    QUAD,
    PauliFromT3,
    PauliRGB,
    build_workflow,
    main,
)
from grdl_rt.execution.builder import WorkflowStep
from grdl_rt.execution.tiled import plan_segments


@pytest.fixture
def quad_product(tmp_path):
    rng = np.random.default_rng(19)
    cube = (rng.normal(size=(4, 47, 61)) + 1j * rng.normal(size=(4, 47, 61))).astype(np.complex64)
    cube[2] = cube[1]
    cube[:, :3] = 0
    path = tmp_path / "quad.h5"
    with h5py.File(path, "w") as file:
        frequency = file.create_group("science/LSAR/RSLC/swaths/frequencyA")
        frequency.create_dataset("listOfPolarizations", data=np.array(QUAD[::-1], dtype="S2"))
        for polarization, channel in zip(QUAD, cube, strict=True):
            frequency.create_dataset(polarization, data=channel)
    return path, cube


def tutorial_rgb(cube):
    filtered = RefinedLeeFilter(kernel_size=7).filter_channels(*cube, matrix_type="T3")
    pauli = PauliDecomposition()
    components = pauli.decompose_from_t3(filtered)
    return pauli.to_rgb(components, representation="magnitude")[0]


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_full_workflow_matches_tutorial_and_preserves_dimensions(quad_product, backend):
    path, cube = quad_product
    result = build_workflow(tile_size=17, workers=2, backend=backend).execute(path)
    expected = tutorial_rgb(cube)
    assert result.result.shape == (3, 47, 61)
    assert result.result.dtype == np.float32
    np.testing.assert_allclose(result.result, expected, rtol=1e-4, atol=1e-5)
    assert np.isfinite(result.result).all()
    assert 0 <= result.result.min() <= result.result.max() <= 1
    assert [m.concurrent for m in result.metrics.step_metrics] == [True, True, True, False]
    assert len(result.metadata_trace) == 4
    for entry in result.metadata_trace:
        meta = entry["metadata"]
        assert (meta.rows, meta.cols) == (47, 61)
    rgb_meta = result.metadata_trace[-1]["metadata"]
    assert rgb_meta.axis_order == "CYX" and rgb_meta.dtype == "float32"
    assert [c.name for c in rgb_meta.channel_metadata] == [
        "double_bounce",
        "volume",
        "surface",
    ]


def test_rgb_stretch_is_global():
    steps = [
        WorkflowStep(fn=CoherencyMatrix(window_size=1), name="T3", gpu_compatible=False),
        WorkflowStep(fn=RefinedLeeFilter(kernel_size=7), name="Refined Lee", gpu_compatible=False),
        WorkflowStep(fn=PauliFromT3(), name="Pauli", gpu_compatible=False),
        WorkflowStep(fn=PauliRGB(), name="RGB", gpu_compatible=False),
    ]
    segments = plan_segments(steps)
    assert [(s.kind, len(s.steps), s.halo) for s in segments] == [
        ("tiled", 3, (3, 3)),
        ("global", 1, (0, 0)),
    ]


def test_pauli_matrix_adapter_rejects_nonfinite_values():
    meta = ImageMetadata(format="T3", rows=8, cols=8, dtype="complex64", axis_order="CCYX")
    matrix = np.zeros((3, 3, 8, 8), dtype=np.complex64)
    matrix[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PauliFromT3().execute(meta, matrix)


def test_zero_matrix_stays_zero():
    meta = ImageMetadata(format="T3", rows=8, cols=9, dtype="complex64", axis_order="CCYX")
    components, component_meta = PauliFromT3().execute(
        meta, np.zeros((3, 3, 8, 9), dtype=np.complex64)
    )
    rgb, rgb_meta = PauliRGB().execute(component_meta, components)
    assert rgb.shape == (3, 8, 9) and not rgb.any()
    assert (rgb_meta.rows, rgb_meta.cols) == (8, 9)


def test_rgb_requires_components():
    meta = ImageMetadata(format="T3", rows=8, cols=9, dtype="complex64", axis_order="CCYX")
    with pytest.raises(TypeError, match="dictionary"):
        PauliRGB().execute(meta, np.zeros((3, 3, 8, 9), dtype=np.complex64))


def test_cli_persists_image_and_report(quad_product, tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    path, cube = quad_product
    output = tmp_path / "results" / "pauli.tif"
    monkeypatch.setattr(
        "sys.argv",
        [
            __file__,
            str(path),
            "--output",
            str(output),
            "--backend",
            "serial",
            "--tile-size",
            "17",
            "--output-type",
            "float32",
        ],
    )
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        main()
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning), rasterio.open(output) as reader:
        actual = reader.read()
        assert reader.count == 3 and reader.shape == (47, 61)
        assert reader.colorinterp == (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
        assert reader.descriptions == ("double_bounce", "volume", "surface")
        np.testing.assert_allclose(actual, tutorial_rgb(cube), rtol=1e-4, atol=1e-5)
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["input_shape"] == [47, 61]
    assert report["output_shape"] == [3, 47, 61]
    assert report["collection"]["mode"] == "quad"
    assert report["reciprocity"]["passed"] is True
    assert report["parameters"]["representation"] == "magnitude"
    assert report["metrics"]["status"] == "success"
    assert Path(report["output"]) == output
    with pytest.raises(FileExistsError, match="overwrite"):
        main()


def test_reader_uses_explicit_channel_order(quad_product):
    path, cube = quad_product
    with NISARReader(path, polarizations=QUAD) as reader:
        assert [c.polarization for c in reader.metadata.channel_metadata] == QUAD
        np.testing.assert_array_equal(reader.read_full(), cube)


@pytest.mark.parametrize("timeline_enabled", [False, True])
def test_cli_chip_mode_writes_viewer_compatible_tiff(
    quad_product, tmp_path, monkeypatch, timeline_enabled
):
    from grdl_rt.execution.config import MemoryConfig, RuntimeConfig

    monkeypatch.setattr(
        "grdl_rt.execution.memory_sampler.get_runtime_config",
        lambda: RuntimeConfig(memory=MemoryConfig(timeline_enabled=timeline_enabled)),
    )
    rasterio = pytest.importorskip("rasterio")
    path, _ = quad_product
    output = tmp_path / "chip.tif"
    monkeypatch.setattr(
        "sys.argv",
        [
            __file__,
            str(path),
            "--output",
            str(output),
            "--chip-size",
            "31",
            "--tile-size",
            "13",
            "--backend",
            "serial",
            "--progress",
            "none",
        ],
    )
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        main()
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning), rasterio.open(output) as reader:
        assert reader.shape == (31, 31)
        assert reader.dtypes == ("uint8", "uint8", "uint8")
        assert reader.colorinterp == (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["input_shape"] == [31, 31]
    assert report["parameters"]["output_type"] == "uint8"
    assert report["chip_window"] == [8, 39, 15, 46]
    assert ("memory_timeline" in report["metrics"]) is timeline_enabled
    if timeline_enabled:
        assert report["metrics"]["memory_timeline"]["timestamps"]
    else:
        assert report["metrics"]["peak_rss_bytes"] == 0
