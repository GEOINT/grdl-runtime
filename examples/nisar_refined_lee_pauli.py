"""
NISAR quad-pol workflow: T3 -> Refined Lee -> Pauli magnitudes -> RGB.

Follows notebooks/grdl/image_processing/sar/full_pol_decomposition_tutorial.ipynb.
Building T3 with window_size=1 and filtering it is equivalent to calling
RefinedLeeFilter.filter_channels(..., matrix_type="T3"). The matrix-aware
Pauli API returns magnitudes, not reconstructed complex scattering channels.

The first three steps run on halo tiles; RGB stretching runs once on the
stitched scene so tile boundaries do not introduce contrast seams. The
output retains every source row and column in the original radar grid.
This is not orthorectification; the TIFF has no map projection.

Requires the staged multi-channel/tiling support in grdl and grdl-runtime,
plus h5py and rasterio. Run with --help for options.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.decomposition import CoherencyMatrix, PauliDecomposition
from grdl.image_processing.filters import RefinedLeeFilter
from grdl.image_processing.versioning import processor_version
from grdl.IO import write
from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.IO.sar import NISARReader, check_reciprocity_reader, validate_collection
from rasterio.enums import ColorInterp

from grdl_rt import Workflow

DEFAULT_INPUT = Path(
    "/data/sar/slc/nisar/l1_rslc/20260105T055924_20260105T055931/"
    "NISAR_L1_PR_RSLC_009_116_D_054_2005_QPDH_A_20260105T055924_"
    "20260105T055931_X05010_N_P_J_001.h5"
)
QUAD = ["HH", "HV", "VH", "VV"]


class WindowedNISARReader(NISARReader):
    """Expose a source window as a dimension-preserving reader."""

    def __init__(self, filepath: Path, window: tuple[int, int, int, int], **kwargs: Any):
        self._window = window
        super().__init__(filepath, **kwargs)

    def _load_metadata(self) -> None:
        super()._load_metadata()
        row_start, row_end, col_start, col_end = self._window
        if not (0 <= row_start < row_end <= self.metadata.rows):
            raise ValueError(f"Invalid chip rows: {self._window[:2]}")
        if not (0 <= col_start < col_end <= self.metadata.cols):
            raise ValueError(f"Invalid chip columns: {self._window[2:]}")
        self.metadata = replace(
            self.metadata,
            rows=row_end - row_start,
            cols=col_end - col_start,
        )

    def read_chip(
        self, row_start: int, row_end: int, col_start: int, col_end: int, bands=None
    ) -> np.ndarray:
        if (
            row_start < 0
            or col_start < 0
            or row_end > self.metadata.rows
            or col_end > self.metadata.cols
        ):
            raise ValueError("Chip indices exceed the selected window")
        window_row, _, window_col, _ = self._window
        source_rows = slice(row_start + window_row, row_end + window_row)
        source_cols = slice(col_start + window_col, col_end + window_col)
        if self._multi_pol:
            return np.stack(
                [
                    self._file[self._imagery_paths[i]][source_rows, source_cols]
                    for i in self._channel_indices(bands)
                ],
                axis=0,
            )
        return self._file[self._imagery_path][source_rows, source_cols]

    def read_full(self, bands=None) -> np.ndarray:
        row_start, row_end, col_start, col_end = self._window
        return self.read_chip(0, row_end - row_start, 0, col_end - col_start, bands=bands)


@processor_version("1.0.0")
class PauliFromT3(PauliDecomposition):
    """Adapt the existing matrix-aware Pauli API to workflow dispatch."""

    def execute(
        self, metadata: ImageMetadata, source: np.ndarray, **kwargs: Any
    ) -> tuple[dict[str, np.ndarray], ImageMetadata]:
        if not np.isfinite(source).all():
            raise ValueError("The filtered T3 matrix must contain only finite values")
        components = self.decompose_from_t3(source)
        updated = replace(
            metadata,
            rows=source.shape[-2],
            cols=source.shape[-1],
            dtype=str(components["surface"].dtype),
            bands=3,
            axis_order="CYX",
            channel_metadata=[
                ChannelMetadata(index=i, name=name, role="decomposition", source_indices=[4 * i])
                for i, name in enumerate(self.component_names)
            ],
        )
        return components, updated


@processor_version("1.0.0")
class PauliRGB(ImageProcessor):
    """Apply the tutorial's scene-wide magnitude percentile stretch."""

    def execute(
        self,
        metadata: ImageMetadata,
        source: np.ndarray | dict[str, np.ndarray],
        **kwargs: Any,
    ) -> tuple[np.ndarray, ImageMetadata]:
        if not isinstance(source, dict):
            raise TypeError("Pauli RGB requires a dictionary of magnitude components")
        rgb, rgb_meta = PauliDecomposition().to_rgb(source, representation="magnitude")
        return rgb, replace(
            metadata,
            format=rgb_meta.format,
            rows=rgb_meta.rows,
            cols=rgb_meta.cols,
            bands=rgb_meta.bands,
            dtype=rgb_meta.dtype,
            axis_order=rgb_meta.axis_order,
            channel_metadata=rgb_meta.channel_metadata,
        )


def build_workflow(
    *,
    frequency: str = "A",
    kernel_size: int = 7,
    nlook: float = 1.0,
    tile_size: int = 512,
    workers: int = 4,
    backend: str = "process",
    scratch_dir: Path | None = None,
    reader_cls: type[NISARReader] = NISARReader,
    reader_kwargs: dict[str, Any] | None = None,
) -> Workflow:
    """Declare a full-scene workflow with no cropping or resampling."""
    return (
        Workflow("NISAR Refined Lee Pauli", version="1.0.0", modalities=["SAR"])
        .reader(
            reader_cls,
            frequency=frequency,
            polarizations=QUAD,
            **(reader_kwargs or {}),
        )
        .validate_collection()
        .step(CoherencyMatrix, window_size=1)
        .step(RefinedLeeFilter, kernel_size=kernel_size, nlook=nlook)
        .step(PauliFromT3)
        .step(PauliRGB)
        .tiling(
            tile_size=tile_size,
            workers=workers,
            backend=backend,
            scratch_dir=scratch_dir,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path("nisar_refined_lee_pauli.tif"))
    parser.add_argument("--frequency", choices=["A", "B"], default="A")
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--nlook", type=float, default=1.0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    parser.add_argument("--scratch-dir", type=Path, help="Keep stitched component memmaps here.")
    parser.add_argument(
        "--chip-size",
        type=int,
        help="Process a centered square chip instead of the full scene.",
    )
    parser.add_argument(
        "--chip-row",
        type=int,
        help="Zero-based chip row start; requires --chip-col and --chip-size.",
    )
    parser.add_argument(
        "--chip-col",
        type=int,
        help="Zero-based chip column start; requires --chip-row and --chip-size.",
    )
    parser.add_argument(
        "--output-type",
        choices=["uint8", "float32"],
        default="uint8",
        help="TIFF sample type; uint8 is compatible with common image viewers.",
    )
    parser.add_argument(
        "--progress",
        choices=["percent", "tiles", "none"],
        default="percent",
        help="Progress output: percent, per-tile updates, or none.",
    )
    args = parser.parse_args()
    if args.output.suffix.lower() not in (".tif", ".tiff"):
        parser.error("--output must have a .tif or .tiff extension")
    if args.tile_size < 1:
        parser.error("--tile-size must be positive")
    if (args.chip_row is None) != (args.chip_col is None):
        parser.error("--chip-row and --chip-col must be supplied together")
    if args.chip_size is not None and args.chip_size < 1:
        parser.error("--chip-size must be positive")
    if args.chip_size is None and (args.chip_row is not None or args.chip_col is not None):
        parser.error("--chip-row and --chip-col require --chip-size")
    report_path = args.output.with_suffix(".json")

    reader_cls: type[NISARReader] = NISARReader
    reader_kwargs: dict[str, Any] = {}
    if args.chip_size is not None:
        with NISARReader(
            args.filepath, frequency=args.frequency, polarizations=QUAD
        ) as source_reader:
            source_rows, source_cols = source_reader.metadata.rows, source_reader.metadata.cols
        if args.chip_row is None:
            chip_row = (source_rows - args.chip_size) // 2
            chip_col = (source_cols - args.chip_size) // 2
        else:
            chip_row, chip_col = args.chip_row, args.chip_col
        chip_window = (
            chip_row,
            chip_row + args.chip_size,
            chip_col,
            chip_col + args.chip_size,
        )
        reader_cls = WindowedNISARReader
        reader_kwargs["window"] = chip_window
    else:
        chip_window = None

    workflow = build_workflow(
        frequency=args.frequency,
        kernel_size=args.kernel_size,
        nlook=args.nlook,
        tile_size=args.tile_size,
        workers=args.workers,
        backend=args.backend,
        scratch_dir=args.scratch_dir,
        reader_cls=reader_cls,
        reader_kwargs=reader_kwargs,
    )
    with reader_cls(
        args.filepath,
        frequency=args.frequency,
        polarizations=QUAD,
        **reader_kwargs,
    ) as reader:
        collection = validate_collection(reader)
        collection.raise_if_invalid()
        if not collection.is_quad_pol:
            raise ValueError(f"Expected quad-pol data, found {collection.polarizations}")
        rows, cols = reader.metadata.rows, reader.metadata.cols
        reciprocity = check_reciprocity_reader(reader)
    print(f"Input: {rows} x {cols}; channels: {collection.polarizations}", flush=True)
    print(f"Sampled reciprocity: {asdict(reciprocity)}", flush=True)

    last_percent = -1
    tile_number = 0

    def progress(fraction: float) -> None:
        nonlocal last_percent, tile_number
        if args.progress == "none":
            return
        if args.progress == "tiles":
            tile_number += 1
            print(f"Progress update {tile_number}: {fraction:.1%}", flush=True)
            return
        percent = int(100 * fraction)
        if percent > last_percent:
            print(f"Progress: {percent}%", flush=True)
            last_percent = percent

    result = workflow.execute(args.filepath, progress_callback=progress)
    rgb_float = result.result
    if args.output_type == "uint8":
        rgb = np.rint(rgb_float * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb = rgb_float.astype(np.float32, copy=False)
    if rgb.shape != (3, rows, cols):
        raise ValueError(f"Output shape {rgb.shape} does not preserve input {(rows, cols)}")
    if not np.isfinite(rgb).all():
        raise ValueError("Output RGB must contain only finite values")
    output_meta = replace(
        result.metadata_trace[-1]["metadata"],
        dtype=str(rgb.dtype),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(rgb, args.output, metadata=output_meta, format="geotiff")
    with rasterio.open(args.output, "r+") as image:
        image.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        for index, channel in enumerate(output_meta.channel_metadata, start=1):
            image.set_band_description(index, channel.name)
    report = {
        "input": str(args.filepath.resolve()),
        "output": str(args.output.resolve()),
        "input_shape": [rows, cols],
        "output_shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "axis_order": "CYX",
        "rgb_channels": [c.name for c in output_meta.channel_metadata],
        "grid": "original radar grid; not orthorectified",
        "parameters": {
            "frequency": args.frequency,
            "kernel_size": args.kernel_size,
            "nlook": args.nlook,
            "tile_size": args.tile_size,
            "workers": args.workers,
            "backend": args.backend,
            "representation": "magnitude",
            "percentiles": [2.0, 98.0],
            "output_type": args.output_type,
            "progress": args.progress,
        },
        "chip_window": list(chip_window) if chip_window is not None else None,
        "collection": asdict(collection),
        "reciprocity": asdict(reciprocity),
        "metrics": result.metrics.to_dict(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
