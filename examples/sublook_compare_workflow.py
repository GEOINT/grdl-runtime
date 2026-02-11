# -*- coding: utf-8 -*-
"""
Sublook Compare (Workflow) - grdl-runtime Workflow version of sublook_compare.

Performs the same processing as ``grdl/example/image_processing/sar/
sublook_compare.py`` — splits a SICD image into 3 sub-aperture looks —
but defines the pipeline as a ``grdl_rt.Workflow`` instead of inline code.

Runtime Benefits
----------------
**IDE support** — Every step references a real Python object.  IDE
autocompletion works on constructor parameters
(``SublookDecomposition(metadata, num_looks=▸``), on bound methods
(``sublook.▸``), and on the builder itself (``Workflow(...).step(▸``).
Type checkers catch errors at edit time rather than at runtime.

**GPU acceleration** — Pass ``prefer_gpu=True`` to ``execute()`` and
compatible steps are transparently dispatched to the GPU via CuPy, with
automatic CPU fallback if the GPU path fails.

**Progress tracking** — A single ``progress_callback`` receives
proportional ``[0.0, 1.0]`` updates as each step completes, useful for
progress bars in notebooks, GUIs, and long-running batch jobs.

**Error isolation** — If a step fails, the raised ``RuntimeError``
includes the workflow name, step index, and step name, making
debugging straightforward without wrapping every call in try/except.

**Batch execution** — ``execute_batch(chips)`` runs the same workflow
on a list of chips in one call, with aggregate progress reporting.

**Composability** — Steps are plain callables: mix GRDL processor
methods, ``ImageTransform`` instances, numpy operations, and custom
functions in a single pipeline without adapter boilerplate.

**Reusable processors** — ``ToDecibels`` and ``PercentileStretch`` are
proper ``ImageTransform`` components with declared parameters, version
metadata, and GPU compatibility, rather than throw-away lambdas.

Usage:
  python sublook_compare_workflow.py <sicd_file>
  python sublook_compare_workflow.py <sicd_file> --chip-size 2048
  python sublook_compare_workflow.py <sicd_file> --plow 1 --phigh 99
  python sublook_compare_workflow.py --help

Dependencies
------------
matplotlib (optional, for visualization)
sarkit or sarpy
grdl-runtime

Author
------
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

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
import numpy as np

# GRDL
from grdl.IO import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar import SublookDecomposition
from grdl.image_processing.intensity import ToDecibels, PercentileStretch

# grdl-runtime
from grdl_rt import Workflow


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Workflow-based sublook comparison. Splits a SICD image "
                    "into 3 sub-aperture looks and displays side-by-side.",
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the SICD file (NITF or other SICD container).",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=5000,
        help="Side length of the center chip in pixels (default: 5000).",
    )
    parser.add_argument(
        "--plow",
        type=float,
        default=2.0,
        help="Lower percentile for contrast stretch (default: 2).",
    )
    parser.add_argument(
        "--phigh",
        type=float,
        default=98.0,
        help="Upper percentile for contrast stretch (default: 98).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap (default: gray).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib plot (requires matplotlib + Qt backend).",
    )
    return parser.parse_args()


# ── Source factory ───────────────────────────────────────────────────


def read_center_chip(
    filepath: Path,
    chip_size: int = 5000,
) -> np.ndarray:
    """Read the center chip from a SICD file.

    This function encapsulates all IO — opening the reader, planning
    the chip region with ``ChipExtractor``, and reading pixels.
    It is designed to be passed to ``Workflow.source()`` as a
    deferred data factory.

    Parameters
    ----------
    filepath : Path
        Path to the SICD file.
    chip_size : int
        Side length of the center chip in pixels.

    Returns
    -------
    np.ndarray
        Complex-valued chip array.
    """
    with SICDReader(filepath) as reader:
        rows, cols = reader.get_shape()
        print(f"  Image size: {rows} x {cols}")

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2,
            row_width=chip_size, col_width=chip_size,
        )

        chip_h = region.row_end - region.row_start
        chip_w = region.col_end - region.col_start
        print(f"  Center chip: [{region.row_start}:{region.row_end}, "
              f"{region.col_start}:{region.col_end}] ({chip_h} x {chip_w})")

        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        print(f"  Chip shape: {chip.shape}, dtype: {chip.dtype}")
        return chip


# ── Main ─────────────────────────────────────────────────────────────


def sublook_compare_workflow(
    filepath: Path,
    chip_size: int = 5000,
    plow: float = 2.0,
    phigh: float = 98.0,
    cmap: str = "gray",
    show: bool = False,
) -> None:
    """Build and execute a sublook-decomposition workflow.

    Parameters
    ----------
    filepath : Path
        Path to the SICD file.
    chip_size : int
        Side length of the center chip.
    plow : float
        Lower percentile for contrast stretch.
    phigh : float
        Upper percentile for contrast stretch.
    cmap : str
        Matplotlib colormap name.
    show : bool
        If ``True``, display the results via matplotlib.
    """
    print(f"Opening: {filepath}")

    # ── Read metadata (needed to configure SublookDecomposition) ──────
    with SICDReader(filepath) as reader:
        meta = reader.metadata

    # ── Build reusable image-processing components ────────────────────
    #
    #   IDE autocomplete works on every constructor parameter.
    #   All are proper ImageTransform instances with declared parameters,
    #   version metadata, and GPU compatibility.
    #
    sublook = SublookDecomposition(
        meta, num_looks=3, dimension='azimuth', overlap=0.0,
    )
    to_db = ToDecibels()
    stretch = PercentileStretch(plow=plow, phigh=phigh)

    # ── Define the sublook workflow ───────────────────────────────────
    #
    #   Workflow.step() accepts:
    #     - ImageTransform instances   (auto-wrapped to .apply())
    #     - Bound methods              (sublook.decompose)
    #     - Plain functions / lambdas  (any ndarray → ndarray callable)
    #
    #   .source() registers a deferred data factory — the reader callback
    #   is invoked automatically when .execute() is called without data.
    #
    sublook_wf = (
        Workflow("Sublook Compare", version="1.0.0", modalities=["SAR"])
        .source(read_center_chip, filepath, chip_size)
        .step(sublook.decompose, name="Sublook Decomposition")
        .step(to_db,             name="Convert to dB")
        .step(stretch,           name="Percentile Stretch")
    )

    # ── Execute with runtime benefits ─────────────────────────────────
    #
    #   prefer_gpu=True  → GPU dispatch for compatible steps (with fallback)
    #   progress_callback → per-step [0.0, 1.0] progress updates
    #
    print("  Executing sublook workflow...")
    looks_stretched = sublook_wf.execute(
        prefer_gpu=True,
        progress_callback=lambda f: print(f"    Progress: {f:.0%}"),
    )
    print(f"  Result shape: {looks_stretched.shape}")

    # ── Also process the full-aperture chip for display ───────────────
    chip = read_center_chip(filepath, chip_size)
    display_wf = (
        Workflow("Full Aperture Display")
        .step(to_db,    name="Convert to dB")
        .step(stretch,  name="Percentile Stretch")
    )
    chip_stretched = display_wf.execute(chip)

    # ── Visualization (optional) ──────────────────────────────────────
    if show:
        from grdl.example.image_processing.sar.sublook_compare import (
            plot_sublook_comparison,
        )

        ci = meta.collection_info
        title_parts = [filepath.name]
        if ci is not None and ci.collector_name:
            title_parts.append(ci.collector_name)
        file_title = "  |  ".join(title_parts)

        plot_sublook_comparison(
            chip_stretched, looks_stretched,
            title=file_title,
            cmap=cmap,
        )


if __name__ == "__main__":
    args = parse_args()
    sublook_compare_workflow(
        args.filepath,
        chip_size=args.chip_size,
        plow=args.plow,
        phigh=args.phigh,
        cmap=args.cmap,
        show=args.show,
    )
