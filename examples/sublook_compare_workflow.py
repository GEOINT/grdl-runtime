# -*- coding: utf-8 -*-
"""
Sublook Compare (Workflow) — Framework-driven sublook decomposition.

Performs the same processing as ``grdl/example/image_processing/sar/
sublook_compare.py`` — splits a SICD image into 3 sub-aperture looks —
but defines the pipeline as a ``grdl_rt.Workflow``.

The entire processing pipeline — from reading SICD data to display-ready
output — is declared as a workflow recipe.  The framework handles all
orchestration: reader lifecycle management, metadata extraction and
injection, chip planning and pixel reading, processor construction,
and GPU-accelerated execution.

Compared to the ~200-line manual script, this example is ~40 lines
total.  The workflow definition itself is 7 lines.

Framework Benefits
------------------
**Zero boilerplate** — Declare what to do, not how to wire it.  The
framework opens the reader, extracts metadata, plans the chip, reads
pixels, constructs processors, and runs the pipeline.

**Automatic metadata injection** — Processors whose constructors accept
a ``metadata`` parameter (e.g., ``SublookDecomposition``) receive it
from the reader automatically — no manual wiring.

**Built-in chip management** — Chip strategy declared once
(``.chip("center", size=5000)``), applied by the framework using
``ChipExtractor`` internally.

**GPU acceleration** — ``prefer_gpu=True`` with transparent CPU
fallback for all ``__gpu_compatible__`` processors.

**Progress tracking** — Proportional ``[0, 1]`` callbacks per step.

**Error isolation** — Step-level error context for debugging.

Usage:
  python sublook_compare_workflow.py <sicd_file>
  python sublook_compare_workflow.py <sicd_file> --chip-size 2048
  python sublook_compare_workflow.py <sicd_file> --plow 1 --phigh 99
  python sublook_compare_workflow.py --help

Dependencies
------------
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

# GRDL processors (IDE autocomplete on every constructor parameter)
from grdl.IO import SICDReader
from grdl.image_processing.sar import SublookDecomposition
from grdl.image_processing.intensity import ToDecibels, PercentileStretch

# grdl-runtime
from grdl_rt import Workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workflow-based sublook comparison.",
    )
    parser.add_argument("filepath", type=Path)
    parser.add_argument("--chip-size", type=int, default=5000)
    parser.add_argument("--plow", type=float, default=2.0)
    parser.add_argument("--phigh", type=float, default=98.0)
    args = parser.parse_args()

    # ── Define the workflow ───────────────────────────────────────────
    #
    #   .reader()  → framework opens reader and extracts metadata
    #   .chip()    → framework plans and reads a center chip
    #   .step()    → processor classes are constructed at execute time;
    #                SublookDecomposition receives metadata automatically
    #
    wf = (
        Workflow("Sublook Compare", version="1.0.0", modalities=["SAR"])
        .reader(SICDReader)
        .chip("center", size=args.chip_size)
        .step(SublookDecomposition, num_looks=3, dimension='azimuth', overlap=0.0)
        .step(ToDecibels)
        .step(PercentileStretch, plow=args.plow, phigh=args.phigh)
    )

    # ── Execute ───────────────────────────────────────────────────────
    result = wf.execute(
        args.filepath,
        prefer_gpu=True,
        progress_callback=lambda f: print(f"  Progress: {f:.0%}"),
    )
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")


if __name__ == "__main__":
    main()
