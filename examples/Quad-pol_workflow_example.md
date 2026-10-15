### Full-scene NISAR Refined Lee / Pauli Workflow Example

[nisar_refined_lee_pauli.py](examples/nisar_refined_lee_pauli.py) runs the
quad-pol NISAR product from 2026-01-05 specified in the script:

```bash
python examples/nisar_refined_lee_pauli.py --output /path/to/results/pauli.tif
```

An optional positional filepath selects another quad-pol product. Use `--help`
for frequency, Refined Lee kernel/looks, tile size, workers, backend, and scratch
directory options. This requires the multi-channel/tiling changes in both local
packages, plus `h5py` and `rasterio`.

The workflow constructs single-look T3, applies Refined Lee, then calls
`PauliDecomposition.decompose_from_t3`. The first three steps use halo tiles;
the magnitude-based 2nd/98th-percentile RGB stretch runs on the stitched scene,
not independently per tile. This matches the full-pol decomposition tutorial.
The full input cube and full T3 matrix are never assembled in memory, but
the stitched magnitude components and final RGB image are scene-sized.
With the default process backend, intermediate components use temporary disk;
`--scratch-dir` instead retains those memmaps in the selected directory.

By default, outputs are a viewer-compatible three-band uint8 TIFF in `[0, 255]`
(red=double-bounce, green=volume, blue=surface). Use `--output-type float32`
to retain the processing values in `[0, 1]`; some desktop image viewers do not
display float32 TIFFs. A same-stem JSON report records dimensions,
processing settings, collection validation, sampled reciprocity, and runtime
metrics. Reciprocity failures warn and are recorded without stopping processing;
NaN/Inf inputs to the reciprocity check are invalid. Zero-only samples fail the
check; jointly zero pixels are excluded from otherwise usable samples.

Every source row and column is retained. The TIFF remains in the original radar
grid, **not** an orthorectified map projection. Existing output files are not
overwritten.

Use `--chip-size 2048` for a centered, multi-tile debug run, or combine
`--chip-size`, `--chip-row`, and `--chip-col` to select an explicit source
window. Use `--progress percent` (default), `--progress tiles`, or
`--progress none`.

The `.tiling()` call configures the whole workflow; it is not a step after
`PauliRGB`. At execution time, the runtime examines each step's `tile_halo`:

1. `CoherencyMatrix` (`halo=0`), `RefinedLeeFilter` (`halo=3`), and
   `PauliFromT3` (`halo=0`) form one tiled segment with cumulative halo 3.
2. Each core tile is expanded by 3 pixels, read from NISAR, processed through
   all three steps, cropped back to its core, and stitched.
3. `PauliRGB` is a global step because percentile stretching needs the
   complete stitched component arrays. It runs once after the tiled segment.

The tiled executor currently bypasses GPU-aware dispatch. In addition,
Refined Lee uses SciPy and is CPU-only, so this example is expected to run on
the CPU even on hosts with CUDA GPUs. GPU acceleration would require a
CuPy-compatible Refined Lee implementation and GPU-aware tiled workers.