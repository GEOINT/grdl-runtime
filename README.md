# grdl-runtime

Execution framework and workflow engine for GRDL processing pipelines.

**grdl-runtime** sits between [grdl](../grdl/) (processing primitives) and [grdk](../grdk/) (Qt/Orange GUI). It is the reference runtime for orchestrating GRDL components into reproducible workflows — handling reader lifecycle management, metadata extraction and injection, chip planning, processor construction, GPU-accelerated execution, and artifact catalog management.

Users declare *what* to process. The framework handles *how* to wire it.

## Architecture

```
grdl  (processing primitives — readers, transforms, detectors, decompositions)
  ↓
grdl-runtime  (execution framework, catalog, GPU orchestration)   ← this package
  ↓
grdk  (Qt/Orange GUI widgets)
```

## Quick Start

### Framework-Driven Workflow (Recommended)

The `Workflow` builder is the primary Python API. Declare a reader, chip strategy, and processing steps — the framework handles everything else:

```python
from grdl.IO import SICDReader
from grdl.image_processing.sar import SublookDecomposition
from grdl.image_processing.intensity import ToDecibels, PercentileStretch
from grdl_rt import Workflow

wf = (
    Workflow("Sublook Compare", version="1.0.0", modalities=["SAR"])
    .reader(SICDReader)
    .chip("center", size=5000)
    .step(SublookDecomposition, num_looks=3, dimension='azimuth', overlap=0.0)
    .step(ToDecibels)
    .step(PercentileStretch, plow=2.0, phigh=98.0)
)

result = wf.execute("image.nitf", prefer_gpu=True)
```

The workflow definition is 7 lines. The framework:
- Opens the reader and manages its lifecycle (single open, automatic close)
- Extracts metadata from the reader
- Plans and reads a center chip using `ChipExtractor`
- Constructs `SublookDecomposition` with automatically injected metadata
- Constructs `ToDecibels` and `PercentileStretch` with the declared kwargs
- Runs the pipeline with GPU acceleration and CPU fallback
- Reports progress via callback

Compare this to the [~200-line manual script](../grdl/grdl/example/image_processing/sar/sublook_compare.py) that does the same thing by hand.

### Direct Array Mode

For simpler pipelines that operate on data already in memory:

```python
wf = (
    Workflow("Display Pipeline")
    .step(ToDecibels())
    .step(PercentileStretch(plow=2.0, phigh=98.0))
)
result = wf.execute(my_array)
```

### Batch Execution

```python
results = wf.execute_batch(
    [array1, array2, array3],
    prefer_gpu=True,
    progress_callback=lambda f: print(f"Progress: {f:.0%}"),
)
```

## Key Concepts

### Workflow as Framework

The `Workflow` class acts as both a recipe builder and an execution framework. When `.execute()` receives a filepath, it orchestrates the full pipeline:

1. **Reader management** — opens the declared reader class, extracts metadata, closes on completion
2. **Chip planning** — uses `grdl.data_prep.ChipExtractor` to plan and read chips based on the declared strategy (`"center"`, `"full"`)
3. **Deferred construction** — processor classes passed to `.step()` are instantiated at execute time, not at build time
4. **Metadata injection** — processors whose constructors accept a `metadata` parameter (e.g., `SublookDecomposition`) receive it from the reader automatically
5. **GPU acceleration** — steps marked `__gpu_compatible__` are dispatched to CuPy with transparent CPU fallback
6. **Progress tracking** — proportional `[0, 1]` callbacks per step
7. **Error isolation** — step-level error context with workflow name, step index, and step name

### Three Step Types

`.step()` accepts three argument forms:

| Form | Example | Behavior |
|------|---------|----------|
| **Class** (deferred) | `.step(SublookDecomposition, num_looks=3)` | Stored as `DeferredStep`; constructed at execute time with metadata injection |
| **Instance** | `.step(ToDecibels(floor_db=-50.0))` | Wrapped immediately; `.apply()` called at execute time |
| **Callable** | `.step(my_function, name="Custom")` | Used directly as-is |

### Metadata Injection

Convention-based: if a processor class's `__init__` has a parameter named `metadata`, the framework injects it from the reader. No decorators, registration, or configuration needed.

```python
# SublookDecomposition.__init__(self, metadata, num_looks=2, ...) → metadata injected
.step(SublookDecomposition, num_looks=3)

# ToDecibels.__init__(self, floor_db=-60.0) → no metadata param, constructed with just kwargs
.step(ToDecibels)

# User-provided metadata kwarg takes precedence over injection
.step(SublookDecomposition, metadata=my_custom_meta, num_looks=3)
```

### Processor Discovery

Scan installed GRDL processors by modality, category, or capability:

```python
from grdl_rt import discover_processors, filter_processors

all_processors = discover_processors()
sar_filters = filter_processors(modality="SAR", category="filters")
```

### Artifact Catalog

SQLite-backed catalog with full-text search for managing processors and workflows:

```python
from grdl_rt import ArtifactCatalog

catalog = ArtifactCatalog("~/.grdl/catalog.db")
results = catalog.search("sublook SAR")
```

Supports SQLite (local), YAML (portable), and federated (multi-source) backends.

## Installation

```bash
pip install -e .
```

With GPU support:

```bash
pip install -e ".[gpu]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Launching the UI

grdl-runtime includes a built-in Tkinter/matplotlib GUI for configuring and running workflows interactively.

### From the command line

```bash
# Launch the UI
grdl-rt ui

# Pre-load a workflow file
grdl-rt ui --workflow path/to/workflow.yaml

# Pre-load a workflow and an input image
grdl-rt ui -w path/to/workflow.yaml -i path/to/image.nitf
```

Alternative invocations:

```bash
# Console script (equivalent to `grdl-rt ui`)
grdl-rt-ui

# Module invocation
python -m grdl_rt.ui
```

### From Python

```python
from grdl_rt.ui import launch

launch()                                                    # empty UI
launch(workflow="workflow.yaml", input_path="image.nitf")   # pre-loaded
```

### UI overview

The GUI provides:

- **Input panel** — workflow/component file picker, input image(s) and folder pickers, output directory, ground-truth GeoJSON, hardware options (GPU preference, worker count, memory limit), and per-step parameter configuration
- **Results panel** — tabbed view with output preview (matplotlib canvas), metrics, and a real-time log console
- **Keyboard shortcuts** — Ctrl+O (open workflow), Ctrl+I (open input), Ctrl+R (run), Escape (cancel)
- **File watcher** — automatic reload when the workflow file changes on disk
- **Run history** — recent runs persisted to `~/.grdl/ui_history.json`

## Subpackages

### `grdl_rt.execution` — Workflow engine and processor orchestration

| Module | Purpose |
|--------|---------|
| `builder.py` | `Workflow` builder — fluent API, framework orchestration, deferred construction, metadata injection |
| `executor.py` | `WorkflowExecutor` — runs `WorkflowDefinition` pipelines (string-based) |
| `workflow.py` | `WorkflowDefinition`, `ProcessingStep` — serializable workflow models |
| `dsl.py` | DSL compiler (Python decorator ↔ YAML bidirectional) |
| `discovery.py` | Processor scanning, tag filtering, modality/category queries |
| `gpu.py` | `GpuBackend` — CuPy GPU dispatch with CPU fallback |
| `tags.py` | `ImageModality`, `WorkflowTags`, `ProjectTags` — taxonomy enums |
| `chip.py` | `Chip`, `ChipSet`, `ChipLabel` — chip data models |
| `config.py` | `GrdkConfig` — runtime configuration |
| `project.py` | `GrdkProject` — project directory model |

### `grdl_rt.catalog` — Artifact storage and discovery

| Module | Purpose |
|--------|---------|
| `database.py` | `SqliteArtifactCatalog` — SQLite + FTS5 full-text search |
| `yaml_catalog.py` | `YamlArtifactCatalog` — portable YAML-based catalog |
| `federated.py` | `FederatedArtifactCatalog` — multi-source catalog aggregation |
| `base.py` | `ArtifactCatalogBase` ABC |
| `models.py` | `Artifact`, `UpdateResult` data models |
| `resolver.py` | Catalog path resolution (env → config → default) |
| `updater.py` | `ArtifactUpdateWorker` — PyPI/Conda update checking |
| `pool.py` | `ThreadExecutorPool` — background task management |

### `grdl_rt.ui` — Interactive workflow GUI

| Module | Purpose |
|--------|---------|
| `__init__.py` | Public API — `launch()` entry point |
| `__main__.py` | `python -m grdl_rt.ui` support |
| `_app.py` | `App(tk.Tk)` — main application window |
| `_runner.py` | `WorkflowRunner` — threaded background execution |
| `_widgets.py` | Custom widgets (FilePickerRow, LogConsole, ProgressBar, etc.) |
| `_config_dialog.py` | Per-step parameter configuration dialog |
| `_metrics_panel.py` | Metrics visualization panel |
| `_importer.py` | Workflow/component introspection and parameter discovery |
| `_accuracy.py` | Accuracy report generation (optional ground-truth comparison) |

### `grdl_rt.api` — Convenience functions

| Function | Purpose |
|----------|---------|
| `load_workflow()` | Load a `WorkflowDefinition` from YAML |
| `execute_workflow()` | Load and execute a workflow in one call |

## Project Structure

```
grdl-runtime/
├── grdl_rt/
│   ├── __init__.py              # Package exports
│   ├── api.py                   # Convenience functions (load_workflow, execute_workflow)
│   ├── execution/
│   │   ├── builder.py           # Workflow builder + framework orchestration
│   │   ├── executor.py          # WorkflowExecutor (string-based pipeline runner)
│   │   ├── workflow.py          # WorkflowDefinition, ProcessingStep models
│   │   ├── dsl.py               # DSL compiler (Python ↔ YAML)
│   │   ├── discovery.py         # Processor scanning and filtering
│   │   ├── gpu.py               # GpuBackend (CuPy dispatch)
│   │   ├── tags.py              # Taxonomy enums (ImageModality, etc.)
│   │   ├── chip.py              # Chip data models
│   │   ├── config.py            # Runtime configuration
│   │   └── project.py           # Project directory model
│   ├── ui/
│   │   ├── __init__.py            # Public API (launch)
│   │   ├── __main__.py            # python -m grdl_rt.ui
│   │   ├── _app.py                # Main application window
│   │   ├── _runner.py             # Threaded workflow execution
│   │   ├── _widgets.py            # Custom Tkinter widgets
│   │   ├── _config_dialog.py      # Parameter configuration dialog
│   │   ├── _metrics_panel.py      # Metrics visualization
│   │   ├── _importer.py           # Workflow introspection
│   │   └── _accuracy.py           # Accuracy reporting
│   └── catalog/
│       ├── base.py              # ArtifactCatalogBase ABC
│       ├── database.py          # SqliteArtifactCatalog (SQLite + FTS5)
│       ├── yaml_catalog.py      # YamlArtifactCatalog
│       ├── federated.py         # FederatedArtifactCatalog
│       ├── models.py            # Artifact, UpdateResult
│       ├── resolver.py          # Catalog path resolution
│       ├── updater.py           # Update checking
│       └── pool.py              # Thread pool management
├── examples/
│   └── sublook_compare_workflow.py  # Framework-driven sublook decomposition
├── tests/                       # Test suite (316 tests)
├── pyproject.toml
├── LICENSE
└── README.md
```

## Testing

```bash
pytest tests/ -v                          # Full suite
pytest tests/test_builder.py -v           # Workflow builder + framework tests
pytest tests/ -p no:napari -x -q          # Quick run (skip napari plugin)
```

## Using AI to Generate Workflow YAML

Large language models can accelerate workflow authoring when given the right context. This section provides a reusable **system prompt**, example **user prompts** for common remote sensing targets, and tips for iterating on the results.

### System Prompt (Copy This)

Paste the following as the system or initial context when starting a conversation with an LLM about grdl-runtime workflows:

> You are a remote sensing workflow engineer. You generate YAML workflow
> definitions for grdl-runtime, the execution framework for GRDL image
> processing pipelines.
>
> **YAML schema (v2.0):**
>
> ```yaml
> schema_version: "2.0"
> name: "<Workflow Name>"
> version: "<semver>"
> description: "<What the workflow does>"
> state: "draft"          # draft | testing | published
> tags:
>   modalities: [...]     # SAR | EO | MSI | HSI | THERMAL
>   niirs_range: [lo, hi] # 0.0–9.0 NIIRS quality bounds
>   day_capable: true
>   night_capable: false
>   detection_types: []   # classification | segmentation | change | anomaly
>   segmentation_types: [] # polygon | pixel | contour
> steps:
>   - processor: "<ProcessorName>"
>     version: "<semver>"
>     id: "<unique_id>"             # optional, auto-assigned if omitted
>     depends_on: ["<step_id>"]     # optional, linear chain inferred if omitted
>     phase: "<execution_phase>"    # optional: io | global_processing | data_prep
>                                   #   tiling | tile_processing | extraction
>                                   #   vector_processing | finalization
>     condition: "<expr>"           # optional runtime guard
>     timeout_seconds: 60.0         # optional
>     retry:                        # optional
>       max_retries: 3
>       initial_delay: 1.0
>       max_delay: 10.0
>       exponential_base: 2.0
>       jitter: true
>     params:
>       key: value
>   - tap_out: "<path>"             # optional intermediate write-to-disk
>     format: "npy"                 # optional, auto-detected from extension
> ```
>
> **Conventions:**
> - Processors that accept a `metadata` parameter get it injected automatically
>   from the reader — never pass it explicitly in `params`.
> - Steps without `depends_on` are chained linearly (step N depends on step N-1).
> - Use `depends_on` with explicit `id` fields to create branching/merging DAGs.
> - `condition` supports Python-like expressions: comparisons, `and`/`or`/`not`,
>   dotted attribute access (`metadata.band_count > 1`), `in`/`not in`.
> - Use `tap_out` steps to write intermediate products for debugging.
> - Set `phase` annotations to help the framework optimize execution order.
>
> **When generating a workflow, always:**
> 1. Ask what modality the input imagery is (SAR, EO, MSI, HSI, thermal)
>    or infer it from context.
> 2. Ask about the target of interest and the desired output format.
> 3. Choose processors appropriate to the modality and target.
> 4. Set realistic `niirs_range` and `day_capable`/`night_capable` tags.
> 5. Annotate `detection_types` and `segmentation_types` when applicable.
> 6. Prefer short, meaningful `id` values for every step.
> 7. Add `timeout_seconds` or `retry` only where the step is expensive or
>    prone to transient failure.
> 8. Output valid YAML and nothing else unless asked to explain.

### Example Prompts by Target Type

#### SAR — Vehicle Detection

> I have single-polarization (HH) SAR SICD imagery at approximately NIIRS 4.
> Generate a grdl-runtime workflow that detects vehicles. The pipeline should
> suppress speckle, convert to decibels, apply an adaptive threshold to isolate
> bright scatterers, and clean up with morphological operations. Output the
> detection mask as a GeoTIFF tap-out.

#### SAR — Ship Detection (Open Water)

> Generate a grdl-runtime workflow for detecting ships in open ocean SAR imagery.
> Assume VV-polarization Sentinel-1 GRD product (NIIRS 2–3). The workflow should
> apply land masking, calibrate to sigma-naught, apply a CFAR detector for bright
> targets against the sea clutter background, and cluster detections into point
> targets. Mark it as day-and-night capable.

#### SAR — Coherent Change Detection

> I have a co-registered pair of SAR SICD collects over the same area taken days
> apart. Generate a grdl-runtime workflow for coherent change detection. The
> pipeline should compute the coherence magnitude between the two collects,
> threshold low-coherence regions as changes, and contour the results into
> vector polygons. Use a branching DAG where each collect is filtered
> independently before the coherence step merges them.

#### SAR — Sub-Aperture Analysis

> Generate a grdl-runtime workflow that performs sub-aperture (sublook)
> decomposition on a SAR SICD image. Split into 3 azimuth looks with no
> overlap, convert each to decibels, and apply a percentile stretch for
> display. This is for visual analysis, not detection — no detection_types
> needed.

#### EO — Building Footprint Extraction

> I have 30 cm panchromatic electro-optical imagery (NIIRS 6). Generate a
> grdl-runtime workflow that extracts building footprints. The pipeline should
> apply edge enhancement, segment candidate regions, filter by area and
> compactness, and produce polygon vectors. Set segmentation_types to
> ["polygon"]. Day-only.

#### EO — Vegetation Health / NDVI

> Generate a grdl-runtime workflow for computing NDVI from a 4-band EO image
> (R, G, B, NIR). The pipeline should extract the Red and NIR bands,
> compute the normalized difference, and apply a color ramp for
> visualization. Include a tap-out of the raw NDVI array before the color
> ramp step. Modality is EO, day-only, no detection types.

#### MSI/HSI — Material Classification

> I have a 224-band hyperspectral image (HSI). Generate a grdl-runtime workflow
> that classifies surface materials. The pipeline should apply atmospheric
> correction, reduce dimensionality (MNF or PCA), run a spectral angle mapper
> or matched filter against a signature library, and threshold the result into a
> classification map. Set detection_types to ["classification"] and
> segmentation_types to ["pixel"]. NIIRS range 2–5, day-only.

#### Thermal — Anomaly Detection

> Generate a grdl-runtime workflow for detecting thermal anomalies (hot spots)
> in LWIR thermal imagery. The pipeline should apply non-uniformity correction,
> convert to apparent temperature, compute a local background model, and flag
> pixels that exceed the background by a configurable number of standard
> deviations. Mark as day-and-night capable. Modality is THERMAL, detection_type
> is ["anomaly"].

#### Multi-Modal — SAR + EO Fusion for Building Change

> I have co-located SAR and EO imagery of the same area. Generate a
> grdl-runtime workflow that fuses both modalities to detect building changes.
> Use a branching DAG: one branch processes SAR (sublook decomposition → dB →
> stretch), the other processes EO (pan-sharpen → edge enhance). Merge the
> branches with a feature-level fusion step, then run change detection and
> contour the results. Set modalities to ["SAR", "EO"], detection_types to
> ["change"], segmentation_types to ["polygon"].

### Tips for Better Results

1. **Provide the modality up front.** SAR, EO, MSI, HSI, and thermal imagery
   have fundamentally different processing chains. Stating the modality early
   prevents the LLM from guessing.

2. **State the NIIRS range or resolution.** Processing choices depend heavily
   on image quality — a 1 m SAR image needs different speckle filtering than
   a 0.3 m spotlight collect.

3. **Describe the target, not the algorithm.** "Detect vehicles in SAR" gives
   the LLM room to choose appropriate processors. "Run a median filter then
   threshold at -15 dB" locks in a specific (possibly wrong) approach.

4. **Ask for a DAG when you have branches.** Multi-source fusion, before/after
   change detection, and parallel filter banks all benefit from explicit
   `depends_on` wiring. Mention "branching DAG" in your prompt.

5. **Iterate.** Ask the LLM to explain each step, then refine:
   - *"Why did you choose a 5x5 kernel for the speckle filter?"*
   - *"What happens if I increase num_looks to 5?"*
   - *"Add a tap_out after the CFAR step so I can inspect the raw detections."*

6. **Validate the output.** Load the generated YAML and run validation:
   ```python
   from grdl_rt.api import load_workflow
   wf = load_workflow("generated_workflow.yaml")
   errors = wf.validate()
   if errors:
       for e in errors:
           print(e)
   ```

7. **Feed errors back.** If validation fails, paste the error messages back
   into the conversation — the LLM can usually fix schema issues, unknown
   processor names, or missing parameters in one round.

## Publishing to PyPI

### Dependency Management

All dependencies are defined in `pyproject.toml`. Keep these files synchronized:

- **`pyproject.toml`** — source of truth for versions and dependencies
- **`requirements.txt`** — regenerate with `pip freeze > requirements.txt` after updating `pyproject.toml`
- **`.github/workflows/publish.yml`** — automated PyPI publication (do not edit manually)

### Releasing a New Version

1. Update the `version` field in `pyproject.toml` (semantic versioning: `major.minor.patch`)
2. Update `requirements.txt` if dependencies changed: `pip install -e ".[all,dev]" && pip freeze > requirements.txt`
3. Commit both files
4. Create a git tag: `git tag v0.2.0` (matches version in `pyproject.toml`)
5. Push to GitHub: `git push && git push --tags`
6. Create a GitHub Release from the tag — this triggers the publish workflow automatically

The workflow:
- Builds wheels and source distribution using `python -m build`
- Publishes to PyPI with OIDC authentication (secure, no API keys)
- Artifacts are available at [pypi.org/p/grdl-runtime](https://pypi.org/p/grdl-runtime)

See [CLAUDE.md](CLAUDE.md#dependency-management) for detailed dependency management guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
