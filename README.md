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

## License

MIT License — see [LICENSE](LICENSE) for details.
