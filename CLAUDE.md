# grdl-runtime Development Guide

## Project Vision

grdl-runtime is the **execution framework** for the GRDL ecosystem. Where grdl is a library ("use what you need, wire it yourself"), grdl-runtime is a framework ("declare what you want, we handle the wiring").

The core insight: GRDL processing pipelines share a common structure — open a reader, extract metadata, plan chips, construct processors (some of which need metadata), execute steps with GPU acceleration, track progress, handle errors. grdl-runtime codifies this structure so users express intent (a 7-line workflow definition) instead of mechanics (200+ lines of boilerplate).

### Design Philosophy

**Framework, not library.** grdl-runtime inverts control. The user declares the recipe; the framework drives execution. Reader lifecycle, metadata routing, chip planning, processor construction, GPU dispatch — all handled internally.

**Convention over configuration.** Metadata injection is convention-based: if a processor's `__init__` has a parameter named `metadata`, the framework injects it from the reader. No decorators, registration, or configuration. This works with existing GRDL processors without modification.

**Deferred construction.** Processor classes (not instances) are passed to `.step()`. The framework constructs them at execute time when it has the context (metadata, GPU backend) needed to do it correctly. This is what makes the 7-line workflow possible.

**Backward compatible.** Every new capability preserves existing behavior. `execute(np.ndarray)`, `step(callable)`, `step(instance)`, `source(factory)` — all continue to work unchanged. New functionality is additive, triggered by new calling patterns (e.g., `execute("filepath")`).

**IDE-first API.** The `Workflow` builder uses concrete types, not strings. IntelliSense, autocomplete, and type checking work out of the box. Processor constructor parameters are visible when writing `.step(SublookDecomposition, ...)`.

### Architecture Layers

```
grdl  (processing primitives — no framework awareness)
  ↓
grdl-runtime  (execution framework)
  ├── execution/  — workflow engine, builder, GPU backend, discovery, DSL
  └── catalog/    — artifact storage, search, updates
  ↓
grdk  (Qt/Orange GUI — uses grdl-runtime for execution)
```

grdl-runtime depends on grdl. grdk depends on grdl-runtime. grdl-runtime has **no GUI dependencies** — it runs headless in CI, containers, notebooks, and scripts.

## Development Environment

**Python Environment:** Use the `starlight` conda environment for all Python operations.

```bash
conda activate starlight
pytest tests/ -p no:napari -x -q
```

The `-p no:napari` flag is required to avoid napari plugin conflicts on Python 3.14.

## Architecture Rules

### Execution Subpackage (`grdl_rt.execution`)

This is the core of the framework. Key modules and their responsibilities:

| Module | Responsibility | Key Types |
|--------|---------------|-----------|
| `builder.py` | Fluent workflow builder + framework orchestration | `Workflow`, `WorkflowStep`, `DeferredStep` |
| `executor.py` | String-based pipeline execution (for YAML workflows) | `WorkflowExecutor` |
| `workflow.py` | Serializable workflow models | `WorkflowDefinition`, `ProcessingStep`, `WorkflowState` |
| `dsl.py` | Python ↔ YAML DSL compilation | `DslCompiler`, `@step`, `@workflow` |
| `discovery.py` | Processor scanning and filtering | `discover_processors()`, `filter_processors()` |
| `gpu.py` | CuPy GPU dispatch with CPU fallback | `GpuBackend` |
| `tags.py` | Taxonomy enums (re-exported from `grdl.vocabulary`) | `ImageModality`, `ExecutionPhase`, `OutputFormat`, `WorkflowTags`, `ProjectTags` |
| `chip.py` | Chip data models | `Chip`, `ChipSet`, `ChipLabel`, `PolygonRegion` |
| `config.py` | Runtime configuration | `GrdkConfig`, `load_config()` |
| `project.py` | Project directory structure | `GrdkProject` |

### Catalog Subpackage (`grdl_rt.catalog`)

Artifact storage with multiple backends:

| Module | Responsibility | Key Types |
|--------|---------------|-----------|
| `base.py` | Catalog ABC | `ArtifactCatalogBase` |
| `database.py` | SQLite + FTS5 catalog | `SqliteArtifactCatalog` |
| `yaml_catalog.py` | YAML-based catalog | `YamlArtifactCatalog` |
| `federated.py` | Multi-source aggregation | `FederatedArtifactCatalog` |
| `models.py` | Data models | `Artifact`, `UpdateResult` |
| `resolver.py` | Path resolution (env → config → default) | `resolve_catalog_path()` |
| `updater.py` | PyPI/Conda update checking | `ArtifactUpdateWorker` |
| `pool.py` | Background task management | `ThreadExecutorPool` |

### Framework Orchestration (`Workflow.execute`)

The framework execution path is triggered when `execute()` receives a filepath. The dispatch logic:

```
execute(source) →
  str/Path  → _execute_from_file()   ← framework mode
  ndarray   → _run_pipeline()         ← direct mode
  None      → _source() or error      ← source mode
```

**Framework mode internals:**

1. `_execute_from_file()` — opens reader as context manager, extracts metadata
2. `_read_chip()` — uses `ChipExtractor` for center/full strategies
3. `_resolve_steps()` — iterates steps, resolves `DeferredStep` instances
4. `_resolve_deferred()` → `_construct_processor()` — inspects `__init__` signature, injects metadata if parameter exists
5. `_run_pipeline()` — sequential execution with GPU dispatch and progress callbacks

### Processor Dispatch Protocol

At runtime, each step is dispatched via `execute_processor()` (in `dispatch.py`), which uses grdl's `ImageProcessor.execute(metadata, source, **kwargs)` protocol. This is the **primary execution path** — it handles `ImageTransform`, `ImageDetector`, `PolarimetricDecomposition`, `WorkflowOperator`, and raw callables polymorphically.

### Metadata Injection Convention (Deferred Construction)

When a processor **class** (not instance) is passed to `.step()`, it becomes a `DeferredStep`. At execute time, `_resolve_deferred()` → `_construct_processor()` inspects `cls.__init__` via `inspect.signature()`. If a parameter named `metadata` exists and the user didn't provide it in kwargs, the framework injects it from the reader:

```python
# This constructor signature → metadata will be injected at construction
class SublookDecomposition:
    def __init__(self, metadata: SICDMetadata, num_looks=2, ...): ...

# This constructor signature → no injection, just kwargs
class ToDecibels:
    def __init__(self, floor_db=-60.0): ...
```

User-provided `metadata` in kwargs always takes precedence over injection. This convention-based injection is a **secondary path** used only during deferred construction — the primary execution protocol is `execute(metadata, source, **kwargs)`.

### No Circular Dependencies

grdl-runtime depends on grdl. grdl does **not** depend on grdl-runtime. All framework-level concepts (workflows, execution, catalog) live here. Examples and tests that demonstrate framework capabilities belong in grdl-runtime, not in grdl.

### No GUI Dependencies

grdl-runtime must run headless. No Qt, no PyQt6, no display requirements. All visualization and GUI concerns belong in grdk.

### Graceful GRDL Fallbacks

grdl is an optional dependency for some grdl-runtime functionality (catalog and config work without it). Use `try/except ImportError` guards at module level:

```python
try:
    from grdl.image_processing.base import ImageTransform
except ImportError:
    ImageTransform = None
```

## Workflow Builder Design

### Step Types

The `Workflow.step()` method accepts three forms:

1. **Class (deferred)** — `step(SublookDecomposition, num_looks=3)` → stored as `DeferredStep`, constructed at execute time with metadata injection. `**kwargs` forwarded to constructor.

2. **Instance** — `step(ToDecibels(floor_db=-50.0))` → `ImageTransform` instances are wrapped to call `.apply()`. GPU compatibility read from `__gpu_compatible__`.

3. **Callable** — `step(my_fn, name="Custom")` → used directly. GPU compatibility inferred from `__self__.__gpu_compatible__` for bound methods.

`**kwargs` are only valid for class-type steps. Passing kwargs with a non-class step raises `TypeError`.

### Builder Methods

| Method | Purpose | When used |
|--------|---------|-----------|
| `.reader(cls)` | Declare reader class | Framework mode (execute with filepath) |
| `.chip(strategy, **kw)` | Declare chip strategy | Framework mode |
| `.step(cls_or_obj, **kw)` | Add processing step | Always |
| `.source(fn, *args, **kw)` | Set deferred data source | Source mode (execute with no args) |
| `.execute(source, ...)` | Run the pipeline | Always |
| `.execute_batch(sources, ...)` | Run on multiple inputs | Batch processing |

### Chip Strategies

| Strategy | Behavior |
|----------|----------|
| `"center"` | Center chip of given `size` (default 5000). Uses `ChipExtractor.chip_at_point()` |
| `"full"` | Read entire image (no chipping) |
| `None` | Same as `"full"` |

## Standards

Follow GRDL's development standards (see `../grdl/CLAUDE.md`):

- **PEP 8/257/484** — naming, docstrings, type hints
- **NumPy-style docstrings** — Parameters, Returns, Raises sections
- **File headers** — encoding, title, author, license, dates
- **Imports** — three groups (stdlib, third-party, grdl-runtime/grdl internal)
- **No global state** — no singletons, no module-level side effects
- **Fail fast** — clear exceptions for missing dependencies or invalid configuration

### File Header Format

```python
# -*- coding: utf-8 -*-
"""
Module title — short description.

Extended description of the module's role in grdl-runtime.

Author
------
<Author Name>

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
YYYY-MM-DD

Modified
--------
YYYY-MM-DD
"""
```

## Testing

Tests live in `tests/` and follow `test_<module>.py` naming.

```bash
pytest tests/ -p no:napari -x -q          # Quick run
pytest tests/ -v                           # Verbose
pytest tests/test_builder.py -v            # Workflow builder tests only
```

### Test Approach

| Module | Strategy |
|--------|----------|
| `builder.py` | Unit tests with mock readers, fake processors, synthetic arrays |
| `executor.py` | Unit tests with mock processors |
| `discovery.py` | Tests against real GRDL processors (integration) |
| `gpu.py` | Tests with CPU fallback (GPU optional) |
| `catalog/` | Unit tests with temp SQLite databases |
| `dsl.py` | Round-trip Python ↔ YAML tests |

All tests use synthetic data. No real imagery files needed.

### Key Test Fixtures

- `_FakeReader` — mock reader with context manager, `metadata`, `get_shape()`, `read_chip()`, `read_full()`
- `_SimpleProcessor` — processor class without metadata parameter
- `_MetadataProcessor` — processor class with metadata parameter (for injection tests)

## Dependency Management

### Source of Truth: `pyproject.toml`

**`pyproject.toml` is the single source of truth** for all dependencies. All package metadata, dependencies, and optional extras are defined here. This file drives PyPI publication and is read by build tools.

### Keeping Files in Sync

Three files must be kept synchronized:

| File | Purpose | How to Update |
|------|---------|---------------|
| `pyproject.toml` | **Source of truth** — package metadata, all dependencies, extras | Edit directly; this is the authoritative definition |
| `requirements.txt` (if it exists) | Development convenience — pinned versions for reproducible environments | `pip freeze > requirements.txt` after updating dependencies in `pyproject.toml` and installing |
| `.github/workflows/publish.yml` | PyPI publication — **DO NOT EDIT this file manually** (it extracts version from `pyproject.toml` automatically) | No action needed; the workflow reads `version` from `pyproject.toml` |

**Workflow:**
1. Update dependencies in `pyproject.toml` (add new packages, change versions, create/rename extras)
2. Install dependencies: `pip install -e ".[all,dev]"` (or appropriate extras for your work)
3. If `requirements.txt` exists in this project, regenerate it: `pip freeze > requirements.txt`
4. Commit both files
5. When creating a release, bump the `version` field in `pyproject.toml` (semantic versioning: `major.minor.patch`)
6. Create a git tag (e.g., `v0.2.0`) and push — the publish workflow triggers automatically

### Versioning for PyPI

- Versions follow **semantic versioning**: `major.minor.patch` (e.g., `0.1.0`, `1.2.3`)
- Update `version = "X.Y.Z"` in `pyproject.toml` before creating a release
- The publish workflow extracts the version automatically — no manual version extraction needed

## Git Practices

Same as GRDL: imperative commit messages, one change per commit, `<type>/<description>` branches.

## Adding New Functionality

### New Chip Strategy

1. Add handling in `Workflow._read_chip()` in `builder.py`
2. Add tests in `test_builder.py` under `TestWorkflowChip` and `TestExecuteFromFile`
3. Document in README.md under "Chip Strategies"

### New Execution Mode

1. Add dispatch logic in `Workflow.execute()`
2. Add internal implementation method (e.g., `_execute_from_<mode>()`)
3. Add tests under a new `TestExecuteFrom<Mode>` class
4. Preserve backward compatibility — existing modes must not break

### New Catalog Backend

1. Implement `ArtifactCatalogBase` ABC in a new module under `catalog/`
2. Add exports to `catalog/__init__.py` and `grdl_rt/__init__.py`
3. Add tests following the existing `test_catalog_*.py` pattern
