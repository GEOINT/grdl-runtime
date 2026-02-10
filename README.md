# grdl-runtime

Headless execution engine for GRDL workflows.

**grdl-runtime** is the reference implementation of a runtime environment for [grdl](../grdl/) components and [grdk](../grdk/) orchestrated processing worklflows. It also includes the base domain model for gdrl component catalog and orchestrated workflow, purely to keep things simple.

grdl-runtime interprets an image processing workflow, determines the optimal execution path given the execution environment (available hardware, image-specific handling, etc) and executes it, handling cross-component hetergenity (ie CPU-GPU, nparray-tensor, etc). 


## Architecture

```
grdl  (processing primitives)
  ↓
grdl-runtime  (workflow engine, catalog, GPU orchestration)   ← this package
  ↓
grdk  (Qt/Orange GUI widgets)
```

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

- **`grdl_rt.execution`** — Workflow definition, DSL compilation, step execution, GPU backend, processor discovery
- **`grdl_rt.catalog`** — Artifact storage (SQLite + FTS5), resolver, update management, connection pooling
