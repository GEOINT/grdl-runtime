# grdl-runtime

Headless execution engine for GRDL workflows.

**grdl-runtime** sits between [grdl](../grdl/) (the processing library) and
[grdk](../grdk/) (the GUI toolkit), providing workflow execution, artifact
catalog management, and GPU orchestration — all without any GUI framework
dependencies.

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
