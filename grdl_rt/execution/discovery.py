"""
Processor Discovery - Catalog-backed processor lookup and filtering.

Provides functions for discovering, resolving, and filtering GRDL
processor classes using an ArtifactCatalogBase backend.  The catalog
can be explicitly configured via :func:`init_discovery`; if not
configured, a default :class:`SqliteArtifactCatalog` is created lazily.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-10
"""

# Standard library
import importlib
import importlib.metadata
import threading
from pathlib import Path

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.federated import FederatedArtifactCatalog
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level catalog state
# ---------------------------------------------------------------------------

_catalog: ArtifactCatalogBase | None = None
_catalog_lock = threading.Lock()


def init_discovery(catalog: ArtifactCatalogBase) -> None:
    """Configure the discovery module with a catalog backend.

    Must be called before any discovery functions are used.  If not
    called, a default :class:`SqliteArtifactCatalog` is created lazily
    on first access.

    Parameters
    ----------
    catalog : ArtifactCatalogBase
        The catalog to use for processor lookup.
    """
    global _catalog
    with _catalog_lock:
        _catalog = catalog


def discover_entry_point_catalogs(
    group: str = "grdl.catalogs",
) -> list[ArtifactCatalogBase]:
    """Load catalogs registered via Python entry points (``grdl.catalogs``).

    Each entry point may resolve to:
    - a callable returning a ``Path``/``str`` to a ``catalog.yaml`` file (or an
      ``ArtifactCatalogBase`` instance), or
    - a ``Path``/``str`` directly.

    If a returned ``catalog.yaml`` path does not exist on disk, falls back to
    dynamic decorator introspection via :mod:`grdl.discovery.catalog_builder`.
    """
    catalogs: list[ArtifactCatalogBase] = []
    eps = importlib.metadata.entry_points(group=group)

    for ep in sorted(eps, key=lambda e: e.name):
        try:
            resolved = None
            try:
                target = ep.load()
                resolved = target() if callable(target) else target
            except (AttributeError, ImportError) as load_exc:
                # PEP 420 namespace shadowing (e.g. CWD contains an uninstalled
                # source folder with the same top-level package name) can mask
                # entry-point targets; locate bundled catalog.yaml via __path__.
                pkg_name = ep.module.split(".")[0]
                pkg = importlib.import_module(pkg_name)
                for p in getattr(pkg, "__path__", []):
                    for candidate in (
                        Path(p) / pkg_name / "catalog.yaml",
                        Path(p) / "catalog.yaml",
                    ):
                        if candidate.exists():
                            resolved = candidate
                            break
                    if resolved is not None:
                        break
                if resolved is None:
                    raise load_exc

            if isinstance(resolved, ArtifactCatalogBase):
                catalogs.append(resolved)
                continue

            cat_path = Path(resolved)
            if cat_path.exists():
                catalogs.append(YamlArtifactCatalog(file_path=cat_path))
                logger.info("Loaded entry-point catalog", name=ep.name, path=str(cat_path))
                continue

            # Dynamic introspection fallback when catalog.yaml hasn't been exported
            from grdl.discovery.catalog_builder import build_catalog_data

            pkg_name = ep.module.split(".")[0]
            data = build_catalog_data(pkg_name)
            mem_cat = YamlArtifactCatalog.__new__(YamlArtifactCatalog)
            mem_cat._file_path = cat_path
            mem_cat._meta = data.get("catalog_meta", {"format_version": 1, "next_id": 1})
            mem_cat._artifacts = data.get("artifacts", [])
            mem_cat._remote_versions = data.get("remote_versions", [])
            catalogs.append(mem_cat)
            logger.info("Synthesized entry-point catalog via introspection", name=ep.name)
        except Exception as exc:
            logger.warning("Failed to load entry-point catalog", name=ep.name, error=str(exc))

    return catalogs


def build_default_catalog() -> ArtifactCatalogBase:
    """Build the default federated catalog from local DB/YAML + entry points.

    Resolution priority (first-catalog-wins):
    1. Local user SQLite catalog (``SqliteArtifactCatalog``)
    2. Local user YAML catalog (``~/.grdl/catalog.yaml`` if present)
    3. Installed ``grdl.catalogs`` entry-point packages (``grdl``, ``grdl-sar``,
       ``grdl-eo``, and third-party extensions)
    """
    from grdl_rt.catalog.database import SqliteArtifactCatalog

    backends: list[ArtifactCatalogBase] = [SqliteArtifactCatalog()]

    user_yaml = Path.home() / ".grdl" / "catalog.yaml"
    if user_yaml.exists():
        backends.append(YamlArtifactCatalog(file_path=user_yaml))

    backends.extend(discover_entry_point_catalogs())

    if len(backends) == 1:
        return backends[0]
    return FederatedArtifactCatalog(backends)


def _get_catalog() -> ArtifactCatalogBase:
    """Return the active catalog, creating the default federated catalog if needed."""
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = build_default_catalog()
            logger.info("Discovery initialised with default federated catalog")
        return _catalog


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _import_class(fqn: str) -> type:
    """Import a class by its fully-qualified dotted name.

    Parameters
    ----------
    fqn : str
        Dotted path, e.g. ``'grdl.image_processing.filters.LeeFilter'``.

    Returns
    -------
    type
        The imported class.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the module does not contain the named attribute.
    """
    module_path, _, class_name = fqn.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid fully-qualified name '{fqn}': no module path.")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_processor_class(name: str) -> type:
    """Resolve a processor name to its Python class.

    Resolution order:

    1. If *name* contains a dot, attempt a direct fully-qualified import.
    2. Search the catalog for a ``grdl_processor`` artifact whose short
       class name or artifact name matches *name*.
    3. Raise :class:`ImportError` if resolution fails.

    Parameters
    ----------
    name : str
        Processor short name (e.g. ``'PauliDecomposition'``) or
        fully-qualified class path (e.g.
        ``'grdl.image_processing.decomposition.pauli.PauliDecomposition'``).

    Returns
    -------
    type
        The processor class.

    Raises
    ------
    ImportError
        If the processor cannot be resolved.
    """
    # Phase 1: Direct FQN import
    if "." in name:
        try:
            return _import_class(name)
        except (ImportError, AttributeError):
            pass  # fall through to catalog lookup

    # Phase 2: Catalog lookup
    catalog = _get_catalog()
    for artifact in catalog.list_artifacts(artifact_type="grdl_processor"):
        if not artifact.processor_class:
            continue
        fqn = artifact.processor_class
        short = fqn.rsplit(".", 1)[-1]
        if short == name or artifact.name == name:
            try:
                return _import_class(fqn)
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    f"Found catalog entry for '{name}' with class "
                    f"'{fqn}' but import failed: {exc}"
                ) from exc

    # Phase 3: Nothing matched
    raise ImportError(f"Cannot resolve processor '{name}': not found in catalog.")


def discover_processors() -> dict[str, type]:
    """Discover all known processor classes from the catalog.

    Returns a mapping of processor short class name to its Python class.
    Artifacts without a ``processor_class`` or whose class cannot be
    imported are silently skipped (logged as warnings).

    Returns
    -------
    Dict[str, type]
        Mapping of short name to class.
    """
    catalog = _get_catalog()
    processors: dict[str, type] = {}
    for artifact in catalog.list_artifacts(artifact_type="grdl_processor"):
        if not artifact.processor_class:
            continue
        try:
            cls = _import_class(artifact.processor_class)
            short_name = artifact.processor_class.rsplit(".", 1)[-1]
            processors[short_name] = cls
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Cannot import processor",
                artifact_name=artifact.name,
                processor_class=artifact.processor_class,
                error=str(exc),
            )
    return processors


def get_processor_tags(cls: type) -> dict:
    """Get tag metadata from a processor class.

    Reads the ``__processor_tags__`` attribute typically set by the
    ``@processor_tags`` decorator in ``grdl.image_processing.versioning``.

    Parameters
    ----------
    cls : type
        A processor class.

    Returns
    -------
    Dict
        Tag dictionary, or empty dict if no tags are declared.
    """
    return getattr(cls, "__processor_tags__", {})


def has_global_pass(cls: type) -> bool:
    """Check whether a processor class requires a global pass.

    Reads the ``__has_global_pass__`` flag set by
    ``ImageProcessor.__init_subclass__`` when any method is decorated
    with ``@globalprocessor``.

    Parameters
    ----------
    cls : type
        A processor class.

    Returns
    -------
    bool
        True if the processor has global-pass callbacks.
    """
    return bool(getattr(cls, "__has_global_pass__", False))


def get_gpu_capability(cls: type) -> str:
    """Determine a processor's GPU capability.

    Reads ``__processor_tags__['gpu_capability']`` if present.  Falls
    back to the legacy ``__gpu_compatible__`` class attribute:

    - ``__gpu_compatible__ = True`` → ``"preferred"``
    - ``__gpu_compatible__ = False`` → ``"cpu_only"``
    - No attribute → ``"preferred"`` (optimistic, matching GpuBackend)

    Parameters
    ----------
    cls : type
        A processor class.

    Returns
    -------
    str
        One of ``"required"``, ``"preferred"``, ``"cpu_only"``.
    """
    tags = getattr(cls, "__processor_tags__", {})
    gpu_cap = tags.get("gpu_capability")
    if gpu_cap is not None:
        # It's a GpuCapability enum — extract its value
        return gpu_cap.value if hasattr(gpu_cap, "value") else str(gpu_cap)

    # Legacy fallback: __gpu_compatible__
    gpu_compat = getattr(cls, "__gpu_compatible__", None)
    if gpu_compat is True:
        return "preferred"
    if gpu_compat is False:
        return "cpu_only"

    # No information — assume preferred (optimistic)
    return "preferred"


def get_all_modalities() -> set:
    """Collect all unique modality values from known processors.

    Returns
    -------
    Set
        Set of modality values found across all discoverable processors.
    """
    modalities: set = set()
    for _name, cls in discover_processors().items():
        tags = get_processor_tags(cls)
        modalities.update(tags.get("modalities", ()))
    return modalities


def get_all_categories() -> set:
    """Collect all unique category values from known processors.

    Returns
    -------
    Set
        Set of category values found across all discoverable processors.
    """
    categories: set = set()
    for _name, cls in discover_processors().items():
        tags = get_processor_tags(cls)
        cat = tags.get("category")
        if cat is not None:
            categories.add(cat)
    return categories


def filter_processors(
    *,
    modality: str | None = None,
    category: str | None = None,
    processor_type: str | None = None,
) -> dict[str, type]:
    """Filter known processors by tag criteria.

    Parameters
    ----------
    modality : Optional[str]
        Filter to processors supporting this modality value.
    category : Optional[str]
        Filter to processors in this category value.
    processor_type : Optional[str]
        Filter to processors of this type (``'transform'``,
        ``'detector'``, ``'decomposition'``).

    Returns
    -------
    Dict[str, type]
        Filtered mapping of processor short name to class.
    """
    catalog = _get_catalog()
    result: dict[str, type] = {}

    for artifact in catalog.list_artifacts(artifact_type="grdl_processor"):
        if not artifact.processor_class:
            continue

        if processor_type and artifact.processor_type != processor_type:
            continue

        try:
            cls = _import_class(artifact.processor_class)
        except (ImportError, AttributeError):
            continue

        tags = get_processor_tags(cls)

        if modality:
            modality_values = {
                m.value if hasattr(m, "value") else m for m in tags.get("modalities", ())
            }
            if modality not in modality_values:
                continue

        if category:
            cat = tags.get("category")
            if cat is None:
                continue
            cat_value = cat.value if hasattr(cat, "value") else cat
            if cat_value != category:
                continue

        short_name = artifact.processor_class.rsplit(".", 1)[-1]
        result[short_name] = cls

    return result
