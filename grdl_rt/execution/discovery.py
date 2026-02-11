# -*- coding: utf-8 -*-
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
import logging
from typing import Dict, Optional, Set

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level catalog state
# ---------------------------------------------------------------------------

_catalog: Optional[ArtifactCatalogBase] = None


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
    _catalog = catalog


def _get_catalog() -> ArtifactCatalogBase:
    """Return the active catalog, creating a default if needed."""
    global _catalog
    if _catalog is None:
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        _catalog = SqliteArtifactCatalog()
        logger.info("Discovery initialised with default SqliteArtifactCatalog")
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


def discover_processors() -> Dict[str, type]:
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
    processors: Dict[str, type] = {}
    for artifact in catalog.list_artifacts(artifact_type="grdl_processor"):
        if not artifact.processor_class:
            continue
        try:
            cls = _import_class(artifact.processor_class)
            short_name = artifact.processor_class.rsplit(".", 1)[-1]
            processors[short_name] = cls
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Cannot import processor '%s' (%s): %s",
                artifact.name,
                artifact.processor_class,
                exc,
            )
    return processors


def get_processor_tags(cls: type) -> Dict:
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


def get_all_modalities() -> Set:
    """Collect all unique modality values from known processors.

    Returns
    -------
    Set
        Set of modality values found across all discoverable processors.
    """
    modalities: Set = set()
    for _name, cls in discover_processors().items():
        tags = get_processor_tags(cls)
        modalities.update(tags.get("modalities", ()))
    return modalities


def get_all_categories() -> Set:
    """Collect all unique category values from known processors.

    Returns
    -------
    Set
        Set of category values found across all discoverable processors.
    """
    categories: Set = set()
    for _name, cls in discover_processors().items():
        tags = get_processor_tags(cls)
        cat = tags.get("category")
        if cat is not None:
            categories.add(cat)
    return categories


def filter_processors(
    *,
    modality: Optional[str] = None,
    category: Optional[str] = None,
    processor_type: Optional[str] = None,
) -> Dict[str, type]:
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
    result: Dict[str, type] = {}

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
                m.value if hasattr(m, "value") else m
                for m in tags.get("modalities", ())
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
