# -*- coding: utf-8 -*-
"""
Processor Discovery - Scan GRDL modules for available processors.

Scans both ``grdl.image_processing`` and ``grdl.coregistration`` for
concrete processor classes that can be used in workflows.

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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
import importlib
import inspect
from typing import Any, Dict, List, Optional, Set, Type


def discover_processors() -> Dict[str, Any]:
    """Discover available GRDL ImageTransform/Detector/CoRegistration classes.

    Scans ``grdl.image_processing`` for classes with an ``apply`` method
    and ``grdl.coregistration`` for classes with an ``estimate`` method.

    Returns
    -------
    Dict[str, Any]
        Mapping of class name to class object.
    """
    processors: Dict[str, Any] = {}

    # Scan grdl.image_processing
    try:
        ip_module = importlib.import_module('grdl.image_processing')
        for name, obj in inspect.getmembers(ip_module, inspect.isclass):
            if hasattr(obj, 'apply') and not inspect.isabstract(obj):
                processors[name] = obj
    except ImportError:
        pass

    # Scan grdl.coregistration
    try:
        coreg_module = importlib.import_module('grdl.coregistration')
        for name, obj in inspect.getmembers(coreg_module, inspect.isclass):
            if hasattr(obj, 'estimate') and not inspect.isabstract(obj):
                processors[name] = obj
    except ImportError:
        pass

    return processors


def resolve_processor_class(processor_name: str) -> Type:
    """Resolve a processor class name to the actual class.

    Supports both short names (e.g., ``"PauliDecomposition"``) resolved
    by scanning GRDL modules, and fully-qualified names (e.g.,
    ``"grdl.image_processing.decomposition.pauli.PauliDecomposition"``).

    Parameters
    ----------
    processor_name : str
        Short or fully-qualified processor class name.

    Returns
    -------
    Type
        The processor class.

    Raises
    ------
    ImportError
        If the processor class cannot be found.
    """
    # Try fully-qualified import first
    if '.' in processor_name:
        module_path, class_name = processor_name.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # Try scanning known GRDL modules
    all_processors = discover_processors()
    if processor_name in all_processors:
        return all_processors[processor_name]

    raise ImportError(
        f"Cannot resolve processor class '{processor_name}'. "
        f"Use a fully-qualified name (e.g., "
        f"'grdl.image_processing.ortho.ortho.Orthorectifier')."
    )


def get_processor_tags(proc_class: Any) -> Dict[str, Any]:
    """Extract ``__processor_tags__`` from a GRDL processor class.

    Parameters
    ----------
    proc_class : Any
        A GRDL processor class.

    Returns
    -------
    Dict[str, Any]
        Tags dict with keys ``modalities`` (tuple of str),
        ``category`` (str or None), ``description`` (str or None).
        Returns empty dict if no tags are present.
    """
    tags = getattr(proc_class, '__processor_tags__', {})
    tags['processor_version'] = getattr(proc_class, '__processor_version__', 'unknown')
    return tags


def get_all_modalities() -> Set[str]:
    """Collect all unique modality strings across discovered processors.

    Returns
    -------
    Set[str]
        e.g. ``{'SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'}``
    """
    modalities: Set[str] = set()
    for cls in discover_processors().values():
        tags = get_processor_tags(cls)
        modalities.update(tags.get('modalities', ()))
    return modalities


def get_all_categories() -> Set[str]:
    """Collect all unique category strings across discovered processors.

    Returns
    -------
    Set[str]
        e.g. ``{'spatial_filter', 'contrast_enhancement', ...}``
    """
    categories: Set[str] = set()
    for cls in discover_processors().values():
        tags = get_processor_tags(cls)
        cat = tags.get('category')
        if cat:
            categories.add(cat)
    return categories


def filter_processors(
    modality: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Return processors matching the given modality and/or category.

    Parameters
    ----------
    modality : Optional[str]
        If set, only return processors whose ``__processor_tags__``
        include this modality.
    category : Optional[str]
        If set, only return processors with this category.

    Returns
    -------
    Dict[str, Any]
        Filtered mapping of class name to class object.
    """
    result: Dict[str, Any] = {}
    for name, cls in discover_processors().items():
        tags = get_processor_tags(cls)
        if modality and modality not in tags.get('modalities', ()):
            continue
        if category and tags.get('category') != category:
            continue
        result[name] = cls
    return result
