"""
Polymorphic Processor Dispatch — Central dispatch for ``execute()`` protocol.

Provides ``execute_processor()`` which calls the GRDL ``execute(metadata,
source, **kwargs)`` protocol on any ``ImageProcessor``, with fallback for
raw callables and legacy processors that only have ``apply()``.

Also provides ``supports_gpu_transfer()`` to determine whether CuPy GPU
array transfer is appropriate for a given processor.

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
2026-02-12
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from grdl_rt.execution.context import get_logger

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = get_logger(__name__)

# GRDL base classes (optional — graceful fallback)
try:
    from grdl.image_processing.base import ImageTransform
except ImportError:
    ImageTransform = None  # type: ignore[misc,assignment]

try:
    from grdl.image_processing.base import ImageProcessor
except ImportError:
    ImageProcessor = None  # type: ignore[misc,assignment]

try:
    from grdl.image_processing.vector.base import VectorProcessor
except ImportError:
    VectorProcessor = None  # type: ignore[misc,assignment]


def execute_processor(
    processor: Any,
    metadata: ImageMetadata,
    source: Any,
    **kwargs: Any,
) -> tuple[Any, ImageMetadata]:
    """Call processor.execute() with fallback for legacy processors.

    Dispatch order:

    1. ``processor.execute(metadata, source, **kwargs)`` — GRDL ``execute()``
       protocol.  Preferred path for all ``ImageProcessor`` subclasses.
    2. ``processor.apply(source, **kwargs)`` — Legacy fallback for
       third-party processors or old code that only has ``apply()``.
    3. ``processor(source, **kwargs)`` — Raw callable fallback.

    Parameters
    ----------
    processor : Any
        An ``ImageProcessor`` instance, a legacy processor with ``apply()``,
        or any callable ``(source, **kwargs) -> result``.
    metadata : ImageMetadata
        Input image metadata.
    source : Any
        Input data (typically ``np.ndarray``).
    **kwargs
        Additional arguments forwarded to the processor.

    Returns
    -------
    tuple[Any, ImageMetadata]
        ``(result, updated_metadata)``.

    Raises
    ------
    TypeError
        If the processor cannot be dispatched.
    """
    # Path 1: GRDL ImageProcessor with execute() protocol.
    # Check isinstance first to avoid false positives from MagicMock
    # and other objects that auto-create attributes on access.
    if ImageProcessor is not None and isinstance(processor, ImageProcessor):
        return processor.execute(metadata, source, **kwargs)

    # Path 2: Non-GRDL object with explicit execute() defined on its class.
    if (
        hasattr(processor, "execute")
        and callable(processor.execute)
        and "execute" in type(processor).__dict__
    ):
        return processor.execute(metadata, source, **kwargs)

    # Path 3: Legacy .apply() fallback
    if hasattr(processor, "apply") and callable(processor.apply):
        logger.debug(
            "dispatch_legacy_apply",
            processor=type(processor).__name__,
        )
        result = processor.apply(source, **kwargs)
        return result, metadata

    # Path 4: Raw callable
    if callable(processor):
        logger.debug(
            "dispatch_raw_callable",
            processor=getattr(processor, "__name__", repr(processor)),
        )
        result = processor(source, **kwargs)
        return result, metadata

    raise TypeError(
        f"Cannot dispatch to {type(processor).__name__}: no execute(), "
        f"apply(), or __call__ method found."
    )


def supports_gpu_transfer(processor: Any) -> bool:
    """Check if a processor benefits from CuPy GPU array transfer.

    Any ``ImageProcessor`` subclass that explicitly sets
    ``__gpu_compatible__ = True`` will receive a CuPy array.  The flag
    defaults to ``False`` on ``ImageProcessor``, so only processors that
    opt in are affected.  Processors that use scipy, shapely, or other
    CPU-only libraries must leave the flag at its default ``False``.
    Processors that set the flag but fail on GPU are caught by
    ``GpuBackend``'s existing warning + CPU fallback.

    Parameters
    ----------
    processor : Any
        Processor instance to check.

    Returns
    -------
    bool
    """
    # VectorProcessor subclasses never benefit from GPU transfer
    if VectorProcessor is not None and isinstance(processor, VectorProcessor):
        return False
    return bool(getattr(processor, "__gpu_compatible__", False))


def _minimal_metadata(source: np.ndarray) -> ImageMetadata:
    """Synthesize minimal ``ImageMetadata`` from an ndarray.

    Used as a backward-compatibility shim when callers don't provide
    metadata.  The result has enough fields for ``execute()`` to work
    but no semantic content (no CRS, no format info).

    Parameters
    ----------
    source : np.ndarray
        Input array to derive metadata from.

    Returns
    -------
    ImageMetadata
    """
    from grdl.IO.models.base import ImageMetadata

    rows = source.shape[0] if source.ndim >= 2 else 1
    cols = source.shape[1] if source.ndim >= 2 else source.shape[0]
    bands = source.shape[2] if source.ndim == 3 else 1

    return ImageMetadata(
        format="synthetic",
        rows=rows,
        cols=cols,
        dtype=str(source.dtype),
        bands=bands,
    )
