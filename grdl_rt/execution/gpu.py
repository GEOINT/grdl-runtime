"""
GPU Backend Abstraction - Unified GPU dispatch for CuPy and PyTorch.

Provides a GpuBackend class that transparently dispatches array
operations to CuPy (for GRDL ImageTransform acceleration) or
PyTorch (for ML model inference), with automatic CPU fallback
when GPU libraries are not available.

Dependencies
------------
cupy (optional, for GPU-accelerated array operations)
torch (optional, for ML model inference)

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
from typing import Any

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)


def _check_cupy() -> bool:
    """Check if CuPy is available."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _check_torch() -> bool:
    """Check if PyTorch with CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class GpuBackend:
    """Manages GPU device selection and array transfer.

    Provides methods to transfer arrays between CPU and GPU, apply GRDL
    ImageTransform operations with GPU acceleration, and run PyTorch
    models.

    CuPy arrays are API-compatible with numpy, so GRDL processors that
    only use numpy operations get automatic GPU acceleration. Processors
    using scipy fall back to CPU.

    Parameters
    ----------
    prefer_gpu : bool
        If True (default), use GPU when available. If False, always
        use CPU.
    """

    def __init__(self, prefer_gpu: bool = True) -> None:
        self._prefer_gpu = prefer_gpu
        self._cupy_available = _check_cupy() if prefer_gpu else False
        self._torch_available = _check_torch() if prefer_gpu else False
        # Per-call tracking — updated by apply_transform()
        self.last_gpu_used: bool = False
        self.last_gpu_memory_bytes: int | None = None

    @property
    def cupy_available(self) -> bool:
        """Whether CuPy GPU acceleration is available."""
        return self._cupy_available

    @property
    def torch_available(self) -> bool:
        """Whether PyTorch CUDA is available."""
        return self._torch_available

    @property
    def gpu_available(self) -> bool:
        """Whether any GPU backend is available."""
        return self._cupy_available or self._torch_available

    def to_gpu(self, arr: np.ndarray) -> Any:
        """Transfer a numpy array to GPU memory via CuPy.

        Parameters
        ----------
        arr : np.ndarray
            CPU array to transfer.

        Returns
        -------
        cupy.ndarray or np.ndarray
            GPU array if CuPy available, otherwise the original array.
        """
        if self._cupy_available:
            import cupy as cp

            return cp.asarray(arr)
        return arr

    def to_cpu(self, arr: Any) -> np.ndarray:
        """Transfer an array from GPU to CPU numpy.

        Parameters
        ----------
        arr : cupy.ndarray or np.ndarray
            Array to transfer.

        Returns
        -------
        np.ndarray
            CPU numpy array.
        """
        if self._cupy_available:
            import cupy as cp

            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        return np.asarray(arr)

    def apply_transform(
        self,
        transform: Any,
        source: np.ndarray,
        *,
        metadata: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a processor with optional GPU acceleration.

        Uses the GRDL ``execute(metadata, source, **kwargs)`` protocol
        via ``execute_processor()``.  CuPy GPU transfer is only attempted
        for ``ImageTransform`` instances with ``__gpu_compatible__ = True``.

        Parameters
        ----------
        transform : Any
            GRDL ``ImageProcessor`` instance, legacy processor, or callable.
        source : np.ndarray
            Input image array.
        metadata : ImageMetadata, optional
            Image metadata.  If ``None``, a minimal synthetic metadata
            is derived from the source array shape/dtype.
        **kwargs
            Additional parameters forwarded to the processor.

        Returns
        -------
        tuple[Any, ImageMetadata]
            ``(result, updated_metadata)`` when *metadata* was provided.
        np.ndarray
            Raw result when *metadata* was ``None`` (legacy path).
        """
        from grdl_rt.execution.dispatch import (
            _minimal_metadata,
            execute_processor,
            supports_gpu_transfer,
        )

        self.last_gpu_used = False
        self.last_gpu_memory_bytes = None

        have_metadata = metadata is not None
        if not have_metadata:
            # Non-ndarray source (e.g., dict from DAG fan-in) — skip
            # metadata synthesis, use None placeholder.
            metadata = _minimal_metadata(source) if isinstance(source, np.ndarray) else None

        if self._cupy_available and supports_gpu_transfer(transform):
            try:
                import cupy as cp

                mem_before = cp.cuda.Device().mem_info[0]
                gpu_source = self.to_gpu(source)
                result, updated_meta = execute_processor(
                    transform,
                    metadata,
                    gpu_source,
                    **kwargs,
                )
                cpu_result = self.to_cpu(result)
                mem_after = cp.cuda.Device().mem_info[0]
                self.last_gpu_used = True
                self.last_gpu_memory_bytes = max(0, mem_before - mem_after)
                if have_metadata:
                    return cpu_result, updated_meta
                return cpu_result
            except Exception as e:
                logger.warning(
                    "GPU execution failed, falling back to CPU",
                    transform=type(transform).__name__,
                    error=str(e),
                )
        elif self._cupy_available:
            # Log skip for non-GPU-transferable processors
            gpu_ok = getattr(transform, "__gpu_compatible__", None)
            if gpu_ok is False:
                logger.debug(
                    "Skipping GPU (__gpu_compatible__=False)",
                    transform=type(transform).__name__,
                )

        result, updated_meta = execute_processor(
            transform,
            metadata,
            source,
            **kwargs,
        )
        if have_metadata:
            return result, updated_meta
        return result

    def apply_torch_model(
        self,
        model_path: str,
        source: np.ndarray,
        device: str | None = None,
    ) -> np.ndarray:
        """Run a PyTorch model on source imagery.

        Parameters
        ----------
        model_path : str
            Path to the saved PyTorch model (.pt file).
        source : np.ndarray
            Input image array. Will be converted to a torch tensor.
        device : Optional[str]
            Device to run on ('cuda', 'cpu'). If None, auto-selects.

        Returns
        -------
        np.ndarray
            Model output as a numpy array.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for model inference. " "Install with: pip install torch"
            ) from None

        if device is None:
            device = "cuda" if self._torch_available else "cpu"

        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()

        tensor = torch.from_numpy(source).float().to(device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():
            output = model(tensor)

        return output.cpu().numpy()

    @property
    def device_info(self) -> dict[str, Any]:
        """Current GPU device information.

        Returns
        -------
        Dict[str, Any]
            Device name, memory info, and availability flags.
        """
        info: dict[str, Any] = {
            "cupy_available": self._cupy_available,
            "torch_available": self._torch_available,
        }

        if self._cupy_available:
            import cupy as cp

            dev = cp.cuda.Device()
            info["cupy_device"] = {
                "id": dev.id,
                "name": str(dev),
            }

        if self._torch_available:
            import torch

            info["torch_device"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_mem,
            }

        return info
