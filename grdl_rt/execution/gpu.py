# -*- coding: utf-8 -*-
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
import logging
from typing import Any, Dict, Optional, Union

# Third-party
import numpy as np

logger = logging.getLogger(__name__)


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
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply a GRDL ImageTransform with optional GPU acceleration.

        Attempts to run the transform on GPU by converting the source
        to a CuPy array. If the transform raises an error (e.g., due
        to scipy usage), falls back to CPU execution.

        Parameters
        ----------
        transform : ImageTransform
            GRDL image transform instance.
        source : np.ndarray
            Input image array.
        **kwargs
            Tunable parameters passed to transform.apply().

        Returns
        -------
        np.ndarray
            Transformed image (always on CPU).
        """
        if self._cupy_available:
            # Check GRDL __gpu_compatible__ flag — skip GPU for
            # scipy-dependent processors that will always fail.
            gpu_ok = getattr(transform, '__gpu_compatible__', None)
            if gpu_ok is False:
                logger.debug(
                    "Skipping GPU for %s (__gpu_compatible__=False)",
                    type(transform).__name__,
                )
            else:
                try:
                    gpu_source = self.to_gpu(source)
                    result = transform.apply(gpu_source, **kwargs)
                    return self.to_cpu(result)
                except Exception as e:
                    logger.warning(
                        "GPU execution failed for %s, falling back to CPU: %s",
                        type(transform).__name__, e,
                    )

        return transform.apply(source, **kwargs)

    def apply_torch_model(
        self,
        model_path: str,
        source: np.ndarray,
        device: Optional[str] = None,
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
                "PyTorch is required for model inference. "
                "Install with: pip install torch"
            )

        if device is None:
            device = 'cuda' if self._torch_available else 'cpu'

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
    def device_info(self) -> Dict[str, Any]:
        """Current GPU device information.

        Returns
        -------
        Dict[str, Any]
            Device name, memory info, and availability flags.
        """
        info: Dict[str, Any] = {
            'cupy_available': self._cupy_available,
            'torch_available': self._torch_available,
        }

        if self._cupy_available:
            import cupy as cp
            dev = cp.cuda.Device()
            info['cupy_device'] = {
                'id': dev.id,
                'name': str(dev),
            }

        if self._torch_available:
            import torch
            info['torch_device'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_mem,
            }

        return info
