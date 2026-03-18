"""
Hardware Context — Abstract hardware capability description.

Provides ``HardwareContext`` (ABC describing available CPU, memory, and GPU
resources), ``LocalHardwareContext`` (queries the local machine), and
``GpuDeviceInfo`` (per-device GPU metadata).

The resolver uses ``HardwareContext`` to plan execution paths — selecting
GPU processors when hardware is available and substituting CPU alternatives
when it is not.

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
2026-02-11
"""

# Standard library
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# grdl-runtime internal
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GpuDeviceInfo:
    """Information about a single GPU device.

    Attributes
    ----------
    name : str
        Human-readable device name (e.g., ``'NVIDIA RTX 4090'``).
    memory_bytes : int
        Total device memory in bytes.
    device_index : int
        Zero-based device index.
    """

    name: str
    memory_bytes: int
    device_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "name": self.name,
            "memory_bytes": self.memory_bytes,
            "device_index": self.device_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GpuDeviceInfo":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        GpuDeviceInfo
        """
        return cls(
            name=data["name"],
            memory_bytes=data["memory_bytes"],
            device_index=data.get("device_index", 0),
        )


class HardwareContext(ABC):
    """Abstract description of available hardware resources.

    Concrete implementations query the environment (local machine,
    cluster node, container) and expose a uniform interface for
    the execution resolver to plan against.
    """

    @property
    @abstractmethod
    def cpu_count(self) -> int:
        """Number of logical CPUs available."""
        ...

    @property
    @abstractmethod
    def total_memory_bytes(self) -> int:
        """Total system RAM in bytes."""
        ...

    @property
    @abstractmethod
    def available_memory_bytes(self) -> int:
        """Currently available system RAM in bytes."""
        ...

    @property
    @abstractmethod
    def gpu_available(self) -> bool:
        """Whether at least one GPU device is accessible."""
        ...

    @property
    @abstractmethod
    def gpu_devices(self) -> list[GpuDeviceInfo]:
        """List of available GPU devices."""
        ...

    @property
    def gpu_memory_bytes(self) -> int:
        """Total GPU memory across all devices.

        Returns
        -------
        int
            Sum of ``memory_bytes`` across all ``gpu_devices``.
        """
        return sum(d.memory_bytes for d in self.gpu_devices)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the hardware context to a dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        ...


class LocalHardwareContext(HardwareContext):
    """Hardware context for the local machine.

    Queries CPU count via ``os.cpu_count()``, memory via ``psutil``
    (with graceful fallback), and GPU availability via ``torch.cuda``
    and CuPy introspection.  All values are cached at construction
    time — hardware does not change during a run.
    
    GPU device detection is cached at the class level so the probe
    (and any associated warnings) only runs once per process regardless
    of how many ``LocalHardwareContext`` instances are created.
    """

    _gpu_cache: list | None = None  # class-level cache; None = not yet probed

    def __init__(self) -> None:
        self._cpu_count = os.cpu_count() or 1
        self._total_memory, self._available_memory = self._query_memory()
        if LocalHardwareContext._gpu_cache is None:
            LocalHardwareContext._gpu_cache = self._query_gpu_devices()
        self._gpu_devices = LocalHardwareContext._gpu_cache

    @staticmethod
    def _query_memory() -> tuple:
        """Query system memory via psutil with fallback."""
        if _psutil is not None:
            mem = _psutil.virtual_memory()
            return mem.total, mem.available
        # Fallback: unknown memory
        logger.warning(
            "psutil not installed; RAM will report as 0. "
            "Install psutil for accurate memory detection."
        )
        return 0, 0

    @staticmethod
    def _query_gpu_devices() -> list[GpuDeviceInfo]:
        """Discover GPU devices via torch.cuda and CuPy."""
        devices: list[GpuDeviceInfo] = []

        # Try PyTorch CUDA first (richer device info)
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append(
                        GpuDeviceInfo(
                            name=props.name,
                            memory_bytes=props.total_memory,
                            device_index=i,
                        )
                    )
                return devices
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("PyTorch GPU detection failed: %s", exc)

        # Fallback to CuPy
        try:
            import cupy as cp

            for i in range(cp.cuda.runtime.getDeviceCount()):
                with cp.cuda.Device(i):
                    free, total = cp.cuda.runtime.memGetInfo()
                    devices.append(
                        GpuDeviceInfo(
                            name=f"CUDA Device {i}",
                            memory_bytes=total,
                            device_index=i,
                        )
                    )
            return devices
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("CuPy GPU detection failed: %s", exc)

        return devices

    @property
    def cpu_count(self) -> int:
        return self._cpu_count

    @property
    def total_memory_bytes(self) -> int:
        return self._total_memory

    @property
    def available_memory_bytes(self) -> int:
        return self._available_memory

    @property
    def gpu_available(self) -> bool:
        return len(self._gpu_devices) > 0

    @property
    def gpu_devices(self) -> list[GpuDeviceInfo]:
        return list(self._gpu_devices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "gpu_available": self.gpu_available,
            "gpu_devices": [d.to_dict() for d in self._gpu_devices],
            "gpu_memory_bytes": self.gpu_memory_bytes,
        }
