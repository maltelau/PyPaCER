"""GPU utilities and availability detection for PyPaCER."""

import warnings
from typing import Any, Optional, Union

import numpy as np

# PyTorch is the only GPU backend now
try:
    import torch

    PYTORCH_AVAILABLE = True
    PYTORCH_GPU_AVAILABLE = torch.cuda.is_available()
    if not PYTORCH_GPU_AVAILABLE:
        warnings.warn(
            "PyTorch is installed but no CUDA-capable GPU was found. "
            "Falling back to CPU processing."
        )
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False
    PYTORCH_GPU_AVAILABLE = False


def gpu_available() -> bool:
    """Check if GPU acceleration is available (PyTorch with CUDA)."""
    return pytorch_gpu_available()


def pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return PYTORCH_AVAILABLE


def pytorch_gpu_available() -> bool:
    """Check if PyTorch with GPU support is available."""
    return PYTORCH_AVAILABLE and PYTORCH_GPU_AVAILABLE


def get_array_module(array: Optional[np.ndarray] = None, use_gpu: bool = False):
    """
    Get the appropriate array module (numpy for now, PyTorch operations handled separately).

    Args:
        array: Optional array to get module from
        use_gpu: Whether to use GPU if available

    Returns:
        numpy module (PyTorch tensors are handled separately in processing code)
    """
    # For now, always return numpy as PyTorch operations are handled separately
    return np


def to_gpu(
    array: Union[np.ndarray, Any], dtype: Optional[np.dtype] = None
) -> Union[np.ndarray, Any]:
    """
    Transfer array to GPU if available, otherwise return as-is.
    Note: This is now a compatibility function. Actual GPU transfers should use PyTorch directly.

    Args:
        array: Input array
        dtype: Optional dtype to cast to

    Returns:
        Original array (GPU transfers should be done with PyTorch tensors)
    """
    return array.astype(dtype) if dtype else array


def from_gpu(array: Union[np.ndarray, Any]) -> np.ndarray:
    """
    Transfer array from GPU to CPU.
    Note: This is now a compatibility function. Actual GPU transfers should use PyTorch directly.

    Args:
        array: Input array (GPU or CPU)

    Returns:
        CPU numpy array
    """
    if hasattr(array, "cpu"):
        # PyTorch tensor - transfer to CPU
        return array.cpu().numpy()
    else:
        # Already CPU array
        return np.asarray(array)


def ensure_gpu_array(
    array: Union[np.ndarray, Any], use_gpu: bool = True
) -> Union[np.ndarray, Any]:
    """
    Ensure array is on the correct device (GPU or CPU).
    Note: This is now a compatibility function. Actual GPU operations should use PyTorch directly.

    Args:
        array: Input array
        use_gpu: Whether to put on GPU

    Returns:
        Array on appropriate device
    """
    # For compatibility, just return the array as-is
    # Actual GPU operations should use PyTorch tensors directly
    return array


class GPUMemoryManager:
    """Manage GPU memory allocation and transfers."""

    def __init__(self, max_memory_mb: int = 1024):
        """
        Initialize GPU memory manager.

        Args:
            max_memory_mb: Maximum GPU memory to use in MB
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.use_gpu = PYTORCH_GPU_AVAILABLE

        if self.use_gpu and torch is not None:
            # PyTorch memory management is handled automatically
            # Can set memory fraction if needed
            pass

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.use_gpu and torch is not None:
            torch.cuda.empty_cache()

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage information."""
        if not self.use_gpu or torch is None:
            return {"available": False}

        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024

        return {
            "available": True,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": reserved - allocated,
        }


# Utility functions for common operations (compatibility layer)
def gpu_zeros(shape, dtype=np.float32, use_gpu=True):
    """Create zeros array on GPU or CPU."""
    # For compatibility, return numpy array
    # Actual GPU operations should use torch.zeros directly
    return np.zeros(shape, dtype=dtype)


def gpu_ones(shape, dtype=np.float32, use_gpu=True):
    """Create ones array on GPU or CPU."""
    # For compatibility, return numpy array
    # Actual GPU operations should use torch.ones directly
    return np.ones(shape, dtype=dtype)


def gpu_arange(start, stop, step=1, dtype=np.float32, use_gpu=True):
    """Create arange array on GPU or CPU."""
    # For compatibility, return numpy array
    # Actual GPU operations should use torch.arange directly
    return np.arange(start, stop, step, dtype=dtype)
