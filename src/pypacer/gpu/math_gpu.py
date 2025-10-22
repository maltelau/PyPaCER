"""GPU-accelerated mathematical operations for PyPaCER."""

from typing import Union

import numpy as np

from .gpu_utils import pytorch_available

# Only PyTorch is used for GPU acceleration now
if pytorch_available():
    pass


def polyval3_gpu(
    polynomial: np.ndarray, t_values: Union[float, np.ndarray], use_gpu: bool = True
) -> np.ndarray:
    """
    Evaluate 3D polynomial using PyTorch if available.

    Args:
        polynomial: Polynomial coefficients (degree+1, 3)
        t_values: Parameter values to evaluate
        use_gpu: Whether to use GPU

    Returns:
        Points in 3D space
    """
    # For now, fall back to CPU implementation
    # PyTorch polynomial evaluation is handled directly in the processing code
    from ..utils.math_helpers import polyval3

    return polyval3(polynomial, t_values)


def polyval3_batch_gpu(
    polynomial: np.ndarray, t_values: np.ndarray, use_gpu: bool = True
) -> np.ndarray:
    """
    Batch evaluate 3D polynomial using PyTorch if available.

    Args:
        polynomial: Polynomial coefficients (degree+1, 3)
        t_values: Array of parameter values
        use_gpu: Whether to use GPU

    Returns:
        Array of points in 3D space
    """
    # For now, fall back to CPU implementation
    # PyTorch polynomial evaluation is handled directly in the processing code
    from ..utils.math_helpers import polyval3

    return polyval3(polynomial, t_values)
