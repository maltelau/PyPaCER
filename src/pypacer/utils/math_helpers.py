"""Mathematical utility functions for polynomial operations."""

from typing import Tuple, Union

import numpy as np
from scipy.special import comb


def polyval3(coeffs: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Evaluate 3D polynomial at parameter value(s).

    Args:
        coeffs: Polynomial coefficients (degree+1 x 3)
        t: Parameter value(s) to evaluate at

    Returns:
        3D point(s) at parameter t
    """
    t = np.atleast_1d(t)
    is_scalar = t.shape == (1,)

    t = t.reshape(-1, 1)
    powers = t ** np.arange(coeffs.shape[0])[::-1]
    result = powers @ coeffs

    # Return scalar result if input was scalar
    if is_scalar:
        return result[0]
    return result


def poly_arc_length_3d(
    coeffs: np.ndarray, t0: float, t1: float, num_samples: int = 10000
) -> float:
    """
    Calculate arc length of 3D polynomial between t0 and t1.

    Args:
        coeffs: Polynomial coefficients (degree+1 x 3)
        t0: Start parameter
        t1: End parameter
        num_samples: Number of samples for numerical integration

    Returns:
        Arc length in same units as polynomial
    """
    # Use adaptive sampling based on parameter range
    # For very small ranges, we don't need as many samples
    param_range = abs(t1 - t0)
    if param_range < 0.01:
        num_samples = max(100, int(num_samples * param_range))

    t = np.linspace(t0, t1, num_samples)
    points = polyval3(coeffs, t)

    # Calculate distances between consecutive points
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    return np.sum(distances)


def inv_poly_arc_length_3d(
    coeffs: np.ndarray, target_length: float, t0: float = 0.0, tolerance: float = 1e-6
) -> float:
    """
    Find parameter t where arc length from t0 equals target_length.

    Uses binary search for efficiency.

    Args:
        coeffs: Polynomial coefficients (degree+1 x 3)
        target_length: Desired arc length from t0
        t0: Starting parameter (default 0)
        tolerance: Convergence tolerance

    Returns:
        Parameter t where arc length equals target_length
    """
    # Binary search
    t_min, t_max = t0, 1.0

    while t_max - t_min > tolerance:
        t_mid = (t_min + t_max) / 2
        current_length = poly_arc_length_3d(coeffs, t0, t_mid)

        if current_length < target_length:
            t_min = t_mid
        else:
            t_max = t_mid

    return (t_min + t_max) / 2


def fit_polynomial_3d(
    points: np.ndarray, degree: int = 8, weights: np.ndarray = None
) -> np.ndarray:
    """
    Fit parametric 3D polynomial to points.

    Args:
        points: Nx3 array of 3D points
        degree: Polynomial degree
        weights: Optional weights for each point

    Returns:
        Polynomial coefficients (degree+1 x 3)
    """
    # Parameterize by cumulative chord length
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0], np.cumsum(distances)])
    t = t / t[-1]  # Normalize to [0, 1]

    # Fit each dimension independently
    coeffs = []
    for dim in range(3):
        if weights is not None:
            p = np.polyfit(t, points[:, dim], degree, w=weights)
        else:
            p = np.polyfit(t, points[:, dim], degree)
        coeffs.append(p)

    return np.array(coeffs).T


def get_orthogonal_vectors(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get two orthogonal vectors to the given tangent vector.

    Args:
        tangent: 3D tangent vector (normalized)

    Returns:
        Tuple of two orthogonal unit vectors
    """
    # Find the axis with smallest component
    min_axis = np.argmin(np.abs(tangent))

    # Create a vector not parallel to tangent
    v = np.zeros(3)
    v[min_axis] = 1.0

    # Gram-Schmidt orthogonalization
    v1 = v - np.dot(v, tangent) * tangent
    v1 = v1 / np.linalg.norm(v1)

    # Cross product for second orthogonal vector
    v2 = np.cross(tangent, v1)

    return v1, v2


def resample_polynomial(coeffs: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Resample polynomial at uniform arc length intervals.

    Args:
        coeffs: Polynomial coefficients
        num_points: Number of resampled points

    Returns:
        Nx3 array of uniformly spaced points
    """
    total_length = poly_arc_length_3d(coeffs, 0, 1)
    target_lengths = np.linspace(0, total_length, num_points)

    points = []
    for length in target_lengths:
        t = inv_poly_arc_length_3d(coeffs, length)
        points.append(polyval3(coeffs, t))

    return np.array(points)


def reparameterize_polynomial_3d(
    coeffs: np.ndarray, t_start: float, t_end: float
) -> np.ndarray:
    """
    Reparameterize a 3D polynomial to map [0, 1] to [t_start, t_end].

    This preserves the exact shape of the trajectory - it only changes
    the parameterization. The new polynomial p_new(s) = p_old(t_start + s * (t_end - t_start))
    where s ∈ [0, 1] maps to t ∈ [t_start, t_end].

    Args:
        coeffs: Original polynomial coefficients (degree+1 x 3)
        t_start: Start parameter in original polynomial
        t_end: End parameter in original polynomial

    Returns:
        New polynomial coefficients with s ∈ [0, 1] mapping to the
        portion of the original polynomial from t_start to t_end
    """
    degree = coeffs.shape[0] - 1
    scale = t_end - t_start

    # For each dimension (x, y, z)
    new_coeffs = np.zeros_like(coeffs)

    for dim in range(3):
        # Original polynomial for this dimension: p(t) = sum(c[i] * t^(n-i))
        # New polynomial: p_new(s) = p(t_start + s * scale)
        # We need to expand (t_start + s * scale)^k for each power k

        for k in range(degree + 1):
            # Coefficient for t^(degree-k) in original polynomial
            orig_coeff = coeffs[k, dim]

            # Expand (t_start + s * scale)^(degree-k) using binomial theorem
            # (a + b)^n = sum(C(n,j) * a^(n-j) * b^j)
            power = degree - k

            for j in range(power + 1):
                # Binomial coefficient
                binom = comb(power, j, exact=True)
                # Contribution to s^j term
                contribution = (
                    orig_coeff * binom * (t_start ** (power - j)) * (scale**j)
                )
                # Add to coefficient of s^j (which is at index degree-j)
                new_coeffs[degree - j, dim] += contribution

    return new_coeffs
