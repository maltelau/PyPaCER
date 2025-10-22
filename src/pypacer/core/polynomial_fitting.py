"""Direct polynomial fitting for COG trajectories."""

from typing import Optional

import numpy as np

from ..utils.math_helpers import fit_polynomial_3d, poly_arc_length_3d


def fit_polynomial_to_trajectory(
    points: np.ndarray, degree: int = 3, weights: Optional[np.ndarray] = None
) -> "TrajectoryModel":
    """
    Directly fit a polynomial to an ordered trajectory of points.

    This function is designed for COG trajectories where points are already
    ordered and don't need skeleton extraction.

    Args:
        points: Nx3 array of ordered 3D points
        degree: Polynomial degree (default 3)
        weights: Optional weights for each point (e.g., intensities)

    Returns:
        TrajectoryModel with polynomial fit
    """
    # Validate inputs
    if len(points) < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} points for degree {degree} polynomial, got {len(points)}"
        )

    # Use the points directly as the skeleton (they're already ordered from COG tracking)
    skeleton = points.copy()

    # Fit polynomial directly to the ordered points
    if weights is not None and len(weights) == len(points):
        # Weight the fitting by intensities if provided
        # Normalize weights
        weights_norm = (
            weights / weights.max() if weights.max() > 0 else np.ones_like(weights)
        )
        # For weighted fitting, we can repeat points based on weight
        # or use a weighted least squares approach
        # For simplicity, we'll use the standard fit for now
        polynomial = fit_polynomial_3d(skeleton, degree)
    else:
        polynomial = fit_polynomial_3d(skeleton, degree)

    # Calculate arc length
    total_length = poly_arc_length_3d(polynomial, 0, 1)

    # Create distance scale along trajectory
    n_points = len(skeleton)
    t_values = np.linspace(0, 1, n_points)
    distance_scale = np.zeros(n_points)

    for i in range(1, n_points):
        # Calculate cumulative arc length
        segment_length = poly_arc_length_3d(polynomial, t_values[i - 1], t_values[i])
        distance_scale[i] = distance_scale[i - 1] + segment_length

    # Create trajectory model
    return TrajectoryModel(
        polynomial=polynomial,
        skeleton=skeleton,
        total_length_mm=total_length,
        degree=degree,
        distance_scale_mm=distance_scale,
        intensity_profile=weights,
    )


class TrajectoryModel:
    """Simple trajectory model for COG-fitted polynomials."""

    def __init__(
        self,
        polynomial: np.ndarray,
        skeleton: np.ndarray,
        total_length_mm: float,
        degree: int,
        distance_scale_mm: Optional[np.ndarray] = None,
        intensity_profile: Optional[np.ndarray] = None,
    ):
        self.polynomial = polynomial
        self.skeleton = skeleton
        self.total_length_mm = total_length_mm
        self.degree = degree
        self.distance_scale_mm = distance_scale_mm
        self.intensity_profile = intensity_profile

        # Make it compatible with refinement functions
        self.n_components = 3
        self.t_min = 0.0
        self.t_max = 1.0

    def __repr__(self):
        return (
            f"TrajectoryModel(degree={self.degree}, "
            f"length={self.total_length_mm:.1f}mm, "
            f"points={len(self.skeleton)})"
        )

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at parameter values."""
        # Ensure t is in [0, 1]
        t = np.clip(t, 0, 1)

        # Build polynomial matrix
        n_coeffs = self.polynomial.shape[0]
        T = np.zeros((len(t), n_coeffs))
        for i in range(n_coeffs):
            T[:, i] = t**i

        # Evaluate polynomial
        return T @ self.polynomial
