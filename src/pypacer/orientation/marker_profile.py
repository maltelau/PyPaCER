"""Marker region profile sampling.

Samples circular intensity profiles along the marker region of an electrode
trajectory for debug visualisation (heatmaps, contour plots).
"""

from typing import List, Optional

import numpy as np

from .circular_sampling import sample_circular_intensity
from .orientation_analysis import _evaluate_polynomial_at_distance


def sample_full_marker_profile(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode_polynomial: np.ndarray,
    trajectory_direction_func,
    distance_start_mm: float,
    distance_end_mm: float,
    distance_step_mm: float = 0.2,
    radii_mm: Optional[List[float]] = None,
    angle_increment_deg: float = 1.0,
    axial_plane: bool = False,
) -> dict:
    """
    Sample circular intensity profiles along the full marker region.

    Args:
        ct_data: 3D CT volume
        affine: Affine transformation matrix
        electrode_polynomial: Polynomial coefficients for trajectory
        trajectory_direction_func: Callable(polynomial, distance_mm) -> direction vector
        distance_start_mm: Start of marker region (mm from tip)
        distance_end_mm: End of marker region (mm from tip)
        distance_step_mm: Step size along trajectory (default: 0.2mm)
        radii_mm: Sampling radii (default: [0.75, 1.0, 1.25])
        angle_increment_deg: Angular resolution (default: 1.0 deg)
        axial_plane: If True, sample in axial plane instead of trajectory-perpendicular.

    Returns:
        Dict with keys:
            distances: 1D array of distance positions
            angles_deg: 1D array of angles
            intensity_grid: 2D array (n_distances x n_angles) of mean intensities
    """
    if radii_mm is None:
        radii_mm = [0.75, 1.0, 1.25]

    distances = np.arange(distance_start_mm, distance_end_mm + distance_step_mm / 2, distance_step_mm)
    n_distances = len(distances)

    all_intensities = []
    angles_deg = None

    for i, dist in enumerate(distances):
        center_world = _evaluate_polynomial_at_distance(electrode_polynomial, dist)

        if axial_plane:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = trajectory_direction_func(electrode_polynomial, dist)

        result = sample_circular_intensity(
            ct_data=ct_data,
            affine=affine,
            center_world=center_world,
            trajectory_direction=direction,
            radii_mm=radii_mm,
            angle_increment_deg=angle_increment_deg,
        )

        if angles_deg is None:
            angles_deg = result.angles_deg

        all_intensities.append(result.mean_intensity_by_angle)

        if (i + 1) % 10 == 0 or i == n_distances - 1:
            print(f"  Sampled {i + 1}/{n_distances} positions ({dist:.1f}mm)")

    intensity_grid = np.array(all_intensities)

    return {
        "distances": distances,
        "angles_deg": angles_deg,
        "intensity_grid": intensity_grid,
    }
