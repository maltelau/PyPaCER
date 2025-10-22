"""Trajectory fitting algorithms for electrode reconstruction."""

import numpy as np

from ..utils.math_helpers import fit_polynomial_3d, poly_arc_length_3d


def fit_initial_trajectory(
    points_world: np.ndarray,
    intensities: np.ndarray,
    degree: int = 8,
    reverse_dir: bool = False,
    use_intensity_weighting: bool = True,
) -> "InitialTrajectory":
    """
    Fit initial polynomial trajectory to electrode point cloud.

    Based on MATLAB electrodePointCloudModelEstimate.m

    Args:
        points_world: Nx3 array of 3D points in world coordinates
        intensities: N array of CT intensities at each point
        degree: Polynomial degree (default 8 as in MATLAB)
        reverse_dir: Reverse Z direction for special cases
        use_intensity_weighting: Use intensity-weighted centroids

    Returns:
        InitialTrajectory object with polynomial and skeleton
    """
    print("  Step 1: Extracting skeleton from point cloud...")
    # Extract skeleton using axial slicing approach
    skeleton = _extract_skeleton_axial(
        points_world, intensities, use_intensity_weighting
    )
    print(f"    Extracted {len(skeleton)} skeleton points")

    # Validate skeleton
    if len(skeleton) < degree + 1:
        print(
            f"Warning: Only {len(skeleton)} skeleton points for degree {degree} polynomial. "
            f"Reducing degree to {len(skeleton) - 1}"
        )
        degree = len(skeleton) - 1

    # Reverse if needed
    if reverse_dir:
        skeleton = skeleton[::-1]

    # Fit polynomial
    print(f"  Step 2: Fitting degree-{degree} polynomial to skeleton...")
    polynomial = fit_polynomial_3d(skeleton, degree)
    print(f"    Polynomial shape: {polynomial.shape}")

    # Calculate total length
    total_length = poly_arc_length_3d(polynomial, 0, 1)

    return InitialTrajectory(
        polynomial=polynomial,
        skeleton=skeleton,
        total_length_mm=total_length,
        degree=degree,
    )


def _extract_skeleton_axial(
    points: np.ndarray,
    intensities: np.ndarray,
    use_intensity_weighting: bool = True,
    z_tolerance: float = 0.1,
) -> np.ndarray:
    """
    Extract skeleton points by finding centroids in axial slices.

    Args:
        points: Nx3 array of points
        intensities: N array of intensities
        use_intensity_weighting: Use intensity-weighted centroids
        z_tolerance: Tolerance for grouping points in Z slices

    Returns:
        Skeleton points array
    """
    # Find unique Z planes
    z_coords = points[:, 2]
    z_planes = np.unique(z_coords)

    # Check if planes are well-defined
    if len(z_planes) >= len(points):
        # Try with tolerance
        z_planes = []
        z_sorted = np.sort(z_coords)
        current_plane = z_sorted[0]
        z_planes.append(current_plane)

        for z in z_sorted[1:]:
            if abs(z - current_plane) > z_tolerance:
                current_plane = z
                z_planes.append(current_plane)

        z_planes = np.array(z_planes)

    if len(z_planes) >= len(points):
        raise ValueError("Could not find distinct Z planes. Check CT orientation.")

    # Extract skeleton points
    skeleton = []
    sum_in_plane = []

    for z_plane in z_planes:
        # Get points in this plane
        plane_mask = np.abs(z_coords - z_plane) <= z_tolerance
        plane_points = points[plane_mask]

        if len(plane_points) > 1:
            if use_intensity_weighting and intensities is not None:
                plane_intensities = intensities[plane_mask].astype(float)
                # Intensity-weighted centroid
                centroid = np.average(plane_points, axis=0, weights=plane_intensities)
                skeleton.append(centroid)
                sum_in_plane.append(np.sum(plane_intensities))
            else:
                # Simple centroid
                skeleton.append(np.mean(plane_points, axis=0))
                if intensities is not None:
                    sum_in_plane.append(np.sum(intensities[plane_mask]))

    skeleton = np.array(skeleton)

    # Filter skeleton for low-intensity planes
    if use_intensity_weighting and len(sum_in_plane) > 0:
        sum_in_plane = np.array(sum_in_plane)
        median_intensity = np.median(sum_in_plane)
        valid_planes = sum_in_plane >= (median_intensity / 1.5)

        if np.sum(~valid_planes) > 0:
            print(f"Filtered {np.sum(~valid_planes)} low-intensity skeleton points")
            skeleton = skeleton[valid_planes]

    if len(skeleton) == 0:
        raise ValueError("Empty skeleton. Check CT orientation and metal detection.")

    return skeleton


class InitialTrajectory:
    """Container for initial trajectory fit results."""

    def __init__(
        self,
        polynomial: np.ndarray,
        skeleton: np.ndarray,
        total_length_mm: float,
        degree: int,
    ):
        self.polynomial = polynomial
        self.skeleton = skeleton
        self.total_length_mm = total_length_mm
        self.degree = degree

    def __repr__(self):
        return (
            f"InitialTrajectory(degree={self.degree}, "
            f"length={self.total_length_mm:.1f}mm, "
            f"skeleton_points={len(self.skeleton)})"
        )
