"""Circular intensity sampling around electrode trajectory."""

from typing import List, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


class CircularSamplingResult:
    """Result of circular intensity sampling."""

    def __init__(
        self,
        angles_deg: np.ndarray,
        radii_mm: np.ndarray,
        intensities: np.ndarray,
        mean_intensity_by_angle: np.ndarray,
        center_world: np.ndarray,
        normal_vector: np.ndarray,
    ):
        """
        Args:
            angles_deg: Array of angles in degrees (0-360)
            radii_mm: Array of radii in mm
            intensities: 2D array of intensities (n_angles x n_radii)
            mean_intensity_by_angle: Mean intensity across radii for each angle
            center_world: 3D center point in world coordinates
            normal_vector: Normal vector of the sampling plane (trajectory direction)
        """
        self.angles_deg = angles_deg
        self.radii_mm = radii_mm
        self.intensities = intensities
        self.mean_intensity_by_angle = mean_intensity_by_angle
        self.center_world = center_world
        self.normal_vector = normal_vector


def sample_circular_intensity(
    ct_data: np.ndarray,
    affine: np.ndarray,
    center_world: np.ndarray,
    trajectory_direction: np.ndarray,
    radii_mm: List[float] = None,
    angle_increment_deg: float = 5.0,
    interpolation_order: int = 1,
) -> CircularSamplingResult:
    """
    Sample CT intensities in a circular pattern perpendicular to electrode trajectory.

    Args:
        ct_data: 3D CT volume
        affine: Affine transformation matrix (world to voxel)
        center_world: Center point in world coordinates (3D)
        trajectory_direction: Unit vector of electrode trajectory direction
        radii_mm: List of radii to sample in mm (default: [0.5, 0.75, 1.0])
        angle_increment_deg: Angular increment in degrees
        interpolation_order: Order for interpolation (0=nearest, 1=linear, 3=cubic)

    Returns:
        CircularSamplingResult with sampled intensities
    """
    if radii_mm is None:
        radii_mm = [0.5, 0.75, 1.0]

    # Generate angles
    angles_deg = np.arange(0, 360, angle_increment_deg)
    angles_rad = np.deg2rad(angles_deg)

    # Normalize trajectory direction
    trajectory_unit = trajectory_direction / np.linalg.norm(trajectory_direction)

    # Create orthonormal basis for the plane perpendicular to trajectory
    # Find two orthogonal vectors in the plane
    u, v = _create_orthonormal_basis(trajectory_unit)

    # Pre-compute inverse affine once (avoid recomputing per sample point)
    inv_affine = np.linalg.inv(affine)

    # Sample points in circular pattern
    n_angles = len(angles_deg)
    n_radii = len(radii_mm)
    intensities = np.zeros((n_angles, n_radii))

    for i, angle in enumerate(angles_rad):
        for j, radius in enumerate(radii_mm):
            # Calculate point in plane using parametric circle equation
            # P = center + radius * (cos(θ) * u + sin(θ) * v)
            point_world = center_world + radius * (np.cos(angle) * u + np.sin(angle) * v)

            # Transform to voxel coordinates
            point_voxel = _world_to_voxel(point_world, inv_affine)

            # Sample intensity using interpolation
            if _is_within_volume(point_voxel, ct_data.shape):
                intensity = map_coordinates(
                    ct_data,
                    point_voxel.reshape(3, 1),
                    order=interpolation_order,
                    mode="constant",
                    cval=0,
                )[0]
                intensities[i, j] = intensity
            else:
                intensities[i, j] = 0

    # Calculate mean intensity across radii for each angle
    mean_intensity_by_angle = np.mean(intensities, axis=1)

    return CircularSamplingResult(
        angles_deg=angles_deg,
        radii_mm=np.array(radii_mm),
        intensities=intensities,
        mean_intensity_by_angle=mean_intensity_by_angle,
        center_world=center_world,
        normal_vector=trajectory_unit,
    )


def _create_orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create orthonormal basis vectors perpendicular to given normal vector.

    Args:
        normal: Normal vector (should be normalized)

    Returns:
        Tuple of two orthonormal vectors (u, v) perpendicular to normal
    """
    # Find a vector not parallel to normal
    if abs(normal[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    # First basis vector using cross product
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)

    # Second basis vector
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    return u, v


def _world_to_voxel(point_world: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    """
    Transform world coordinates to voxel coordinates.

    Args:
        point_world: 3D point in world coordinates
        inv_affine: Pre-computed inverse of the 4x4 affine transformation matrix

    Returns:
        3D point in voxel coordinates
    """
    point_homogeneous = np.append(point_world, 1)
    point_voxel = inv_affine @ point_homogeneous

    return point_voxel[:3]


def _is_within_volume(point_voxel: np.ndarray, shape: Tuple[int, int, int]) -> bool:
    """Check if voxel coordinates are within volume bounds."""
    return (
        0 <= point_voxel[0] < shape[0]
        and 0 <= point_voxel[1] < shape[1]
        and 0 <= point_voxel[2] < shape[2]
    )


