"""
Generate orthogonal slice views along electrode trajectories.

This module provides utilities to generate 2D slice views from 3D medical imaging
data (e.g., CT scans) along electrode trajectories, showing all contacts with
customizable visualization options.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def create_volume_interpolator(volume_data: np.ndarray) -> RegularGridInterpolator:
    """
    Create a RegularGridInterpolator for sampling 3D volume data.

    Parameters
    ----------
    volume_data : np.ndarray
        3D array of volume data (e.g., CT scan)

    Returns
    -------
    RegularGridInterpolator
        Interpolator for sampling the volume at arbitrary coordinates
    """
    voxel_coords = [np.arange(s) for s in volume_data.shape]
    return RegularGridInterpolator(
        voxel_coords, volume_data, method="linear", bounds_error=False, fill_value=0
    )


def get_world_to_voxel_transform(affine: np.ndarray):
    """
    Create a function to transform world coordinates to voxel coordinates.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header

    Returns
    -------
    callable
        Function that takes world coordinates and returns voxel coordinates
    """
    inv_affine = np.linalg.inv(affine)

    def world_to_voxel(world_coords):
        """Convert world coordinates to voxel coordinates."""
        world_coords = np.atleast_2d(world_coords)
        homogeneous = np.column_stack([world_coords, np.ones(len(world_coords))])
        voxel_coords = (inv_affine @ homogeneous.T).T[:, :3]
        return voxel_coords

    return world_to_voxel


def compute_orthogonal_vectors(
    electrode_axis: np.ndarray, affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two orthogonal vectors perpendicular to the electrode axis.

    This function finds the two image-space directions (from the affine matrix)
    that are most perpendicular to the electrode axis, creating lateral and
    frontal viewing planes.

    Parameters
    ----------
    electrode_axis : np.ndarray
        Normalized 3D vector representing the electrode direction
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header

    Returns
    -------
    lateral_vec : np.ndarray
        Normalized vector perpendicular to electrode (lateral view direction)
    frontal_vec : np.ndarray
        Normalized vector perpendicular to electrode (frontal view direction)
    """
    # Get image-space directions from affine matrix
    i_axis = affine[:3, 0] / np.linalg.norm(affine[:3, 0])  # Left-Right
    j_axis = affine[:3, 1] / np.linalg.norm(affine[:3, 1])  # Posterior-Anterior
    k_axis = affine[:3, 2] / np.linalg.norm(affine[:3, 2])  # Inferior-Superior

    # Find which image axes are most perpendicular to electrode
    dots = [
        abs(np.dot(electrode_axis, i_axis)),
        abs(np.dot(electrode_axis, j_axis)),
        abs(np.dot(electrode_axis, k_axis)),
    ]

    # Use the two most perpendicular axes
    sorted_indices = np.argsort(dots)
    axes = [i_axis, j_axis, k_axis]
    lateral_vec = axes[sorted_indices[0]].copy()  # Most perpendicular
    frontal_vec = axes[sorted_indices[1]].copy()  # Second most perpendicular

    # Make sure they're orthogonal to electrode axis (project out parallel component)
    lateral_vec = lateral_vec - np.dot(lateral_vec, electrode_axis) * electrode_axis
    lateral_vec = lateral_vec / np.linalg.norm(lateral_vec)

    frontal_vec = frontal_vec - np.dot(frontal_vec, electrode_axis) * electrode_axis
    frontal_vec = frontal_vec / np.linalg.norm(frontal_vec)

    return lateral_vec, frontal_vec


def generate_slice_plane(
    volume_interpolator: RegularGridInterpolator,
    world_to_voxel: callable,
    center_point: np.ndarray,
    electrode_axis: np.ndarray,
    perpendicular_vec: np.ndarray,
    width: float = 32.0,
    height: float = 48.0,
    resolution: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate a 2D slice plane through 3D volume data along an electrode trajectory.

    Parameters
    ----------
    volume_interpolator : RegularGridInterpolator
        Interpolator for the 3D volume data
    world_to_voxel : callable
        Function to convert world coordinates to voxel coordinates
    center_point : np.ndarray
        3D world coordinates of the plane center (typically electrode midpoint)
    electrode_axis : np.ndarray
        Normalized 3D vector along the electrode direction
    perpendicular_vec : np.ndarray
        Normalized 3D vector perpendicular to electrode (defines plane orientation)
    width : float, optional
        Width of plane perpendicular to electrode in mm (default: 32.0)
    height : float, optional
        Height of plane along electrode in mm (default: 48.0)
    resolution : float, optional
        Sampling resolution in mm (default: 0.1)

    Returns
    -------
    intensity_map : np.ndarray
        2D array of sampled intensities
    extent : dict
        Dictionary with keys 'u_min', 'u_max', 'v_min', 'v_max' for plotting extent
    """
    # Create sampling grid
    # u: perpendicular to electrode (horizontal axis)
    # v: along electrode axis (vertical axis)
    perp_coords = np.arange(-width / 2, width / 2 + resolution, resolution)
    along_coords = np.arange(-height / 2, height / 2 + resolution, resolution)

    u_grid, v_grid = np.meshgrid(perp_coords, along_coords)

    # Sample points in world space
    sample_points = (
        center_point[np.newaxis, np.newaxis, :]
        + u_grid[:, :, np.newaxis] * perpendicular_vec
        + v_grid[:, :, np.newaxis] * electrode_axis
    )

    # Flatten for sampling
    sample_points_flat = sample_points.reshape(-1, 3)

    # Convert to voxel coordinates and sample intensities
    voxel_coords = world_to_voxel(sample_points_flat)
    intensities = volume_interpolator(voxel_coords)
    intensity_map = intensities.reshape(u_grid.shape)

    extent = {
        "u_min": -width / 2,
        "u_max": width / 2,
        "v_min": -height / 2,
        "v_max": height / 2,
    }

    return intensity_map, extent


def project_contacts_to_plane(
    contact_positions: np.ndarray,
    center_point: np.ndarray,
    electrode_axis: np.ndarray,
    perpendicular_vec: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Project 3D contact positions onto a 2D plane.

    Parameters
    ----------
    contact_positions : np.ndarray
        Nx3 array of contact positions in world coordinates
    center_point : np.ndarray
        3D world coordinates of the plane center
    electrode_axis : np.ndarray
        Normalized 3D vector along the electrode direction
    perpendicular_vec : np.ndarray
        Normalized 3D vector perpendicular to electrode (defines plane orientation)

    Returns
    -------
    List[Tuple[float, float]]
        List of (u, v) coordinates for each contact in the plane coordinate system
    """
    projected_coords = []

    for contact_pos in contact_positions:
        # Calculate position relative to center
        rel_pos = contact_pos - center_point

        # Project onto plane coordinates
        u_coord = np.dot(rel_pos, perpendicular_vec)
        v_coord = np.dot(rel_pos, electrode_axis)

        projected_coords.append((u_coord, v_coord))

    return projected_coords


def generate_orthogonal_slice_views(
    volume_data: np.ndarray,
    affine: np.ndarray,
    contact_positions: np.ndarray,
    width: float = 32.0,
    height: float = 48.0,
    resolution: float = 0.1,
) -> Dict[str, Dict]:
    """
    Generate orthogonal slice views along an electrode trajectory.

    This is the main high-level function that generates two orthogonal slice views
    (lateral and frontal) showing the electrode trajectory and all contacts.

    Parameters
    ----------
    volume_data : np.ndarray
        3D array of volume data (e.g., CT scan)
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header
    contact_positions : np.ndarray
        Nx3 array of contact positions in world coordinates
    width : float, optional
        Width of plane perpendicular to electrode in mm (default: 32.0)
    height : float, optional
        Height of plane along electrode in mm (default: 48.0)
    resolution : float, optional
        Sampling resolution in mm (default: 0.1)

    Returns
    -------
    Dict[str, Dict]
        Dictionary with keys 'lateral' and 'frontal', each containing:
        - 'intensity_map': 2D numpy array of sampled intensities
        - 'extent': dict with u_min, u_max, v_min, v_max for plotting
        - 'contact_coords': list of (u, v) tuples for contact positions
        - 'center_point': 3D world coordinates of plane center
        - 'electrode_axis': normalized electrode direction vector
        - 'perpendicular_vec': normalized perpendicular vector

    Examples
    --------
    >>> import nibabel as nib
    >>> ct_nii = nib.load('ct_scan.nii.gz')
    >>> ct_data = ct_nii.get_fdata()
    >>> affine = ct_nii.affine
    >>> contact_positions = np.array([[x1,y1,z1], [x2,y2,z2], ...])
    >>> views = generate_orthogonal_slice_views(ct_data, affine, contact_positions)
    >>> lateral_image = views['lateral']['intensity_map']
    >>> frontal_image = views['frontal']['intensity_map']
    """
    # Calculate electrode axis (from bottom to top contact)
    electrode_axis = contact_positions[-1] - contact_positions[0]
    electrode_axis = electrode_axis / np.linalg.norm(electrode_axis)

    # Center point of electrode
    center_point = (contact_positions[0] + contact_positions[-1]) / 2.0

    # Create volume interpolator and coordinate transform
    volume_interpolator = create_volume_interpolator(volume_data)
    world_to_voxel = get_world_to_voxel_transform(affine)

    # Compute orthogonal viewing directions
    lateral_vec, frontal_vec = compute_orthogonal_vectors(electrode_axis, affine)

    # Generate both planes
    results = {}

    for plane_name, perp_vec in [("lateral", lateral_vec), ("frontal", frontal_vec)]:
        intensity_map, extent = generate_slice_plane(
            volume_interpolator,
            world_to_voxel,
            center_point,
            electrode_axis,
            perp_vec,
            width=width,
            height=height,
            resolution=resolution,
        )

        contact_coords = project_contacts_to_plane(
            contact_positions, center_point, electrode_axis, perp_vec
        )

        results[plane_name] = {
            "intensity_map": intensity_map,
            "extent": extent,
            "contact_coords": contact_coords,
            "center_point": center_point,
            "electrode_axis": electrode_axis,
            "perpendicular_vec": perp_vec,
        }

    return results
