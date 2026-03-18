"""CT subvolume extraction for electrode visualization."""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def extract_contacts_subvolume(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    padding_mm: float = 10.0,
    extra_positions: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Extract subvolume around contact positions only.

    Args:
        extra_positions: Optional Nx3 array of additional world-space positions
            to include in the bounding box (e.g. marker positions).
    """
    if "contact_positions_3d" not in electrode:
        return None, None, None

    contacts_3d = np.array(electrode["contact_positions_3d"])

    # Optionally extend bbox to include extra positions (e.g. markers)
    if extra_positions is not None and len(extra_positions) > 0:
        all_positions = np.vstack([contacts_3d, np.atleast_2d(extra_positions)])
    else:
        all_positions = contacts_3d

    # Calculate bounding box around all positions with padding
    bbox = {
        "min": all_positions.min(axis=0) - padding_mm,
        "max": all_positions.max(axis=0) + padding_mm,
        "center": all_positions.mean(axis=0),
        "size": all_positions.max(axis=0) - all_positions.min(axis=0) + 2 * padding_mm,
    }

    return _extract_subvolume(ct_data, affine, bbox)


def extract_markers_subvolume(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    padding_mm: float = 3.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Extract subvolume around marker positions from orientation data."""
    orientation = electrode.get("orientation", {})
    markers = orientation.get("markers", {})
    if not markers:
        return None, None, None

    # Collect all marker 3D positions
    positions = []
    for marker_data in markers.values():
        pos = marker_data.get("position_xyz")
        if pos is not None:
            positions.append(pos)

    if not positions:
        return None, None, None

    positions = np.array(positions)

    bbox = {
        "min": positions.min(axis=0) - padding_mm,
        "max": positions.max(axis=0) + padding_mm,
        "center": positions.mean(axis=0),
        "size": positions.max(axis=0) - positions.min(axis=0) + 2 * padding_mm,
    }

    return _extract_subvolume(ct_data, affine, bbox)


def extract_electrode_subvolume(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    padding_mm: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Extract subvolume for a single electrode from CT data."""

    # Use the bounding box from the reconstruction JSON if available
    if "bounding_box" in electrode:
        bbox_data = electrode["bounding_box"]
        # Apply padding to the existing bounding box
        bbox = {
            "min": np.array(bbox_data["min"]) - padding_mm,
            "max": np.array(bbox_data["max"]) + padding_mm,
            "center": (np.array(bbox_data["min"]) + np.array(bbox_data["max"])) / 2,
            "size": np.array(bbox_data["max"])
            - np.array(bbox_data["min"])
            + 2 * padding_mm,
        }
    elif "contact_positions_3d" in electrode:
        # Fallback to calculating from contacts if no bounding box is provided
        contacts_3d = np.array(electrode["contact_positions_3d"])

        # Calculate bounding box with padding
        bbox = {
            "min": contacts_3d.min(axis=0) - padding_mm,
            "max": contacts_3d.max(axis=0) + padding_mm,
            "center": contacts_3d.mean(axis=0),
            "size": contacts_3d.max(axis=0) - contacts_3d.min(axis=0) + 2 * padding_mm,
        }
    else:
        return None, None, None

    result = _extract_subvolume(ct_data, affine, bbox)

    return result


def _extract_subvolume(
    ct_data: np.ndarray,
    affine: np.ndarray,
    bbox: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Shared implementation for subvolume extraction given a bounding box."""
    # Convert world coordinates to voxel indices
    inv_affine = np.linalg.inv(affine)

    # Transform bbox corners to voxel space
    min_world = np.append(bbox["min"], 1)
    max_world = np.append(bbox["max"], 1)

    min_voxel = inv_affine @ min_world
    max_voxel = inv_affine @ max_world

    # Handle flipped axes by ensuring voxel min < max
    voxel_min = np.minimum(min_voxel[:3], max_voxel[:3])
    voxel_max = np.maximum(min_voxel[:3], max_voxel[:3])

    # Get integer voxel indices (with bounds checking)
    min_idx = np.maximum(0, np.floor(voxel_min).astype(int))
    max_idx = np.minimum(ct_data.shape, np.ceil(voxel_max).astype(int))

    # Extract subvolume
    subvolume_data = ct_data[
        min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2]
    ]

    # Handle negative axes (following native renderer approach)
    subvolume_affine = affine.copy()

    for axis in range(3):
        if affine[axis, axis] < 0:
            # Flip the data along this axis
            subvolume_data = np.flip(subvolume_data, axis=axis)
            # Make spacing positive
            subvolume_affine[axis, axis] = -affine[axis, axis]
            # Adjust origin (accounting for flip)
            subvolume_affine[axis, 3] = affine[axis, 3] + affine[axis, axis] * (
                max_idx[axis] - 1
            )
        else:
            # Positive spacing - just adjust origin
            subvolume_affine[axis, 3] = (
                affine[axis, 3] + affine[axis, axis] * min_idx[axis]
            )

    # IMPORTANT: Adjust origin from voxel center to voxel corner
    # NiBabel affine maps voxel centers, PyVista expects corner at (-0.5, -0.5, -0.5)
    voxel_shift = np.array([-0.5, -0.5, -0.5, 1])
    corner_shift = subvolume_affine @ voxel_shift
    subvolume_affine[:3, 3] = corner_shift[:3]

    return subvolume_data, subvolume_affine, bbox
