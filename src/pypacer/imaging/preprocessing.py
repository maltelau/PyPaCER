"""Preprocessing functions for CT images."""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


def detect_metal_artifacts(
    ct_data: np.ndarray,
    brain_mask: np.ndarray,
    metal_threshold: float = 2000,
    erosion_radius: float = 3.0,
    voxel_sizes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect metal artifacts in CT image within brain region.

    Based on MATLAB extractElectrodePointclouds.m

    Args:
        ct_data: CT image data in HU
        brain_mask: Binary brain mask
        metal_threshold: Hounsfield unit threshold for metal (default 800)
        erosion_radius: Radius in mm for brain mask erosion (default 3mm)
        voxel_sizes: Voxel dimensions in mm for erosion scaling

    Returns:
        Binary mask of metal artifacts
    """
    # Handle CT offset (some CTs are shifted by 1024 to be positive)
    if np.min(ct_data) >= 0:
        metal_threshold = metal_threshold + 1024

    # Erode brain mask to avoid skull
    if voxel_sizes is not None:
        # Scale erosion by voxel size
        radius_voxels = int(np.ceil(erosion_radius / np.max(voxel_sizes)))
    else:
        radius_voxels = int(erosion_radius)

    if radius_voxels > 0:
        struct_elem = ndimage.generate_binary_structure(3, 1)
        struct_elem = ndimage.iterate_structure(struct_elem, radius_voxels)
        brain_mask_eroded = ndimage.binary_erosion(brain_mask, struct_elem)
    else:
        brain_mask_eroded = brain_mask

    # Threshold for metal
    metal_mask = (ct_data > metal_threshold) & brain_mask_eroded

    print(
        f"  Metal voxels found: {np.sum(metal_mask)} "
        f"({np.sum(metal_mask) / np.sum(brain_mask_eroded) * 100:.2f}% of brain)"
    )

    return metal_mask


def extract_brain_mask(
    ct_data: np.ndarray,
    voxel_sizes: np.ndarray,
    hu_threshold: float = -100,
    use_convex_hull: bool = True,
) -> np.ndarray:
    """
    Extract brain region from CT scan.

    Simplified version - assumes preprocessed brain mask will be provided.
    Full implementation would use convex hull approach from MATLAB.

    Args:
        ct_data: CT image data
        voxel_sizes: Voxel dimensions
        hu_threshold: HU threshold for initial segmentation
        use_convex_hull: Whether to use convex hull (not implemented)

    Returns:
        Binary brain mask
    """
    # Simple thresholding approach
    # In practice, brain mask should be provided pre-computed
    brain_mask = ct_data > hu_threshold

    # Remove small components
    brain_mask = ndimage.binary_opening(brain_mask)

    # Get largest connected component
    labeled, num_features = ndimage.label(brain_mask)
    if num_features > 0:
        sizes = ndimage.sum(brain_mask, labeled, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1
        brain_mask = labeled == max_label

    return brain_mask


def filter_metal_components(
    metal_mask: np.ndarray,
    ct_data: np.ndarray,
    voxel_sizes: np.ndarray,
    min_voxels: Optional[int] = None,
    max_voxels: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Filter connected components to find electrode candidates.

    Args:
        metal_mask: Binary mask of metal artifacts
        ct_data: Original CT data for intensity values
        voxel_sizes: Voxel dimensions in mm
        min_voxels: Minimum voxel count for valid electrode
        max_voxels: Maximum voxel count for valid electrode

    Returns:
        Labeled array and number of electrode candidates
    """
    # Calculate expected electrode volume constraints if not provided
    if min_voxels is None:
        # Electrode diameter ~1.27mm, minimum length ~40mm in brain
        electrode_radius_mm = 1.27 / 2
        min_volume_mm3 = np.pi * electrode_radius_mm**2 * 40
        min_voxels = int(min_volume_mm3 / np.prod(voxel_sizes))

    if max_voxels is None:
        # Maximum ~80mm in brain with 3x partial volume
        electrode_radius_mm = 1.27 / 2
        max_volume_mm3 = np.pi * electrode_radius_mm**2 * 80 * 3
        max_voxels = int(max_volume_mm3 / np.prod(voxel_sizes))

    # Label connected components
    labeled, num_components = ndimage.label(metal_mask, structure=np.ones((3, 3, 3)))
    print(f"  Found {num_components} connected metal components")

    # Filter by size
    valid_labels = []
    for label_id in range(1, num_components + 1):
        component_mask = labeled == label_id
        voxel_count = np.sum(component_mask)

        if min_voxels <= voxel_count <= max_voxels:
            valid_labels.append(label_id)
            print(f"  Component {label_id}: {voxel_count} voxels (valid)")
        else:
            print(
                f"  Component {label_id}: {voxel_count} voxels (rejected - expected {min_voxels}-{max_voxels})"
            )

    # Create filtered labeled array
    filtered_labeled = np.zeros_like(labeled)
    for i, label_id in enumerate(valid_labels, 1):
        filtered_labeled[labeled == label_id] = i

    return filtered_labeled, len(valid_labels)
