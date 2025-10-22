"""Electrode detection from metal artifacts in CT images."""

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ElectrodePointCloud:
    """Container for electrode point cloud data."""

    points_voxel: np.ndarray  # Nx3 voxel coordinates
    points_world: np.ndarray  # Nx3 world coordinates
    intensities: np.ndarray  # N intensity values
    component_id: int
    pca_variance: np.ndarray
    estimated_length_mm: float


def extract_electrode_pointclouds(
    labeled_metal: np.ndarray,
    ct_data: np.ndarray,
    voxel_sizes: np.ndarray,
    affine: np.ndarray,
    min_length_mm: float = 40.0,
) -> List[ElectrodePointCloud]:
    """
    Extract electrode point clouds from labeled metal components.

    Args:
        labeled_metal: Labeled array of metal components
        ct_data: Original CT data for intensity values
        voxel_sizes: Voxel dimensions in mm
        affine: Affine transformation matrix to world coordinates
        min_length_mm: Minimum electrode length in mm to keep (default: 40.0)

    Returns:
        List of electrode point clouds
    """
    num_components = int(np.max(labeled_metal)) if labeled_metal.any() else 0
    electrodes = []

    if num_components == 0:
        print("No electrode components found in labeled metal mask")
        return electrodes

    for component_id in range(1, num_components + 1):
        # Get component voxels
        voxel_coords = np.argwhere(labeled_metal == component_id)

        # Extract intensities
        intensities = ct_data[
            voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
        ]

        # Convert to world coordinates
        voxel_coords_homogeneous = np.column_stack(
            [voxel_coords, np.ones(len(voxel_coords))]
        )
        world_coords = (affine @ voxel_coords_homogeneous.T).T[:, :3]

        # Perform PCA for characterization
        pca = PCA(n_components=3)
        pca.fit(world_coords)

        # Estimate length from PCA
        estimated_length = 2 * np.sqrt(pca.explained_variance_[0])

        electrode = ElectrodePointCloud(
            points_voxel=voxel_coords,
            points_world=world_coords,
            intensities=intensities,
            component_id=component_id,
            pca_variance=pca.explained_variance_,
            estimated_length_mm=estimated_length,
        )

        # Only keep electrodes above the length threshold
        if estimated_length >= min_length_mm:
            electrodes.append(electrode)
            print(
                f"Electrode {len(electrodes)}: {len(voxel_coords)} voxels, "
                f"~{estimated_length:.1f}mm length"
            )
        else:
            print(
                f"Filtered out component {component_id}: {len(voxel_coords)} voxels, "
                f"~{estimated_length:.1f}mm length (below {min_length_mm}mm threshold)"
            )

    return electrodes


def _validate_electrode_geometry(
    voxel_coords: np.ndarray, voxel_sizes: np.ndarray, linearity_threshold: float = 10.0
) -> bool:
    """
    Validate if component has electrode-like geometry using PCA.

    Args:
        voxel_coords: Nx3 array of voxel coordinates
        voxel_sizes: Voxel dimensions
        linearity_threshold: Ratio threshold for linear structure

    Returns:
        True if component appears electrode-like
    """
    if len(voxel_coords) < 10:  # Too few points
        return False

    # Scale to physical coordinates
    physical_coords = voxel_coords * voxel_sizes

    # PCA to check linearity
    pca = PCA(n_components=3)
    pca.fit(physical_coords)

    variances = pca.explained_variance_

    # Check if structure is linear (high variance in one direction)
    if len(variances) < 3:
        return False

    linearity_ratio = variances[0] / (variances[1] + 1e-10)

    # Check physical dimensions
    length = 2 * np.sqrt(variances[0])  # Approximate length
    diameter = 2 * np.sqrt(variances[1])  # Approximate diameter

    # Electrode criteria:
    # - Linear structure
    # - Appropriate length
    # - Appropriate diameter
    is_linear = linearity_ratio > linearity_threshold
    has_valid_length = 30 < length < 100  # mm
    has_valid_diameter = diameter < 3  # mm

    return is_linear and has_valid_length and has_valid_diameter


def filter_electrodes_by_distance(
    electrodes: List[ElectrodePointCloud], min_separation_mm: float = 10.0
) -> List[ElectrodePointCloud]:
    """
    Filter out duplicate/nearby electrodes.

    Sometimes artifacts can be split into multiple components.
    This function merges nearby components.

    Args:
        electrodes: List of electrode point clouds
        min_separation_mm: Minimum separation between electrodes

    Returns:
        Filtered list of electrodes
    """
    if len(electrodes) <= 1:
        return electrodes

    # Calculate pairwise distances between electrode centroids
    centroids = [np.mean(e.points_world, axis=0) for e in electrodes]

    # Greedy merging of nearby electrodes
    merged = []
    used = set()

    for i, e1 in enumerate(electrodes):
        if i in used:
            continue

        # Check for nearby electrodes
        merge_group = [i]
        c1 = centroids[i]

        for j, e2 in enumerate(electrodes[i + 1 :], i + 1):
            if j in used:
                continue

            c2 = centroids[j]
            dist = np.linalg.norm(c1 - c2)

            if dist < min_separation_mm:
                merge_group.append(j)
                used.add(j)

        # If multiple components, merge them
        if len(merge_group) > 1:
            merged_electrode = _merge_electrode_components(
                [electrodes[idx] for idx in merge_group]
            )
            merged.append(merged_electrode)
        else:
            merged.append(e1)

        used.add(i)

    return merged


def _merge_electrode_components(
    components: List[ElectrodePointCloud],
) -> ElectrodePointCloud:
    """Merge multiple electrode components into one."""
    # Concatenate all points
    all_voxels = np.vstack([c.points_voxel for c in components])
    all_world = np.vstack([c.points_world for c in components])
    all_intensities = np.hstack([c.intensities for c in components])

    # Recompute PCA
    pca = PCA(n_components=3)
    pca.fit(all_world)

    return ElectrodePointCloud(
        points_voxel=all_voxels,
        points_world=all_world,
        intensities=all_intensities,
        component_id=components[0].component_id,
        pca_variance=pca.explained_variance_,
        estimated_length_mm=2 * np.sqrt(pca.explained_variance_[0]),
    )
