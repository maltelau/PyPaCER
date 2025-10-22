"""GPU-accelerated trajectory-based point cloud extraction using PyTorch."""

from typing import Tuple

import numpy as np
import torch

from ..utils.math_helpers import polyval3


def extract_trajectory_pointcloud_gpu(
    initial_trajectory,
    ct_data: np.ndarray,
    affine: np.ndarray,
    refinement_threshold: float,
    refinement_radius_mm: float = 3.5,
    sample_spacing_mm: float = 0.5,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated extraction of point cloud around electrode trajectory.

    This function samples points along the trajectory and extracts all voxels
    within a specified radius that exceed the refinement threshold.

    Args:
        initial_trajectory: Initial trajectory with polynomial
        ct_data: CT volume data
        affine: Affine transformation matrix
        refinement_threshold: Intensity threshold for voxel inclusion
        refinement_radius_mm: Radius around trajectory in mm
        sample_spacing_mm: Spacing between trajectory samples in mm
        use_gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (points_world, intensities) for the extracted point cloud
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"    Using PyTorch on {device} for trajectory point cloud extraction")

    # Sample points along the initial trajectory
    n_samples = int(initial_trajectory.total_length_mm / sample_spacing_mm)
    t_values = np.linspace(0, 1, n_samples)
    trajectory_points = polyval3(initial_trajectory.polynomial, t_values)

    print(
        f"      Sampling {n_samples} points along {initial_trajectory.total_length_mm:.1f}mm trajectory"
    )

    # Convert to PyTorch tensors
    ct_tensor = torch.from_numpy(ct_data).float().to(device)
    affine_tensor = torch.from_numpy(affine).float().to(device)
    affine_inv = torch.inverse(affine_tensor)
    trajectory_tensor = torch.from_numpy(trajectory_points).float().to(device)

    # Calculate voxel sizes from affine matrix
    voxel_sizes = torch.abs(torch.diag(affine_tensor[:3, :3]))

    # Create a 3D grid of voxel coordinates
    ct_shape = ct_data.shape
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(ct_shape[0], device=device),
        torch.arange(ct_shape[1], device=device),
        torch.arange(ct_shape[2], device=device),
        indexing="ij",
    )

    # Stack into a single tensor of voxel coordinates
    voxel_coords = torch.stack([z_coords, y_coords, x_coords], dim=-1).float()

    # Convert voxel coordinates to world coordinates
    # Reshape to (N, 4) for matrix multiplication
    voxel_coords_flat = voxel_coords.reshape(-1, 3)
    voxel_coords_homo = torch.cat(
        [voxel_coords_flat, torch.ones((voxel_coords_flat.shape[0], 1), device=device)],
        dim=1,
    )

    # Transform to world coordinates (N, 4) @ (4, 4).T = (N, 4)
    world_coords_homo = voxel_coords_homo @ affine_tensor.T
    world_coords = world_coords_homo[:, :3]

    # Create mask for voxels above threshold
    ct_flat = ct_tensor.reshape(-1)
    threshold_mask = ct_flat > refinement_threshold

    # Filter world coordinates and intensities by threshold
    world_coords_thresh = world_coords[threshold_mask]
    intensities_thresh = ct_flat[threshold_mask]

    print(
        f"      Found {threshold_mask.sum().item()} voxels above {refinement_threshold} HU"
    )

    # Now check distance to trajectory for each thresholded voxel
    # We'll process in chunks to manage memory
    chunk_size = 1000000  # Process 1M voxels at a time
    n_voxels = world_coords_thresh.shape[0]

    within_radius_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)

    for chunk_start in range(0, n_voxels, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_voxels)
        chunk_coords = world_coords_thresh[chunk_start:chunk_end]  # (chunk_size, 3)

        # Compute distance from each voxel in chunk to each trajectory point
        # Use broadcasting: (chunk_size, 1, 3) - (1, n_samples, 3) = (chunk_size, n_samples, 3)
        distances = torch.norm(
            chunk_coords.unsqueeze(1) - trajectory_tensor.unsqueeze(0), dim=2
        )  # (chunk_size, n_samples)

        # Find minimum distance to trajectory for each voxel
        min_distances, _ = distances.min(dim=1)  # (chunk_size,)

        # Check which voxels are within radius
        chunk_mask = min_distances <= refinement_radius_mm
        within_radius_mask[chunk_start:chunk_end] = chunk_mask

        if chunk_start % (chunk_size * 5) == 0:
            print(f"        Processed {min(chunk_end, n_voxels)}/{n_voxels} voxels...")

    # Extract final point cloud
    final_world_coords = world_coords_thresh[within_radius_mask]
    final_intensities = intensities_thresh[within_radius_mask]

    # Convert back to numpy
    points_world = final_world_coords.cpu().numpy()
    intensities = final_intensities.cpu().numpy()

    print(f"      Trajectory-based point cloud: {len(points_world)} points")
    if len(intensities) > 0:
        print(
            f"      Intensity range: [{intensities.min():.0f}, {intensities.max():.0f}] HU"
        )

        # Count distribution
        below_2000 = (intensities < 2000).sum()
        below_1500 = (intensities < 1500).sum()
        below_1000 = (intensities < 1000).sum()
        print(
            f"      Points below 2000 HU: {below_2000} ({below_2000/len(intensities)*100:.1f}%)"
        )
        print(
            f"      Points below 1500 HU: {below_1500} ({below_1500/len(intensities)*100:.1f}%)"
        )
        print(
            f"      Points below 1000 HU: {below_1000} ({below_1000/len(intensities)*100:.1f}%)"
        )

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return points_world, intensities


def extract_trajectory_pointcloud_gpu_optimized(
    initial_trajectory,
    ct_data: np.ndarray,
    affine: np.ndarray,
    refinement_threshold: float,
    refinement_radius_mm: float = 3.5,
    sample_spacing_mm: float = 0.5,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized GPU version that only processes voxels near the trajectory.

    This version creates a bounding box around the trajectory first,
    then only processes voxels within that box.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(
        f"    Using PyTorch on {device} for optimized trajectory point cloud extraction"
    )

    # Sample points along the initial trajectory
    n_samples = int(initial_trajectory.total_length_mm / sample_spacing_mm)
    t_values = np.linspace(0, 1, n_samples)
    trajectory_points = polyval3(initial_trajectory.polynomial, t_values)

    print(
        f"      Sampling {n_samples} points along {initial_trajectory.total_length_mm:.1f}mm trajectory"
    )

    # Get bounding box of trajectory with padding
    traj_min = trajectory_points.min(axis=0) - refinement_radius_mm
    traj_max = trajectory_points.max(axis=0) + refinement_radius_mm

    # Convert to voxel coordinates
    affine_inv = np.linalg.inv(affine)
    min_voxel = (affine_inv @ np.append(traj_min, 1))[:3]
    max_voxel = (affine_inv @ np.append(traj_max, 1))[:3]

    # Handle flipped axes
    voxel_coords = np.vstack([min_voxel, max_voxel])
    min_voxel = np.minimum(voxel_coords[0], voxel_coords[1]).astype(int)
    max_voxel = np.maximum(voxel_coords[0], voxel_coords[1]).astype(int) + 1

    # Ensure valid bounds
    min_voxel = np.maximum(min_voxel, 0)
    max_voxel = np.minimum(max_voxel, ct_data.shape)

    # Extract ROI
    roi = ct_data[
        min_voxel[0] : max_voxel[0],
        min_voxel[1] : max_voxel[1],
        min_voxel[2] : max_voxel[2],
    ]

    print(f"      ROI shape: {roi.shape} (from full volume {ct_data.shape})")

    # Convert to PyTorch
    roi_tensor = torch.from_numpy(roi).float().to(device)
    affine_tensor = torch.from_numpy(affine).float().to(device)
    trajectory_tensor = torch.from_numpy(trajectory_points).float().to(device)

    # Create grid for ROI
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(roi.shape[0], device=device) + min_voxel[0],
        torch.arange(roi.shape[1], device=device) + min_voxel[1],
        torch.arange(roi.shape[2], device=device) + min_voxel[2],
        indexing="ij",
    )

    # Stack into voxel coordinates
    voxel_coords = torch.stack([z_coords, y_coords, x_coords], dim=-1).float()
    voxel_coords_flat = voxel_coords.reshape(-1, 3)

    # Convert to world coordinates
    voxel_coords_homo = torch.cat(
        [voxel_coords_flat, torch.ones((voxel_coords_flat.shape[0], 1), device=device)],
        dim=1,
    )
    world_coords = (voxel_coords_homo @ affine_tensor.T)[:, :3]

    # Apply threshold
    roi_flat = roi_tensor.reshape(-1)
    threshold_mask = roi_flat > refinement_threshold

    world_coords_thresh = world_coords[threshold_mask]
    intensities_thresh = roi_flat[threshold_mask]

    print(
        f"      Found {threshold_mask.sum().item()} voxels above {refinement_threshold} HU in ROI"
    )

    if threshold_mask.sum() == 0:
        return np.array([]), np.array([])

    # Compute distances using matrix operations
    # For memory efficiency, process in chunks
    chunk_size = min(500000, world_coords_thresh.shape[0])
    within_radius_indices = []

    for i in range(0, world_coords_thresh.shape[0], chunk_size):
        chunk = world_coords_thresh[i : i + chunk_size]

        # Compute pairwise distances
        distances = torch.cdist(chunk.unsqueeze(0), trajectory_tensor.unsqueeze(0))[0]

        # Find minimum distance to trajectory
        min_distances, _ = distances.min(dim=1)

        # Get indices within radius
        chunk_mask = min_distances <= refinement_radius_mm
        chunk_indices = torch.arange(
            i, min(i + chunk_size, world_coords_thresh.shape[0]), device=device
        )[chunk_mask]
        within_radius_indices.append(chunk_indices)

    # Combine all indices
    if within_radius_indices:
        all_indices = torch.cat(within_radius_indices)

        # Extract final point cloud
        points_world = world_coords_thresh[all_indices].cpu().numpy()
        intensities = intensities_thresh[all_indices].cpu().numpy()
    else:
        points_world = np.array([])
        intensities = np.array([])

    print(f"      Trajectory-based point cloud: {len(points_world)} points")
    if len(intensities) > 0:
        print(
            f"      Intensity range: [{intensities.min():.0f}, {intensities.max():.0f}] HU"
        )

        # Count distribution
        below_2000 = (intensities < 2000).sum()
        below_1500 = (intensities < 1500).sum()
        below_1000 = (intensities < 1000).sum()
        print(
            f"      Points below 2000 HU: {below_2000} ({below_2000/len(intensities)*100:.1f}%)"
        )
        print(
            f"      Points below 1500 HU: {below_1500} ({below_1500/len(intensities)*100:.1f}%)"
        )
        print(
            f"      Points below 1000 HU: {below_1000} ({below_1000/len(intensities)*100:.1f}%)"
        )

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return points_world, intensities
