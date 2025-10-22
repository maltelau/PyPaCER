"""Subvolume CT interpolation for memory-efficient GPU processing."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class SubvolumeInterpolator:
    """
    Memory-efficient CT volume interpolator that only loads a subvolume.

    This is particularly useful for large CT scans where loading the entire
    volume would consume too much GPU memory.
    """

    def __init__(
        self,
        ct_volume: np.ndarray,
        affine: np.ndarray,
        bbox_world: Tuple[np.ndarray, np.ndarray],
        padding_mm: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize subvolume interpolator.

        Args:
            ct_volume: Full CT volume data (will extract subvolume)
            affine: 4x4 affine transformation matrix
            bbox_world: (min_coords, max_coords) bounding box in world coordinates
            padding_mm: Additional padding around bounding box in mm
            device: Torch device
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Store full volume shape and affine
        self.full_volume_shape = ct_volume.shape
        self.affine = affine
        self.affine_inv = np.linalg.inv(affine)

        # Check affine orientation
        affine_det = np.linalg.det(affine[:3, :3])
        if affine_det < 0:
            print(
                f"  Note: Affine has negative determinant ({affine_det:.2f}), coordinate system is flipped"
            )

        # Convert bounding box to voxel coordinates with padding
        min_world, max_world = bbox_world

        # Add padding
        padding_world = np.array([padding_mm, padding_mm, padding_mm])
        min_world = min_world - padding_world
        max_world = max_world + padding_world

        # Convert to voxel coordinates
        min_voxel = self._world_to_voxel_numpy(min_world)
        max_voxel = self._world_to_voxel_numpy(max_world)

        # Debug output
        print(f"  World bbox: min={min_world}, max={max_world}")
        print(f"  Voxel bbox: min={min_voxel}, max={max_voxel}")

        # Handle negative affine transforms by ensuring min < max for each axis
        voxel_min = np.minimum(min_voxel, max_voxel)
        voxel_max = np.maximum(min_voxel, max_voxel)

        # Ensure integer indices and clip to volume bounds
        min_idx = np.maximum(0, np.floor(voxel_min).astype(int))
        max_idx = np.minimum(self.full_volume_shape, np.ceil(voxel_max).astype(int) + 1)

        # Ensure we have a valid subvolume
        if np.any(max_idx <= min_idx):
            print("  Warning: Invalid subvolume bounds - using full volume")
            print(f"  min_idx: {min_idx}, max_idx: {max_idx}")
            # Fall back to full volume
            min_idx = np.array([0, 0, 0])
            max_idx = np.array(self.full_volume_shape)

        # Extract subvolume
        self.subvolume_offset = min_idx
        self.subvolume = ct_volume[
            min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2]
        ].copy()  # Copy to ensure contiguous memory

        # Calculate memory savings
        full_size_mb = np.prod(self.full_volume_shape) * 4 / (1024**2)  # float32
        sub_size_mb = np.prod(self.subvolume.shape) * 4 / (1024**2)
        reduction = 100 * (1 - sub_size_mb / full_size_mb)

        print(
            f"  Subvolume extraction: {self.subvolume.shape} from {self.full_volume_shape}"
        )
        print(
            f"  Memory reduction: {reduction:.1f}% ({full_size_mb:.1f}MB -> {sub_size_mb:.1f}MB)"
        )

        # Convert subvolume to torch tensor
        self.subvolume_torch = torch.from_numpy(self.subvolume).float().to(self.device)
        self.subvolume_torch = self.subvolume_torch.unsqueeze(0).unsqueeze(0)

        # Store torch versions of affine matrices
        self.affine_torch = torch.from_numpy(self.affine).float().to(self.device)
        self.affine_inv_torch = (
            torch.from_numpy(self.affine_inv).float().to(self.device)
        )

        # Subvolume shape
        self.subvolume_shape = torch.tensor(
            self.subvolume.shape, device=self.device, dtype=torch.float32
        )

    def _world_to_voxel_numpy(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world to voxel coordinates using numpy."""
        if world_coords.ndim == 1:
            world_homo = np.append(world_coords, 1)
            voxel_homo = self.affine_inv @ world_homo
            return voxel_homo[:3]
        else:
            world_homo = np.column_stack([world_coords, np.ones(len(world_coords))])
            voxel_homo = (self.affine_inv @ world_homo.T).T
            return voxel_homo[:, :3]

    def world_to_voxel(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel coordinates."""
        # Add homogeneous coordinate
        ones = torch.ones(world_coords.shape[0], 1, device=self.device)
        world_homo = torch.cat([world_coords, ones], dim=1)

        # Transform to voxel space
        voxel_homo = (self.affine_inv_torch @ world_homo.T).T
        voxel_coords = voxel_homo[:, :3]

        return voxel_coords

    def interpolate(
        self, world_coords: Union[np.ndarray, torch.Tensor], return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Interpolate CT values at world coordinates using subvolume.

        Args:
            world_coords: (N, 3) array of world coordinates
            return_numpy: Whether to return numpy array

        Returns:
            Interpolated CT values
        """
        # Convert to tensor
        if isinstance(world_coords, np.ndarray):
            world_tensor = torch.from_numpy(world_coords).float().to(self.device)
        else:
            world_tensor = world_coords.float().to(self.device)

        # Convert world to voxel coordinates
        voxel_coords = self.world_to_voxel(world_tensor)

        # Adjust for subvolume offset
        offset_tensor = torch.tensor(
            self.subvolume_offset, device=self.device, dtype=torch.float32
        )
        voxel_coords_sub = voxel_coords - offset_tensor

        # Normalize to [-1, 1] for grid_sample (relative to subvolume)
        normalized_coords = 2.0 * voxel_coords_sub / (self.subvolume_shape - 1) - 1.0

        # Swap coordinates for grid_sample
        normalized_coords = normalized_coords[:, [2, 1, 0]]

        # Prepare for grid_sample
        n_points = normalized_coords.shape[0]
        sample_grid = normalized_coords.view(1, 1, 1, n_points, 3)

        # Interpolate
        interpolated = F.grid_sample(
            self.subvolume_torch,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Extract values
        interpolated = interpolated.squeeze()

        # Handle single point
        if interpolated.ndim == 0:
            interpolated = interpolated.unsqueeze(0)

        if return_numpy:
            return interpolated.cpu().numpy()
        else:
            return interpolated

    def __call__(self, world_coords):
        """Make callable like scipy interpolators."""
        return self.interpolate(world_coords, return_numpy=True)

    def batch_interpolate(
        self,
        world_coords: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Batch interpolation for large query sets.

        Args:
            world_coords: World coordinates
            batch_size: Process in chunks if specified

        Returns:
            Interpolated values
        """
        if batch_size is not None and batch_size > 0:
            n_points = world_coords.shape[0]
            results = []

            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batch = world_coords[i:end_idx]
                batch_result = self.interpolate(batch, return_numpy=True)
                results.append(batch_result)

            return np.concatenate(results)
        else:
            return self.interpolate(world_coords, return_numpy=True)


def create_subvolume_interpolator(
    ct_volume: np.ndarray,
    affine: np.ndarray,
    points_world: np.ndarray,
    initial_trajectory: np.ndarray,
    padding_mm: float = 15.0,
    device: Optional[torch.device] = None,
) -> SubvolumeInterpolator:
    """
    Create a subvolume interpolator based on electrode trajectory and points.

    Args:
        ct_volume: Full CT volume
        affine: Affine transformation matrix
        points_world: Point cloud around electrode
        initial_trajectory: Initial polynomial trajectory
        padding_mm: Padding around bounding box
        device: Torch device

    Returns:
        SubvolumeInterpolator instance
    """
    from ..utils.math_helpers import polyval3

    # Get bounding box from points
    min_coords = points_world.min(axis=0)
    max_coords = points_world.max(axis=0)

    # Also include trajectory points to ensure full coverage
    t_values = np.linspace(-0.1, 1.1, 50)  # Include lookahead
    trajectory_points = polyval3(initial_trajectory, t_values)

    # Update bounding box
    min_coords = np.minimum(min_coords, trajectory_points.min(axis=0))
    max_coords = np.maximum(max_coords, trajectory_points.max(axis=0))

    # Create interpolator
    return SubvolumeInterpolator(
        ct_volume=ct_volume,
        affine=affine,
        bbox_world=(min_coords, max_coords),
        padding_mm=padding_mm,
        device=device,
    )
