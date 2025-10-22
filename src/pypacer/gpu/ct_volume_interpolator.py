"""Direct trilinear interpolation from CT volume data."""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


class CTVolumeInterpolator:
    """
    Direct trilinear interpolation from CT volume using PyTorch.

    This interpolator works directly with the 3D CT volume data,
    avoiding the need to create point clouds.
    """

    def __init__(
        self,
        ct_volume: Union[np.ndarray, torch.Tensor],
        affine: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize CT volume interpolator.

        Args:
            ct_volume: 3D CT volume data (nx, ny, nz)
            affine: 4x4 affine transformation matrix from voxel to world coordinates
            device: Torch device (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Convert CT volume to torch tensor
        if isinstance(ct_volume, np.ndarray):
            self.ct_volume = torch.from_numpy(ct_volume).float().to(self.device)
        else:
            self.ct_volume = ct_volume.float().to(self.device)

        # Add batch and channel dimensions for grid_sample
        # Shape: (1, 1, nx, ny, nz)
        self.ct_volume = self.ct_volume.unsqueeze(0).unsqueeze(0)

        # Store affine and its inverse
        self.affine = torch.from_numpy(affine).float().to(self.device)
        self.affine_inv = torch.linalg.inv(self.affine)

        # Get volume dimensions
        self.volume_shape = torch.tensor(
            ct_volume.shape, device=self.device, dtype=torch.float32
        )

        print(
            f"  CT volume interpolator initialized: {ct_volume.shape} on {self.device}"
        )

    def world_to_voxel(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel coordinates."""
        # Add homogeneous coordinate
        ones = torch.ones(world_coords.shape[0], 1, device=self.device)
        world_homo = torch.cat([world_coords, ones], dim=1)

        # Transform to voxel space
        voxel_homo = (self.affine_inv @ world_homo.T).T
        voxel_coords = voxel_homo[:, :3]

        return voxel_coords

    def interpolate(
        self, world_coords: Union[np.ndarray, torch.Tensor], return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Interpolate CT values at world coordinates using trilinear interpolation.

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

        # Normalize voxel coordinates to [-1, 1] for grid_sample
        # grid_sample expects coordinates in range [-1, 1]
        normalized_coords = 2.0 * voxel_coords / (self.volume_shape - 1) - 1.0

        # Swap coordinates because grid_sample expects (z, y, x) order
        normalized_coords = normalized_coords[:, [2, 1, 0]]

        # Prepare for grid_sample: (1, 1, 1, N, 3)
        n_points = normalized_coords.shape[0]
        sample_grid = normalized_coords.view(1, 1, 1, n_points, 3)

        # Interpolate using grid_sample
        # Input: (1, 1, nx, ny, nz)
        # Grid: (1, 1, 1, N, 3)
        # Output: (1, 1, 1, 1, N)
        interpolated = F.grid_sample(
            self.ct_volume,
            sample_grid,
            mode="bilinear",  # Trilinear for 3D
            padding_mode="zeros",  # Return 0 for out-of-bounds
            align_corners=True,
        )

        # Extract values: (1, 1, 1, 1, N) -> (N,)
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
