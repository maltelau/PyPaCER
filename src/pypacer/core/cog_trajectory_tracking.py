"""Center of gravity based trajectory tracking for electrode detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class TrajectoryPoint:
    """Single point along an electrode trajectory."""

    slice_idx: int
    center_of_gravity: np.ndarray  # 3D world coordinates
    center_voxel: np.ndarray  # 3D voxel coordinates
    num_voxels: int
    mean_intensity: float
    direction_vector: Optional[np.ndarray] = None
    direction_change_angle: Optional[float] = None


class CenterOfGravityTracker:
    """
    Track electrode trajectory by following center of gravity through CT slices.

    This algorithm starts from a seed point and tracks the center of gravity
    of metal artifacts through adjacent slices, monitoring direction changes
    to detect when the electrode exits the skull.
    """

    def __init__(
        self,
        ct_data: np.ndarray,
        affine: np.ndarray,
        metal_threshold: float = 2000.0,
        search_radius_voxels: int = 10,
        max_direction_change_deg: float = 30.0,
        min_voxels_per_slice: int = 3,
        smoothing_window: int = 3,
        search_radius_mm: Optional[float] = None,
    ):
        """
        Initialize the tracker.

        Args:
            ct_data: 3D CT volume
            affine: Affine transformation matrix to world coordinates
            metal_threshold: HU threshold for metal detection
            search_radius_voxels: Radius around COG to search in next slice (deprecated, use search_radius_mm)
            max_direction_change_deg: Maximum allowed direction change between slices
            min_voxels_per_slice: Minimum voxels required to continue tracking
            smoothing_window: Window size for direction vector smoothing
            search_radius_mm: Physical search radius in mm (overrides search_radius_voxels)
        """
        self.ct_data = ct_data
        self.affine = affine
        self.metal_threshold = metal_threshold
        self.max_direction_change = np.radians(max_direction_change_deg)
        self.min_voxels = min_voxels_per_slice
        self.smoothing_window = smoothing_window

        # Get voxel sizes from affine matrix
        self.voxel_sizes = np.abs(np.diagonal(affine[:3, :3]))

        # Set search radius - prefer physical units if specified
        if search_radius_mm is not None:
            self.search_radius_mm = search_radius_mm
            # Also store voxel-based radius for backward compatibility
            avg_voxel_size = np.mean(self.voxel_sizes)
            self.search_radius = self.search_radius_mm / avg_voxel_size
        else:
            # Use voxel-based radius (for backward compatibility)
            self.search_radius = search_radius_voxels
            # Convert to approximate mm using average voxel size
            avg_voxel_size = np.mean(self.voxel_sizes)
            self.search_radius_mm = search_radius_voxels * avg_voxel_size

    def track_from_seed(
        self, seed_voxel: Tuple[int, int, int], slice_axis: str = "axial"
    ) -> List[TrajectoryPoint]:
        """
        Track electrode trajectory from a seed point.

        Args:
            seed_voxel: Starting voxel coordinates (i, j, k)
            slice_axis: Which axis to track along ('axial', 'sagittal', 'coronal')

        Returns:
            List of trajectory points along the electrode
        """
        # Determine slice axis
        axis_map = {"axial": 2, "sagittal": 0, "coronal": 1}
        slice_axis_idx = axis_map.get(slice_axis, 2)

        # Start from seed slice
        seed_slice = seed_voxel[slice_axis_idx]

        # Find initial center of gravity in seed slice
        initial_cog = self._find_center_of_gravity_in_slice(
            seed_voxel, slice_axis_idx, seed_slice
        )

        if initial_cog is None:
            print("No metal voxels found at seed point")
            return []

        # Create initial trajectory point
        initial_point = TrajectoryPoint(
            slice_idx=seed_slice,
            center_of_gravity=self._voxel_to_world(initial_cog["center"]),
            center_voxel=initial_cog["center"],
            num_voxels=initial_cog["num_voxels"],
            mean_intensity=initial_cog["mean_intensity"],
        )

        trajectory = [initial_point]

        # Track downward (decreasing slice index)
        print(f"Tracking downward from slice {seed_slice}...")
        down_points = self._track_direction(
            initial_cog["center"], seed_slice, slice_axis_idx, direction=-1
        )

        # Track upward (increasing slice index)
        print(f"Tracking upward from slice {seed_slice}...")
        up_points = self._track_direction(
            initial_cog["center"], seed_slice, slice_axis_idx, direction=1
        )

        # Combine trajectories (reverse down points to maintain order)
        trajectory = down_points[::-1] + [initial_point] + up_points

        # Calculate direction vectors and changes
        self._calculate_direction_vectors(trajectory)

        # Smooth direction vectors
        self._smooth_direction_vectors(trajectory)

        print(f"Tracked {len(trajectory)} points along electrode")

        return trajectory

    def _track_direction(
        self, start_cog: np.ndarray, start_slice: int, slice_axis: int, direction: int
    ) -> List[TrajectoryPoint]:
        """
        Track in one direction from starting point.

        Args:
            start_cog: Starting center of gravity (voxel coords)
            start_slice: Starting slice index
            slice_axis: Which axis we're slicing along (0, 1, or 2)
            direction: +1 for up, -1 for down

        Returns:
            List of trajectory points
        """
        points = []
        current_cog = start_cog.copy()
        current_slice = start_slice

        # Store recent direction vectors for smoothing
        recent_directions = []

        # Get slice limits
        max_slice = self.ct_data.shape[slice_axis]

        while True:
            # Move to next slice
            current_slice += direction

            # Check bounds
            if current_slice < 0 or current_slice >= max_slice:
                print(f"Reached volume boundary at slice {current_slice}")
                break

            # Find center of gravity in new slice
            cog_result = self._find_center_of_gravity_in_slice(
                current_cog, slice_axis, current_slice
            )

            if cog_result is None:
                print(f"No metal voxels found in slice {current_slice}")
                break

            # Check minimum voxel count
            if cog_result["num_voxels"] < self.min_voxels:
                print(
                    f"Too few voxels ({cog_result['num_voxels']}) in slice {current_slice}"
                )
                break

            # Create trajectory point
            point = TrajectoryPoint(
                slice_idx=current_slice,
                center_of_gravity=self._voxel_to_world(cog_result["center"]),
                center_voxel=cog_result["center"],
                num_voxels=cog_result["num_voxels"],
                mean_intensity=cog_result["mean_intensity"],
            )

            # Calculate direction vector in world space for orientation-independent angle calculation
            if len(points) > 0:
                # Convert voxel coordinates to world coordinates
                current_world = self._voxel_to_world(current_cog)
                new_world = self._voxel_to_world(cog_result["center"])
                previous_world = self._voxel_to_world(points[-1].center_voxel)

                # Calculate direction vector in world space
                direction_vec_world = new_world - current_world
                direction_vec_world_norm = direction_vec_world / (
                    np.linalg.norm(direction_vec_world) + 1e-6
                )

                # Calculate step size in mm (physical distance between slices along trajectory)
                step_distance_mm = np.linalg.norm(direction_vec_world)

                # Check for sudden direction change
                if len(recent_directions) >= 2:
                    # Average recent directions for stability (in world space)
                    avg_direction = np.mean(
                        recent_directions[-min(3, len(recent_directions)) :], axis=0
                    )
                    avg_direction /= np.linalg.norm(avg_direction) + 1e-6

                    # Calculate angle change in world space
                    cos_angle = np.dot(direction_vec_world_norm, avg_direction)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.arccos(cos_angle)

                    # Normalize angle change by step distance to get angle change per mm
                    # This makes the threshold independent of trajectory angle relative to slice plane
                    angle_change_per_mm = angle_change / (step_distance_mm + 1e-6)

                    # Convert threshold to per-mm basis (assuming 1mm nominal step size)
                    max_angle_per_mm = self.max_direction_change / 1.0  # radians per mm

                    if angle_change_per_mm > max_angle_per_mm:
                        print(
                            f"Direction change too large ({np.degrees(angle_change):.1f}° over {step_distance_mm:.2f}mm = {np.degrees(angle_change_per_mm):.1f}°/mm) at slice {current_slice}"
                        )
                        print(f"  Threshold: {np.degrees(max_angle_per_mm):.1f}°/mm")
                        print("  Likely reached skull exit point")
                        break

                    point.direction_change_angle = np.degrees(angle_change)

                recent_directions.append(direction_vec_world_norm)
                # Store direction in voxel space for compatibility
                point.direction_vector = cog_result["center"] - current_cog

            points.append(point)
            current_cog = cog_result["center"]

            # Limit tracking length to prevent runaway
            if len(points) > 200:  # ~200mm maximum at 1mm slices
                print("Reached maximum tracking length")
                break

        return points

    def _find_center_of_gravity_in_slice(
        self, seed_point: np.ndarray, slice_axis: int, slice_idx: int
    ) -> Optional[Dict]:
        """
        Find center of gravity of metal voxels near seed point in a slice.

        Args:
            seed_point: Seed point in voxel coordinates
            slice_axis: Which axis we're slicing along
            slice_idx: Index of the slice

        Returns:
            Dictionary with center of gravity info, or None if no voxels found
        """
        # Extract slice
        if slice_axis == 0:  # Sagittal
            slice_data = self.ct_data[slice_idx, :, :]
            seed_2d = (seed_point[1], seed_point[2])
        elif slice_axis == 1:  # Coronal
            slice_data = self.ct_data[:, slice_idx, :]
            seed_2d = (seed_point[0], seed_point[2])
        else:  # Axial (2)
            slice_data = self.ct_data[:, :, slice_idx]
            seed_2d = (seed_point[0], seed_point[1])

        # Create search region around seed point using physical radius
        # Convert mm radius to voxels for each axis
        if slice_axis == 0:  # Sagittal - searching in YZ plane
            radius_y_voxels = self.search_radius_mm / self.voxel_sizes[1]
            radius_x_voxels = self.search_radius_mm / self.voxel_sizes[2]
        elif slice_axis == 1:  # Coronal - searching in XZ plane
            radius_y_voxels = self.search_radius_mm / self.voxel_sizes[0]
            radius_x_voxels = self.search_radius_mm / self.voxel_sizes[2]
        else:  # Axial (2) - searching in XY plane
            radius_y_voxels = self.search_radius_mm / self.voxel_sizes[0]
            radius_x_voxels = self.search_radius_mm / self.voxel_sizes[1]

        y_min = max(0, int(seed_2d[0] - radius_y_voxels))
        y_max = min(slice_data.shape[0], int(seed_2d[0] + radius_y_voxels + 1))
        x_min = max(0, int(seed_2d[1] - radius_x_voxels))
        x_max = min(slice_data.shape[1], int(seed_2d[1] + radius_x_voxels + 1))

        # Extract region
        region = slice_data[y_min:y_max, x_min:x_max]

        # Find metal voxels
        metal_mask = region >= self.metal_threshold

        if not metal_mask.any():
            return None

        # Calculate center of gravity
        # Weight by intensity for better centering
        weights = region * metal_mask
        cog_local = ndimage.center_of_mass(weights)

        # Convert back to full slice coordinates
        cog_2d = (cog_local[0] + y_min, cog_local[1] + x_min)

        # Convert to 3D voxel coordinates
        if slice_axis == 0:  # Sagittal
            cog_3d = np.array([slice_idx, cog_2d[0], cog_2d[1]])
        elif slice_axis == 1:  # Coronal
            cog_3d = np.array([cog_2d[0], slice_idx, cog_2d[1]])
        else:  # Axial
            cog_3d = np.array([cog_2d[0], cog_2d[1], slice_idx])

        # Calculate statistics
        num_voxels = int(metal_mask.sum())
        mean_intensity = float(weights.sum() / metal_mask.sum())

        return {
            "center": cog_3d,
            "num_voxels": num_voxels,
            "mean_intensity": mean_intensity,
        }

    def _voxel_to_world(self, voxel_coord: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates to world coordinates."""
        voxel_homogeneous = np.append(voxel_coord, 1)
        world_coord = (self.affine @ voxel_homogeneous)[:3]
        return world_coord

    def _calculate_direction_vectors(self, trajectory: List[TrajectoryPoint]) -> None:
        """Calculate direction vectors between consecutive points in world coordinates."""
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i - 1]
            curr_point = trajectory[i]

            # Direction in world coordinates (already in world coords)
            direction = curr_point.center_of_gravity - prev_point.center_of_gravity
            curr_point.direction_vector = direction

            # Calculate angle change if previous direction exists
            if i > 1 and trajectory[i - 1].direction_vector is not None:
                prev_dir = trajectory[i - 1].direction_vector
                curr_dir = direction

                # Normalize
                prev_norm = prev_dir / (np.linalg.norm(prev_dir) + 1e-6)
                curr_norm = curr_dir / (np.linalg.norm(curr_dir) + 1e-6)

                # Calculate angle
                cos_angle = np.dot(prev_norm, curr_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                curr_point.direction_change_angle = np.degrees(angle)

    def _smooth_direction_vectors(self, trajectory: List[TrajectoryPoint]) -> None:
        """Apply smoothing to direction vectors to reduce noise."""
        if len(trajectory) < self.smoothing_window:
            return

        # Smooth using moving average
        for i in range(len(trajectory)):
            if trajectory[i].direction_vector is None:
                continue

            # Get window of vectors
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(trajectory), i + self.smoothing_window // 2 + 1)

            vectors = []
            for j in range(start_idx, end_idx):
                if trajectory[j].direction_vector is not None:
                    vectors.append(trajectory[j].direction_vector)

            if vectors:
                # Average vectors
                smooth_vector = np.mean(vectors, axis=0)
                trajectory[i].direction_vector = smooth_vector

    def refine_skull_exit_detection(
        self,
        trajectory: List[TrajectoryPoint],
        polynomial_coeffs: np.ndarray,
        step_size_mm: float = 1.0,
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Refine skull exit detection by resampling along polynomial at regular intervals.

        Args:
            trajectory: Original trajectory points
            polynomial_coeffs: Fitted polynomial coefficients
            step_size_mm: Step size in mm for resampling (default 1mm)

        Returns:
            Tuple of (exit_index, exit_arc_length) or (None, None) if no exit found
        """
        if len(trajectory) < 10:
            return None, None

        # Get world coordinates of trajectory
        world_points = np.array([p.center_of_gravity for p in trajectory])

        # Calculate arc length along original trajectory
        arc_lengths = np.zeros(len(world_points))
        for i in range(1, len(world_points)):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(
                world_points[i] - world_points[i - 1]
            )

        total_length = arc_lengths[-1]

        # Resample at regular intervals
        sample_distances = np.arange(0, total_length, step_size_mm)

        # Skip first few mm to avoid noise at tip
        start_idx = max(5, int(5.0 / step_size_mm))  # Start 5mm from tip

        # Calculate angles at each sample point
        angles = []
        for i in range(start_idx + 2, len(sample_distances)):
            # Get 3 points: current, 2mm back, 2mm forward
            s_prev = sample_distances[i - 2]
            s_curr = sample_distances[i]
            s_next = min(
                (
                    sample_distances[i + 2]
                    if i + 2 < len(sample_distances)
                    else total_length
                ),
                total_length,
            )

            # Evaluate polynomial at these points
            # Assuming polynomial is parameterized by arc length (need to convert)
            # For now, use linear interpolation on original points
            p_prev = self._interpolate_point(world_points, arc_lengths, s_prev)
            p_curr = self._interpolate_point(world_points, arc_lengths, s_curr)
            p_next = self._interpolate_point(world_points, arc_lengths, s_next)

            # Calculate vectors
            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            # Normalize
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

            # Calculate angle
            cos_angle = np.dot(v1_norm, v2_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            angles.append((s_curr, np.degrees(angle)))

        # Find maximum angle change
        if not angles:
            return None, None

        max_angle_idx = np.argmax([a[1] for a in angles])
        max_angle_dist, max_angle_deg = angles[max_angle_idx]

        # Check if angle exceeds threshold
        if max_angle_deg > np.degrees(self.max_direction_change):
            # Find corresponding index in original trajectory
            exit_idx = np.searchsorted(arc_lengths, max_angle_dist)
            exit_idx = min(exit_idx, len(trajectory) - 1)

            print("  Refined skull exit detection:")
            print(
                f"    Maximum angle change: {max_angle_deg:.1f}° at {max_angle_dist:.1f}mm along trajectory"
            )
            print(
                f"    Original exit at index {exit_idx} (slice {trajectory[exit_idx].slice_idx})"
            )

            return exit_idx, max_angle_dist

        return None, None

    def _interpolate_point(
        self, points: np.ndarray, arc_lengths: np.ndarray, target_length: float
    ) -> np.ndarray:
        """Linearly interpolate point at given arc length."""
        if target_length <= 0:
            return points[0]
        if target_length >= arc_lengths[-1]:
            return points[-1]

        # Find bracketing indices
        idx = np.searchsorted(arc_lengths, target_length)
        if idx == 0:
            return points[0]

        # Linear interpolation
        t = (target_length - arc_lengths[idx - 1]) / (
            arc_lengths[idx] - arc_lengths[idx - 1]
        )
        return points[idx - 1] + t * (points[idx] - points[idx - 1])


def extract_electrode_from_seed(
    ct_data: np.ndarray,
    affine: np.ndarray,
    seed_voxel: Tuple[int, int, int],
    metal_threshold: float = 2000.0,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to extract electrode trajectory from a seed point.

    Args:
        ct_data: 3D CT volume
        affine: Affine transformation matrix
        seed_voxel: Starting voxel coordinates
        metal_threshold: HU threshold for metal
        **kwargs: Additional parameters for CenterOfGravityTracker

    Returns:
        Nx3 array of trajectory points in world coordinates
    """
    tracker = CenterOfGravityTracker(
        ct_data=ct_data, affine=affine, metal_threshold=metal_threshold, **kwargs
    )

    trajectory_points = tracker.track_from_seed(seed_voxel)

    if not trajectory_points:
        return np.array([])

    # Extract world coordinates
    world_points = np.array([p.center_of_gravity for p in trajectory_points])

    return world_points
