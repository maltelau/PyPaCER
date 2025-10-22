"""GPU-accelerated trajectory refinement using Orthogonal Optimal Resampling (OOR)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.math_helpers import (
    fit_polynomial_3d,
    poly_arc_length_3d,
    polyval3,
    reparameterize_polynomial_3d,
)
from .gpu_utils import (
    ensure_gpu_array,
    get_array_module,
    gpu_available,
    pytorch_available,
)

# CuPy is no longer used - all GPU processing uses PyTorch

# PyTorch is required for GPU acceleration
PYTORCH_INTERPOLATION_AVAILABLE = pytorch_available()


@dataclass
class RefinedTrajectoryGPU:
    """Container for GPU-refined trajectory results."""

    polynomial: np.ndarray
    skeleton: np.ndarray
    intensity_profile: np.ndarray
    distance_scale_mm: np.ndarray
    orthogonal_volume: Optional[np.ndarray] = None
    electrode_type: Optional[str] = None
    total_length_mm: Optional[float] = None
    # Pass 2 data for visualization
    pass2_intensities: Optional[np.ndarray] = None
    pass2_distances_mm: Optional[np.ndarray] = None
    pass2_tip_threshold: Optional[float] = None
    pass2_tip_param: Optional[float] = None
    # Skeleton deviation data
    skeleton_deviations_mm: Optional[np.ndarray] = None
    skeleton_deviation_distances_mm: Optional[np.ndarray] = None
    # Debug polynomial before tip detection
    polynomial_before_tip_detection: Optional[np.ndarray] = None
    # Full Pass 2 data for debug visualization
    pass2_intensities_full: Optional[np.ndarray] = None
    pass2_distances_mm_full: Optional[np.ndarray] = None
    # Original t=0 position in tip-relative coordinates
    original_t0_distance_mm: Optional[float] = None


def _create_orthogonal_grid_gpu(
    center: np.ndarray,
    normal: np.ndarray,
    binormal: np.ndarray,
    grid_size: float,
    n_points: int,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Create orthogonal sampling grid on GPU.

    Args:
        center: 3D center point
        normal: Normal vector
        binormal: Binormal vector
        grid_size: Grid extent in mm
        n_points: Number of points per dimension
        use_gpu: Whether to use GPU

    Returns:
        Grid points in 3D space
    """
    xp = get_array_module(use_gpu=use_gpu)

    # Create 2D grid coordinates
    u = xp.linspace(-grid_size, grid_size, n_points, dtype=xp.float32)
    v = xp.linspace(-grid_size, grid_size, n_points, dtype=xp.float32)
    uu, vv = xp.meshgrid(u, v)

    # Convert to correct device
    center_dev = ensure_gpu_array(center, use_gpu)
    normal_dev = ensure_gpu_array(normal, use_gpu)
    binormal_dev = ensure_gpu_array(binormal, use_gpu)

    # Generate 3D grid points
    grid_points = (
        center_dev
        + uu.ravel()[:, None] * normal_dev
        + vv.ravel()[:, None] * binormal_dev
    )

    return grid_points


def _orthogonal_optimal_resampling_gpu_pytorch(
    polynomial: np.ndarray,
    points: np.ndarray,
    intensities: np.ndarray,
    grid_size: float = 2.0,
    xy_resolution: float = 0.1,
    t_start: float = -0.1,
    t_end: float = 1.0,
    t_step: float = 0.001,
    intensity_threshold: float = 1500,
    use_gpu: bool = True,
    electrode_idx: Optional[int] = None,
    debug_output_dir: Optional[str] = None,
    pass_num: Optional[int] = None,
    z_axis_scale: float = 8.0,
    ct_volume: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    use_subvolume: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch-optimized OOR with batch processing.

    This implementation processes all orthogonal grids in parallel for
    massive speedup compared to sequential processing.

    Args:
        Same as phase1 version

    Returns:
        Tuple of (skeleton_points, median_intensities, orthogonal_volume, distance_scale)
    """
    import torch

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"  Using PyTorch on {device}")

    # Direct CT volume interpolation is required
    if ct_volume is None or affine is None:
        raise ValueError("CT volume and affine matrix are required for GPU refinement")

    # Check if we should use subvolume interpolation
    # use_subvolume parameter comes from the parent function

    if use_subvolume and points is not None and len(points) > 0:
        print("  Using subvolume CT interpolation (memory-efficient)")
        from .subvolume_interpolator import create_subvolume_interpolator

        # Use the point cloud to determine bounding box
        interpolator = create_subvolume_interpolator(
            ct_volume=ct_volume,
            affine=affine,
            points_world=points,
            initial_trajectory=polynomial,
            padding_mm=15.0,
            device=device,
        )
    else:
        print("  Using direct CT volume interpolation (trilinear)")
        from .ct_volume_interpolator import CTVolumeInterpolator

        interpolator = CTVolumeInterpolator(ct_volume, affine, device=device)

    # Generate all t values
    t_values = np.arange(t_start, t_end, t_step, dtype=np.float32)
    n_samples = len(t_values)

    print(f"  Batch processing {n_samples} sample points with PyTorch...")

    # TESTING: Use CPU for polynomial evaluation to isolate interpolation impact
    # Batch compute all trajectory points and derivatives
    # PyTorch handles polynomial evaluation directly in the processing code
    #
    #     # Compute all trajectory points at once
    #     trajectory_points = polyval3_batch_gpu(polynomial_gpu, t_values_gpu, use_gpu=True)
    #     trajectory_points = from_gpu(trajectory_points)
    # else:
    #     # CPU fallback
    #     from ..utils.math_helpers import polyval3
    #     trajectory_points = np.array([polyval3(polynomial, t) for t in t_values])

    # CPU only for testing
    trajectory_points = np.array([polyval3(polynomial, t) for t in t_values])

    # Convert to torch tensor
    # Ensure trajectory_points is a numpy array
    if not isinstance(trajectory_points, np.ndarray):
        print(
            f"  Warning: trajectory_points is {type(trajectory_points)}, converting to numpy array"
        )
        trajectory_points = np.asarray(trajectory_points, dtype=np.float32)
    trajectory_points_torch = torch.from_numpy(trajectory_points).float().to(device)

    # Compute tangent vectors using finite differences
    tangents = torch.zeros_like(trajectory_points_torch)
    tangents[1:] = trajectory_points_torch[1:] - trajectory_points_torch[:-1]
    tangents[0] = tangents[1]  # Use forward difference for first point
    tangents = tangents / torch.norm(tangents, dim=1, keepdim=True)

    # Create orthogonal basis for all points at once
    # Find perpendicular vectors
    perp = torch.zeros_like(tangents)
    mask_x = torch.abs(tangents[:, 0]) < 0.9
    perp[mask_x] = torch.tensor([1.0, 0.0, 0.0], device=device)
    perp[~mask_x] = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Compute normal and binormal vectors
    normals = torch.cross(tangents, perp, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    binormals = torch.cross(tangents, normals, dim=1)

    # Create 2D grid coordinates matching CPU version
    # Use arange with xy_resolution step
    u = torch.arange(
        -grid_size,
        grid_size + xy_resolution,
        xy_resolution,
        device=device,
        dtype=torch.float32,
    )
    v = torch.arange(
        -grid_size,
        grid_size + xy_resolution,
        xy_resolution,
        device=device,
        dtype=torch.float32,
    )
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    grid_coords = torch.stack([uu.flatten(), vv.flatten()], dim=1)  # (grid_points^2, 2)
    grid_points_total = len(u) * len(v)

    # Debug: Print grid dimensions
    print(
        f"  Orthogonal grid: {len(u)}x{len(v)} = {grid_points_total} points (grid_size={grid_size}, xy_res={xy_resolution})"
    )

    # Batch create all orthogonal grids
    # Shape: (n_samples, grid_points^2, 3)
    all_grids = (
        trajectory_points_torch.unsqueeze(1)
        + grid_coords[:, 0].unsqueeze(0).unsqueeze(-1) * normals.unsqueeze(1)
        + grid_coords[:, 1].unsqueeze(0).unsqueeze(-1) * binormals.unsqueeze(1)
    )

    # Flatten for interpolation
    all_grids_flat = all_grids.reshape(-1, 3)  # (n_samples * grid_points^2, 3)

    # GPU batch interpolation
    print(f"  Interpolating {all_grids_flat.shape[0]} points on GPU...")

    # Use batch interpolation with memory management
    all_intensities_flat = interpolator.batch_interpolate(
        all_grids_flat, batch_size=50000  # Reduced batch size to avoid OOM
    )

    # Convert to torch tensor if needed
    if isinstance(all_intensities_flat, np.ndarray):
        all_intensities_flat = torch.from_numpy(all_intensities_flat).float().to(device)

    print("  GPU interpolation complete")

    # Reshape back
    all_intensities = all_intensities_flat.reshape(n_samples, grid_points_total)

    # Process results for each sample point
    skeleton_points = []
    median_intensities = []
    orthogonal_volumes = []
    distance_scales = []

    # Process all points without chunking to avoid boundary artifacts
    # If memory is an issue, increase chunk_size significantly (e.g., 1000+)
    for i in range(n_samples):
        grid_intensities = all_intensities[i]
        grid_points_3d = all_grids[i]

        # Handle NaN values and apply intensity threshold (matching CPU version)
        valid_mask = ~torch.isnan(grid_intensities) & (
            grid_intensities >= intensity_threshold
        )
        if valid_mask.any():
            valid_intensities = grid_intensities[valid_mask]
            valid_points = grid_points_3d[valid_mask]

            if len(valid_intensities) > 0:
                # Compute weighted centroid (intensity-weighted average like CPU)
                weights = valid_intensities / valid_intensities.sum()
                centroid = (valid_points * weights.unsqueeze(-1)).sum(dim=0)

                # Median intensity includes all non-NaN values (not just threshold-filtered)
                non_nan_mask = ~torch.isnan(grid_intensities)
                if non_nan_mask.any():
                    median_intensity = torch.median(
                        grid_intensities[non_nan_mask]
                    ).item()
                else:
                    median_intensity = 0.0

                skeleton_points.append(centroid.cpu().numpy())
                median_intensities.append(median_intensity)
                orthogonal_volumes.append(len(valid_intensities))
                distance_scales.append(t_values[i])
        # else: skip this point entirely if no valid intensities (matching CPU behavior)

    print("  PyTorch batch processing complete")

    # Convert to numpy arrays
    if skeleton_points:
        skeleton_points = np.array(skeleton_points)
        median_intensities = np.array(median_intensities)
        orthogonal_volumes = np.array(orthogonal_volumes)
        distance_scales = np.array(distance_scales)
    else:
        # Return empty arrays with correct shape if no valid points found
        skeleton_points = np.empty((0, 3), dtype=np.float32)
        median_intensities = np.empty(0, dtype=np.float32)
        orthogonal_volumes = np.empty(0, dtype=np.float32)
        distance_scales = np.empty(0, dtype=np.float32)

    return skeleton_points, median_intensities, orthogonal_volumes, distance_scales


def refine_electrode_trajectory_gpu(
    initial_trajectory,
    points_world: np.ndarray,
    intensities: np.ndarray,
    ct_data: np.ndarray,
    affine: np.ndarray,
    final_degree: int = 3,
    xy_resolution: float = 0.1,
    z_resolution: float = 0.025,
    grid_size: float = 2.0,
    use_gpu: bool = True,
    electrode_idx: Optional[int] = None,
    debug_output_dir: Optional[str] = None,
    refinement_threshold: Optional[float] = None,
    refinement_radius_mm: float = 3.5,
    debug_z_axis_scale: float = 8.0,
    use_subvolume: bool = True,
    auto_select_degree: bool = False,
    contact_region_mm: float = 20.0,
) -> RefinedTrajectoryGPU:
    """
    GPU-accelerated electrode trajectory refinement.

    Phase 1 implementation using PyTorch GPU acceleration.

    Args:
        initial_trajectory: Initial trajectory fit
        points_world: Point cloud in world coordinates
        intensities: Intensity values
        ct_data: CT volume data
        affine: Affine transformation matrix
        final_degree: Final polynomial degree
        xy_resolution: XY plane resolution in mm
        z_resolution: Z axis resolution in mm
        grid_size: Orthogonal grid size in mm
        use_gpu: Whether to use GPU acceleration
        electrode_idx: Optional electrode index for debug output
        debug_output_dir: Optional debug output directory
        refinement_threshold: Optional threshold for refinement point cloud
        refinement_radius_mm: Radius around trajectory for refinement
        debug_z_axis_scale: Scale factor for z-axis in debug NIfTI

    Returns:
        Refined trajectory model
    """
    if not use_gpu:
        raise ValueError("GPU processing was explicitly disabled (use_gpu=False)")

    if not gpu_available():
        import torch

        cuda_available = torch.cuda.is_available() if pytorch_available() else False

        error_msg = f"""
        GPU acceleration is required but not available!
        
        PyTorch available: {pytorch_available()}
        CUDA available: {cuda_available}
        
        This function requires GPU acceleration for reasonable performance.
        Possible issues:
        1. PyTorch is not installed or installed without CUDA support
        2. No NVIDIA GPU present or CUDA drivers not installed
        3. GPU is in use by another process
        
        To check: python -c "import torch; print(torch.cuda.is_available())"
        """
        raise RuntimeError(error_msg)

    print("  Starting GPU-accelerated trajectory refinement...")
    print("  *** TESTING MODE: Using CPU for all operations except interpolation ***")

    # Store original point cloud in case we need to fall back
    original_points_world = points_world.copy()
    original_intensities = intensities.copy()

    # Apply refinement threshold if specified
    if refinement_threshold is not None:
        print(
            f"    Extracting point cloud along trajectory (threshold: {refinement_threshold} HU, radius: {refinement_radius_mm} mm)..."
        )

        # Check if PyTorch is available for GPU acceleration
        try:
            import torch

            if torch.cuda.is_available() and use_gpu:
                # Use GPU-accelerated extraction
                from .trajectory_pointcloud_gpu import (
                    extract_trajectory_pointcloud_gpu_optimized,
                )

                points_world, intensities = extract_trajectory_pointcloud_gpu_optimized(
                    initial_trajectory,
                    ct_data,
                    affine,
                    refinement_threshold,
                    refinement_radius_mm,
                    sample_spacing_mm=0.5,
                    use_gpu=True,
                )

                if len(points_world) == 0:
                    print(
                        "      Warning: No voxels found along trajectory, using original point cloud"
                    )
                    # Reset to original if no points found
                    points_world = original_points_world
                    intensities = original_intensities
            else:
                raise ImportError("PyTorch not available or GPU not requested")

        except (ImportError, RuntimeError) as e:
            print(f"      GPU extraction not available ({e}), using CPU fallback...")

            # CPU fallback implementation
            # Sample points along the initial trajectory
            n_samples = int(
                initial_trajectory.total_length_mm / 0.5
            )  # Sample every 0.5mm
            t_values = np.linspace(0, 1, n_samples)
            trajectory_points = polyval3(initial_trajectory.polynomial, t_values)

            print(
                f"      Sampling {n_samples} points along {initial_trajectory.total_length_mm:.1f}mm trajectory"
            )

            # Convert trajectory points to voxel coordinates
            affine_inv = np.linalg.inv(affine)
            trajectory_voxels = []
            for tp in trajectory_points:
                tv = (affine_inv @ np.append(tp, 1))[:3]
                trajectory_voxels.append(tv)
            trajectory_voxels = np.array(trajectory_voxels)

            # Calculate radius in voxels
            voxel_sizes = np.abs(np.diag(affine[:3, :3]))
            radius_voxels = refinement_radius_mm / voxel_sizes
            max_radius_voxels = int(np.ceil(radius_voxels.max()))

            print(
                f"      Search radius: {refinement_radius_mm}mm = ~{max_radius_voxels} voxels"
            )

            # Collect all voxels within radius of trajectory
            tube_voxels = set()

            for i, tv in enumerate(trajectory_voxels):
                if i % 50 == 0:
                    print(
                        f"        Processing trajectory point {i}/{len(trajectory_voxels)}..."
                    )

                # Get integer center
                center = tv.astype(int)

                # Define search bounds
                min_bound = np.maximum(center - max_radius_voxels, 0)
                max_bound = np.minimum(center + max_radius_voxels + 1, ct_data.shape)

                # Check each voxel in the bounding box
                for x in range(min_bound[0], max_bound[0]):
                    for y in range(min_bound[1], max_bound[1]):
                        for z in range(min_bound[2], max_bound[2]):
                            voxel = np.array([x, y, z])
                            # Check if within radius (in mm)
                            voxel_world = (affine @ np.append(voxel, 1))[:3]
                            traj_world = (affine @ np.append(tv, 1))[:3]
                            dist_mm = np.linalg.norm(voxel_world - traj_world)

                            if dist_mm <= refinement_radius_mm:
                                # Check intensity threshold
                                if ct_data[x, y, z] > refinement_threshold:
                                    tube_voxels.add((x, y, z))

            # Convert tube voxels to arrays
            if len(tube_voxels) > 0:
                tube_voxels_array = np.array(list(tube_voxels))

                # Extract intensities
                tube_intensities = ct_data[
                    tube_voxels_array[:, 0],
                    tube_voxels_array[:, 1],
                    tube_voxels_array[:, 2],
                ]

                # Convert to world coordinates
                tube_voxels_homogeneous = np.column_stack(
                    [tube_voxels_array, np.ones(len(tube_voxels_array))]
                )
                tube_points_world = (affine @ tube_voxels_homogeneous.T).T[:, :3]

                print(
                    f"      Trajectory-based point cloud: {len(tube_points_world)} points"
                )
                print(
                    f"      Intensity range: [{tube_intensities.min():.0f}, {tube_intensities.max():.0f}] HU"
                )

                # Count distribution
                below_2000 = (tube_intensities < 2000).sum()
                below_1500 = (tube_intensities < 1500).sum()
                below_1000 = (tube_intensities < 1000).sum()
                print(
                    f"      Points below 2000 HU: {below_2000} ({below_2000/len(tube_intensities)*100:.1f}%)"
                )
                print(
                    f"      Points below 1500 HU: {below_1500} ({below_1500/len(tube_intensities)*100:.1f}%)"
                )
                print(
                    f"      Points below 1000 HU: {below_1000} ({below_1000/len(tube_intensities)*100:.1f}%)"
                )

                # Use trajectory-based point cloud
                points_world = tube_points_world
                intensities = tube_intensities
            else:
                print(
                    "      Warning: No voxels found along trajectory, using original point cloud"
                )
    else:
        print("    No refinement threshold specified, using original point cloud")

    # Calculate lookahead in parameter space (matching CPU)
    lookahead_mm = 3.0
    arc_length = initial_trajectory.total_length_mm
    lookahead_param = lookahead_mm / arc_length if arc_length > 0 else 0.1

    # Convert z_resolution to parameter space (matching CPU)
    step_size = z_resolution / arc_length if arc_length > 0 else 0.001

    # Choose best available method
    if PYTORCH_INTERPOLATION_AVAILABLE and pytorch_available():
        print("  Using PyTorch-accelerated interpolation")
        oor_function = _orthogonal_optimal_resampling_gpu_pytorch
    else:
        print("  Using CPU interpolation (PyTorch not available)")
        oor_function = _orthogonal_optimal_resampling_gpu_phase1

    # Pass 2: Initial refinement (matching CPU's 2nd pass)
    print("  Pass 2: Initial OOR refinement...")
    skeleton_2, intensities_2, volumes_2, distances_2 = oor_function(
        initial_trajectory.polynomial,
        points_world,
        intensities,
        grid_size=grid_size,
        xy_resolution=xy_resolution,
        t_start=-lookahead_param,
        t_end=1.0,
        t_step=step_size,
        intensity_threshold=1500,
        use_gpu=True,
        electrode_idx=electrode_idx,
        debug_output_dir=debug_output_dir,
        pass_num=2,
        ct_volume=ct_data,
        affine=affine,
        use_subvolume=use_subvolume,
    )

    # TESTING: Use CPU for polynomial fitting to isolate interpolation impact
    # Fit refined polynomial
    # from .math_gpu import fit_polynomial_3d_gpu

    # Debug: Check skeleton_2 before fitting
    if len(skeleton_2) == 0:
        print("  Warning: No skeleton points returned from Pass 2")
        raise ValueError("No skeleton points found in Pass 2")

    # polynomial_1, t_values_1 = fit_polynomial_3d_gpu(
    #     skeleton_2,
    #     degree=final_degree,
    #     weights=intensities_2,
    #     use_gpu=True
    # )

    # Store the original requested degree for later use after tip detection
    original_final_degree = final_degree

    # CRITICAL: First refit polynomial to the OOR skeleton with degree 8 (matching MATLAB)
    # This ensures the polynomial follows the OOR-shifted trajectory
    print("    Refitting degree-8 polynomial to Pass 2 OOR skeleton...")
    polynomial_2nd_internal = fit_polynomial_3d(
        skeleton_2, degree=8, weights=intensities_2
    )

    # Now run another OOR pass on the refitted polynomial to get intensity profile for tip detection
    # This matches MATLAB line 67: [skeleton3rd, medIntensity, ...] = oor(refittedR3Poly2nd, ...)
    print("    Running OOR on refitted polynomial for intensity profile...")
    skeleton_for_tip, intensities_for_tip, volumes_for_tip, distances_for_tip = (
        oor_function(
            polynomial_2nd_internal,
            points_world,
            intensities,
            grid_size=grid_size,
            xy_resolution=xy_resolution,
            t_start=-lookahead_param,
            t_end=1.0,
            t_step=step_size,
            intensity_threshold=1500,
            use_gpu=True,
            electrode_idx=electrode_idx,
            debug_output_dir=None,  # No debug output for this intermediate step
            pass_num=None,
            ct_volume=ct_data,
            affine=affine,
            use_subvolume=use_subvolume,
        )
    )

    # Use the refitted polynomial for all subsequent operations
    polynomial_1 = polynomial_2nd_internal
    intensities_2 = intensities_for_tip
    distances_2 = distances_for_tip
    skeleton_2 = skeleton_for_tip  # Update skeleton to match

    # Save polynomial before tip detection if debug mode is enabled
    polynomial_before_tip_detection = (
        polynomial_1.copy() if debug_output_dir is not None else None
    )

    # CRITICAL TIP DETECTION STEP
    # Analyze the intensity profile from pass 2 to find the actual electrode tip
    print("  Detecting electrode tip from intensity profile...")

    # Dynamic threshold calculation based on intensity profile
    # Find the maximum intensity region (likely in the contact area)
    # Use a sliding window to find stable high-intensity region
    window_size = 20  # ~0.5mm window at typical resolution
    if len(intensities_2) > window_size:
        # Calculate moving average
        moving_avg = np.convolve(
            intensities_2, np.ones(window_size) / window_size, mode="valid"
        )
        max_avg_idx = np.argmax(moving_avg)
        max_avg_intensity = moving_avg[max_avg_idx]

        # Also get the overall intensity statistics
        intensity_median = np.median(intensities_2)
        intensity_75th = np.percentile(intensities_2, 75)
        intensity_90th = np.percentile(intensities_2, 90)

        print(
            f"  Intensity statistics: median={intensity_median:.0f}, 75th={intensity_75th:.0f}, 90th={intensity_90th:.0f}, max_avg={max_avg_intensity:.0f} HU"
        )

        # Dynamic threshold: use percentage of maximum stable intensity
        # This adapts to different grid sizes automatically
        threshold_percentage = 0.5  # 50% of maximum average intensity

        # Use median as tip threshold to detect transition from background to lead body
        # This is more reliable than trying to detect the high-intensity contact region
        tip_threshold = intensity_median

        print(f"  Using median as tip threshold: {tip_threshold:.0f} HU")

    else:
        # Fallback for very short trajectories
        tip_threshold = 1500
        print(
            f"  Using default tip threshold: {tip_threshold:.0f} HU (short trajectory)"
        )

    # Find the first point where intensity rises above threshold AND stays above
    # Starting from the beginning (negative lookahead region)
    tip_idx = None
    min_consecutive = 5  # Require at least 5 consecutive points above threshold

    for i in range(len(intensities_2) - min_consecutive):
        # Check if we have sustained high intensity
        if all(intensities_2[i : i + min_consecutive] >= tip_threshold):
            # Found a sustained high-intensity region - this is likely the real tip
            tip_idx = i
            break

    # If we couldn't find sustained high intensity, fall back to single point
    if tip_idx is None:
        for i in range(len(intensities_2)):
            if intensities_2[i] >= tip_threshold:
                tip_idx = i
                print(
                    "  Warning: No sustained high intensity found, using single-point detection"
                )
                break

    if tip_idx is None:
        print("  Warning: Could not detect electrode tip, trying lower threshold")
        # Try with a lower threshold (25th percentile)
        fallback_threshold = np.percentile(intensities_2, 25)
        for i in range(len(intensities_2)):
            if intensities_2[i] >= fallback_threshold:
                tip_idx = i
                break

        if tip_idx is None:
            print(
                "  Warning: Still could not detect electrode tip, using full trajectory"
            )
            tip_param = -lookahead_param
        else:
            tip_param = distances_2[tip_idx]
            print(
                f"  Electrode tip detected with fallback threshold at t={tip_param:.4f}"
            )
    else:
        # Get the parameter value at the detected tip
        # distances_2 contains the t values for each point
        tip_param = distances_2[tip_idx]
        print(
            f"  Electrode tip detected at parameter t={tip_param:.4f} (index {tip_idx}/{len(intensities_2)})"
        )
        print(f"  DEBUG: Intensity at detected tip: {intensities_2[tip_idx]:.1f} HU")
        print(
            f"  DEBUG: Intensities around tip: {intensities_2[max(0, tip_idx-2):tip_idx+3]}"
        )

        # Calculate arc length to tip (using refitted polynomial)
        if tip_param < 0:
            tip_distance_mm = -poly_arc_length_3d(polynomial_2nd_internal, tip_param, 0)
        else:
            tip_distance_mm = poly_arc_length_3d(polynomial_2nd_internal, 0, tip_param)
        print(f"  Tip position: {tip_distance_mm:.2f}mm from t=0")

    # Pass 3: Final OOR for intensity profile (matching CPU's 3rd pass)
    # Optimize by only processing first 20mm for contact detection
    print("  Pass 3: Final OOR for intensity profile (first 20mm only)...")

    # IMPORTANT: Use the refitted polynomial for Pass 3, not the original
    polynomial_for_pass3 = polynomial_2nd_internal

    # Find parameter value that corresponds to 20mm from tip
    target_distance_mm = 20.0
    t_low, t_high = tip_param, 1.0
    for _ in range(20):  # Binary search
        t_mid = (t_low + t_high) / 2
        dist = poly_arc_length_3d(polynomial_for_pass3, tip_param, t_mid)
        if dist < target_distance_mm:
            t_low = t_mid
        else:
            t_high = t_mid

    # Use the found parameter as end point (limited to first 20mm)
    t_end_20mm = min(t_mid, 1.0)

    skeleton_3, intensities_3, volumes_3, distances_3 = oor_function(
        polynomial_for_pass3,
        points_world,
        intensities,
        grid_size=grid_size,
        xy_resolution=xy_resolution,
        t_start=tip_param,  # Start from detected tip
        t_end=t_end_20mm,  # Only process up to 20mm from tip
        t_step=step_size,
        intensity_threshold=1500,
        use_gpu=True,
        electrode_idx=electrode_idx,
        debug_output_dir=debug_output_dir,
        pass_num=3,
        ct_volume=ct_data,
        affine=affine,
        use_subvolume=use_subvolume,
    )

    print(
        f"  Pass 3 range: {poly_arc_length_3d(polynomial_for_pass3, tip_param, t_end_20mm):.1f}mm"
    )

    # Debug: Check what intensities we got from Pass 3
    if len(intensities_3) > 0:
        print(
            f"  Pass 3 intensity range: [{intensities_3.min():.1f}, {intensities_3.max():.1f}] HU"
        )
        print(f"  Pass 3 first few intensities: {intensities_3[:5]}")

        # DEBUG: Check if Pass 3 starting point matches Pass 2 detected tip
        if len(skeleton_3) > 0 and len(skeleton_2) > 0:
            # Find the Pass 2 point at tip_param
            pass2_tip_idx = None
            for i, d in enumerate(distances_2):
                if abs(d - tip_param) < 1e-6:
                    pass2_tip_idx = i
                    break

            if pass2_tip_idx is not None:
                pass2_tip_point = skeleton_2[pass2_tip_idx]
                pass3_first_point = skeleton_3[0]
                distance_diff = np.linalg.norm(pass2_tip_point - pass3_first_point)
                print(
                    f"  DEBUG: Distance between Pass 2 tip and Pass 3 start: {distance_diff:.3f}mm"
                )
                print(f"  DEBUG: Pass 2 tip point: {pass2_tip_point}")
                print(f"  DEBUG: Pass 3 first point: {pass3_first_point}")

                # Also check distance from polynomial
                poly_at_tip = polyval3(polynomial_for_pass3, tip_param)
                skeleton_to_poly = np.linalg.norm(pass2_tip_point - poly_at_tip)
                print(
                    f"  DEBUG: Pass 2 skeleton deviation from polynomial at tip: {skeleton_to_poly:.3f}mm"
                )
                print(f"  DEBUG: Polynomial at tip_param: {poly_at_tip}")
            else:
                print("  DEBUG: Could not find exact tip_param in Pass 2 distances")

    # Re-zero the polynomial to start at the detected tip
    # IMPORTANT: We reparameterize the polynomial WITHOUT changing its shape
    # The new polynomial p_new(s) = p_old(tip_param + s * (1 - tip_param))
    # where s ∈ [0, 1] maps to t ∈ [tip_param, 1] in the original polynomial
    print(
        f"\n  Reparameterizing polynomial to start at detected tip (t={tip_param:.4f})..."
    )

    # For a polynomial p(t) = sum(c_i * t^i), the reparameterized polynomial
    # p_new(s) = p(tip_param + s * (1 - tip_param)) preserves the exact shape

    # This is mathematically exact and preserves the trajectory shape perfectly
    polynomial_rezeroed = reparameterize_polynomial_3d(
        polynomial_for_pass3, tip_param, 1.0
    )

    # Verify the reparameterization
    original_length = poly_arc_length_3d(polynomial_for_pass3, tip_param, 1.0)
    rezeroed_length = poly_arc_length_3d(polynomial_rezeroed, 0, 1)
    print("  Reparameterized polynomial created.")
    print(f"    Original length from tip: {original_length:.2f}mm")
    print(f"    Reparameterized length: {rezeroed_length:.2f}mm")

    # Check that the endpoints match
    original_tip = polyval3(polynomial_for_pass3, tip_param)
    rezeroed_tip = polyval3(polynomial_rezeroed, 0)
    tip_diff = np.linalg.norm(original_tip - rezeroed_tip)
    print(f"    Tip position difference: {tip_diff:.6f}mm (should be ~0)")

    # Filter skeleton and intensities to only include points from tip onwards
    # Find which skeleton points are at or after the tip
    tip_mask = distances_3 >= tip_param
    skeleton_final = skeleton_3[tip_mask]
    intensities_final = intensities_3[tip_mask]
    volumes_final = volumes_3[tip_mask] if volumes_3 is not None else None

    # Now perform auto-selection AFTER tip detection, using the tip-corrected skeleton
    if auto_select_degree:
        print(
            f"\n  Auto-selecting polynomial degree based on bottom {contact_region_mm}mm after tip detection..."
        )

        # Get skeleton points from tip onwards for evaluation
        tip_skeleton_mask = distances_for_tip >= tip_param
        tip_skeleton = skeleton_for_tip[tip_skeleton_mask]
        tip_intensities = intensities_for_tip[tip_skeleton_mask]

        # Calculate arc length for each skeleton point from tip
        skeleton_distances = np.zeros(len(tip_skeleton))
        for i in range(1, len(tip_skeleton)):
            skeleton_distances[i] = skeleton_distances[i - 1] + np.linalg.norm(
                tip_skeleton[i] - tip_skeleton[i - 1]
            )

        # Find points in the bottom region (from tip)
        bottom_mask = skeleton_distances <= contact_region_mm
        bottom_skeleton = tip_skeleton[bottom_mask]
        bottom_intensities = tip_intensities[bottom_mask]

        print(
            f"    Using {len(bottom_skeleton)} points in bottom {contact_region_mm}mm for evaluation"
        )

        # Test multiple polynomial degrees
        degrees_to_test = range(
            2, min(9, len(tip_skeleton))
        )  # Test degrees 2-8 (or max possible)
        best_aic = float("inf")
        best_degree = original_final_degree

        for test_degree in degrees_to_test:
            try:
                # Fit polynomial to skeleton from tip onwards
                test_poly = fit_polynomial_3d(
                    tip_skeleton, degree=test_degree, weights=tip_intensities
                )

                # Evaluate fit only on bottom region
                # Sample polynomial at same resolution as skeleton points
                n_eval_points = len(bottom_skeleton)
                t_eval = np.linspace(
                    0, bottom_mask.sum() / len(tip_skeleton), n_eval_points
                )
                poly_points = polyval3(test_poly, t_eval)

                # Calculate residuals
                residuals = bottom_skeleton - poly_points
                rss = np.sum(residuals**2)  # Residual sum of squares

                # Calculate AIC: 2k - 2ln(L), where k is number of parameters
                # For polynomial fit: k = (degree + 1) * 3 (3D polynomial)
                # Approximate log-likelihood assuming Gaussian errors
                n = len(bottom_skeleton)
                k = (test_degree + 1) * 3

                # Avoid log(0) by adding small epsilon
                if rss < 1e-10:
                    rss = 1e-10

                aic = n * np.log(rss / n) + 2 * k

                print(f"      Degree {test_degree}: RSS={rss:.4f}, AIC={aic:.2f}")

                if aic < best_aic:
                    best_aic = aic
                    best_degree = test_degree

            except Exception as e:
                print(f"      Degree {test_degree}: Failed - {str(e)}")
                continue

        print(f"    Selected degree {best_degree} (lowest AIC: {best_aic:.2f})")
        if best_degree != original_final_degree:
            print(
                f"    Note: Changed from requested degree {original_final_degree} to {best_degree}"
            )
        final_degree = best_degree

        # Refit polynomial with selected degree using tip-corrected skeleton
        print(
            f"    Refitting with degree-{final_degree} polynomial from detected tip..."
        )
        polynomial_final = fit_polynomial_3d(
            tip_skeleton, degree=final_degree, weights=tip_intensities
        )
    else:
        print(f"    Using specified degree: {original_final_degree}")
        # Use the reparameterized polynomial
        polynomial_final = polynomial_rezeroed

    # Debug: Check filtered intensities
    if len(intensities_final) > 0:
        print(
            f"  Filtered intensity range: [{intensities_final.min():.1f}, {intensities_final.max():.1f}] HU"
        )
        print(f"  Filtered first few intensities: {intensities_final[:5]}")

    # TESTING: Use CPU for arc length computation
    # Compute arc length
    # from .math_gpu import poly_arc_length_3d_gpu
    # total_length = poly_arc_length_3d_gpu(
    #     polynomial_final,
    #     t_start=0.0,
    #     t_end=1.0,
    #     use_gpu=True
    # )
    total_length = poly_arc_length_3d(polynomial_final, 0.0, 1.0)

    print(f"  Refinement complete. Total length: {total_length:.2f}mm")

    # Calculate distance scale for the re-zeroed polynomial
    # IMPORTANT: We must calculate actual arc lengths along the final polynomial
    # to ensure consistency with how 3D positions are calculated from distances

    n_points = len(skeleton_final)

    if n_points > 1:
        # We need to find the correspondence between skeleton points and polynomial parameters
        # The skeleton was sampled at regular parameter intervals during OOR
        # For the re-zeroed polynomial, we need to calculate arc lengths properly

        # First, determine the parameter range covered
        # We processed from t=0 to some t_end on the re-zeroed polynomial
        # The total arc length should match what we expect from the sampling
        expected_length = min((n_points - 1) * z_resolution, 20.0)

        # Find the parameter value that gives us the expected arc length
        t_end = 1.0
        actual_total_length = poly_arc_length_3d(polynomial_final, 0.0, t_end)

        # If the actual length is much larger than expected, find the right t_end
        if actual_total_length > expected_length * 1.1:  # Allow 10% tolerance
            # Binary search for the right t_end
            t_low, t_high = 0.0, 1.0
            for _ in range(20):
                t_mid = (t_low + t_high) / 2
                length_at_mid = poly_arc_length_3d(polynomial_final, 0.0, t_mid)
                if length_at_mid < expected_length:
                    t_low = t_mid
                else:
                    t_high = t_mid
            t_end = t_mid
            actual_total_length = poly_arc_length_3d(polynomial_final, 0.0, t_end)

        # Now calculate arc lengths for uniformly spaced parameters
        t_values = np.linspace(0.0, t_end, n_points)
        distance_scale_mm = np.zeros(n_points)

        for i in range(n_points):
            if i == 0:
                distance_scale_mm[i] = 0.0
            else:
                distance_scale_mm[i] = poly_arc_length_3d(
                    polynomial_final, 0.0, t_values[i]
                )

        # Debug output
        spacings = np.diff(distance_scale_mm[: min(10, len(distance_scale_mm) - 1)])
        actual_spacing = np.mean(spacings)
        print(
            f"  Debug: Arc length-based distance scale with {actual_spacing:.4f}mm mean spacing"
        )
        print(
            f"  Debug: Total arc length: {distance_scale_mm[-1]:.2f}mm with {n_points} points"
        )
        print(f"  Debug: Parameter range used: t=0 to t={t_end:.4f}")
    else:
        distance_scale_mm = np.array([0.0])

    # Debug: Check the distance scale starts from 0 or close to 0
    if len(distance_scale_mm) > 0:
        print(
            f"  Debug: Distance scale range: [{distance_scale_mm[0]:.2f}, {distance_scale_mm[-1]:.2f}] mm"
        )
        print(f"  Debug: First few distances: {distance_scale_mm[:5]}")

    # Apply MATLAB-style filtering to match refitElec.m behavior
    # This smoothing is critical for matching MATLAB's intensity profiles
    from scipy.signal import filtfilt

    z_resolution_mm = step_size * arc_length  # Convert from parameter space to mm
    filter_width = int((0.25 / z_resolution_mm) + 1)

    if len(intensities_final) > filter_width:
        # Create filter coefficients for moving average
        b = np.ones(filter_width) / filter_width
        # Apply zero-phase filtering as in MATLAB filtfilt
        intensities_final_smoothed = filtfilt(b, 1, intensities_final)
        print(
            f"      Applied intensity smoothing (filter width: {filter_width} samples)"
        )
    else:
        intensities_final_smoothed = intensities_final

    # Convert pass 2 distances to mm for visualization
    distances_2_mm = np.zeros_like(distances_2)
    for i, t in enumerate(distances_2):
        if t < tip_param:
            # Before tip (negative distance)
            distances_2_mm[i] = -poly_arc_length_3d(
                polynomial_2nd_internal, t, tip_param
            )
        else:
            # After tip (positive distance)
            distances_2_mm[i] = poly_arc_length_3d(
                polynomial_2nd_internal, tip_param, t
            )

    # Save full Pass 2 data for debug visualization (including lookahead)
    pass2_intensities_full = (
        intensities_2.copy() if debug_output_dir is not None else None
    )
    pass2_distances_mm_full = (
        distances_2_mm.copy() if debug_output_dir is not None else None
    )

    # For debug: calculate where original t=0 falls in the new distance coordinate system
    if debug_output_dir is not None:
        # Distance from detected tip to original t=0
        if tip_param < 0:
            # Tip was detected before original t=0
            original_t0_distance_mm = poly_arc_length_3d(
                polynomial_2nd_internal, tip_param, 0
            )
        elif tip_param > 0:
            # Tip was detected after original t=0
            original_t0_distance_mm = -poly_arc_length_3d(
                polynomial_2nd_internal, 0, tip_param
            )
        else:
            # Tip was detected exactly at t=0
            original_t0_distance_mm = 0.0
        print(
            f"  DEBUG: Original t=0 is at {original_t0_distance_mm:.2f}mm in tip-relative coordinates"
        )

    # Filter pass2 data to only include points from tip onwards for regular visualization
    pass2_tip_mask = distances_2 >= tip_param
    pass2_intensities_filtered = intensities_2[pass2_tip_mask]
    pass2_distances_mm_filtered = distances_2_mm[pass2_tip_mask]
    # Shift distances to start from 0 at the tip
    if len(pass2_distances_mm_filtered) > 0:
        pass2_distances_mm_filtered = (
            pass2_distances_mm_filtered - pass2_distances_mm_filtered[0]
        )

    # Calculate deviation between original polynomial and skeleton points BEFORE re-zeroing
    # This ensures we're comparing apples to apples
    skeleton_deviations_mm_all = np.zeros(len(skeleton_3))
    for i, skel_pt in enumerate(skeleton_3):
        # Get the parameter value for this skeleton point
        t = distances_3[i]

        # Get polynomial point at this parameter on the original polynomial
        poly_pt = polyval3(polynomial_1, t)

        # Calculate deviation
        skeleton_deviations_mm_all[i] = np.linalg.norm(skel_pt - poly_pt)

    # Now filter the deviations to only include points from tip onwards
    skeleton_deviations_mm = skeleton_deviations_mm_all[tip_mask]

    # Print statistics for debugging - show all points and from tip
    if len(skeleton_deviations_mm) > 0:
        print("  Skeleton-to-polynomial deviation (all points):")
        print(f"    Mean: {np.mean(skeleton_deviations_mm):.3f}mm")
        print(f"    Max: {np.max(skeleton_deviations_mm):.3f}mm")
        print(f"    Min: {np.min(skeleton_deviations_mm):.3f}mm")

        # Also show stats from tip for reference
        tip_onwards_mask = distance_scale_mm >= 0
        if np.any(tip_onwards_mask):
            deviations_from_tip = skeleton_deviations_mm[tip_onwards_mask]
            print(f"  From tip onwards ({np.sum(tip_onwards_mask)} points):")
            print(f"    Mean: {np.mean(deviations_from_tip):.3f}mm")
            print(f"    Max: {np.max(deviations_from_tip):.3f}mm")

    return RefinedTrajectoryGPU(
        polynomial=polynomial_final,
        skeleton=skeleton_final,
        intensity_profile=intensities_final_smoothed,
        distance_scale_mm=distance_scale_mm,
        orthogonal_volume=volumes_final,
        total_length_mm=total_length,
        # Pass 2 data for visualization (filtered to start from tip)
        pass2_intensities=pass2_intensities_filtered,
        pass2_distances_mm=pass2_distances_mm_filtered,
        pass2_tip_threshold=tip_threshold,
        pass2_tip_param=tip_param,
        # Deviation data - full arrays aligned with intensity profile
        skeleton_deviations_mm=skeleton_deviations_mm,
        skeleton_deviation_distances_mm=distance_scale_mm,  # Use same distance scale as intensity
        # Debug polynomial before tip detection (only when debug_output_dir is set)
        polynomial_before_tip_detection=polynomial_before_tip_detection,
        # Full Pass 2 data for debug visualization (including lookahead)
        pass2_intensities_full=pass2_intensities_full,
        pass2_distances_mm_full=pass2_distances_mm_full,
        # Original t=0 position in tip-relative coordinates
        original_t0_distance_mm=(
            original_t0_distance_mm if debug_output_dir is not None else None
        ),
    )
