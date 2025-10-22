"""Trajectory refinement algorithms including Orthogonal Optimal Resampling (OOR)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import filtfilt

from ..utils.math_helpers import (
    fit_polynomial_3d,
    get_orthogonal_vectors,
    poly_arc_length_3d,
    polyval3,
)
from .trajectory_fit import InitialTrajectory


@dataclass
class RefinedTrajectory:
    """Container for refined trajectory results."""

    polynomial: np.ndarray
    skeleton: np.ndarray
    intensity_profile: np.ndarray
    distance_scale_mm: np.ndarray
    orthogonal_volume: Optional[np.ndarray] = None
    electrode_type: Optional[str] = None
    total_length_mm: Optional[float] = None
    # Pass 2 data for visualization (if from GPU version)
    pass2_intensities: Optional[np.ndarray] = None
    pass2_distances_mm: Optional[np.ndarray] = None
    pass2_tip_threshold: Optional[float] = None
    pass2_tip_param: Optional[float] = None
    # Skeleton deviation data
    skeleton_deviations_mm: Optional[np.ndarray] = None
    skeleton_deviation_distances_mm: Optional[np.ndarray] = None
    # Full Pass 2 data for debug visualization
    pass2_intensities_full: Optional[np.ndarray] = None
    pass2_distances_mm_full: Optional[np.ndarray] = None
    # Debug polynomial before tip detection
    polynomial_before_tip_detection: Optional[np.ndarray] = None
    # Original t=0 position in tip-relative coordinates
    original_t0_distance_mm: Optional[float] = None


def refine_electrode_trajectory(
    initial_trajectory: InitialTrajectory,
    points_world: np.ndarray,
    intensities: np.ndarray,
    ct_data: np.ndarray,
    affine: np.ndarray,
    final_degree: int = 3,
    xy_resolution: float = 0.1,  # mm
    z_resolution: float = 0.025,  # mm
    grid_size: float = 1.5,  # mm radius for orthogonal sampling
    intensity_threshold: float = 1500,  # HU threshold for skeleton refinement
    electrode_idx: Optional[int] = None,
    debug_output_dir: Optional[str] = None,
    refinement_threshold: Optional[
        float
    ] = None,  # Lower threshold for refinement point cloud
    refinement_radius_mm: float = 3.5,  # Radius around trajectory for refinement
    debug_z_axis_scale: float = 8.0,  # Scale factor for z-axis in debug NIfTI files
    auto_select_degree: bool = False,  # Automatically select best polynomial degree
    contact_region_mm: float = 20.0,  # Length from tip to evaluate for auto degree selection
) -> RefinedTrajectory:
    """
    Refine electrode trajectory using Orthogonal Optimal Resampling (OOR).

    Based on MATLAB refitElec.m and oor.m

    Args:
        initial_trajectory: Initial polynomial fit
        points_world: Point cloud in world coordinates
        intensities: Intensity values at each point
        ct_data: Full CT volume for interpolation
        affine: Affine transformation matrix
        final_degree: Final polynomial degree (default 3)
        xy_resolution: Resolution for orthogonal grid sampling
        z_resolution: Resolution along trajectory
        grid_size: Size of orthogonal sampling grid
        intensity_threshold: Threshold for valid intensity values
        electrode_idx: Optional electrode index for debug output
        debug_output_dir: Optional directory for debug outputs
        refinement_threshold: Optional lower threshold for refinement point cloud (e.g., 800 HU)

    Returns:
        RefinedTrajectory with improved polynomial and intensity profile
    """
    print("  Step 3: Refining trajectory with OOR...")
    print(f"    Initial length: {initial_trajectory.total_length_mm:.1f}mm")

    # Skip point cloud extraction when using volume interpolator
    # The RegularGridInterpolator can directly sample from the CT volume
    # without needing to extract a point cloud first
    if refinement_threshold is not None:
        print(
            f"    Using refinement threshold: {refinement_threshold} HU (with direct volume interpolation)"
        )
    else:
        print("    No refinement threshold specified, using original point cloud")

    # Always use volume interpolation when CT data is available
    print("    Creating volume interpolator...")

    # Create coordinate arrays for the volume
    x_coords = np.arange(ct_data.shape[0])
    y_coords = np.arange(ct_data.shape[1])
    z_coords = np.arange(ct_data.shape[2])

    # Create the interpolator
    volume_interpolator = RegularGridInterpolator(
        (x_coords, y_coords, z_coords),
        ct_data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Create a wrapper to convert world to voxel coordinates
    affine_inv = np.linalg.inv(affine)

    def interpolator(world_points):
        # Convert world coordinates to voxel coordinates
        world_homo = np.column_stack([world_points, np.ones(len(world_points))])
        voxel_coords = (affine_inv @ world_homo.T).T[:, :3]
        # Query the volume interpolator
        return volume_interpolator(voxel_coords)

    # Create orthogonal sampling grid
    x_grid, y_grid = np.meshgrid(
        np.arange(-grid_size, grid_size + xy_resolution, xy_resolution),
        np.arange(-grid_size, grid_size + xy_resolution, xy_resolution),
    )

    # Convert z_resolution to parameter space
    total_length = initial_trajectory.total_length_mm
    step_size = z_resolution / total_length  # Approximate parameter step

    # Run OOR on initial trajectory (2nd pass)
    print("    Running 2nd pass OOR...")
    skeleton_2nd, intensity_profile, ortho_volume, distance_scale = (
        _orthogonal_optimal_resampling(
            initial_trajectory.polynomial,
            interpolator,
            x_grid,
            y_grid,
            step_size,
            intensity_threshold,
            electrode_idx=electrode_idx,
            debug_output_dir=debug_output_dir,
            pass_num=2,
            z_axis_scale=debug_z_axis_scale,
        )
    )
    print(f"    Refined to {len(skeleton_2nd)} points")

    # Save Pass 2 data for visualization (will be updated after refitted polynomial OOR)

    # Fit refined polynomial to new skeleton
    if len(skeleton_2nd) < final_degree + 1:
        print(
            f"Warning: Only {len(skeleton_2nd)} points for degree {final_degree} polynomial. "
            f"Using degree {len(skeleton_2nd) - 1}"
        )
        final_degree = len(skeleton_2nd) - 1

    # Store the original requested degree for later use after tip detection
    original_final_degree = final_degree

    # INTERMEDIATE STEP (matching GPU/MATLAB): Refit with degree 8 for better tip detection
    print(
        "    Refitting with degree 8 for tip detection (matching MATLAB refittedR3Poly2nd)..."
    )
    polynomial_2nd_internal = fit_polynomial_3d(skeleton_2nd, degree=8)

    # Run OOR again on the refitted polynomial to get better intensity profile for tip detection
    print("    Running OOR on refitted polynomial for tip detection...")
    skeleton_for_tip, intensity_for_tip, _, distance_for_tip = (
        _orthogonal_optimal_resampling(
            polynomial_2nd_internal,
            interpolator,
            x_grid,
            y_grid,
            step_size,
            intensity_threshold,
            electrode_idx=electrode_idx,
            debug_output_dir=None,  # Don't save debug for intermediate pass
            pass_num=None,
            return_t_values=True,  # Return parameter values for tip detection
        )
    )

    # Save Pass 2 data from refitted polynomial (matching GPU behavior)
    pass2_intensities = intensity_for_tip.copy()
    # Note: distance_for_tip contains t_values (parameters), not distances
    # We'll convert to mm after tip detection

    # Save full Pass 2 data for debug visualization (including lookahead)
    pass2_intensities_full = (
        intensity_for_tip.copy() if debug_output_dir is not None else None
    )
    # We'll convert these to mm distances after tip detection

    # Save polynomial before tip detection for debugging
    polynomial_before_tip_detection = (
        polynomial_2nd_internal.copy() if debug_output_dir is not None else None
    )

    # CRITICAL TIP DETECTION STEP - matching GPU implementation
    # Analyze the intensity profile from refitted polynomial to find the actual electrode tip
    print("  Detecting electrode tip from intensity profile...")

    # Find the actual tip based on intensity threshold
    # Use the intensity profile from the refitted polynomial (matching GPU)
    intensity_profile = intensity_for_tip
    distance_scale = distance_for_tip

    # Threshold calculation matching GPU implementation
    intensity_median = np.median(intensity_profile)
    intensity_75th = np.percentile(intensity_profile, 75)
    intensity_90th = np.percentile(intensity_profile, 90)

    print(
        f"  Intensity statistics: median={intensity_median:.0f}, 75th={intensity_75th:.0f}, 90th={intensity_90th:.0f} HU"
    )

    # Use median as tip threshold (matching GPU implementation)
    # This is more reliable than trying to detect the high-intensity contact region
    tip_threshold = intensity_median
    print(f"  Using median tip threshold: {tip_threshold:.0f} HU")

    # Save tip threshold for visualization
    pass2_tip_threshold = tip_threshold

    # Find sustained intensity above threshold (matching GPU implementation)
    # Require 5 consecutive points above threshold for more robust detection
    min_consecutive = 5
    tip_idx = None

    for i in range(len(intensity_profile) - min_consecutive):
        if all(intensity_profile[i : i + min_consecutive] >= tip_threshold):
            # Found sustained intensity above threshold
            tip_idx = i
            break

    if tip_idx is None:
        # Fallback to single point detection if sustained intensity not found
        print("  Warning: No sustained intensity found, using single point detection")
        for i in range(len(intensity_profile)):
            if intensity_profile[i] >= tip_threshold:
                tip_idx = i
                break

    if tip_idx is None:
        print("  Warning: Could not detect electrode tip, using start of trajectory")
        tip_param = 0.0
        perform_rezero = False
    else:
        # Get the parameter value at the detected tip
        # distance_scale now contains t_values (parameter values) from return_t_values=True
        tip_param = distance_scale[tip_idx]
        # Calculate the actual distance for logging
        if tip_param < 0:
            tip_distance_mm = -poly_arc_length_3d(polynomial_2nd_internal, tip_param, 0)
        else:
            tip_distance_mm = poly_arc_length_3d(polynomial_2nd_internal, 0, tip_param)
        perform_rezero = True
        print(
            f"  Electrode tip detected at parameter t={tip_param:.4f} (distance {tip_distance_mm:.2f}mm)"
        )

    # Save tip parameter for visualization
    pass2_tip_param = tip_param if "tip_param" in locals() else 0.0

    # Import reparameterize function
    from ..utils.math_helpers import reparameterize_polynomial_3d

    # Reparameterize the polynomial to start at detected tip (matching GPU behavior)
    print(
        f"\n  Reparameterizing polynomial to start at detected tip (t={tip_param:.4f})..."
    )

    # This preserves the exact shape while remapping parameters
    polynomial_rezeroed = reparameterize_polynomial_3d(
        polynomial_2nd_internal, tip_param, 1.0
    )

    # Verify the reparameterization
    original_length = poly_arc_length_3d(polynomial_2nd_internal, tip_param, 1.0)
    rezeroed_length = poly_arc_length_3d(polynomial_rezeroed, 0, 1)
    print("  Reparameterized polynomial created.")
    print(f"    Original length from tip: {original_length:.2f}mm")
    print(f"    Reparameterized length: {rezeroed_length:.2f}mm")

    # Now perform auto-selection AFTER tip detection, using the tip-corrected skeleton
    if auto_select_degree:
        print(
            f"\n  Auto-selecting polynomial degree based on bottom {contact_region_mm}mm after tip detection..."
        )

        # Get skeleton points from tip onwards for evaluation
        tip_skeleton_mask = distance_for_tip >= tip_param
        tip_skeleton = skeleton_for_tip[tip_skeleton_mask]
        tip_intensities = intensity_for_tip[tip_skeleton_mask]

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
        final_polynomial = fit_polynomial_3d(
            tip_skeleton, degree=final_degree, weights=tip_intensities
        )
    else:
        print(f"    Using specified degree: {original_final_degree}")
        # Use the reparameterized polynomial
        final_polynomial = polynomial_rezeroed

    t_start_final = 0.0  # Start from the new zero (tip)

    # Convert pass 2 distances to mm for visualization (matching GPU)
    distances_2_mm = np.zeros_like(distance_for_tip)
    for i, t in enumerate(distance_for_tip):
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

    # Save full Pass 2 data with mm distances
    pass2_distances_mm_full = (
        distances_2_mm.copy() if debug_output_dir is not None else None
    )

    # Filter pass2 data to only include points from tip onwards for regular visualization
    pass2_tip_mask = distance_for_tip >= tip_param
    pass2_intensities_filtered = intensity_for_tip[pass2_tip_mask]
    pass2_distances_mm_filtered = distances_2_mm[pass2_tip_mask]
    # Shift distances to start from 0 at the tip
    if len(pass2_distances_mm_filtered) > 0:
        pass2_distances_mm_filtered = (
            pass2_distances_mm_filtered - pass2_distances_mm_filtered[0]
        )

    # Update pass2 data for return
    pass2_intensities = pass2_intensities_filtered
    pass2_distances_mm = pass2_distances_mm_filtered

    # For debug: calculate where original t=0 falls in the new distance coordinate system
    original_t0_distance_mm = None
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

    # Run OOR again for final intensity profile (3rd pass)
    # Limit to first 20mm for contact detection (matching GPU behavior)
    print("    Running 3rd pass OOR for intensity profile (first 20mm only)...")

    # Find parameter value that corresponds to 20mm from tip
    target_distance_mm = 20.0
    t_low, t_high = 0.0, 1.0  # Since polynomial is reparameterized to start at tip
    for _ in range(20):  # Binary search
        t_mid = (t_low + t_high) / 2
        dist = poly_arc_length_3d(final_polynomial, 0.0, t_mid)
        if dist < target_distance_mm:
            t_low = t_mid
        else:
            t_high = t_mid

    # Use the found parameter as end point (limited to first 20mm)
    t_end_20mm = min(t_mid, 1.0)

    skeleton_3rd, intensity_profile, ortho_volume, distance_scale = (
        _orthogonal_optimal_resampling(
            final_polynomial,
            interpolator,
            x_grid,
            y_grid,
            step_size,
            intensity_threshold,
            electrode_idx=electrode_idx,
            debug_output_dir=debug_output_dir,
            pass_num=3,
            z_axis_scale=debug_z_axis_scale,
            t_start=0.0,  # Start from tip (reparameterized to 0)
            t_end=t_end_20mm,  # Only process up to 20mm
        )
    )
    print(f"    Final skeleton: {len(skeleton_3rd)} points")

    # Calculate final length
    final_length = poly_arc_length_3d(final_polynomial, 0, 1)

    print(
        f"Refinement complete: {initial_trajectory.total_length_mm:.1f}mm -> {final_length:.1f}mm"
    )

    # Calculate deviation between final polynomial and final skeleton points
    # This measures how well the polynomial fits the refined skeleton
    skeleton_deviations_mm = np.zeros(len(skeleton_3rd))
    for i, skel_pt in enumerate(skeleton_3rd):
        # Sample polynomial densely to find nearest point
        t_samples = np.linspace(0, 1, 1000)
        poly_samples = np.array([polyval3(final_polynomial, t) for t in t_samples])

        # Find minimum distance from skeleton point to polynomial
        distances_to_poly = np.linalg.norm(poly_samples - skel_pt, axis=1)
        skeleton_deviations_mm[i] = np.min(distances_to_poly)

    # Print statistics for debugging
    if len(skeleton_deviations_mm) > 0:
        print("  Skeleton-to-polynomial deviation (all points):")
        print(f"    Mean: {np.mean(skeleton_deviations_mm):.3f}mm")
        print(f"    Max: {np.max(skeleton_deviations_mm):.3f}mm")
        print(f"    Min: {np.min(skeleton_deviations_mm):.3f}mm")

    return RefinedTrajectory(
        polynomial=final_polynomial,
        skeleton=skeleton_3rd,
        intensity_profile=intensity_profile,
        distance_scale_mm=distance_scale,
        orthogonal_volume=ortho_volume,
        total_length_mm=final_length,
        # Pass 2 data for visualization (matching GPU behavior)
        pass2_intensities=pass2_intensities,
        pass2_distances_mm=pass2_distances_mm,
        pass2_tip_threshold=pass2_tip_threshold,
        pass2_tip_param=pass2_tip_param,
        # Full Pass 2 data for debug visualization
        pass2_intensities_full=pass2_intensities_full,
        pass2_distances_mm_full=pass2_distances_mm_full,
        # Deviation data - full arrays aligned with intensity profile
        skeleton_deviations_mm=skeleton_deviations_mm,
        skeleton_deviation_distances_mm=distance_scale,  # Use same distance scale as intensity
        # Debug polynomial before tip detection
        polynomial_before_tip_detection=polynomial_before_tip_detection,
        # Original t=0 position in tip-relative coordinates
        original_t0_distance_mm=original_t0_distance_mm,
    )


def _orthogonal_optimal_resampling(
    polynomial: np.ndarray,
    interpolator,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    step_size: float,
    intensity_threshold: float = 1500,
    lookahead_mm: float = 3.0,
    electrode_idx: Optional[int] = None,
    debug_output_dir: Optional[str] = None,
    pass_num: Optional[int] = None,
    z_axis_scale: float = 4.0,  # Scale factor for z-axis in debug NIfTI
    t_start: Optional[float] = None,  # Override start parameter
    t_end: Optional[float] = None,  # Override end parameter
    return_t_values: bool = False,  # Return t_values instead of distances (for tip detection)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Orthogonal Optimal Resampling along polynomial trajectory.

    Based on MATLAB oor.m

    Args:
        polynomial: Polynomial coefficients
        interpolator: Interpolation function for intensity values
        x_grid: X coordinates of orthogonal sampling grid
        y_grid: Y coordinates of orthogonal sampling grid
        step_size: Step size in parameter space
        intensity_threshold: Minimum intensity for valid points
        lookahead_mm: Extra distance to sample beyond endpoints

    Returns:
        Tuple of (improved_skeleton, median_intensity, orthogonal_volume, distance_scale)
    """
    # Calculate lookahead in parameter space
    arc_length = poly_arc_length_3d(polynomial, 0, 1)
    lookahead_param = lookahead_mm / arc_length

    # Use provided start/end or default values
    if t_start is None:
        t_start = -lookahead_param
    if t_end is None:
        t_end = 1.0

    # Sample points along trajectory
    t_values = np.arange(t_start, t_end, step_size)
    print(f"      Sampling {len(t_values)} points along trajectory...")

    improved_skeleton = []
    median_intensities = []
    orthogonal_volumes = []
    valid_t_values = []

    # For debug volume saving
    all_intensity_grids = []
    all_sample_points = []

    # Get polynomial derivatives for tangent calculation
    # Each row is multiplied by its power (degree, degree-1, ..., 1)
    powers = np.arange(len(polynomial) - 1, 0, -1)
    poly_deriv = polynomial[:-1] * powers[:, np.newaxis]

    for i, t in enumerate(t_values):
        if i % 100 == 0:
            print(
                f"      Processing point {i}/{len(t_values)} ({i/len(t_values)*100:.1f}%)"
            )

        # Current point on trajectory
        current_point = polyval3(polynomial, t)

        # Tangent direction
        tangent = polyval3(poly_deriv, t)
        tangent = tangent / np.linalg.norm(tangent)

        # Get orthogonal vectors
        ortho1, ortho2 = get_orthogonal_vectors(tangent)

        # Create orthogonal sampling points
        grid_points = (
            x_grid.ravel()[:, np.newaxis] * ortho1
            + y_grid.ravel()[:, np.newaxis] * ortho2
        )
        sample_points = current_point + grid_points

        # Interpolate intensities at sample points
        intensities = interpolator(sample_points)

        # Handle NaN and threshold
        valid_mask = ~np.isnan(intensities) & (intensities >= intensity_threshold)

        if not np.any(valid_mask):
            continue

        valid_intensities = intensities[valid_mask]
        valid_points = sample_points[valid_mask]

        # Calculate intensity-weighted centroid
        skeleton_point = np.average(valid_points, axis=0, weights=valid_intensities)

        improved_skeleton.append(skeleton_point)
        median_intensities.append(np.median(intensities[~np.isnan(intensities)]))

        # Reshape intensities to grid
        intensity_map = intensities.reshape(x_grid.shape)
        orthogonal_volumes.append(intensity_map)
        valid_t_values.append(t)

        # Store for debug output
        all_intensity_grids.append(intensity_map)
        all_sample_points.append(sample_points)

    improved_skeleton = np.array(improved_skeleton)
    median_intensities = np.array(median_intensities)
    orthogonal_volume = np.stack(orthogonal_volumes, axis=2)

    # Apply MATLAB-style filtering to match refitElec.m behavior
    # filterWidth = (0.25 / Z_RESOLUTION) + 1; where Z_RESOLUTION = 0.025
    # This is approximately 0.25mm smoothing window
    z_resolution_mm = step_size * arc_length  # Convert from parameter space to mm
    filter_width = int((0.25 / z_resolution_mm) + 1)

    if len(median_intensities) > filter_width:
        # Create filter coefficients for moving average
        b = np.ones(filter_width) / filter_width
        # Apply zero-phase filtering as in MATLAB filtfilt
        median_intensities = filtfilt(b, 1, median_intensities)
        print(
            f"      Applied intensity smoothing (filter width: {filter_width} samples)"
        )

    # DEBUG: Save interpolated volume as NIfTI
    print(
        f"      DEBUG CHECK (CPU): debug_output_dir={debug_output_dir}, electrode_idx={electrode_idx}"
    )

    # Calculate distance scale in mm
    valid_t_values = np.array(valid_t_values)
    distance_scale = np.zeros_like(valid_t_values)

    # Use the actual start point as reference (usually 0, but might be different for Pass 3)
    ref_t = max(0, t_start) if t_start is not None else 0

    for i, t in enumerate(valid_t_values):
        if t < ref_t:
            # Negative distance before reference point
            distance_scale[i] = -poly_arc_length_3d(polynomial, t, ref_t)
        else:
            distance_scale[i] = poly_arc_length_3d(polynomial, ref_t, t)

    # Return t_values or distance_scale based on caller's needs
    if return_t_values:
        return improved_skeleton, median_intensities, orthogonal_volume, valid_t_values
    else:
        return improved_skeleton, median_intensities, orthogonal_volume, distance_scale
