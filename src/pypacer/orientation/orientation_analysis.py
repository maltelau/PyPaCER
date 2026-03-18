"""Orientation analysis for directional electrode markers."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .circular_sampling import (
    CircularSamplingResult,
    sample_circular_intensity,
)


class MarkerOrientationResult:
    """Result of marker orientation determination."""

    def __init__(
        self,
        marker_location_mm: float,
        peak_angle_deg: float,
        confidence: float,
        orientation_vector_local: np.ndarray,
        orientation_vector_world: np.ndarray,
        sampling_result: CircularSamplingResult,
        analysis_metadata: Optional[Dict] = None,
    ):
        """
        Args:
            marker_location_mm: Distance from tip where marker is located
            peak_angle_deg: Angle of peak intensity in local plane coordinates
            confidence: Confidence score (0-1)
            orientation_vector_local: 3D unit vector in local plane coordinates
            orientation_vector_world: 3D unit vector in world coordinates
            sampling_result: Circular sampling result
            analysis_metadata: Additional metadata (peak contrast, etc.)
        """
        self.marker_location_mm = marker_location_mm
        self.peak_angle_deg = peak_angle_deg
        self.confidence = confidence
        self.orientation_vector_local = orientation_vector_local
        self.orientation_vector_world = orientation_vector_world
        self.sampling_result = sampling_result
        self.analysis_metadata = analysis_metadata or {}

    def __repr__(self):
        return (
            f"MarkerOrientationResult("
            f"location={self.marker_location_mm:.2f}mm, "
            f"angle={self.peak_angle_deg:.1f}°, "
            f"confidence={self.confidence:.2f})"
        )


def determine_marker_orientation(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode_polynomial: np.ndarray,
    marker_location_mm: float,
    trajectory_direction: np.ndarray,
    radii_mm: List[float] = None,
    angle_increment_deg: float = 0.1,
    smoothing_window: int = 5,
    check_for_bias: bool = True,
    bias_opposite_peak_threshold: float = 0.7,
) -> MarkerOrientationResult:
    """
    Determine orientation of a directional marker.

    Samples a single circular intensity profile at the given position
    and finds the peak direction.

    Args:
        ct_data: 3D CT volume
        affine: Affine transformation matrix
        electrode_polynomial: Polynomial coefficients for trajectory
        marker_location_mm: Distance from tip to sample position
        trajectory_direction: Direction vector at marker location
        radii_mm: Radii for circular sampling (default: [1.25, 1.5, 1.75])
        angle_increment_deg: Angular increment for sampling (default: 0.1)
        smoothing_window: Window size for smoothing angular profile
        check_for_bias: Check for trajectory bias toward marker
        bias_opposite_peak_threshold: Ratio threshold for detecting opposite peaks

    Returns:
        MarkerOrientationResult with detected orientation
    """
    if radii_mm is None:
        radii_mm = [1.25, 1.5, 1.75]

    center_world = _evaluate_polynomial_at_distance(
        electrode_polynomial, marker_location_mm
    )

    sampling_result = sample_circular_intensity(
        ct_data=ct_data,
        affine=affine,
        center_world=center_world,
        trajectory_direction=trajectory_direction,
        radii_mm=radii_mm,
        angle_increment_deg=angle_increment_deg,
    )

    # Smooth the angular profile to reduce noise
    smoothed_intensity = _smooth_circular_profile(
        sampling_result.mean_intensity_by_angle, window_size=smoothing_window
    )

    # Find peak angle using intensity-weighted center
    peak_angle = _calculate_angular_peak_center(
        sampling_result.angles_deg, smoothed_intensity
    )

    # Check for bias (opposite peak at ~180 degrees)
    bias_detected = False
    if check_for_bias:
        bias_detected = _check_trajectory_bias(
            smoothed_intensity,
            sampling_result.angles_deg,
            peak_angle,
            bias_opposite_peak_threshold,
        )

    # Calculate orientation vector in world coordinates
    orientation_vector_world = angle_to_vector(
        peak_angle, trajectory_direction, affine
    )

    # Calculate confidence score
    confidence = _calculate_orientation_confidence(
        sampling_result.mean_intensity_by_angle,
        peak_angle,
        sampling_result.angles_deg,
        bias_detected,
    )

    # Metadata
    peak_intensity = np.max(smoothed_intensity)
    mean_intensity = np.mean(smoothed_intensity)
    contrast = (peak_intensity - mean_intensity) / mean_intensity if mean_intensity > 0 else 0

    metadata = {
        "peak_intensity": float(peak_intensity),
        "mean_intensity": float(mean_intensity),
        "contrast": float(contrast),
        "bias_detected": bias_detected,
    }

    return MarkerOrientationResult(
        marker_location_mm=marker_location_mm,
        peak_angle_deg=peak_angle,
        confidence=confidence,
        orientation_vector_local=np.array(
            [np.cos(np.deg2rad(peak_angle)), np.sin(np.deg2rad(peak_angle)), 0]
        ),
        orientation_vector_world=orientation_vector_world,
        sampling_result=sampling_result,
        analysis_metadata=metadata,
    )


def validate_marker_pair(
    marker1_result: MarkerOrientationResult,
    marker2_result: MarkerOrientationResult,
    min_separation_deg: float = 120.0,
    max_separation_deg: float = 150.0,
) -> Tuple[bool, float]:
    """
    Validate that two markers have expected angular separation.

    For Medtronic directional leads, markers are physically 120° apart.
    CT artifact spread causes the detected separation to appear wider,
    so valid range is 120°–150° (never narrower than the true separation).

    Args:
        marker1_result: First marker orientation result
        marker2_result: Second marker orientation result
        min_separation_deg: Minimum valid angular separation
        max_separation_deg: Maximum valid angular separation

    Returns:
        Tuple of (is_valid, angular_separation)
    """
    # Calculate angular separation between orientation vectors
    dot_product = np.dot(
        marker1_result.orientation_vector_world,
        marker2_result.orientation_vector_world,
    )

    # Clamp to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angular_separation_rad = np.arccos(dot_product)
    angular_separation_deg = np.rad2deg(angular_separation_rad)

    is_valid = min_separation_deg <= angular_separation_deg <= max_separation_deg

    return is_valid, angular_separation_deg


def fit_constrained_marker_directions(
    marker1_result: MarkerOrientationResult,
    marker2_result: MarkerOrientationResult,
    angular_constraint_deg: float = 120.0,
) -> Tuple[float, float]:
    """
    Fit marker directions constrained to fixed angular separation.

    Computes the angular midpoint between the two detected peak angles,
    then places the fitted directions at midpoint ± half the constraint.
    Each fitted direction is assigned to the closer detected peak.

    Args:
        marker1_result: First marker orientation result
        marker2_result: Second marker orientation result
        angular_constraint_deg: Fixed angular separation (default: 120 deg)

    Returns:
        Tuple of (fitted_angle_marker1_deg, fitted_angle_marker2_deg)
    """
    a1 = marker1_result.peak_angle_deg
    a2 = marker2_result.peak_angle_deg
    half = angular_constraint_deg / 2.0

    # Angular midpoint via shortest arc (shift a1 to 0, find direction to a2)
    diff = (a2 - a1) % 360.0
    if diff > 180:
        diff -= 360.0
    midpoint = (a1 + diff / 2.0) % 360.0

    # Two fitted directions
    f1 = (midpoint - half) % 360.0
    f2 = (midpoint + half) % 360.0

    # Assign each fitted direction to the closer detected peak
    dist_f1_to_a1 = abs(((f1 - a1 + 180) % 360) - 180)
    dist_f2_to_a1 = abs(((f2 - a1 + 180) % 360) - 180)

    if dist_f1_to_a1 <= dist_f2_to_a1:
        return float(f1), float(f2)
    else:
        return float(f2), float(f1)


def _evaluate_polynomial_at_distance(
    polynomial: np.ndarray, distance_mm: float
) -> np.ndarray:
    """
    Evaluate polynomial trajectory at given distance from tip.

    The polynomial coefficients are stored in descending order of powers.
    Parameter t ranges from 0 (tip) to 1 (entry).

    Args:
        polynomial: Polynomial coefficients [n_coeffs, 3] in descending power order
        distance_mm: Distance from tip

    Returns:
        3D point in world coordinates
    """
    from ..utils.math_helpers import polyval3

    # Sample trajectory at many points to build distance mapping
    t_values = np.linspace(0, 1, 1000)
    points = polyval3(polynomial, t_values)

    # Calculate cumulative distance along trajectory
    distances = np.zeros(len(t_values))
    for i in range(1, len(t_values)):
        distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])

    # Handle edge cases
    if distance_mm <= 0:
        return polyval3(polynomial, 0.0)
    if distance_mm >= distances[-1]:
        return polyval3(polynomial, 1.0)

    # Interpolate to find t value for target distance
    target_t = np.interp(distance_mm, distances, t_values)

    # Evaluate polynomial at target t
    target_point = polyval3(polynomial, target_t)

    return target_point


def _smooth_circular_profile(
    intensity_profile: np.ndarray, window_size: int = 5
) -> np.ndarray:
    """
    Smooth circular intensity profile with periodic boundary conditions.

    Args:
        intensity_profile: 1D array of intensities at different angles
        window_size: Size of smoothing window (must be odd)

    Returns:
        Smoothed intensity profile
    """
    if window_size < 3:
        return intensity_profile

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Pad with periodic boundary conditions
    pad_size = window_size // 2
    padded = np.concatenate(
        [intensity_profile[-pad_size:], intensity_profile, intensity_profile[:pad_size]]
    )

    # Apply Savitzky-Golay filter
    smoothed_padded = savgol_filter(padded, window_size, polyorder=2)

    # Remove padding
    smoothed = smoothed_padded[pad_size:-pad_size]

    return smoothed


def _calculate_angular_peak_center(angles_deg: np.ndarray, intensities: np.ndarray) -> float:
    """
    Calculate intensity-weighted angular center of the peak region.

    Finds the contiguous above-threshold region around the intensity peak
    and computes the weighted center. Handles wrapping at the 360/0 degree
    boundary by shifting angles so the peak is centered at 180 degrees.

    The threshold is set at 50% between the global minimum and peak intensity.
    Weights are (intensity - threshold) so points well above threshold
    contribute more than points barely above it. The region is limited by
    threshold only (no arbitrary angular width cap); contiguous expansion
    from the peak prevents inclusion of secondary peaks.

    Args:
        angles_deg: Array of angles in degrees
        intensities: Intensity values at each angle

    Returns:
        Weighted center angle in degrees [0, 360)
    """
    peak_idx = np.argmax(intensities)
    peak_angle = angles_deg[peak_idx]
    peak_intensity = intensities[peak_idx]
    min_intensity = np.min(intensities)

    # Threshold at 50% between global min and peak (half-maximum)
    threshold = (peak_intensity + min_intensity) / 2.0

    # Shift angles so peak is at 180 deg to handle 0/360 wrapping
    shift = 180.0 - peak_angle
    shifted_angles = (angles_deg + shift) % 360.0

    # Sort by shifted angle for contiguous region detection
    sort_idx = np.argsort(shifted_angles)
    sorted_shifted = shifted_angles[sort_idx]
    sorted_intensities = intensities[sort_idx]

    # Find peak in sorted array (should be near 180 deg)
    peak_sorted_idx = np.argmin(np.abs(sorted_shifted - 180.0))

    # Expand outward from peak to find contiguous above-threshold region
    # No angular width cap - the threshold is the sole limiter;
    # contiguous expansion prevents jumping to secondary peaks
    above = sorted_intensities >= threshold
    left = peak_sorted_idx
    right = peak_sorted_idx

    while left > 0 and above[left - 1]:
        left -= 1
    while right < len(above) - 1 and above[right + 1]:
        right += 1

    region_shifted = sorted_shifted[left:right + 1]
    region_intensities = sorted_intensities[left:right + 1]

    if len(region_shifted) < 2:
        return float(peak_angle)

    # Weights: intensity above threshold
    weights = np.maximum(region_intensities - threshold, 0.0)

    if weights.sum() <= 0:
        return float(peak_angle)

    # Weighted mean in shifted space (peak near 180 deg, no wrapping issue)
    center_shifted = np.average(region_shifted, weights=weights)

    # Shift back to original coordinates
    center_deg = (center_shifted - shift) % 360.0

    return float(center_deg)


def _angular_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest angular difference between two angles in degrees.

    Returns value in range [0, 180].

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        Absolute angular difference in degrees
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def _check_trajectory_bias(
    intensity_profile: np.ndarray,
    angles_deg: np.ndarray,
    peak_angle_deg: float,
    threshold: float = 0.7,
) -> bool:
    """
    Check if trajectory may be biased toward marker.

    Detects if there's a significant peak at ~180° from main peak,
    which could indicate trajectory center is offset from true center.

    Args:
        intensity_profile: Smoothed intensity profile
        angles_deg: Angle values
        peak_angle_deg: Main peak angle
        threshold: Ratio threshold for opposite peak

    Returns:
        True if bias detected
    """
    # Find opposite angle (±180°)
    opposite_angle = (peak_angle_deg + 180) % 360

    # Find intensity at opposite angle (interpolate if needed)
    opposite_intensity = np.interp(opposite_angle, angles_deg, intensity_profile)

    # Get peak intensity
    peak_intensity = np.max(intensity_profile)

    # Check if opposite peak is significant
    ratio = opposite_intensity / peak_intensity if peak_intensity > 0 else 0

    return ratio > threshold


def angle_to_vector(
    angle_deg: float, trajectory_direction: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    """
    Convert angle in sampling plane to 3D orientation vector.

    Args:
        angle_deg: Angle in degrees (in plane perpendicular to trajectory)
        trajectory_direction: Trajectory direction vector
        affine: Affine transformation matrix

    Returns:
        3D unit vector in world coordinates
    """
    # Normalize trajectory
    trajectory_unit = trajectory_direction / np.linalg.norm(trajectory_direction)

    # Create orthonormal basis
    if abs(trajectory_unit[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u = np.cross(trajectory_unit, arbitrary)
    u = u / np.linalg.norm(u)

    v = np.cross(trajectory_unit, u)
    v = v / np.linalg.norm(v)

    # Convert angle to vector
    angle_rad = np.deg2rad(angle_deg)
    vector = np.cos(angle_rad) * u + np.sin(angle_rad) * v

    return vector / np.linalg.norm(vector)


def _calculate_orientation_confidence(
    intensity_profile: np.ndarray,
    peak_angle_deg: float,
    angles_deg: np.ndarray,
    bias_detected: bool,
) -> float:
    """
    Calculate confidence score for orientation detection.

    Based on:
    - Peak prominence
    - Peak sharpness
    - Contrast ratio
    - Whether bias was detected

    Args:
        intensity_profile: Intensity values at different angles
        peak_angle_deg: Peak angle
        angles_deg: Angle values
        bias_detected: Whether trajectory bias was detected

    Returns:
        Confidence score (0-1)
    """
    peak_intensity = np.max(intensity_profile)
    mean_intensity = np.mean(intensity_profile)
    min_intensity = np.min(intensity_profile)

    # Contrast score
    if mean_intensity > 0:
        contrast = (peak_intensity - mean_intensity) / mean_intensity
        contrast_score = min(1.0, contrast / 0.5)  # Normalize by expected contrast
    else:
        contrast_score = 0.0

    # Prominence score (peak vs minimum)
    if peak_intensity > 0:
        prominence = (peak_intensity - min_intensity) / peak_intensity
        prominence_score = prominence
    else:
        prominence_score = 0.0

    # Sharpness score (how narrow is the peak)
    # Find width at half maximum
    half_max = (peak_intensity + min_intensity) / 2
    above_half_max = intensity_profile > half_max
    peak_width_samples = np.sum(above_half_max)
    peak_width_deg = peak_width_samples * (angles_deg[1] - angles_deg[0]) if len(angles_deg) > 1 else 360
    sharpness_score = max(0.0, 1.0 - peak_width_deg / 180.0)  # Narrower is better

    # Combine scores
    confidence = (contrast_score + prominence_score + sharpness_score) / 3.0

    # Penalize if bias detected
    if bias_detected:
        confidence *= 0.7

    return confidence
