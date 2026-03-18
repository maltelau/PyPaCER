"""Marker detection for directional electrodes."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


class MarkerDetectionResult:
    """Result of marker detection analysis."""

    def __init__(
        self,
        has_markers: bool,
        confidence: float,
        marker_peak_locations: Optional[List[float]] = None,
        marker_region_center: Optional[float] = None,
        marker_region_stats: Optional[Dict] = None,
        detection_method: Optional[str] = None,
    ):
        self.has_markers = has_markers
        self.confidence = confidence
        self.marker_peak_locations = marker_peak_locations or []
        self.marker_region_center = marker_region_center
        self.marker_region_stats = marker_region_stats or {}
        self.detection_method = detection_method

    def __repr__(self):
        return (
            f"MarkerDetectionResult(has_markers={self.has_markers}, "
            f"confidence={self.confidence:.2f}, "
            f"peaks={len(self.marker_peak_locations)}, "
            f"method={self.detection_method})"
        )


def detect_directional_markers(
    distance_scale: np.ndarray,
    intensity_profile: np.ndarray,
    skeleton_deviations: Optional[np.ndarray],
    contact_positions: List[float],
    marker_offset_mm: float = 2.5,
    max_distance_mm: float = 20.0,
    deviation_threshold: float = 0.08,
    min_peak_distance_mm: float = 2.0,
    expected_num_peaks: int = 2,
) -> MarkerDetectionResult:
    """
    Detect if electrode has directional markers (radiopaque markers).

    Searches the region from last contact + offset to the end of the
    distance scale (capped at max_distance_mm). Markers are identified
    by finding peaks that stand out within the region (deviation or
    intensity based).

    Args:
        distance_scale: Distance from electrode tip in mm
        intensity_profile: CT intensity values along trajectory (HU)
        skeleton_deviations: Polynomial fit deviations in mm (preferred for peak detection)
        contact_positions: Contact positions in mm from tip
        marker_offset_mm: Distance above last contact (furthest from tip) where markers start
        max_distance_mm: Maximum distance from tip to include in marker region
        deviation_threshold: Minimum deviation value to consider as peak
        min_peak_distance_mm: Minimum distance between peaks
        expected_num_peaks: Expected number of marker peaks

    Returns:
        MarkerDetectionResult with detection outcome and confidence
    """
    # Define marker region: from last contact + offset to end of data (capped)
    last_contact = max(contact_positions)
    marker_start = last_contact + marker_offset_mm
    marker_end = min(distance_scale[-1], max_distance_mm)

    marker_mask = (distance_scale >= marker_start) & (distance_scale <= marker_end)

    if not marker_mask.any():
        return MarkerDetectionResult(
            has_markers=False,
            confidence=0.0,
            detection_method="insufficient_data",
        )

    marker_distances = distance_scale[marker_mask]
    marker_intensities = intensity_profile[marker_mask]

    marker_stats = {
        "mean": float(np.mean(marker_intensities)),
        "max": float(np.max(marker_intensities)),
        "std": float(np.std(marker_intensities)),
        "region": (marker_start, marker_end),
    }

    # Try deviation-based detection first (more reliable)
    if skeleton_deviations is not None and len(skeleton_deviations) == len(
        distance_scale
    ):
        result = _detect_markers_from_deviations(
            marker_distances,
            skeleton_deviations[marker_mask],
            deviation_threshold,
            min_peak_distance_mm,
            expected_num_peaks,
            marker_stats,
        )
        if result is not None:
            if result.has_markers:
                result.marker_region_center = _compute_region_center(
                    marker_distances, marker_intensities
                )
            return result

    # Fallback to intensity-based detection
    result = _detect_markers_from_intensity(
        marker_distances,
        marker_intensities,
        min_peak_distance_mm,
        expected_num_peaks,
        marker_stats,
    )
    if result.has_markers:
        result.marker_region_center = _compute_region_center(
            marker_distances, marker_intensities
        )
    return result


def _detect_markers_from_deviations(
    marker_distances: np.ndarray,
    marker_deviations: np.ndarray,
    deviation_threshold: float,
    min_peak_distance_mm: float,
    expected_num_peaks: int,
    marker_stats: Dict,
) -> Optional[MarkerDetectionResult]:
    """Detect markers using skeleton deviation peaks."""
    distance_step = np.median(np.diff(marker_distances)) if len(marker_distances) > 1 else 0.025
    min_distance_samples = max(1, int(min_peak_distance_mm / distance_step))

    peaks, properties = find_peaks(
        marker_deviations,
        height=deviation_threshold,
        distance=min_distance_samples,
        prominence=0.02,
    )

    if len(peaks) == 0:
        # No deviation peaks found — strong evidence against markers.
        # Return explicit "no markers" to prevent intensity fallback from
        # picking up noise on flat profiles (non-directional electrodes).
        return MarkerDetectionResult(
            has_markers=False,
            confidence=0.0,
            marker_region_stats=marker_stats,
            detection_method="deviation_peaks",
        )

    peak_locations = marker_distances[peaks].tolist()
    peak_heights = properties["peak_heights"]

    # Sort by height and keep top peaks
    sorted_indices = np.argsort(peak_heights)[::-1]
    peak_locations = [peak_locations[i] for i in sorted_indices[:expected_num_peaks]]
    top_heights = peak_heights[sorted_indices[:expected_num_peaks]]

    # Sort peak locations by distance (ascending) so marker 1 is closest to contacts
    peak_locations = sorted(peak_locations)

    # Confidence based on:
    # 1. Number of peaks found vs expected
    # 2. Peak prominence relative to threshold
    # 3. Peak height relative to region median
    num_peaks_score = min(1.0, len(peaks) / expected_num_peaks)
    prominence_score = min(1.0, np.mean(top_heights) / (deviation_threshold * 2))
    median_dev = np.median(marker_deviations)
    contrast_score = min(1.0, (np.mean(top_heights) - median_dev) / median_dev) if median_dev > 0.01 else 0.0

    confidence = (num_peaks_score + prominence_score + contrast_score) / 3.0

    has_markers = len(peaks) >= expected_num_peaks

    return MarkerDetectionResult(
        has_markers=has_markers,
        confidence=confidence,
        marker_peak_locations=peak_locations,
        marker_region_stats=marker_stats,
        detection_method="deviation_peaks",
    )


def _detect_markers_from_intensity(
    marker_distances: np.ndarray,
    marker_intensities: np.ndarray,
    min_peak_distance_mm: float,
    expected_num_peaks: int,
    marker_stats: Dict,
) -> MarkerDetectionResult:
    """Detect markers using intensity peaks (fallback method).

    Uses within-region statistics to identify peaks that stand out
    from the local intensity distribution.
    """
    distance_step = np.median(np.diff(marker_distances)) if len(marker_distances) > 1 else 0.025
    min_distance_samples = max(1, int(min_peak_distance_mm / distance_step))

    # Use within-region statistics for thresholds
    region_median = np.median(marker_intensities)
    region_std = np.std(marker_intensities)

    # Reject flat profiles: real markers produce substantial intensity variation.
    # Coefficient of variation < 3% or dynamic range < 50 HU → essentially flat.
    cv = region_std / abs(region_median) if abs(region_median) > 1.0 else 0.0
    intensity_range = float(np.ptp(marker_intensities))
    if cv < 0.03 and intensity_range < 50.0:
        return MarkerDetectionResult(
            has_markers=False,
            confidence=0.0,
            marker_region_stats=marker_stats,
            detection_method="intensity_peaks",
        )

    # Normalize relative to region median
    normalized_intensity = marker_intensities - region_median

    # Peaks must be at least 1.5 std above median, with some prominence
    min_height = max(region_std * 1.5, 5.0)

    peaks, properties = find_peaks(
        normalized_intensity,
        height=min_height,
        distance=min_distance_samples,
        prominence=max(region_std * 0.5, 3.0),
    )

    peak_locations = []
    if len(peaks) > 0:
        peak_heights = properties["peak_heights"]
        sorted_indices = np.argsort(peak_heights)[::-1]
        peak_locations = [
            marker_distances[peaks[i]] for i in sorted_indices[:expected_num_peaks]
        ]
        peak_locations = sorted(peak_locations)

    # Confidence from number of peaks and how much they stand out
    num_peaks_score = min(1.0, len(peaks) / expected_num_peaks)
    if len(peaks) > 0 and region_std > 0:
        top_heights = properties["peak_heights"][np.argsort(properties["peak_heights"])[::-1][:expected_num_peaks]]
        contrast_score = min(1.0, np.mean(top_heights) / (region_std * 2))
    else:
        contrast_score = 0.0

    confidence = (num_peaks_score + contrast_score) / 2.0

    has_markers = len(peaks) >= expected_num_peaks

    return MarkerDetectionResult(
        has_markers=has_markers,
        confidence=confidence,
        marker_peak_locations=peak_locations,
        marker_region_stats=marker_stats,
        detection_method="intensity_peaks",
    )


def _compute_region_center(
    distances: np.ndarray,
    intensities: np.ndarray,
) -> float:
    """Compute intensity-weighted center of the marker region.

    Uses intensities above the median as weights so that the
    high-intensity marker areas dominate the center calculation.
    """
    median_intensity = np.median(intensities)
    weights = np.maximum(intensities - median_intensity, 0.0)

    if weights.sum() > 0:
        return float(np.average(distances, weights=weights))
    # Fallback to simple midpoint
    return float((distances[0] + distances[-1]) / 2.0)


def classify_electrode_type(
    contact_positions: List[float],
    marker_detection_result: MarkerDetectionResult,
) -> str:
    """
    Classify electrode type based on contact spacing and marker presence.

    Args:
        contact_positions: Contact positions in mm from tip
        marker_detection_result: Result from marker detection

    Returns:
        Electrode type string
    """
    # Calculate average contact spacing
    if len(contact_positions) < 2:
        return "Unknown"

    sorted_contacts = sorted(contact_positions)
    spacings = np.diff(sorted_contacts)
    avg_spacing = np.mean(spacings)

    # Classify by spacing and markers
    has_markers = marker_detection_result.has_markers

    # Center-to-center spacing: 3389/B33005 = 2.0mm, 3387/B33015 = 3.0mm
    if avg_spacing > 2.5:
        return "Medtronic B33015" if has_markers else "Medtronic 3387"
    else:
        return "Medtronic B33005" if has_markers else "Medtronic 3389"
