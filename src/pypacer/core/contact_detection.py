"""Contact detection algorithms for electrode reconstruction."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import ndimage, signal


@dataclass
class ContactDetectionResult:
    """Results from contact detection."""

    contact_positions: np.ndarray  # Contact positions in mm from tip
    method_used: str
    peak_locations: Optional[np.ndarray] = None
    peak_values: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    contact_area_center: Optional[float] = None
    contact_area_width: Optional[float] = None
    intensity_profile: Optional[np.ndarray] = None
    distance_scale: Optional[np.ndarray] = None


def detect_contacts(
    refined_model: Any,  # Will be RefinedTrajectory type
    method: str = "contactAreaCenter",
    electrode_type: Optional[str] = None,
    display_profile: bool = False,
    limit_search_mm: float = 20.0,
    run_all_methods: bool = False,
) -> np.ndarray:
    """
    Detect electrode contacts from intensity profile.

    Based on MATLAB getIntensityPeaks.m

    Args:
        refined_model: Refined trajectory model with intensity profile
        method: Detection method ('peak', 'peakWaveCenter', 'contactAreaCenter')
        electrode_type: Electrode type for validation
        display_profile: Whether to display intensity profile
        limit_search_mm: Limit contact search to first N mm
        run_all_methods: Run all detection methods and save results (debug mode)

    Returns:
        Array of contact positions in mm from tip
    """
    print("  Step 4: Detecting contacts...")
    print(f"    run_all_methods: {run_all_methods}")

    # Extract intensity profile and scale
    intensity_profile = refined_model.intensity_profile
    distance_scale = refined_model.distance_scale_mm
    print(f"    Intensity profile length: {len(intensity_profile)} points")

    # Limit search range
    search_mask = distance_scale <= limit_search_mm

    # Run all methods if requested (debug mode)
    if run_all_methods:
        print("    DEBUG MODE: Running all contact detection methods")
        all_results = {}
        methods = ["contactAreaCenter", "peak", "peakWaveCenter"]

        for test_method in methods:
            print(f"    Running method: {test_method}")
            try:
                if test_method == "contactAreaCenter":
                    result = _detect_contacts_area_center(
                        intensity_profile[search_mask],
                        distance_scale[search_mask],
                        electrode_type=electrode_type,
                    )
                elif test_method in ["peak", "peakWaveCenter"]:
                    result = _detect_contacts_peaks(
                        intensity_profile[search_mask],
                        distance_scale[search_mask],
                        use_wave_centers=(test_method == "peakWaveCenter"),
                    )

                # Store result with method name
                all_results[test_method] = {
                    "contact_positions": result.contact_positions.tolist(),
                    "threshold": (
                        float(result.threshold)
                        if result.threshold is not None
                        else None
                    ),
                    "peak_locations": (
                        result.peak_locations.tolist()
                        if result.peak_locations is not None
                        else None
                    ),
                    "peak_values": (
                        result.peak_values.tolist()
                        if result.peak_values is not None
                        else None
                    ),
                    "contact_area_center": (
                        float(result.contact_area_center)
                        if result.contact_area_center is not None
                        else None
                    ),
                    "contact_area_width": (
                        float(result.contact_area_width)
                        if result.contact_area_width is not None
                        else None
                    ),
                }

                # Use the requested method as the primary result
                if test_method == method:
                    primary_result = result

            except Exception as e:
                print(f"      Method {test_method} failed: {str(e)}")
                all_results[test_method] = {"error": str(e)}

        # Store all results in refined model for saving later
        refined_model.contact_detection_results = all_results

        # Use primary result for continuation
        result = (
            primary_result
            if "primary_result" in locals()
            else list(all_results.values())[0]
        )

    else:
        # Normal single method detection
        print(f"    Using method: {method}")
        if method == "contactAreaCenter":
            result = _detect_contacts_area_center(
                intensity_profile[search_mask],
                distance_scale[search_mask],
                electrode_type=electrode_type,
            )
        elif method in ["peak", "peakWaveCenter"]:
            result = _detect_contacts_peaks(
                intensity_profile[search_mask],
                distance_scale[search_mask],
                use_wave_centers=(method == "peakWaveCenter"),
            )
        else:
            raise ValueError(f"Unknown contact detection method: {method}")

    print(f"    Initial detection: {len(result.contact_positions)} contacts")

    # Validate and determine electrode type if needed
    if electrode_type is None:
        electrode_type = _determine_electrode_type(result.contact_positions)
        print(f"Auto-detected electrode type: {electrode_type}")

    # Store electrode type in refined model for later use
    refined_model.electrode_type = electrode_type

    # Get expected contact positions for validation
    from ..models.electrode import ELECTRODE_GEOMETRIES

    # Handle combined electrode types
    if electrode_type in ELECTRODE_GEOMETRIES:
        expected_positions = ELECTRODE_GEOMETRIES[electrode_type].contact_centers_mm
    else:
        # Try to find first matching type from combined string
        electrode_types = [t.strip() for t in electrode_type.split("/")]
        expected_positions = None
        for et in electrode_types:
            if et in ELECTRODE_GEOMETRIES:
                expected_positions = ELECTRODE_GEOMETRIES[et].contact_centers_mm
                break

        if expected_positions is None:
            # Default to Medtronic 3389
            expected_positions = ELECTRODE_GEOMETRIES[
                "Medtronic 3389"
            ].contact_centers_mm

    # If detection failed or wrong number of contacts, use expected positions
    if len(result.contact_positions) != len(expected_positions):
        print(
            f"Warning: Detected {len(result.contact_positions)} contacts, "
            f"expected {len(expected_positions)}. Using geometry-based positions."
        )
        return expected_positions

    # Plotting disabled to prevent hanging
    # if display_profile:
    #     _plot_intensity_profile(result)

    return result.contact_positions


def _detect_contacts_peaks(
    intensity: np.ndarray,
    distance: np.ndarray,
    use_wave_centers: bool = True,
    min_peak_distance: float = 1.4,  # mm
    min_peak_height_factor: float = 1.1,
    min_peak_prominence_factor: float = 0.01,
) -> ContactDetectionResult:
    """
    Detect contacts using peak detection.

    Args:
        intensity: Intensity profile
        distance: Distance scale in mm
        use_wave_centers: Use wave centers instead of peak locations
        min_peak_distance: Minimum distance between peaks
        min_peak_height_factor: Factor of mean for minimum peak height
        min_peak_prominence_factor: Factor of mean for minimum prominence

    Returns:
        ContactDetectionResult
    """
    # Find peaks
    mean_intensity = np.nanmean(intensity)

    # Convert distance to indices for peak detection
    distance_spacing = np.mean(np.diff(distance))
    min_distance_samples = int(min_peak_distance / distance_spacing)

    peaks, properties = signal.find_peaks(
        intensity,
        distance=min_distance_samples,
        height=min_peak_height_factor * mean_intensity,
        prominence=min_peak_prominence_factor * mean_intensity,
    )

    if len(peaks) == 0:
        print("No peaks found, falling back to contact area method")
        return _detect_contacts_area_center(intensity, distance)

    peak_locations = distance[peaks]
    peak_values = intensity[peaks]
    peak_prominences = properties["prominences"]

    # Find wave centers if requested
    if use_wave_centers and len(peaks) >= 4:
        try:
            # Threshold at minimum of first 4 peaks minus quarter prominence
            threshold = np.min(peak_values[:4]) - (np.min(peak_prominences[:4]) / 4)

            # Find regions above threshold
            above_threshold = intensity >= threshold
            labeled, num_features = ndimage.label(above_threshold)

            # Get center of mass for first 4 regions
            wave_centers = []
            for i in range(1, min(5, num_features + 1)):  # First 4 contact regions
                region_mask = labeled == i
                if np.any(region_mask):
                    region_distances = distance[region_mask]
                    region_intensities = intensity[region_mask]
                    # Intensity-weighted center
                    center = np.average(region_distances, weights=region_intensities)
                    wave_centers.append(center)

            if len(wave_centers) >= 4:
                contact_positions = np.array(wave_centers[:4])
            else:
                contact_positions = peak_locations[:4]
        except Exception as e:
            print(f"Wave center detection failed: {e}. Using peak locations.")
            contact_positions = peak_locations[:4]
            threshold = None
    else:
        contact_positions = peak_locations[:4]
        threshold = None

    return ContactDetectionResult(
        contact_positions=contact_positions,
        method_used="peak" if not use_wave_centers else "peakWaveCenter",
        peak_locations=peak_locations,
        peak_values=peak_values,
        threshold=threshold,
        intensity_profile=intensity,
        distance_scale=distance,
    )


def _detect_contacts_area_center(
    intensity: np.ndarray, distance: np.ndarray, electrode_type: Optional[str] = None
) -> ContactDetectionResult:
    """
    Detect contacts using contact area center method.

    This is a fallback for low SNR where individual contacts aren't visible.

    Args:
        intensity: Intensity profile
        distance: Distance scale in mm

    Returns:
        ContactDetectionResult
    """
    # Try multiple thresholds to find the best one
    # We want a region that's close to the expected 7.5mm for typical electrodes
    expected_region_size = 7.5  # mm for Medtronic 3389 (4 contacts)

    # Try percentiles from 50th to 70th
    best_threshold = None
    best_region_distances = None
    best_region_intensities = None
    best_score = float("inf")

    for percentile in [50, 55, 60, 65, 70]:
        threshold_test = np.percentile(intensity, percentile)
        above_threshold_test = intensity >= threshold_test

        # Find connected regions
        labeled_test, num_features_test = ndimage.label(above_threshold_test)

        if num_features_test == 0:
            continue

        # Find all regions and their extents
        region_candidates = []
        for i in range(1, num_features_test + 1):
            region_mask_i = labeled_test == i
            if np.sum(region_mask_i) < 3:  # Skip tiny regions
                continue
            region_distances_i = distance[region_mask_i]
            if len(region_distances_i) >= 2:
                width_i = region_distances_i[-1] - region_distances_i[0]
                region_candidates.append((i, width_i, region_distances_i))

        if not region_candidates:
            continue

        # Choose the region with width closest to expected (not necessarily largest by pixel count)
        best_region_for_threshold = min(
            region_candidates, key=lambda x: abs(x[1] - expected_region_size)
        )
        region_id, region_width_test, region_distances_test = best_region_for_threshold
        region_mask_test = labeled_test == region_id

        if len(region_distances_test) < 2:
            continue

        # Width already calculated above

        # Score based on how close to expected size (prefer slightly larger over smaller)
        if region_width_test < expected_region_size:
            # Penalize undersized regions more
            score = abs(region_width_test - expected_region_size) * 2
        else:
            # Smaller penalty for oversized regions up to 2x expected
            if region_width_test < expected_region_size * 2:
                score = abs(region_width_test - expected_region_size) * 0.5
            else:
                # Heavy penalty for very large regions
                score = abs(region_width_test - expected_region_size) * 3

        # Update best if this is better
        if score < best_score:
            best_score = score
            best_threshold = threshold_test
            best_region_distances = region_distances_test
            best_region_intensities = intensity[region_mask_test]
            best_percentile = percentile

    # If no good threshold found, fall back to 60th percentile
    if best_threshold is None:
        threshold = np.percentile(intensity, 60)
        above_threshold = intensity >= threshold
        labeled, num_features = ndimage.label(above_threshold)

        if num_features == 0:
            raise ValueError("No metal regions found above threshold")

        region_sizes = []
        for i in range(1, num_features + 1):
            region_sizes.append(np.sum(labeled == i))
        largest_region = np.argmax(region_sizes) + 1
        region_mask = labeled == largest_region
        region_distances = distance[region_mask]
        region_intensities = intensity[region_mask]
        best_percentile = 60
    else:
        threshold = best_threshold
        region_distances = best_region_distances
        region_intensities = best_region_intensities

    # Calculate geometric center (unweighted) for better centering in plateau regions
    # This avoids bias towards intensity variations within the plateau
    geometric_center = (region_distances[0] + region_distances[-1]) / 2
    contact_area_width = region_distances[-1] - region_distances[0]

    # Also calculate weighted center for comparison (can be useful for diagnostics)
    weighted_center = np.average(region_distances, weights=region_intensities)

    print("      Contact area detection:")
    print(
        f"        Threshold: {threshold:.1f} ({best_percentile}th percentile - adaptive)"
    )
    print(f"        Region width: {contact_area_width:.2f}mm")
    print(f"        Geometric center: {geometric_center:.2f}mm")
    print(f"        Weighted center: {weighted_center:.2f}mm (not used)")
    print("        Using geometric center for better plateau centering")

    # Use geometric center for distributing contacts
    # This provides better centering in smooth high intensity regions
    contact_positions = _distribute_contacts_in_area(
        geometric_center, contact_area_width, electrode_type=electrode_type
    )

    return ContactDetectionResult(
        contact_positions=contact_positions,
        method_used="contactAreaCenter",
        threshold=threshold,
        contact_area_center=geometric_center,  # Use geometric center
        contact_area_width=contact_area_width,
        intensity_profile=intensity,
        distance_scale=distance,
    )


def _distribute_contacts_in_area(
    center: float, width: float, electrode_type: Optional[str] = None
) -> np.ndarray:
    """
    Distribute contacts using proper electrode geometry.

    Args:
        center: Center of contact area
        width: Width of contact area
        electrode_type: Type of electrode (e.g., 'Medtronic 3389')

    Returns:
        Contact positions in mm from tip
    """
    # Import electrode geometries
    from ..models.electrode import ELECTRODE_GEOMETRIES

    # Get electrode geometry or use default
    # Handle combined electrode types (e.g., "Medtronic 3389/B33005")
    if electrode_type:
        # Try exact match first
        if electrode_type in ELECTRODE_GEOMETRIES:
            geometry = ELECTRODE_GEOMETRIES[electrode_type]
            print(f"        Using electrode geometry: {electrode_type}")
        else:
            # Try to find first matching type from combined string
            electrode_types = [t.strip() for t in electrode_type.split("/")]
            geometry = None
            for et in electrode_types:
                if et in ELECTRODE_GEOMETRIES:
                    geometry = ELECTRODE_GEOMETRIES[et]
                    print(
                        f"        Using electrode geometry: {et} (from {electrode_type})"
                    )
                    break

            if geometry is None:
                # Default to Medtronic 3389
                geometry = ELECTRODE_GEOMETRIES["Medtronic 3389"]
                print("        Using default geometry: Medtronic 3389")
    else:
        # Default to Medtronic 3389
        geometry = ELECTRODE_GEOMETRIES["Medtronic 3389"]
        print("        Using default geometry: Medtronic 3389")

    # Get contact specifications from geometry
    num_contacts = geometry.num_contacts
    contact_length = geometry.contact_length_mm
    contact_spacing = geometry.contact_spacing_mm

    # Calculate center-to-center spacing
    center_to_center_spacing = contact_length + contact_spacing

    # Calculate total extent of contacts
    # From center of first contact to center of last contact
    total_contact_extent = (num_contacts - 1) * center_to_center_spacing

    # Calculate total artifact region
    # This includes the full extent of all contacts plus half contact on each end
    total_artifact_region = total_contact_extent + contact_length

    print(
        f"        Electrode specs: {num_contacts} contacts, {contact_length}mm length, {contact_spacing}mm spacing"
    )
    print(f"        Center-to-center spacing: {center_to_center_spacing}mm")
    print(f"        Expected artifact region: {total_artifact_region}mm")
    print(f"        Detected region width: {width:.2f}mm")

    # Position contacts based on detected region
    # The detected high intensity region IS where the contacts are
    # We need to position them within this region using proper geometry

    if width < total_artifact_region * 0.8:
        # Region is smaller than expected - might be partial detection
        print(
            f"        WARNING: Detected region ({width:.2f}mm) smaller than expected ({total_artifact_region}mm)"
        )

    # Position contacts within the detected region
    # The key insight: the high intensity region corresponds to the actual contacts
    # So we should fit our contact array within this region

    if width > total_artifact_region * 1.5:
        # Large region detected - contacts are likely at the beginning (lower) portion
        # Position first contact near the start of the detected region
        start_offset = center - width / 2 + contact_length / 2  # Small offset from edge
    else:
        # Region size is reasonable - center the contact array within it
        # This centers the entire contact array in the detected region
        start_offset = center - total_contact_extent / 2

    # Generate contact positions with proper spacing
    positions = []
    for i in range(num_contacts):
        position = start_offset + i * center_to_center_spacing
        positions.append(position)

    positions = np.array(positions)

    print(f"        Contact positions: {positions}")
    print(
        f"        First contact at {positions[0]:.2f}mm, last at {positions[-1]:.2f}mm"
    )
    print(f"        Actual spacings: {np.diff(positions)}")

    return positions


def _determine_electrode_type(contact_positions: np.ndarray) -> str:
    """
    Determine electrode type based on contact spacing.

    Note: We cannot distinguish between standard and directional electrodes,
    so we return combined type strings.

    Args:
        contact_positions: Detected contact positions

    Returns:
        Electrode type string (may be combined, e.g. 'Medtronic 3389/B33005')
    """
    if len(contact_positions) < 2:
        return "Medtronic 3389/B33005"  # Default to 0.5mm spacing types

    # Calculate average spacing
    spacings = np.diff(contact_positions)
    avg_spacing = np.mean(spacings)

    # Match to known electrode types
    # We can't distinguish standard from directional, so return both possibilities
    if 0.3 < avg_spacing < 0.7:
        # 0.5mm spacing - could be 3389 (standard) or B33005 (directional)
        return "Medtronic 3389/B33005"
    elif 1.2 < avg_spacing < 1.8:
        # 1.5mm spacing - could be 3387 (standard) or B33015 (directional)
        return "Medtronic 3387/B33015"
    else:
        # Default to 0.5mm spacing types
        return "Medtronic 3389/B33005"


def _plot_intensity_profile(result: ContactDetectionResult):
    """Plot intensity profile with detected contacts."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Force non-interactive backend
        from datetime import datetime
        from pathlib import Path

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            result.distance_scale, result.intensity_profile, "b-", label="Intensity"
        )

        # Plot threshold
        if result.threshold is not None:
            plt.axhline(result.threshold, color="r", linestyle="--", label="Threshold")

        # Plot peaks
        if result.peak_locations is not None:
            plt.plot(
                result.peak_locations,
                result.peak_values,
                "ro",
                markersize=8,
                label="Peaks",
            )

        # Plot detected contacts
        contact_intensities = np.interp(
            result.contact_positions, result.distance_scale, result.intensity_profile
        )
        plt.plot(
            result.contact_positions,
            contact_intensities,
            "g^",
            markersize=10,
            label="Contacts",
        )

        plt.xlabel("Distance from tip (mm)")
        plt.ylabel("Intensity (HU)")
        plt.title(f"Contact Detection ({result.method_used})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save to file instead of showing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd()  # Current working directory
        filename = (
            output_dir / f"contact_detection_{result.method_used}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Contact detection plot saved to: {filename.name}")

    except ImportError:
        print("Matplotlib not available for plotting")
