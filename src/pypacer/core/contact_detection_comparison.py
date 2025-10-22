"""Compare different contact detection methods."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .contact_detection import (
    ContactDetectionResult,
    _detect_contacts_area_center,
    _detect_contacts_peaks,
)


@dataclass
class ContactComparisonResult:
    """Results from comparing contact detection methods."""

    method_results: Dict[str, ContactDetectionResult]
    statistics: Dict[str, float]
    refined_model: Any


def compare_contact_detection_methods(
    refined_model: Any,
    electrode_type: Optional[str] = None,
    limit_search_mm: float = 20.0,
    save_plots: bool = True,
    output_dir: Optional[str] = None,
) -> ContactComparisonResult:
    """
    Run multiple contact detection methods and compare results.

    Args:
        refined_model: Refined trajectory model
        electrode_type: Electrode type for validation
        limit_search_mm: Limit search to first N mm
        save_plots: Whether to save comparison plots
        output_dir: Optional output directory for plots

    Returns:
        Comparison results with statistics
    """
    print("\n  Comparing contact detection methods...")

    # Methods to compare
    methods = ["contactAreaCenter", "peakWaveCenter"]
    results = {}

    # Extract common data
    intensity_profile = refined_model.intensity_profile
    distance_scale = refined_model.distance_scale_mm

    # Apply search limit
    search_mask = distance_scale <= limit_search_mm
    limited_intensity = intensity_profile[search_mask]
    limited_distance = distance_scale[search_mask]

    # Run each method
    for method in methods:
        print(f"\n    Running {method}...")

        if method == "contactAreaCenter":
            result = _detect_contacts_area_center(limited_intensity, limited_distance)
        else:  # peakWaveCenter
            result = _detect_contacts_peaks(
                limited_intensity, limited_distance, use_wave_centers=True
            )

        # Store full intensity profile for plotting
        result.intensity_profile = intensity_profile
        result.distance_scale = distance_scale

        results[method] = result

        # Print contact positions
        print("    Contact positions (mm from tip):")
        for i, pos in enumerate(result.contact_positions):
            print(f"      Contact {i+1}: {pos:.2f} mm")

    # Calculate statistics
    statistics = calculate_comparison_statistics(results)

    # Print comparison
    print_comparison_summary(results, statistics)

    # Save comparison plots if requested
    if save_plots:
        save_comparison_plots(results, electrode_type, output_dir)

    return ContactComparisonResult(
        method_results=results, statistics=statistics, refined_model=refined_model
    )


def calculate_comparison_statistics(
    results: Dict[str, ContactDetectionResult],
) -> Dict[str, float]:
    """Calculate statistics comparing the methods."""
    stats = {}

    # Get contact positions
    area_positions = results["contactAreaCenter"].contact_positions
    wave_positions = results["peakWaveCenter"].contact_positions

    # Ensure same number of contacts for comparison
    n_contacts = min(len(area_positions), len(wave_positions))
    area_positions = area_positions[:n_contacts]
    wave_positions = wave_positions[:n_contacts]

    # Calculate differences
    if n_contacts > 0:
        differences = wave_positions - area_positions

        stats["mean_difference"] = np.mean(differences)
        stats["std_difference"] = np.std(differences)
        stats["max_difference"] = np.max(np.abs(differences))
        stats["rmse"] = np.sqrt(np.mean(differences**2))

        # Per-contact differences
        for i in range(n_contacts):
            stats[f"contact_{i+1}_diff"] = differences[i]
    else:
        stats["mean_difference"] = 0
        stats["std_difference"] = 0
        stats["max_difference"] = 0
        stats["rmse"] = 0

    return stats


def print_comparison_summary(
    results: Dict[str, ContactDetectionResult], statistics: Dict[str, float]
):
    """Print comparison summary to terminal."""
    print("\n  ============= Contact Detection Comparison =============")

    # Get positions
    area_positions = results["contactAreaCenter"].contact_positions
    wave_positions = results["peakWaveCenter"].contact_positions

    n_contacts = min(len(area_positions), len(wave_positions))

    print(f"\n  Number of contacts detected: {n_contacts}")
    print("\n  Contact positions (mm from tip):")
    print("  " + "-" * 60)
    print(
        f"  {'Contact':<10} {'ContactAreaCenter':<20} {'PeakWaveCenter':<20} {'Difference':<15}"
    )
    print("  " + "-" * 60)

    for i in range(n_contacts):
        diff = wave_positions[i] - area_positions[i] if i < len(wave_positions) else 0
        print(
            f"  {i+1:<10} {area_positions[i]:<20.2f} {wave_positions[i]:<20.2f} {diff:<15.2f}"
        )

    print("  " + "-" * 60)

    # Print statistics
    print("\n  Statistical differences (PeakWaveCenter - ContactAreaCenter):")
    print(f"    Mean difference:     {statistics['mean_difference']:6.2f} mm")
    print(f"    Std deviation:       {statistics['std_difference']:6.2f} mm")
    print(f"    Max absolute diff:   {statistics['max_difference']:6.2f} mm")
    print(f"    RMSE:               {statistics['rmse']:6.2f} mm")

    # Special notes
    if "contact_area_center" in results["contactAreaCenter"].__dict__:
        center = results["contactAreaCenter"].contact_area_center
        width = results["contactAreaCenter"].contact_area_width
        print(f"\n  Contact area center: {center:.2f} mm")
        print(f"  Contact area width:  {width:.2f} mm")

    print("\n  ======================================================\n")


def save_comparison_plots(
    results: Dict[str, ContactDetectionResult],
    electrode_type: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Save comparison plots showing both methods using Plotly."""
    try:
        from datetime import datetime
        from pathlib import Path

        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Common data
        distance = results["contactAreaCenter"].distance_scale
        intensity = results["contactAreaCenter"].intensity_profile

        # Get all contact positions from both methods to determine x-axis range
        all_contacts = np.concatenate(
            [
                results["contactAreaCenter"].contact_positions,
                results["peakWaveCenter"].contact_positions,
            ]
        )
        min_contact = np.min(all_contacts)
        max_contact = np.max(all_contacts)

        # Set x-axis limits to contact area ± 1mm
        x_min = min_contact - 1.0
        x_max = max_contact + 1.0

        # Find the intensity values at all contact positions
        contact_intensities_area = np.interp(
            results["contactAreaCenter"].contact_positions, distance, intensity
        )
        contact_intensities_wave = np.interp(
            results["peakWaveCenter"].contact_positions, distance, intensity
        )
        all_contact_intensities = np.concatenate(
            [contact_intensities_area, contact_intensities_wave]
        )

        # Set y-axis limits based on contact intensities with padding
        intensity_min = np.min(all_contact_intensities)
        intensity_max = np.max(all_contact_intensities)
        intensity_range = intensity_max - intensity_min
        y_padding = intensity_range * 0.2  # 20% padding
        y_min = intensity_min - y_padding
        y_max = intensity_max + y_padding

        # Limit data to visible region for plotting
        mask = (distance >= x_min) & (distance <= x_max)
        distance_limited = distance[mask]
        intensity_limited = intensity[mask]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "<b>ContactAreaCenter Method</b>",
                "<b>PeakWaveCenter Method</b>",
            ),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
            shared_xaxes=True,
        )

        # Color scheme
        profile_color = "#1f77b4"
        area_contact_color = "#2ca02c"
        wave_contact_color = "#9467bd"
        peak_color = "#d62728"
        threshold_color = "#ff7f0e"

        # Plot 1: ContactAreaCenter
        area_result = results["contactAreaCenter"]

        # Intensity profile
        fig.add_trace(
            go.Scatter(
                x=distance_limited,
                y=intensity_limited,
                mode="lines",
                name="Intensity Profile",
                line=dict(color=profile_color, width=2),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Threshold line
        if area_result.threshold is not None:
            fig.add_hline(
                y=area_result.threshold,
                line=dict(color=threshold_color, width=2, dash="dash"),
                annotation_text=f"Threshold ({area_result.threshold:.0f} HU)",
                annotation_position="top right",
                row=1,
                col=1,
            )

        # Contact area shading
        if hasattr(area_result, "contact_area_center"):
            center = area_result.contact_area_center
            width = area_result.contact_area_width

            # Add shaded area
            fig.add_vrect(
                x0=center - width / 2,
                x1=center + width / 2,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Contact Area",
                annotation_position="top",
                row=1,
                col=1,
            )

        # Contact markers
        area_intensities = np.interp(area_result.contact_positions, distance, intensity)
        fig.add_trace(
            go.Scatter(
                x=area_result.contact_positions,
                y=area_intensities,
                mode="markers+text",
                name="Area Center Contacts",
                marker=dict(
                    color=area_contact_color,
                    size=14,
                    symbol="triangle-up",
                    line=dict(color="darkgreen", width=2),
                ),
                text=[f"C{i+1}" for i in range(len(area_result.contact_positions))],
                textposition="top center",
                textfont=dict(size=11, color=area_contact_color),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Plot 2: PeakWaveCenter
        wave_result = results["peakWaveCenter"]

        # Intensity profile
        fig.add_trace(
            go.Scatter(
                x=distance_limited,
                y=intensity_limited,
                mode="lines",
                name="Intensity Profile",
                line=dict(color=profile_color, width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Plot peaks
        if wave_result.peak_locations is not None:
            peak_mask = (wave_result.peak_locations >= x_min) & (
                wave_result.peak_locations <= x_max
            )
            if np.any(peak_mask):
                fig.add_trace(
                    go.Scatter(
                        x=wave_result.peak_locations[peak_mask],
                        y=wave_result.peak_values[peak_mask],
                        mode="markers",
                        name="Detected Peaks",
                        marker=dict(
                            color=peak_color,
                            size=10,
                            symbol="circle",
                            line=dict(color="darkred", width=1.5),
                        ),
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )

        # Contact markers
        wave_intensities = np.interp(wave_result.contact_positions, distance, intensity)
        fig.add_trace(
            go.Scatter(
                x=wave_result.contact_positions,
                y=wave_intensities,
                mode="markers+text",
                name="Wave Center Contacts",
                marker=dict(
                    color=wave_contact_color,
                    size=14,
                    symbol="triangle-up",
                    line=dict(color="darkmagenta", width=2),
                ),
                text=[f"C{i+1}" for i in range(len(wave_result.contact_positions))],
                textposition="top center",
                textfont=dict(size=11, color=wave_contact_color),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        # Update layout
        title_text = "<b>Contact Detection Method Comparison</b>"
        if electrode_type:
            title_text += f"<br><sub>Electrode Type: {electrode_type}</sub>"

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=18), x=0.5, xanchor="center"),
            height=800,
            width=900,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
            template="plotly_white",
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(
            title_text="Distance from tip (mm)",
            title_font=dict(size=14),
            range=[x_min, x_max],
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Intensity (HU)", title_font=dict(size=14), range=[y_min, y_max]
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            save_dir = Path.cwd()
        else:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Create descriptive filename
        electrode_suffix = (
            f"_{electrode_type.replace(' ', '_')}" if electrode_type else ""
        )

        # Save as interactive HTML
        html_filename = (
            save_dir
            / f"contact_detection_comparison{electrode_suffix}_{timestamp}.html"
        )
        fig.write_html(str(html_filename))

        # Save as static PNG
        png_filename = (
            save_dir / f"contact_detection_comparison{electrode_suffix}_{timestamp}.png"
        )
        fig.write_image(str(png_filename), width=900, height=800, scale=2)

        print("\n  Comparison plots saved:")
        print(f"    - Interactive: {html_filename.name}")
        print(f"    - Static: {png_filename.name}")

    except ImportError as e:
        print(f"  Required library not available: {e}")
    except Exception as e:
        print(f"  Error creating comparison plot: {e}")
