"""Debug visualization tools for orientation detection development.

These visualizations are useful during development and testing but are not
needed for final report generation. They can be safely excluded from
production builds.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def visualize_multi_slice_debug(
    slice_distances: np.ndarray,
    all_slice_intensities: List[np.ndarray],
    angles_deg: np.ndarray,
    averaged_intensity: np.ndarray,
    final_peak_angle: float,
    marker_location_mm: float,
    marker_index: int = 0,
    angular_diffs_from_base: Optional[List[float]] = None,
    fitted_direction_deg: Optional[float] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Visualize multi-slice sampling debug information.

    Shows angular intensity profiles at each sampled slice level to understand
    how marker intensity varies along the trajectory.

    Args:
        slice_distances: Array of distances where slices were sampled
        all_slice_intensities: List of intensity profiles for each slice
        angles_deg: Angular positions
        averaged_intensity: Final averaged intensity profile
        final_peak_angle: Detected peak angle
        marker_location_mm: Center marker location
        output_path: Path to save figure
        show: Whether to show figure

    Returns:
        Path to saved figure if output_path provided
    """
    n_slices = len(slice_distances)

    # Create figure with better spacing
    fig = plt.figure(figsize=(18, 20))

    # Create grid with more vertical space
    # Row 0-1: Top overlay plot (spans 2 rows)
    # Row 2-3: Polar plots for individual slices (spans 2 rows)
    # Row 4: Bottom analysis plots
    # Row 5-6: 3D surface contour plot
    gs = fig.add_gridspec(7, 3, hspace=0.5, wspace=0.35,
                          height_ratios=[1, 1, 1, 1, 1, 1.2, 1.2])

    # Color map for slices
    colors = plt.cm.viridis(np.linspace(0, 1, n_slices))

    # Plot 1: All slice profiles overlaid (Cartesian) - LARGER with more space
    ax_overlay = fig.add_subplot(gs[0:2, :])

    # Calculate peak angles for each slice for markers
    peak_angles_per_slice = []
    for intensities in all_slice_intensities:
        peak_idx = np.argmax(intensities)
        peak_angles_per_slice.append(angles_deg[peak_idx])

    # Calculate average peak angle to center the plot
    avg_peak_angle = np.mean(peak_angles_per_slice)

    # Center plot on average peak ±90 degrees, handling wraparound
    plot_center = avg_peak_angle
    plot_range = 90  # Show ±90° around peak
    xlim_min = plot_center - plot_range
    xlim_max = plot_center + plot_range

    for i, (dist, intensities) in enumerate(zip(slice_distances, all_slice_intensities)):
        is_center = abs(dist - marker_location_mm) < 0.01
        linewidth = 3 if is_center else 1.5
        alpha = 1.0 if is_center else 0.6
        label = f'{dist:.2f}mm' + (' (center)' if is_center else '')

        # Handle wraparound: if xlim crosses 360/0 boundary, duplicate data
        if xlim_min < 0 or xlim_max > 360:
            # Plot with shifted angles for continuity
            angles_shifted = angles_deg.copy()
            # Shift angles so they're continuous in the viewing window
            angles_shifted = np.where(angles_shifted > plot_center + 180,
                                     angles_shifted - 360,
                                     angles_shifted)
            angles_shifted = np.where(angles_shifted < plot_center - 180,
                                     angles_shifted + 360,
                                     angles_shifted)
            ax_overlay.plot(angles_shifted, intensities,
                           color=colors[i], linewidth=linewidth, alpha=alpha,
                           label=label)
        else:
            ax_overlay.plot(angles_deg, intensities,
                           color=colors[i], linewidth=linewidth, alpha=alpha,
                           label=label)

        # Add circular marker at detected peak for each slice
        peak_idx = np.argmax(intensities)
        peak_angle = angles_deg[peak_idx]
        # Shift peak angle if needed for display
        if xlim_min < 0 or xlim_max > 360:
            if peak_angle > plot_center + 180:
                peak_angle = peak_angle - 360
            elif peak_angle < plot_center - 180:
                peak_angle = peak_angle + 360
        peak_intensity = intensities[peak_idx]
        ax_overlay.plot(peak_angle, peak_intensity, 'o',
                       color=colors[i], markersize=8, markeredgecolor='black',
                       markeredgewidth=1.5, zorder=10)

    # Final peak angle line (shifted if needed)
    final_peak_shifted = final_peak_angle
    if xlim_min < 0 or xlim_max > 360:
        if final_peak_angle > plot_center + 180:
            final_peak_shifted = final_peak_angle - 360
        elif final_peak_angle < plot_center - 180:
            final_peak_shifted = final_peak_angle + 360
    ax_overlay.axvline(final_peak_shifted, color='red', linestyle='--',
                      linewidth=2.5, label=f'Final Peak ({final_peak_angle:.1f}°)', zorder=5)

    ax_overlay.set_xlim(xlim_min, xlim_max)

    ax_overlay.set_xlabel('Angle (degrees)', fontsize=12)
    ax_overlay.set_ylabel('Intensity (HU)', fontsize=12)
    ax_overlay.set_title(f'Intensity Profiles at All Slice Levels (Centered on Peak ~{avg_peak_angle:.0f}°)',
                         fontsize=14, fontweight='bold')
    ax_overlay.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=2)
    ax_overlay.grid(True, alpha=0.3)

    # Plot 2: Individual slice polar plots - LARGER with more space
    slice_indices_to_plot = []
    if n_slices <= 3:
        slice_indices_to_plot = list(range(n_slices))
    else:
        # Show first, middle, and last
        slice_indices_to_plot = [0, n_slices // 2, n_slices - 1]

    for plot_idx, slice_idx in enumerate(slice_indices_to_plot):
        ax_polar = fig.add_subplot(gs[2:4, plot_idx], projection='polar')

        angles_rad = np.deg2rad(angles_deg)
        intensities = all_slice_intensities[slice_idx]
        dist = slice_distances[slice_idx]

        ax_polar.plot(angles_rad, intensities, 'b-', linewidth=2)

        # Mark peak
        peak_idx = np.argmax(intensities)
        peak_angle_rad = angles_rad[peak_idx]
        peak_intensity = intensities[peak_idx]
        ax_polar.plot([peak_angle_rad], [peak_intensity], 'r*', markersize=15)

        # Arrow to peak (from visible center of polar plot)
        r_min = ax_polar.get_ylim()[0]
        ax_polar.annotate('', xy=(peak_angle_rad, peak_intensity),
                         xytext=(peak_angle_rad, r_min),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))

        # Add fitted direction line if provided
        if fitted_direction_deg is not None:
            fitted_rad = np.deg2rad(fitted_direction_deg)
            ax_polar.plot([fitted_rad, fitted_rad], [r_min, peak_intensity * 1.1],
                         'g--', linewidth=2.5, label=f'Fitted ({fitted_direction_deg:.1f}°)')

        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_title(f'Slice at {dist:.2f}mm\nPeak: {np.rad2deg(peak_angle_rad):.1f}°',
                          fontsize=11, fontweight='bold')
        ax_polar.grid(True, alpha=0.3)

    # Plot 3: Averaged profile (Cartesian) - BOTTOM ROW
    ax_avg_cart = fig.add_subplot(gs[4, 0])
    ax_avg_cart.plot(angles_deg, averaged_intensity, 'k-', linewidth=2.5, label='Averaged')
    ax_avg_cart.axvline(final_peak_angle, color='red', linestyle='--',
                       linewidth=2, label=f'Peak ({final_peak_angle:.1f}°)')
    ax_avg_cart.set_xlabel('Angle (degrees)', fontsize=11)
    ax_avg_cart.set_ylabel('Intensity (HU)', fontsize=11)
    ax_avg_cart.set_title('Averaged Profile (All Slices)', fontsize=12, fontweight='bold')
    ax_avg_cart.legend()
    ax_avg_cart.grid(True, alpha=0.3)

    # Plot 4: Averaged profile (Polar) - BOTTOM ROW
    ax_avg_polar = fig.add_subplot(gs[4, 1], projection='polar')
    angles_rad = np.deg2rad(angles_deg)
    ax_avg_polar.plot(angles_rad, averaged_intensity, 'k-', linewidth=2.5)

    peak_angle_rad = np.deg2rad(final_peak_angle)
    peak_intensity = np.max(averaged_intensity)
    ax_avg_polar.plot([peak_angle_rad], [peak_intensity], 'r*', markersize=20)
    r_min = ax_avg_polar.get_ylim()[0]
    ax_avg_polar.annotate('', xy=(peak_angle_rad, peak_intensity),
                         xytext=(peak_angle_rad, r_min),
                         arrowprops=dict(arrowstyle='->', color='red', lw=3))

    # Add fitted direction line if provided
    if fitted_direction_deg is not None:
        fitted_rad = np.deg2rad(fitted_direction_deg)
        ax_avg_polar.plot([fitted_rad, fitted_rad], [r_min, peak_intensity * 1.1],
                         'g--', linewidth=3, label=f'Fitted ({fitted_direction_deg:.1f}°)')
        ax_avg_polar.legend(fontsize=10, loc='upper right')

    ax_avg_polar.set_theta_zero_location('N')
    ax_avg_polar.set_theta_direction(-1)
    ax_avg_polar.set_title('Averaged Profile (Polar)', fontsize=12, fontweight='bold')
    ax_avg_polar.grid(True, alpha=0.3)

    # Plot 5: Angular difference from base (triangle geometry) - BOTTOM ROW
    ax_angular_diff = fig.add_subplot(gs[4, 2])

    if angular_diffs_from_base is not None:
        # Show angular difference from base of triangle
        base_text = "Base (closest to contacts)" if marker_index == 0 else "Base (furthest from contacts)"
        ax_angular_diff.plot(slice_distances, angular_diffs_from_base, 'mo-',
                            linewidth=2, markersize=8, label='Angular diff from base')
        ax_angular_diff.axhline(0, color='green', linestyle='--',
                               linewidth=1, alpha=0.5, label=base_text)
        ax_angular_diff.axhline(120, color='red', linestyle=':',
                               linewidth=1, alpha=0.5, label='Expected pair separation (120°)')
        ax_angular_diff.set_ylabel('Angular difference (degrees)', fontsize=11)
        ax_angular_diff.set_title(f'Angular Drift from Triangle Base\n(Marker {marker_index + 1})',
                                  fontsize=12, fontweight='bold')
    else:
        # Fallback to peak angle variation
        peak_angles_per_slice = [angles_deg[np.argmax(intensities)]
                                 for intensities in all_slice_intensities]
        ax_angular_diff.plot(slice_distances, peak_angles_per_slice, 'bo-',
                            linewidth=2, markersize=8, label='Peak angle')
        ax_angular_diff.axhline(final_peak_angle, color='red', linestyle='--',
                               linewidth=2, label=f'Final ({final_peak_angle:.1f}°)')
        ax_angular_diff.set_ylabel('Peak angle (degrees)', fontsize=11)
        ax_angular_diff.set_title('Peak Angle Variation Across Slices',
                                 fontsize=12, fontweight='bold')

    ax_angular_diff.axvline(marker_location_mm, color='orange', linestyle=':',
                           linewidth=2, alpha=0.5, label='Marker center')
    ax_angular_diff.set_xlabel('Distance from tip (mm)', fontsize=11)
    ax_angular_diff.legend(fontsize=9)
    ax_angular_diff.grid(True, alpha=0.3)

    # Plot 6: 3D surface contour of intensity across slices and angles
    ax_3d = fig.add_subplot(gs[5:7, :], projection='3d')

    angle_grid, dist_grid = np.meshgrid(angles_deg, slice_distances)
    intensity_grid = np.array(all_slice_intensities)

    surf = ax_3d.plot_surface(
        angle_grid, dist_grid, intensity_grid,
        cmap='viridis', alpha=0.85, edgecolor='none',
        rstride=1, cstride=max(1, len(angles_deg) // 72),
    )

    # Add contour projection on the bottom
    z_floor = intensity_grid.min() - (intensity_grid.max() - intensity_grid.min()) * 0.05
    ax_3d.contourf(
        angle_grid, dist_grid, intensity_grid,
        zdir='z', offset=z_floor, cmap='viridis', alpha=0.4, levels=20,
    )

    # Mark the final peak angle as a vertical plane slice
    ax_3d.plot(
        [final_peak_angle, final_peak_angle],
        [slice_distances[0], slice_distances[-1]],
        [z_floor, z_floor],
        'r--', linewidth=2, alpha=0.8,
    )

    ax_3d.set_xlabel('Angle (°)', fontsize=11, labelpad=10)
    ax_3d.set_ylabel('Distance from tip (mm)', fontsize=11, labelpad=10)
    ax_3d.set_zlabel('Intensity (HU)', fontsize=11, labelpad=10)
    ax_3d.set_title('3D Intensity Surface Across Slices', fontsize=13, fontweight='bold')

    fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=15, label='Intensity (HU)')

    # Set a good viewing angle
    ax_3d.view_init(elev=25, azim=-60)

    # Add overall title
    fig.suptitle(f'Multi-Slice Sampling Debug: Marker at {marker_location_mm:.2f}mm\n'
                f'Sampled {n_slices} slices from {slice_distances[0]:.2f} to {slice_distances[-1]:.2f}mm',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    if show:
        plt.show()
    else:
        plt.close()

    return None
