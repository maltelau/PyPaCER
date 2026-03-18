"""Visualization tools for directional electrode orientation."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .orientation_analysis import MarkerOrientationResult


def visualize_marker_orientations(
    ct_data: np.ndarray,
    affine: np.ndarray,
    marker_positions: List[np.ndarray],
    marker_directions: List[np.ndarray],
    orientation_results: List[MarkerOrientationResult],
    labels: List[str],
    electrode_idx: int = 0,
    fitted_angles: Optional[List[float]] = None,
    plane_size_mm: float = 5.0,
    resolution_mm: float = 0.1,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Combined debug visualization for marker orientation detection.

    Layout (2 rows x N+1 columns when 2 markers, 2 x N otherwise):
    - Columns 0..N-1: one per marker (plane slice on top, polar profile below)
    - Column N (when 2 markers): comparison column (dual-axis Cartesian on top,
      combined polar below)

    Args:
        ct_data: 3D CT volume
        affine: Affine transformation matrix
        marker_positions: World-coordinate center for each marker
        marker_directions: Trajectory direction vector at each marker
        orientation_results: Orientation results for each marker
        labels: Label for each marker (e.g. ["B", "A"])
        electrode_idx: Electrode index for title
        fitted_angles: Fitted (constrained) angles per marker, same order as
            orientation_results. Used in comparison column.
        plane_size_mm: Size of plane to visualize
        resolution_mm: Resolution for sampling
        output_path: Path to save figure
        show: Whether to show figure interactively

    Returns:
        Path to saved figure if output_path provided
    """
    from scipy.ndimage import map_coordinates

    n_markers = len(orientation_results)
    has_comparison = n_markers == 2
    n_cols = n_markers + 1 if has_comparison else n_markers

    fig = plt.figure(figsize=(10 * n_cols, 18))
    gs = fig.add_gridspec(2, n_cols, hspace=0.3, wspace=0.3)

    inv_affine = np.linalg.inv(affine)

    # Consistent marker colours used across the whole figure
    marker_colors = ['#1f77b4', '#d62728']  # blue, red

    # --- Top row: orthogonal plane slices ---
    for mi in range(n_markers):
        ax = fig.add_subplot(gs[0, mi])
        orient_result = orientation_results[mi]
        center_world = marker_positions[mi]
        trajectory_direction = marker_directions[mi]

        # Normalize trajectory direction
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

        # Create grid in plane
        half_size = plane_size_mm / 2
        x_coords = np.arange(-half_size, half_size, resolution_mm)
        y_coords = np.arange(-half_size, half_size, resolution_mm)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Sample CT intensities
        # Map display axes to match polar convention (0°=up/N, CW):
        #   display x (horizontal) = v direction (sin component, 90°=right)
        #   display y (vertical)   = u direction (cos component, 0°=up)
        plane_intensities = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point_world = center_world + X[i, j] * v + Y[i, j] * u
                point_homogeneous = np.append(point_world, 1)
                point_voxel = (inv_affine @ point_homogeneous)[:3]

                if (0 <= point_voxel[0] < ct_data.shape[0] and
                    0 <= point_voxel[1] < ct_data.shape[1] and
                    0 <= point_voxel[2] < ct_data.shape[2]):
                    plane_intensities[i, j] = map_coordinates(
                        ct_data, point_voxel.reshape(3, 1),
                        order=1, mode='constant', cval=0
                    )[0]

        # Display plane
        im = ax.imshow(plane_intensities,
                       extent=[-half_size, half_size, -half_size, half_size],
                       cmap='gray', origin='lower', interpolation='bilinear')
        fig.colorbar(im, ax=ax, label='HU', shrink=0.8)

        # Mark center
        ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=3)

        # Draw orientation arrow (polar convention: 0°=up, CW)
        arrow_length = plane_size_mm * 0.3
        peak_angle_rad = np.deg2rad(orient_result.peak_angle_deg)
        arrow_x = arrow_length * np.sin(peak_angle_rad)
        arrow_y = arrow_length * np.cos(peak_angle_rad)
        ax.arrow(0, 0, arrow_x, arrow_y,
                head_width=0.3, head_length=0.2, fc='red', ec='red',
                linewidth=3, alpha=0.8)

        # Sampling radii
        for radius in orient_result.sampling_result.radii_mm:
            circle = Circle((0, 0), radius, fill=False, edgecolor='yellow',
                           linestyle='--', linewidth=1, alpha=0.6)
            ax.add_patch(circle)

        ax.set_xlabel('90\u00b0 (mm)', fontsize=10)
        ax.set_ylabel('0\u00b0 (mm)', fontsize=10)
        ax.set_title(
            f'Marker {labels[mi]} at {orient_result.marker_location_mm:.1f}mm\n'
            f'Angle: {orient_result.peak_angle_deg:.1f}\u00b0  '
            f'Conf: {orient_result.confidence:.2f}',
            fontsize=12, fontweight='bold',
        )
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # --- Bottom row: per-marker polar plots ---
    for mi in range(n_markers):
        ax = fig.add_subplot(gs[1, mi], projection='polar')
        orient_result = orientation_results[mi]
        sampling_result = orient_result.sampling_result
        angles_rad = np.deg2rad(sampling_result.angles_deg)

        # Plot intensity for each radius
        for i, radius in enumerate(sampling_result.radii_mm):
            intensities = sampling_result.intensities[:, i]
            ax.plot(angles_rad, intensities, '-', linewidth=1.5, alpha=0.7,
                    label=f'r={radius:.2f}mm')

        # Plot mean intensity
        ax.plot(angles_rad, sampling_result.mean_intensity_by_angle, 'k-',
               linewidth=2.5, label='Mean', zorder=10)

        # Mark peak
        peak_angle_rad = np.deg2rad(orient_result.peak_angle_deg)
        peak_intensity = np.max(sampling_result.mean_intensity_by_angle)
        ax.plot([peak_angle_rad], [peak_intensity], 'r*', markersize=20,
               label=f'Peak ({orient_result.peak_angle_deg:.1f}\u00b0)', zorder=15)
        r_min = ax.get_ylim()[0]
        ax.annotate('', xy=(peak_angle_rad, peak_intensity),
                   xytext=(peak_angle_rad, r_min),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.7))

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'Marker {labels[mi]} - Circular Profile',
                    fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- 3rd column: comparison plots (only when 2 markers) ---
    if has_comparison:
        r0 = orientation_results[0]
        r1 = orientation_results[1]
        s0 = r0.sampling_result
        s1 = r1.sampling_result

        # Top: dual-axis Cartesian (raw HU)
        ax_cart = fig.add_subplot(gs[0, n_markers])
        ax_cart.plot(s0.angles_deg, s0.mean_intensity_by_angle, '-',
                     color=marker_colors[0], linewidth=2,
                     label=f'Marker {labels[0]}')
        ax_cart_r = ax_cart.twinx()
        ax_cart_r.plot(s1.angles_deg, s1.mean_intensity_by_angle, '-',
                       color=marker_colors[1], linewidth=2,
                       label=f'Marker {labels[1]}')

        # Detected direction lines
        ax_cart.axvline(r0.peak_angle_deg, color=marker_colors[0],
                        linestyle='--', linewidth=2, alpha=0.8)
        ax_cart.axvline(r1.peak_angle_deg, color=marker_colors[1],
                        linestyle='--', linewidth=2, alpha=0.8)

        # Fitted direction lines
        if fitted_angles and len(fitted_angles) >= 2:
            ax_cart.axvline(fitted_angles[0], color=marker_colors[0],
                            linestyle='-.', linewidth=2.5, alpha=0.9)
            ax_cart.axvline(fitted_angles[1], color=marker_colors[1],
                            linestyle='-.', linewidth=2.5, alpha=0.9)

        ax_cart.set_xlabel('Angle (degrees)', fontsize=10)
        ax_cart.set_ylabel(f'Marker {labels[0]} Intensity (HU)', fontsize=10,
                           color=marker_colors[0])
        ax_cart_r.set_ylabel(f'Marker {labels[1]} Intensity (HU)', fontsize=10,
                             color=marker_colors[1])
        ax_cart.tick_params(axis='y', labelcolor=marker_colors[0])
        ax_cart_r.tick_params(axis='y', labelcolor=marker_colors[1])
        ax_cart.set_title('Both Markers - Dual Axis (Raw HU)',
                          fontsize=12, fontweight='bold')
        ax_cart.grid(True, alpha=0.3)

        # Bottom: polar raw intensities (both markers overlaid)
        ax_polar = fig.add_subplot(gs[1, n_markers], projection='polar')
        angles0_rad = np.deg2rad(s0.angles_deg)
        angles1_rad = np.deg2rad(s1.angles_deg)

        ax_polar.plot(angles0_rad, s0.mean_intensity_by_angle, '-',
                      color=marker_colors[0], linewidth=2,
                      label=f'Marker {labels[0]}')
        ax_polar.plot(angles1_rad, s1.mean_intensity_by_angle, '-',
                      color=marker_colors[1], linewidth=2,
                      label=f'Marker {labels[1]}')

        # Detected peaks
        for mi, (res, color) in enumerate(
            zip([r0, r1], marker_colors)
        ):
            peak_rad = np.deg2rad(res.peak_angle_deg)
            peak_int = np.max(res.sampling_result.mean_intensity_by_angle)
            ax_polar.plot([peak_rad], [peak_int], '*', color=color,
                         markersize=20, zorder=15,
                         label=f'{labels[mi]} det: {res.peak_angle_deg:.1f}\u00b0')
            r_min = ax_polar.get_ylim()[0]
            ax_polar.annotate('', xy=(peak_rad, peak_int),
                             xytext=(peak_rad, r_min),
                             arrowprops=dict(arrowstyle='->', color=color,
                                            lw=2.5, alpha=0.7))

        # Fitted direction radial lines
        if fitted_angles and len(fitted_angles) >= 2:
            r_max = max(np.max(s0.mean_intensity_by_angle),
                        np.max(s1.mean_intensity_by_angle)) * 1.05
            for mi, (fa, color) in enumerate(
                zip(fitted_angles, marker_colors)
            ):
                ax_polar.plot([np.deg2rad(fa)] * 2, [0, r_max],
                             '-.', color=color, linewidth=2.5, alpha=0.9,
                             label=f'{labels[mi]} fit: {fa:.1f}\u00b0')

        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_title('Both Markers - Polar (Raw HU)',
                          fontsize=12, fontweight='bold', pad=20)
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax_polar.grid(True, alpha=0.3)

    # Suptitle
    suptitle = f'Marker Orientation Debug - Electrode {electrode_idx + 1}'
    if has_comparison:
        detected_sep = abs(r1.peak_angle_deg - r0.peak_angle_deg)
        if detected_sep > 180:
            detected_sep = 360 - detected_sep
        suptitle += f'\nDetected separation: {detected_sep:.1f}\u00b0'
        if fitted_angles and len(fitted_angles) >= 2:
            fitted_sep = abs(fitted_angles[1] - fitted_angles[0])
            if fitted_sep > 180:
                fitted_sep = 360 - fitted_sep
            suptitle += (f'  |  Fitted separation: {fitted_sep:.1f}\u00b0'
                         f' (constrained 120\u00b0)')
        suptitle += '\nDashed = detected  |  Dash-dot = constrained fit'

    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
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


