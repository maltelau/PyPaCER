"""Marker profile visualization tools for orientation detection.

Visualises heatmaps, contour plots, and collapsed 1D profiles from
marker region intensity sampling.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def visualize_full_profile(
    profile_data: dict,
    marker_peak_locations: Optional[List[float]] = None,
    marker_region_center: Optional[float] = None,
    contact_positions: Optional[List[float]] = None,
    fitted_angles: Optional[List[float]] = None,
    detected_angles: Optional[List[float]] = None,
    grid_fitted_angles: Optional[Tuple[float, float]] = None,
    collapsed_profile: Optional[np.ndarray] = None,
    electrode_idx: int = 0,
    output_path: Optional[Path] = None,
    show: bool = False,
    region_label: str = "Marker Region",
) -> Optional[Path]:
    """
    Visualize full profile as heatmap, contour, and collapsed 1D profile.

    Layout (3 rows):
        Row 0: 3D surface (spans both columns)
        Row 1 left:  Unwrapped intensity heatmap
        Row 1 right: Filled contour plot
        Row 2 (full width): Collapsed 1D angular profile from find_peak_from_profile_grid
                            (only shown when collapsed_profile is provided)

    Args:
        profile_data: Output from sample_full_marker_profile
        marker_peak_locations: Detected marker peak positions (mm from tip)
        marker_region_center: Center of marker region (mm from tip)
        contact_positions: Detected contact positions (mm from tip)
        fitted_angles: Fitted marker directions [angle1, angle2] in degrees
        detected_angles: Single-slice detected directions [angle1, angle2] in degrees
        grid_fitted_angles: Two-peak 120-deg-constrained fit from find_peak_from_profile_grid,
            as (phi1_deg, phi2_deg). Both lines are drawn in lime green.
        collapsed_profile: 1D weighted-mean profile from find_peak_from_profile_grid
        electrode_idx: Electrode index for title
        output_path: Path to save figure
        show: Whether to show figure
        region_label: Label for the region (e.g. "Marker Region")

    Returns:
        Path to saved figure if output_path provided
    """
    distances = profile_data["distances"]
    angles_deg = profile_data["angles_deg"]
    intensity_grid = profile_data["intensity_grid"]

    angle_grid, dist_grid = np.meshgrid(angles_deg, distances)

    has_collapsed = collapsed_profile is not None
    n_rows = 3 if has_collapsed else 2
    fig = plt.figure(figsize=(20, 8 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.38, wspace=0.3)

    # ------------------------------------------------------------------
    # Helper: add direction-line overlays to 2D axes
    # ------------------------------------------------------------------
    def _add_overlays(ax):
        if marker_peak_locations:
            for pi, peak_dist in enumerate(marker_peak_locations):
                ax.axhline(peak_dist, color="orange", linestyle="-",
                           linewidth=1.5, alpha=0.7,
                           label=f"Peak {pi+1}: {peak_dist:.1f}mm")
        if contact_positions:
            for ci, contact_dist in enumerate(contact_positions):
                ax.axhline(contact_dist, color="cyan", linestyle="-",
                           linewidth=1.5, alpha=0.7,
                           label=f"Contact {ci+1}: {contact_dist:.1f}mm")
        if marker_region_center is not None:
            ax.axhline(marker_region_center, color="white", linestyle=":",
                       linewidth=1.5, alpha=0.7,
                       label=f"Region center: {marker_region_center:.1f}mm")
        if fitted_angles:
            for ai, angle in enumerate(fitted_angles):
                color = "red" if ai == 0 else "magenta"
                ax.axvline(angle, color=color, linestyle="--",
                           linewidth=2, alpha=0.8,
                           label=f"Fitted M{ai+1}: {angle:.1f}\u00b0")
        if detected_angles:
            for ai, angle in enumerate(detected_angles):
                color = "red" if ai == 0 else "magenta"
                ax.axvline(angle, color=color, linestyle="-",
                           linewidth=1.5, alpha=0.6,
                           label=f"Single-slice M{ai+1}: {angle:.1f}\u00b0")
        if grid_fitted_angles is not None:
            for gi, ga in enumerate(grid_fitted_angles):
                ax.axvline(ga, color="lime", linestyle="-.",
                           linewidth=2.5, alpha=0.9,
                           label=f"Grid 120\u00b0 fit M{gi+1}: {ga:.1f}\u00b0")

    # ------------------------------------------------------------------
    # Row 0: 3D surface (span both columns)
    # ------------------------------------------------------------------
    ax3d = fig.add_subplot(gs[0, :], projection="3d")

    surf = ax3d.plot_surface(
        angle_grid, dist_grid, intensity_grid,
        cmap="viridis", alpha=0.85, edgecolor="none",
        rstride=1, cstride=max(1, len(angles_deg) // 72),
    )

    z_floor = intensity_grid.min() - (intensity_grid.max() - intensity_grid.min()) * 0.05
    ax3d.contourf(
        angle_grid, dist_grid, intensity_grid,
        zdir="z", offset=z_floor, cmap="viridis", alpha=0.5, levels=20,
    )

    if fitted_angles:
        for ai, angle in enumerate(fitted_angles):
            ax3d.plot(
                [angle, angle], [distances[0], distances[-1]], [z_floor, z_floor],
                "r--" if ai == 0 else "m--", linewidth=2.5, alpha=0.9,
                label=f"Fitted M{ai+1}: {angle:.1f}\u00b0",
            )
    if detected_angles:
        for ai, angle in enumerate(detected_angles):
            ax3d.plot(
                [angle, angle], [distances[0], distances[-1]], [z_floor, z_floor],
                "r-" if ai == 0 else "m-", linewidth=1.5, alpha=0.7,
                label=f"Single-slice M{ai+1}: {angle:.1f}\u00b0",
            )
    if grid_fitted_angles is not None:
        for gi, ga in enumerate(grid_fitted_angles):
            ax3d.plot(
                [ga, ga], [distances[0], distances[-1]], [z_floor, z_floor],
                color="lime", linestyle="-.", linewidth=2.5, alpha=0.9,
                label=f"Grid 120\u00b0 fit M{gi+1}: {ga:.1f}\u00b0",
            )
    if marker_peak_locations:
        for pi, peak_dist in enumerate(marker_peak_locations):
            if distances[0] <= peak_dist <= distances[-1]:
                ax3d.plot(
                    [0, 360], [peak_dist, peak_dist], [z_floor, z_floor],
                    "orange", linewidth=1.5, alpha=0.6,
                    label=f"Peak {pi+1}: {peak_dist:.1f}mm",
                )

    ax3d.set_xlabel("Angle (\u00b0)", fontsize=11, labelpad=10)
    ax3d.set_ylabel("Distance from tip (mm)", fontsize=11, labelpad=10)
    ax3d.set_zlabel("Intensity (HU)", fontsize=11, labelpad=10)
    ax3d.set_title("3D Intensity Surface", fontsize=13, fontweight="bold")
    ax3d.legend(fontsize=9, loc="upper left")
    ax3d.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=15, label="Intensity (HU)")

    # ------------------------------------------------------------------
    # Row 1 left: 2D heatmap
    # ------------------------------------------------------------------
    ax_heat = fig.add_subplot(gs[1, 0])
    im = ax_heat.pcolormesh(angles_deg, distances, intensity_grid,
                            cmap="viridis", shading="auto")
    _add_overlays(ax_heat)
    ax_heat.set_xlabel("Angle (\u00b0)", fontsize=11)
    ax_heat.set_ylabel("Distance from tip (mm)", fontsize=11)
    ax_heat.set_title("Unwrapped Intensity Heatmap", fontsize=13, fontweight="bold")
    ax_heat.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax_heat, label="Intensity (HU)")

    # ------------------------------------------------------------------
    # Row 1 right: filled contour
    # ------------------------------------------------------------------
    ax_contour = fig.add_subplot(gs[1, 1])
    levels = 30
    cf = ax_contour.contourf(angles_deg, distances, intensity_grid,
                             levels=levels, cmap="viridis")
    ax_contour.contour(angles_deg, distances, intensity_grid,
                       levels=levels, colors="black", linewidths=0.3, alpha=0.3)
    _add_overlays(ax_contour)
    ax_contour.set_xlabel("Angle (\u00b0)", fontsize=11)
    ax_contour.set_ylabel("Distance from tip (mm)", fontsize=11)
    ax_contour.set_title("Filled Contour Plot", fontsize=13, fontweight="bold")
    fig.colorbar(cf, ax=ax_contour, label="Intensity (HU)")

    # ------------------------------------------------------------------
    # Row 2 (full width): collapsed 1D profile
    # ------------------------------------------------------------------
    if has_collapsed:
        ax_1d = fig.add_subplot(gs[2, :])
        ax_1d.plot(angles_deg, collapsed_profile, "k-", linewidth=2,
                   label="Weighted collapsed profile")

        # Shade the row weights as a visual guide on a twin y-axis
        row_peaks = intensity_grid.max(axis=1)
        gate = np.percentile(row_peaks, 50.0)
        row_weights = np.maximum(row_peaks - gate, 0.0)
        n_used = int(np.sum(row_weights > 0))
        n_total = len(distances)

        if fitted_angles:
            for ai, angle in enumerate(fitted_angles):
                color = "red" if ai == 0 else "magenta"
                ax_1d.axvline(angle, color=color, linestyle="--",
                              linewidth=2, alpha=0.8,
                              label=f"Fitted M{ai+1}: {angle:.1f}\u00b0")
        if detected_angles:
            for ai, angle in enumerate(detected_angles):
                color = "red" if ai == 0 else "magenta"
                ax_1d.axvline(angle, color=color, linestyle="-",
                              linewidth=1.5, alpha=0.6,
                              label=f"Single-slice M{ai+1}: {angle:.1f}\u00b0")
        if grid_fitted_angles is not None:
            for gi, ga in enumerate(grid_fitted_angles):
                ax_1d.axvline(ga, color="lime", linestyle="-.",
                              linewidth=2.5, alpha=0.9,
                              label=f"Grid 120\u00b0 fit M{gi+1}: {ga:.1f}\u00b0")

        # Mark the half-max threshold used inside _calculate_angular_peak_center
        p_max = collapsed_profile.max()
        p_min = collapsed_profile.min()
        threshold = (p_max + p_min) / 2.0
        ax_1d.axhline(threshold, color="gray", linestyle=":", linewidth=1.2,
                      alpha=0.7, label=f"50% threshold ({threshold:.0f} HU)")

        ax_1d.set_xlim(0, 360)
        ax_1d.set_xlabel("Angle (\u00b0)", fontsize=11)
        ax_1d.set_ylabel("Weighted intensity (HU)", fontsize=11)
        ax_1d.set_title(
            f"Collapsed 1D Profile  \u2013  {n_used}/{n_total} slices above median row peak",
            fontsize=13, fontweight="bold",
        )
        ax_1d.legend(fontsize=9, loc="upper right")
        ax_1d.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------
    subtitle = (
        f"Distance: {distances[0]:.1f}\u2013{distances[-1]:.1f}mm  |  "
        f"Step: {distances[1] - distances[0]:.2f}mm  |  "
        f"{len(distances)} slices \u00d7 {len(angles_deg)} angles"
    )
    if grid_fitted_angles is not None:
        subtitle += f"  |  Grid 120\u00b0 fit: {grid_fitted_angles[0]:.1f}\u00b0 / {grid_fitted_angles[1]:.1f}\u00b0"
    fig.suptitle(
        f"Full {region_label} Profile - Electrode {electrode_idx + 1}\n{subtitle}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    if show:
        plt.show()
    else:
        plt.close()

    return None
