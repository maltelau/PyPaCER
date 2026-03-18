"""Oriented CT slice visualization aligned with marker direction."""

import base64
import io
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ...utils.math_helpers import get_orthogonal_vectors, inv_poly_arc_length_3d, polyval3
from ..electrode_sliceview import (
    create_volume_interpolator,
    get_world_to_voxel_transform,
)


def generate_oriented_slice_base64(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    length_mm: float = 20.0,
    width_mm: float = 10.0,
    resolution_mm: float = 0.1,
) -> Optional[str]:
    """Generate a base64-encoded PNG of an oriented CT slice along the trajectory.

    Samples CT intensities along the curved polynomial trajectory at correct
    arc-length intervals, with the perpendicular direction aligned to marker A.
    The y-axis represents true arc-length distance from tip.

    Args:
        ct_data: 3D CT volume array.
        affine: 4x4 NIfTI affine matrix.
        electrode: Electrode dict with ``polynomial`` and ``orientation`` keys.
        length_mm: Length along trajectory to show (from tip).
        width_mm: Width of the slice perpendicular to trajectory.
        resolution_mm: Sampling resolution in mm.

    Returns:
        Base64-encoded PNG string, or None if data is insufficient.
    """
    polynomial = electrode.get("polynomial")
    if polynomial is None:
        return None
    polynomial = np.array(polynomial)

    # Get marker A direction vector (if available) for oriented slice
    orientation = electrode.get("orientation", {})
    markers = orientation.get("markers", {})
    marker_a = markers.get("A", {})
    direction_raw = marker_a.get("direction_vector")
    direction = np.array(direction_raw, dtype=float) if direction_raw is not None else None

    # Polynomial derivative coefficients for tangent computation
    deriv_coeffs = polynomial[:-1] * np.arange(
        len(polynomial) - 1, 0, -1
    )[:, np.newaxis]

    # Set up CT volume sampling
    interpolator = create_volume_interpolator(ct_data)
    world_to_voxel = get_world_to_voxel_transform(affine)

    # Perpendicular offsets (horizontal axis of the image)
    u_offsets = np.arange(-width_mm / 2, width_mm / 2 + resolution_mm, resolution_mm)
    n_perp = len(u_offsets)

    # Arc-length distances from tip (vertical axis of the image)
    d_values = np.arange(0, length_mm + resolution_mm, resolution_mm)
    n_along = len(d_values)

    # Build intensity map row-by-row along the curved trajectory
    intensity_map = np.zeros((n_along, n_perp))

    for i, d_mm in enumerate(d_values):
        # Point on the curved trajectory at this arc-length distance from tip
        t = inv_poly_arc_length_3d(polynomial, d_mm)
        center = polyval3(polynomial, t)

        # Local tangent at this point
        local_tangent = polyval3(deriv_coeffs, t)
        local_tangent = local_tangent / np.linalg.norm(local_tangent)

        # Perpendicular direction: use marker A if available, else arbitrary orthogonal
        if direction is not None:
            u_local = direction - np.dot(direction, local_tangent) * local_tangent
            u_norm = np.linalg.norm(u_local)
            if u_norm < 1e-6:
                u_local, _ = get_orthogonal_vectors(local_tangent)
            else:
                u_local = u_local / u_norm
        else:
            u_local, _ = get_orthogonal_vectors(local_tangent)

        # Sample 1D line of intensities perpendicular to trajectory
        sample_points = center[np.newaxis, :] + u_offsets[:, np.newaxis] * u_local
        voxel_coords = world_to_voxel(sample_points)
        intensities = interpolator(voxel_coords)
        intensity_map[i, :] = intensities

    # Portrait figure: x=marker A direction, y=distance from tip (tip at bottom)
    half_width = width_mm / 2
    fig, ax = plt.subplots(figsize=(2, 5))

    im = ax.imshow(
        intensity_map,
        cmap="turbo",
        aspect="auto",
        extent=[-half_width, half_width, 0, length_mm],
        origin="lower",
        interpolation="bilinear",
    )

    # Overlay: trajectory centerline (vertical at x=0)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.4)

    # Overlay: contact positions — direct mm-from-tip mapping
    contact_positions = electrode.get("contact_positions", [])
    for j, pos_mm in enumerate(contact_positions):
        if 0 <= pos_mm <= length_mm:
            ax.axhline(
                pos_mm, color="white", linewidth=0.8,
                linestyle="--", alpha=0.6,
            )
            ax.text(
                half_width * 0.92, pos_mm,
                f"C{j + 1}", color="white", fontsize=8,
                ha="right", va="bottom", fontweight="bold",
            )

    # Overlay: marker positions — direct mm-from-tip mapping
    marker_colors = {"B": "white", "A": "white"}
    for label in ("B", "A"):
        m = markers.get(label)
        if m is None:
            continue
        dist = m.get("distance_from_tip_mm")
        if dist is None:
            continue
        if 0 <= dist <= length_mm:
            color = marker_colors.get(label, "yellow")
            ax.axhline(
                dist, color=color, linewidth=1.2,
                linestyle="-.", alpha=0.8,
            )
            ax.text(
                -half_width * 0.92, dist,
                f"M{label}", color=color, fontsize=9,
                ha="left", va="bottom", fontweight="bold",
            )

    ax.set_title("OOR", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Render to base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_axial_slice_base64(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
) -> Optional[str]:
    """Generate a base64-encoded PNG of a full axial CT slice at the contact region center.

    Extracts the full transverse (axial) slice from the CT volume at the
    z-coordinate of the contact region midpoint. Marker direction vectors
    (if available) are overlaid as arrows projected onto the axial plane.

    Args:
        ct_data: 3D CT volume array.
        affine: 4x4 NIfTI affine matrix.
        electrode: Electrode dict with ``polynomial`` and ``contact_positions``.

    Returns:
        Base64-encoded PNG string, or None if data is insufficient.
    """
    polynomial = electrode.get("polynomial")
    contact_positions = electrode.get("contact_positions", [])
    if polynomial is None or len(contact_positions) < 1:
        return None
    polynomial = np.array(polynomial)

    # Contact region center in mm from tip
    center_mm = (contact_positions[0] + contact_positions[-1]) / 2.0

    # Find 3D world coordinate at contact center
    t_center = inv_poly_arc_length_3d(polynomial, center_mm)
    center_point = polyval3(polynomial, t_center)

    # Compute volume extent in world x-y to determine sampling grid size
    nx, ny, nz = ct_data.shape
    all_corners_voxel = np.array([
        [0, 0, 0, 1], [nx, 0, 0, 1], [0, ny, 0, 1], [nx, ny, 0, 1],
        [0, 0, nz, 1], [nx, 0, nz, 1], [0, ny, nz, 1], [nx, ny, nz, 1],
    ], dtype=float)
    all_corners_world = (affine @ all_corners_voxel.T).T[:, :3]
    x_min, x_max = all_corners_world[:, 0].min(), all_corners_world[:, 0].max()
    y_min, y_max = all_corners_world[:, 1].min(), all_corners_world[:, 1].max()
    width_x = x_max - x_min
    width_y = y_max - y_min

    # Sample an axial plane at the electrode z-level using world coordinates
    # x-axis = world X, y-axis = world Y, fixed z = center_point[2]
    resolution = 0.5  # mm per pixel (coarser than trajectory slice for speed)
    x_coords = np.arange(x_min, x_max + resolution, resolution)
    y_coords = np.arange(y_min, y_max + resolution, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Build 3D world points at the electrode's z-level
    zz = np.full_like(xx, center_point[2])
    world_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Convert to voxel coordinates and sample
    interpolator = create_volume_interpolator(ct_data)
    world_to_voxel = get_world_to_voxel_transform(affine)
    voxel_pts = world_to_voxel(world_pts)
    intensities = interpolator(voxel_pts)
    axial_slice = intensities.reshape(xx.shape)

    # Square figure
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(
        axial_slice,
        cmap="gray",
        aspect="equal",
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        interpolation="bilinear",
    )

    # Overlay: marker direction arrows projected onto axial (x-y) plane
    orientation = electrode.get("orientation", {})
    markers = orientation.get("markers", {})
    marker_colors = {"B": "#1f77b4", "A": "#ff7f0e"}
    # Arrow length proportional to image size (~8% of the smaller dimension)
    arrow_length_mm = min(width_x, width_y) * 0.08

    for label in ("B", "A"):
        m = markers.get(label)
        if m is None:
            continue
        dir_raw = m.get("direction_vector")
        if dir_raw is None:
            continue
        direction = np.array(dir_raw, dtype=float)

        # Project onto axial (x-y) plane by zeroing z component
        d_proj = direction.copy()
        d_proj[2] = 0.0
        d_norm = np.linalg.norm(d_proj)
        if d_norm < 1e-6:
            continue
        d_proj = d_proj / d_norm

        color = marker_colors.get(label, "yellow")
        tip_x = center_point[0] + d_proj[0] * arrow_length_mm
        tip_y = center_point[1] + d_proj[1] * arrow_length_mm

        # Draw arrow from electrode center to marker direction
        ax.annotate(
            "",
            xy=(tip_x, tip_y),
            xytext=(center_point[0], center_point[1]),
            arrowprops=dict(
                arrowstyle="->,head_width=0.4,head_length=0.3",
                color=color,
                lw=2.5,
            ),
        )
        # Place label at arrow tip, offset outward
        label_offset = arrow_length_mm * 0.3
        ax.text(
            tip_x + d_proj[0] * label_offset,
            tip_y + d_proj[1] * label_offset,
            f"M{label}", color=color, fontsize=11,
            fontweight="bold", ha="center", va="center",
        )

    # Anatomical L/R labels (neurological convention: viewed from above)
    y_mid = (y_min + y_max) / 2
    margin = (x_max - x_min) * 0.02
    ax.text(
        x_min + margin, y_mid, "L", color="white", fontsize=12,
        fontweight="bold", ha="left", va="center",
    )
    ax.text(
        x_max - margin, y_mid, "R", color="white", fontsize=12,
        fontweight="bold", ha="right", va="center",
    )

    ax.set_title("Axial CT slice", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
