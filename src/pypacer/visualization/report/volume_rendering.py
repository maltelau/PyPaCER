"""PyVista volume rendering for electrode visualization."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .subvolume_extraction import (
    extract_contacts_subvolume,
    extract_electrode_subvolume,
)


def generate_volume_rendering_html(data: Dict[str, Any]) -> str:
    """Generate HTML content for volume rendering tab."""

    try:
        # Try to import PyVista
        import nibabel as nib
        import pyvista as pv

        pyvista_available = True
    except ImportError:
        pyvista_available = False

    if not pyvista_available:
        return """
        <div class="plot-section">
            <h2>Volume Rendering</h2>
            <div style="padding: 20px; text-align: center; color: #666;">
                <p>PyVista is not installed. Volume rendering requires PyVista.</p>
                <p>Install with: <code>pip install pyvista</code></p>
            </div>
        </div>
        """

    # Check if CT file path is available
    ct_path = data.get("metadata", {}).get("ct_file")

    if not ct_path or not Path(ct_path).exists():
        return """
        <div class="plot-section">
            <h2>Volume Rendering</h2>
            <div style="padding: 20px; text-align: center; color: #666;">
                <p>CT data file not found. Volume rendering requires access to the original CT scan.</p>
                <p>Expected file: {}</p>
            </div>
        </div>
        """.format(
            ct_path if ct_path else "No CT file path in metadata"
        )

    # Generate volume renderings for each electrode
    html = """
    <div class="plot-section">
        <h2>Volume Rendering</h2>
        <p style="color: #666; margin-bottom: 20px;">
            3D volume renderings of the CT data around each electrode.
            Each rendering shows the electrode subvolume with contact positions marked in red.
        </p>
    """

    try:
        # Load CT data once
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        affine = ct_nii.affine

        # Process each electrode
        for i, electrode in enumerate(data["electrodes"]):
            electrode_type = electrode.get("electrode_type", "Unknown")
            hemisphere = electrode.get("side", "Unknown").capitalize()

            # Generate volume rendering
            volume_result = _generate_electrode_volume_rendering(
                ct_data, affine, electrode, i, electrode_type
            )

            if volume_result:
                if isinstance(volume_result, tuple):
                    static_image, contacts_gif = volume_result

                    html += f"""
                    <div class="electrode-card">
                        <h4>Electrode {i+1} - {electrode_type} - {hemisphere}</h4>
                        <div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
                            <div style="text-align: center;">
                                <p style="margin: 5px 0; font-weight: bold;">Full Volume</p>
                                <div class="volume-image-container">
                                    <img src="data:image/png;base64,{static_image}"
                                         style="max-width: 600px; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                                    <button class="maximize-btn" title="Click to maximize">
                                        <svg viewBox="0 0 24 24" fill="currentColor">
                                            <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                                        </svg>
                                        Maximize
                                    </button>
                                </div>
                            </div>
                    """

                    if contacts_gif:
                        html += f"""
                            <div style="text-align: center;">
                                <p style="margin: 5px 0; font-weight: bold;">Contacts Region (Rotating)</p>
                                <div style="display: inline-block;">
                                    <img src="data:image/gif;base64,{contacts_gif}"
                                         style="max-width: 600px; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                                </div>
                            </div>
                        """

                    html += """
                        </div>
                    </div>
                    """
                else:
                    # Only static image
                    html += f"""
                    <div class="electrode-card">
                        <h4>Electrode {i+1} - {electrode_type} - {hemisphere}</h4>
                        <div style="text-align: center;">
                            <div class="volume-image-container">
                                <img src="data:image/png;base64,{volume_result}"
                                     style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                                <button class="maximize-btn" title="Click to maximize">
                                    <svg viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                                    </svg>
                                    Maximize
                                </button>
                            </div>
                        </div>
                    </div>
                    """
            else:
                html += f"""
                <div class="electrode-card">
                    <h4>Electrode {i+1} - {electrode_type} - {hemisphere}</h4>
                    <p style="color: #666; text-align: center;">
                        Failed to generate volume rendering for this electrode.
                    </p>
                </div>
                """

    except Exception as e:
        html += f"""
        <div style="padding: 20px; text-align: center; color: #666;">
            <p>Error loading CT data: {str(e)}</p>
        </div>
        """

    html += """
    </div>
    """

    return html


def _generate_electrode_volume_rendering(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    electrode_idx: int,
    electrode_type: str,
) -> Optional[Union[str, Tuple[str, str]]]:
    """Generate PyVista volume renderings for a single electrode and return as base64 images."""

    try:
        import pyvista as pv
        from PIL import Image

        # Extract subvolume
        subvolume, sub_affine, bbox = extract_electrode_subvolume(
            ct_data, affine, electrode, padding_mm=5.0
        )

        if subvolume is None or subvolume.size == 0:
            return None

        # Create PyVista grid using corrected data and affine
        # Note: subvolume data is already flipped for negative axes
        # and sub_affine already has positive spacing and correct origin
        grid = pv.ImageData(
            dimensions=(
                subvolume.shape[0] + 1,
                subvolume.shape[1] + 1,
                subvolume.shape[2] + 1,
            )
        )

        # Extract spacing from the corrected affine (should be positive)
        spacing = np.abs(np.diag(sub_affine[:3, :3]))
        grid.spacing = spacing

        # Set origin directly from corrected affine (already at voxel corner)
        grid.origin = sub_affine[:3, 3]

        # Add CT data - no need to flip as it's already corrected
        grid.cell_data["CT"] = subvolume.flatten(order="F")
        grid = grid.cell_data_to_point_data()

        # Generate two images
        images = []

        # Image 1: Full volume view
        plotter1 = pv.Plotter(off_screen=True, window_size=[1600, 1600])
        plotter1.background_color = "white"

        # Add volume
        opacity = [0, 0.0, 0.2, 0.0, 0.5, 0.2, 0.6, 0.2]
        volume_actor1 = plotter1.add_volume(
            grid,
            scalars="CT",
            cmap="viridis",
            opacity=opacity,
            clim=[500, 3000],
            opacity_unit_distance=1.0,
            shade=True,
            show_scalar_bar=False,
        )

        # Set volume properties
        volume_prop1 = volume_actor1.GetProperty()
        volume_prop1.SetInterpolationTypeToLinear()
        volume_prop1.SetAmbient(0.2)
        volume_prop1.SetDiffuse(0.8)
        volume_prop1.SetSpecular(0.2)
        volume_prop1.SetSpecularPower(10)

        # Add contact spheres
        contacts_3d = None
        if "contact_positions_3d" in electrode:
            contacts_3d = np.array(electrode["contact_positions_3d"])
            for i, contact in enumerate(contacts_3d):
                sphere = pv.Sphere(center=contact, radius=0.25)
                plotter1.add_mesh(sphere, color="red", opacity=1.0)

        # Add polynomial trajectory
        if "polynomial" in electrode:
            from ...utils.math_helpers import polyval3

            polynomial = np.array(electrode["polynomial"])

            # Generate points along the polynomial
            t_values = np.linspace(0.0, 1.0, 200)
            trajectory_points = np.array([polyval3(polynomial, t) for t in t_values])

            # Create a polyline from the trajectory points
            poly = pv.PolyData(trajectory_points)
            lines = np.arange(0, len(trajectory_points), dtype=np.int_)
            lines = np.column_stack([np.full(len(lines) - 1, 2), lines[:-1], lines[1:]])
            poly.lines = lines

            # Add the trajectory as a tube for better visibility
            tube = poly.tube(radius=0.15)
            plotter1.add_mesh(tube, color="lightblue", opacity=0.8)

        # Set camera for full volume view
        electrode_axis = None
        if (
            "contact_positions_3d" in electrode
            and len(electrode["contact_positions_3d"]) >= 2
        ):
            contacts_3d = np.array(electrode["contact_positions_3d"])

            # Calculate electrode axis
            first_contact = contacts_3d[0]
            last_contact = contacts_3d[-1]
            electrode_axis = last_contact - first_contact
            electrode_axis = electrode_axis / np.linalg.norm(electrode_axis)

            # Center of full bounding box
            center_full = (np.array(bbox["min"]) + np.array(bbox["max"])) / 2.0

            # Keep Z-axis as up vector
            up_vector = np.array([0, 0, 1])

            # Find view direction perpendicular to electrode axis
            electrode_xy = electrode_axis.copy()
            electrode_xy[2] = 0  # Remove Z component

            if np.linalg.norm(electrode_xy) > 0.1:
                perp_vector = np.cross(up_vector, electrode_xy)
                perp_vector = perp_vector / np.linalg.norm(perp_vector)
            else:
                perp_vector = np.array([1, 0, 0])

            # Camera distance for full view
            bbox_diagonal = np.linalg.norm(bbox["size"])
            distance_full = (
                bbox_diagonal * 4.5
            )  # Increased from 3.0 to show full electrode
            camera_pos_full = center_full + perp_vector * distance_full

            plotter1.camera_position = [camera_pos_full, center_full, up_vector]
        else:
            plotter1.camera_position = "iso"

        plotter1.show_axes()
        plotter1.reset_camera()  # Reset camera to show full scene
        plotter1.reset_camera_clipping_range()

        # Render first image
        plotter1.show(auto_close=False)
        image1 = plotter1.screenshot()
        plotter1.close()

        # Encode static full-volume image
        img1 = Image.fromarray(image1)
        buffer = io.BytesIO()
        img1.save(buffer, format="PNG")
        buffer.seek(0)
        img1_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Image 2: Zoomed contacts view with electrode pointing up
        if (
            contacts_3d is not None
            and electrode_axis is not None
            and len(contacts_3d) >= 2
        ):
            result = _generate_contacts_rotating_view(
                ct_data, affine, electrode, contacts_3d, electrode_axis,
                image1, opacity,
            )
            if result is not None:
                return result

        return img1_base64

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None


def _generate_contacts_rotating_view(
    ct_data, affine, electrode, contacts_3d, electrode_axis, image1, opacity,
):
    """Generate the zoomed contacts view with rotating GIF."""
    import pyvista as pv
    from PIL import Image

    # Collect marker positions to extend the subvolume if orientation data exists
    orientation = electrode.get("orientation", {})
    markers = orientation.get("markers", {})
    marker_positions = []
    for marker_data in markers.values():
        pos = marker_data.get("position_xyz")
        if pos is not None:
            marker_positions.append(pos)
    extra_pos = np.array(marker_positions) if marker_positions else None

    # Extract contact region subvolume (extended to include markers if present)
    contacts_subvolume, contacts_sub_affine, contacts_bbox = (
        extract_contacts_subvolume(
            ct_data, affine, electrode, padding_mm=2.0,
            extra_positions=extra_pos,
        )
    )

    if contacts_subvolume is None:
        return None

    # Create new grid for contacts region
    contacts_grid = pv.ImageData(
        dimensions=(
            contacts_subvolume.shape[0] + 1,
            contacts_subvolume.shape[1] + 1,
            contacts_subvolume.shape[2] + 1,
        )
    )
    contacts_spacing = np.abs(np.diag(contacts_sub_affine[:3, :3]))
    contacts_grid.spacing = contacts_spacing

    # Set origin directly from corrected affine (already at voxel corner)
    contacts_grid.origin = contacts_sub_affine[:3, 3]

    # Add CT data - no need to flip as it's already corrected
    contacts_grid.cell_data["CT"] = contacts_subvolume.flatten(order="F")
    contacts_grid = contacts_grid.cell_data_to_point_data()

    plotter2 = pv.Plotter(off_screen=True, window_size=[600, 600])
    plotter2.background_color = "white"

    # Add volume
    volume_actor2 = plotter2.add_volume(
        contacts_grid,
        scalars="CT",
        cmap="viridis",
        opacity=opacity,
        clim=[500, 3000],
        opacity_unit_distance=1.0,
        shade=True,
        show_scalar_bar=False,
    )

    # Set volume properties
    volume_prop2 = volume_actor2.GetProperty()
    volume_prop2.SetInterpolationTypeToLinear()
    volume_prop2.SetAmbient(0.2)
    volume_prop2.SetDiffuse(0.8)
    volume_prop2.SetSpecular(0.2)
    volume_prop2.SetSpecularPower(10)

    # Add contact spheres
    for i, contact in enumerate(contacts_3d):
        sphere = pv.Sphere(center=contact, radius=0.25)
        plotter2.add_mesh(sphere, color="red", opacity=1.0)

    # Add polynomial trajectory
    if "polynomial" in electrode:
        from ...utils.math_helpers import polyval3

        polynomial = np.array(electrode["polynomial"])

        # Generate points along the polynomial (focus on contact region)
        # Use finer sampling for zoomed view
        t_values = np.linspace(0.0, 1.0, 300)
        trajectory_points = np.array(
            [polyval3(polynomial, t) for t in t_values]
        )

        # Filter to points within the contacts bounding box
        in_bbox = np.all(
            (trajectory_points >= contacts_bbox["min"])
            & (trajectory_points <= contacts_bbox["max"]),
            axis=1,
        )
        trajectory_points_filtered = trajectory_points[in_bbox]

        if len(trajectory_points_filtered) > 1:
            # Create a polyline from the trajectory points
            poly = pv.PolyData(trajectory_points_filtered)
            lines = np.arange(
                0, len(trajectory_points_filtered), dtype=np.int_
            )
            lines = np.column_stack(
                [np.full(len(lines) - 1, 2), lines[:-1], lines[1:]]
            )
            poly.lines = lines

            # Add the trajectory as a tube
            tube = poly.tube(radius=0.15)
            plotter2.add_mesh(tube, color="lightblue", opacity=0.8)

    # Add marker positions and direction vectors if orientation data exists
    marker_colors_map = {"B": "#1f77b4", "A": "#ff7f0e"}
    for label, m_data in markers.items():
        pos = m_data.get("position_xyz")
        if pos is None:
            continue
        pos = np.array(pos)

        # Marker sphere
        color = marker_colors_map.get(label, "yellow")
        sphere = pv.Sphere(center=pos, radius=0.3)
        plotter2.add_mesh(sphere, color=color, opacity=1.0)

        # Direction arrow
        direction = m_data.get("direction_vector")
        if direction is not None:
            direction = np.array(direction)
            arrow = pv.Arrow(
                start=pos,
                direction=direction,
                tip_length=0.3,
                tip_radius=0.12,
                shaft_radius=0.05,
                scale=2.0,
            )
            plotter2.add_mesh(arrow, color=color, opacity=0.9)

    # Center on the full subvolume bounding box (includes markers if present)
    center_contacts = (np.array(contacts_bbox["min"]) + np.array(contacts_bbox["max"])) / 2.0

    # For contacts view, electrode axis should point up
    up_vector_contacts = electrode_axis

    # Find best viewing angle
    # Use a vector perpendicular to the electrode axis
    if abs(electrode_axis[0]) < 0.9:
        view_direction = np.cross(electrode_axis, np.array([1, 0, 0]))
    elif abs(electrode_axis[1]) < 0.9:
        view_direction = np.cross(electrode_axis, np.array([0, 1, 0]))
    else:
        view_direction = np.cross(electrode_axis, np.array([0, 0, 1]))

    view_direction = view_direction / np.linalg.norm(view_direction)

    # Camera distance for contacts view
    contacts_bbox_diagonal = np.linalg.norm(contacts_bbox["size"])
    distance_contacts = contacts_bbox_diagonal * 3.0
    camera_pos_contacts = (
        center_contacts + view_direction * distance_contacts
    )

    plotter2.camera_position = [
        camera_pos_contacts,
        center_contacts,
        up_vector_contacts,
    ]

    plotter2.show_axes()
    plotter2.reset_camera_clipping_range()

    # Generate rotating GIF for second view
    frames = []
    n_frames = 30

    for frame in range(n_frames):
        # Calculate rotation angle
        angle = (frame / n_frames) * 360.0
        angle_rad = np.radians(angle)

        # Rotate camera position around electrode axis
        # Create rotation matrix around electrode axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rodrigues' rotation formula for rotation around arbitrary axis
        K = np.array(
            [
                [0, -electrode_axis[2], electrode_axis[1]],
                [electrode_axis[2], 0, -electrode_axis[0]],
                [-electrode_axis[1], electrode_axis[0], 0],
            ]
        )

        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)

        # Rotate the view direction
        rotated_view_direction = R @ view_direction
        rotated_camera_pos = (
            center_contacts + rotated_view_direction * distance_contacts
        )

        # Update camera position
        plotter2.camera_position = [
            rotated_camera_pos,
            center_contacts,
            up_vector_contacts,
        ]

        # Render frame
        plotter2.render()
        frame_image = plotter2.screenshot()
        frames.append(Image.fromarray(frame_image))

    plotter2.close()

    # Create GIF from frames
    gif_buffer = io.BytesIO()
    frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,  # 40ms per frame = 4 seconds per rotation
        loop=0,  # Infinite loop
    )
    gif_buffer.seek(0)
    gif_base64 = base64.b64encode(gif_buffer.read()).decode("utf-8")

    # Combine static image with GIF in HTML
    img1 = Image.fromarray(image1)
    buffer = io.BytesIO()
    img1.save(buffer, format="PNG")
    buffer.seek(0)
    img1_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Return both images as a tuple for special handling
    return (img1_base64, gif_base64)


