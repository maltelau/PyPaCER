"""3D electrode visualization using Plotly traces."""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from ..isosurface_extraction import extract_electrode_mesh


def add_3d_visualization_direct(
    fig: go.Figure, data: Dict[str, Any], cached_mesh: Optional[go.Mesh3d] = None
):
    """Add 3D electrode visualization directly to figure without subplots."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Try to extract metal mesh if CT data is available
    ct_path = data.get("metadata", {}).get("ct_file")
    if ct_path and Path(ct_path).exists():
        # Get output directory from context if available
        output_dir = None
        if hasattr(add_3d_visualization_direct, "_output_path"):
            output_dir = Path(add_3d_visualization_direct._output_path).parent

        # Extract electrode metal mesh
        try:
            if cached_mesh is not None:
                # Use the pre-computed mesh
                print("Using cached electrode metal mesh for HTML report")
                fig.add_trace(cached_mesh)
            else:
                # Extract mesh on demand
                print("Extracting electrode metal mesh...")
                # Pass electrode data if available for contact-based component selection
                electrode_data = None
                if "electrodes" in data and len(data["electrodes"]) > 0:
                    # Combine all contact positions from all electrodes
                    all_contacts = []
                    for electrode in data["electrodes"]:
                        if "contact_positions_3d" in electrode:
                            all_contacts.extend(electrode["contact_positions_3d"])

                    if all_contacts:
                        electrode_data = {
                            "contact_positions_3d": all_contacts,
                            "electrodes": data["electrodes"],
                        }

                electrode_mesh = extract_electrode_mesh(
                    ct_path, output_dir, electrode_data=electrode_data
                )
                if electrode_mesh:
                    print("Successfully extracted electrode metal mesh")
                    fig.add_trace(electrode_mesh)
                else:
                    print("Electrode metal mesh extraction returned None")
        except Exception as e:
            print(f"Failed to extract electrode metal mesh: {str(e)}")
            warnings.warn(f"Failed to extract electrode metal mesh: {str(e)}")

    # Add CT volume bounding box if available
    if "metadata" in data and "ct_volume_bounding_box" in data["metadata"]:
        ct_bbox = data["metadata"]["ct_volume_bounding_box"]
        add_bounding_box_direct(
            fig, ct_bbox["min"], ct_bbox["max"], color="gray", name="CT Volume"
        )

    for i, electrode in enumerate(data["electrodes"]):
        color = colors[i % len(colors)]

        # Plot electrode trajectory
        if "trajectory_coordinates" in electrode:
            trajectory = np.array(electrode["trajectory_coordinates"])

            # Convert numpy arrays to lists to prevent JSON serialization issues
            x_list = trajectory[:, 0].tolist()
            y_list = trajectory[:, 1].tolist()
            z_list = trajectory[:, 2].tolist()

            hemisphere = electrode.get("side", "Unknown").capitalize()
            fig.add_trace(
                go.Scatter3d(
                    x=x_list,
                    y=y_list,
                    z=z_list,
                    mode="lines",
                    name=f"Electrode {i+1} ({hemisphere})",
                    line=dict(color=color, width=6),
                    showlegend=True,
                    legendgroup=f"electrode{i}",
                )
            )

        # Plot polynomial before tip detection (if available in debug mode)
        if "polynomial_before_tip_detection" in electrode:
            from ...utils.math_helpers import polyval3

            polynomial_before_tip = np.array(
                electrode["polynomial_before_tip_detection"]
            )

            # Generate points along the polynomial
            # Only show from t=0 to t=1 (exclude lookahead region)
            t_values = np.linspace(0.0, 1.0, 500)
            trajectory_before_tip = np.array(
                [polyval3(polynomial_before_tip, t) for t in t_values]
            )

            # Convert to lists
            x_list_before = trajectory_before_tip[:, 0].tolist()
            y_list_before = trajectory_before_tip[:, 1].tolist()
            z_list_before = trajectory_before_tip[:, 2].tolist()

            # Add solid line for polynomial before tip detection as separate entity
            # Use a different color for clear distinction
            # Create complementary color by shifting hue
            pretip_colors = [
                "#d62728",
                "#2ca02c",
                "#ff7f0e",
                "#9467bd",
                "#8c564b",
                "#1f77b4",
            ]
            pretip_color = pretip_colors[i % len(pretip_colors)]

            fig.add_trace(
                go.Scatter3d(
                    x=x_list_before,
                    y=y_list_before,
                    z=z_list_before,
                    mode="lines",
                    name=f"Pre-Tip Polynomial {i+1}",
                    line=dict(
                        color=pretip_color,
                        width=5,
                        dash="solid",  # Solid line for clarity
                    ),
                    showlegend=True,
                    legendgroup=f"pretip{i}",  # Separate legend group
                    opacity=0.8,
                    hovertemplate=(
                        "<b>Pre-Tip Polynomial %{text}</b><br>"
                        + "X: %{x:.1f} mm<br>"
                        + "Y: %{y:.1f} mm<br>"
                        + "Z: %{z:.1f} mm<br>"
                        + "<extra></extra>"
                    ),
                    text=[f"{i+1}"] * len(t_values),
                )
            )

            # Add a marker at t=0 on the pre-tip polynomial to show where tip detection occurred
            tip_point = polyval3(polynomial_before_tip, 0.0)
            fig.add_trace(
                go.Scatter3d(
                    x=[tip_point[0]],
                    y=[tip_point[1]],
                    z=[tip_point[2]],
                    mode="markers",
                    name=f"Tip Detection Point {i+1}",
                    marker=dict(
                        color="red",
                        size=8,
                        symbol="x",
                        line=dict(color="darkred", width=2),
                    ),
                    showlegend=True,
                    legendgroup=f"pretip{i}",
                    hovertemplate=(
                        "<b>Tip Detection Point</b><br>"
                        + "X: %{x:.1f} mm<br>"
                        + "Y: %{y:.1f} mm<br>"
                        + "Z: %{z:.1f} mm<br>"
                        + "This is where t=0 was on the<br>"
                        + "polynomial before re-zeroing<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Plot contact positions
        if "contact_positions_3d" in electrode:
            contacts = np.array(electrode["contact_positions_3d"])

            # Convert numpy arrays to lists to prevent JSON serialization issues
            x_list = contacts[:, 0].tolist()
            y_list = contacts[:, 1].tolist()
            z_list = contacts[:, 2].tolist()

            fig.add_trace(
                go.Scatter3d(
                    x=x_list,
                    y=y_list,
                    z=z_list,
                    mode="markers+text",
                    name=f"Contacts {i+1} ({hemisphere})",
                    marker=dict(
                        color=color,
                        size=8,
                        symbol="circle",
                        line=dict(color="white", width=2),
                    ),
                    text=[f"C{j+1}" for j in range(len(contacts))],
                    textposition="top center",
                    showlegend=False,
                    legendgroup=f"electrode{i}",
                )
            )

        # Plot orientation marker direction vectors
        if "orientation" in electrode and "markers" in electrode["orientation"]:
            _add_marker_direction_vectors(fig, electrode, i)

        # Plot bounding box
        if "bounding_box" in electrode:
            bbox = electrode["bounding_box"]
            add_bounding_box_direct(
                fig, bbox["min"], bbox["max"], color=color, name=f"Bbox {i+1}"
            )


def add_bounding_box_direct(
    fig: go.Figure,
    min_coords: List[float],
    max_coords: List[float],
    color: str,
    name: str,
):
    """Add a wireframe bounding box to the 3D plot without subplots."""
    edges = _get_bounding_box_edges(min_coords, max_coords)

    # Add each edge as a line trace
    for i, (x, y, z) in enumerate(edges):
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=(i == 0),  # Only show legend for first edge
                name=name if i == 0 else None,
                legendgroup=name,  # Group all edges together
                hoverinfo="skip",
            )
        )


def add_3d_visualization(fig: go.Figure, data: Dict[str, Any], row: int, col: int):
    """Add 3D electrode visualization to subplot."""

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Add CT volume bounding box if available
    if "metadata" in data and "ct_volume_bounding_box" in data["metadata"]:
        ct_bbox = data["metadata"]["ct_volume_bounding_box"]
        add_bounding_box(
            fig,
            ct_bbox["min"],
            ct_bbox["max"],
            color="gray",
            name="CT Volume",
            row=row,
            col=col,
        )

    for i, electrode in enumerate(data["electrodes"]):
        color = colors[i % len(colors)]

        # Plot electrode trajectory
        if "trajectory_coordinates" in electrode:
            trajectory = np.array(electrode["trajectory_coordinates"])

            # Convert to lists to ensure proper serialization
            x_data = trajectory[:, 0].tolist()
            y_data = trajectory[:, 1].tolist()
            z_data = trajectory[:, 2].tolist()

            hemisphere = electrode.get("side", "Unknown").capitalize()
            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode="lines",
                    name=f"Electrode {i+1} ({hemisphere})",
                    line=dict(color=color, width=6),
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
        else:
            pass

        # Plot contact positions
        if "contact_positions_3d" in electrode:
            contacts = np.array(electrode["contact_positions_3d"])

            # Convert to lists to ensure proper serialization
            x_data = contacts[:, 0].tolist()
            y_data = contacts[:, 1].tolist()
            z_data = contacts[:, 2].tolist()

            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode="markers+text",
                    name=f"Contacts {i+1} ({hemisphere})",
                    marker=dict(
                        color=color,
                        size=8,
                        symbol="circle",
                        line=dict(color="white", width=2),
                    ),
                    text=[f"C{j+1}" for j in range(len(contacts))],
                    textposition="top center",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        else:
            pass

        # Plot bounding box
        if "bounding_box" in electrode:
            bbox = electrode["bounding_box"]
            add_bounding_box(
                fig,
                bbox["min"],
                bbox["max"],
                color=color,
                name=f"Bbox {i+1}",
                row=row,
                col=col,
            )



def add_bounding_box(
    fig: go.Figure,
    min_coords: List[float],
    max_coords: List[float],
    color: str,
    name: str,
    row: int,
    col: int,
):
    """Add a wireframe bounding box to the 3D plot."""
    edges = _get_bounding_box_edges(min_coords, max_coords)

    # Add each edge as a line trace
    for i, (x, y, z) in enumerate(edges):
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=(i == 0),  # Only show legend for first edge
                name=name if i == 0 else None,
                legendgroup=name,  # Group all edges together
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _get_bounding_box_edges(min_coords, max_coords):
    """Return the 12 edges of a bounding box as coordinate tuples."""
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords

    return [
        # Bottom square
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
        ([x_min, x_min], [y_max, y_min], [z_min, z_min]),
        # Top square
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_max, y_min], [z_max, z_max]),
        # Vertical edges
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
    ]


def _add_marker_direction_vectors(fig: go.Figure, electrode: Dict[str, Any], electrode_idx: int):
    """Add 3D direction vectors for detected orientation markers."""
    markers = electrode["orientation"]["markers"]
    arrow_length_mm = 3.0  # Length of the direction arrow

    marker_colors = {
        "A": "#ff7f0e",  # orange
        "B": "#1f77b4",  # blue
        "C": "#2ca02c",  # green
        "D": "#7f7f7f",  # gray
    }

    for label, marker_data in markers.items():
        if "position_xyz" not in marker_data or "direction_vector" not in marker_data:
            continue
        position = np.array(marker_data["position_xyz"])
        direction = np.array(marker_data["direction_vector"])

        # Normalize direction and scale to arrow length
        dir_norm = direction / np.linalg.norm(direction)
        endpoint = position + dir_norm * arrow_length_mm

        color = marker_colors.get(label, "#ff00ff")
        confidence = marker_data.get("detection_confidence", 0)
        angle = marker_data.get("fitted_angle_axial_deg", marker_data.get("detected_angle_traj_perp_deg", 0))

        # Arrow shaft (line from position along direction)
        fig.add_trace(
            go.Scatter3d(
                x=[position[0], endpoint[0]],
                y=[position[1], endpoint[1]],
                z=[position[2], endpoint[2]],
                mode="lines",
                name=f"Marker {label} (E{electrode_idx+1})",
                line=dict(color=color, width=6),
                showlegend=True,
                legendgroup=f"marker_{electrode_idx}_{label}",
                hovertemplate=(
                    f"<b>Marker {label} (Electrode {electrode_idx+1})</b><br>"
                    + f"Confidence: {confidence:.2f}<br>"
                    + f"Axial angle: {angle:.1f}°<br>"
                    + "Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # Arrowhead (cone at the endpoint)
        fig.add_trace(
            go.Cone(
                x=[endpoint[0]],
                y=[endpoint[1]],
                z=[endpoint[2]],
                u=[dir_norm[0]],
                v=[dir_norm[1]],
                w=[dir_norm[2]],
                sizemode="absolute",
                sizeref=1.0,
                showscale=False,
                colorscale=[[0, color], [1, color]],
                showlegend=False,
                legendgroup=f"marker_{electrode_idx}_{label}",
                hoverinfo="skip",
            )
        )
