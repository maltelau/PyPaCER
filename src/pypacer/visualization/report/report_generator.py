"""Generate interactive HTML reports from PyPaCER reconstruction results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from .html_templates import build_3d_plot_html, build_report_html
from .intensity_profiles import generate_summary_html_with_profiles
from .orientation_tab import generate_orientation_html
from .visualization_3d import add_3d_visualization_direct
from .volume_rendering import generate_volume_rendering_html


def generate_html_report_from_data(
    electrodes: List[Any],
    metadata: Dict[str, Any],
    reconstruction_parameters: Dict[str, Any],
    output_path: str,
    include_intensity_profiles: bool = True,
    include_3d_visualization: bool = True,
    include_contact_comparison: bool = True,
    include_volume_rendering: bool = True,
    cached_mesh: Optional[go.Mesh3d] = None,
) -> str:
    """
    Generate an interactive HTML report from PyPaCER electrode data directly.

    Args:
        electrodes: List of PolynomialElectrodeModel objects
        metadata: Metadata dictionary (CT info, etc.)
        reconstruction_parameters: Parameters used for reconstruction
        output_path: Output path for HTML file
        include_intensity_profiles: Include intensity profile plots
        include_3d_visualization: Include 3D electrode visualization
        include_contact_comparison: Include contact detection comparison if available
        include_volume_rendering: Include volume rendering tab with PyVista visualizations
        cached_mesh: Pre-computed electrode mesh (go.Mesh3d) to avoid re-extraction

    Returns:
        Path to the generated HTML file
    """
    # Convert electrodes to dict format
    electrode_dicts = []
    for i, electrode in enumerate(electrodes):
        electrode_data = electrode.to_dict()
        electrode_data["electrode_index"] = i
        electrode_data["contact_positions_3d"] = (
            electrode.get_contact_positions_3d().tolist()
        )

        # Add trajectory coordinates
        if hasattr(electrode, "_skeleton") and electrode._skeleton is not None:
            electrode_data["trajectory_coordinates"] = electrode._skeleton.tolist()
        elif hasattr(electrode, "get_point_at_parameter"):
            # Sample trajectory at regular intervals
            n_points = 100  # Sample 100 points along trajectory
            t_values = np.linspace(0, 1, n_points)
            trajectory_points = [
                electrode.get_point_at_parameter(t).tolist() for t in t_values
            ]
            electrode_data["trajectory_coordinates"] = trajectory_points

        electrode_dicts.append(electrode_data)

    # Create data structure matching JSON format
    data = {
        "metadata": metadata,
        "reconstruction_parameters": reconstruction_parameters,
        "electrodes": electrode_dicts,
    }

    # Call the implementation function
    return _generate_html_report_impl(
        data,
        output_path,
        include_intensity_profiles,
        include_3d_visualization,
        include_contact_comparison,
        include_volume_rendering,
        cached_mesh,
    )


def generate_html_report(
    reconstruction_json_path: str,
    output_path: Optional[str] = None,
    include_intensity_profiles: bool = True,
    include_3d_visualization: bool = True,
    include_contact_comparison: bool = True,
    include_volume_rendering: bool = True,
    cached_mesh: Optional[go.Mesh3d] = None,
) -> str:
    """
    Generate an interactive HTML report from PyPaCER reconstruction results.

    Args:
        reconstruction_json_path: Path to the reconstruction JSON file
        output_path: Optional output path for HTML file (auto-generated if None)
        include_intensity_profiles: Include intensity profile plots
        include_3d_visualization: Include 3D electrode visualization
        include_contact_comparison: Include contact detection comparison if available
        include_volume_rendering: Include volume rendering tab with PyVista visualizations
        cached_mesh: Pre-computed electrode mesh (go.Mesh3d) to avoid re-extraction

    Returns:
        Path to the generated HTML file
    """
    # Load reconstruction data
    with open(reconstruction_json_path) as f:
        data = json.load(f)

    # Create output filename if not provided
    if output_path is None:
        input_path = Path(reconstruction_json_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"{input_path.stem}_report_{timestamp}.html"
    else:
        output_path = Path(output_path)

    # Call the implementation function
    return _generate_html_report_impl(
        data,
        str(output_path),
        include_intensity_profiles,
        include_3d_visualization,
        include_contact_comparison,
        include_volume_rendering,
        cached_mesh,
    )


def _generate_html_report_impl(
    data: Dict[str, Any],
    output_path: str,
    include_intensity_profiles: bool = True,
    include_3d_visualization: bool = True,
    include_contact_comparison: bool = True,
    include_volume_rendering: bool = True,
    cached_mesh: Optional[go.Mesh3d] = None,
) -> str:
    """Internal implementation of HTML report generation."""
    output_path = Path(output_path)

    # Get metadata
    metadata = data.get("metadata", {})
    ct_file = metadata.get("ct_file", "Unknown")
    timestamp = metadata.get("timestamp", "Unknown")

    # Create 3D visualization
    if include_3d_visualization:
        fig_3d = go.Figure()
        add_3d_visualization_direct(fig_3d, data, cached_mesh)

        # Create visibility lists for toggle buttons
        default_visible = [True] * len(fig_3d.data)

        no_metal_visible = []
        for i, trace in enumerate(fig_3d.data):
            if (
                hasattr(trace, "name")
                and trace.name is not None
                and trace.name == "Electrode Metal"
            ):
                no_metal_visible.append(False)
            else:
                no_metal_visible.append(True)

        no_pretip_visible = []
        for i, trace in enumerate(fig_3d.data):
            if (
                hasattr(trace, "name")
                and trace.name is not None
                and "Pre-Tip Polynomial" in trace.name
            ):
                no_pretip_visible.append(False)
            else:
                no_pretip_visible.append(True)

        trajectories_only_visible = []
        for i, trace in enumerate(fig_3d.data):
            if hasattr(trace, "name") and trace.name is not None:
                if "Electrode" in trace.name or "Pre-Tip Polynomial" in trace.name:
                    if "Metal" not in trace.name and "Contact" not in trace.name:
                        trajectories_only_visible.append(True)
                    else:
                        trajectories_only_visible.append(False)
                else:
                    trajectories_only_visible.append(False)
            else:
                trajectories_only_visible.append(False)

        fig_3d.update_layout(
            title="3D Electrode Visualization",
            height=800,
            showlegend=True,
            template="plotly_white",
            hovermode="closest",
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0),
            scene=dict(
                aspectmode="data",
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"visible": default_visible}],
                            label="All",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": no_metal_visible}],
                            label="Hide Metal",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": no_pretip_visible}],
                            label="Hide Pre-Tip",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": trajectories_only_visible}],
                            label="Trajectories Only",
                            method="update",
                        ),
                    ],
                    pad={"r": 10, "b": 10},
                    showactive=True,
                    x=0.95,
                    xanchor="right",
                    y=0.05,
                    yanchor="bottom",
                ),
            ],
        )

    # Generate summary statistics with embedded intensity profiles
    summary_html = generate_summary_html_with_profiles(
        data, include_intensity_profiles
    )

    # Generate 3D plot HTML if requested
    plot_3d_html = ""
    if include_3d_visualization:
        # Store output path for skull mesh export
        add_3d_visualization_direct._output_path = str(output_path)
        fig_json = fig_3d.to_json()
        plot_3d_html = build_3d_plot_html(fig_json)

    # Prepare JSON data for display
    json_formatted = json.dumps(data, indent=2)

    # Prepare minified JSON (core data only, no large profile arrays)
    _profile_keys = {
        "intensity_profile",
        "distance_scale",
        "skeleton_deviations_mm",
        "refined_intensity_profile",
        "pass2_intensities_full",
        "pass2_distances_mm_full",
        "trajectory_coordinates",
        "polynomial_before_tip_detection",
    }
    mini_data = {
        "metadata": data["metadata"],
        "reconstruction_parameters": data["reconstruction_parameters"],
        "electrodes": [],
    }
    for _edata in data["electrodes"]:
        mini_electrode = {k: v for k, v in _edata.items() if k not in _profile_keys}
        if "orientation" in mini_electrode:
            orient = json.loads(json.dumps(mini_electrode["orientation"]))
            for marker in orient.get("markers", {}).values():
                marker.pop("intensity_profile", None)
            if "contact_intensity_profile" in orient:
                orient["contact_intensity_profile"].pop("intensity", None)
            mini_electrode["orientation"] = orient
        mini_data["electrodes"].append(mini_electrode)
    mini_json_formatted = json.dumps(mini_data, indent=2)

    # Generate volume rendering HTML
    volume_html = ""
    if include_volume_rendering:
        volume_html = generate_volume_rendering_html(data)

    # Check if any electrode has orientation data with markers
    has_orientation = any(
        e.get("orientation", {}).get("markers")
        for e in data.get("electrodes", [])
    )

    # Generate orientation tab content from orientation data
    orientation_html = ""
    if has_orientation:
        orientation_html = generate_orientation_html(data)

    # Build the complete HTML report
    html_content = build_report_html(
        ct_file=ct_file,
        timestamp=timestamp,
        summary_html=summary_html,
        plot_3d_html=plot_3d_html,
        volume_rendering_html=volume_html,
        json_formatted=json_formatted,
        mini_json_formatted=mini_json_formatted,
        include_volume_rendering=include_volume_rendering,
        has_orientation=has_orientation,
        orientation_html=orientation_html,
        version=data.get("metadata", {}).get("pypacer_version", ""),
    )

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)

    return str(output_path)
