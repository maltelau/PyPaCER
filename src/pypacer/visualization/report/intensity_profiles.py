"""Intensity profile plots and electrode summary HTML generation."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go


def generate_summary_html_with_profiles(
    data: Dict[str, Any], include_intensity_profiles: bool
) -> str:
    """Generate HTML summary section with embedded intensity profiles."""
    metadata = data.get("metadata", {})
    params = data.get("reconstruction_parameters", {})

    # Calculate summary statistics
    num_electrodes = len(data["electrodes"])
    total_contacts = sum(
        len(e.get("contact_positions", [])) for e in data["electrodes"]
    )

    # Extract processing info
    processing_type = params.get("processing_type", "Unknown")
    interface_type = params.get("interface", "Unknown")
    detection_method = params.get("method", params.get("detection_method", "Unknown"))
    timestamp = metadata.get("timestamp", "Unknown")

    # Extract resolution parameters
    xy_res = params.get("xy_resolution", "N/A")
    z_res = params.get("z_resolution", "N/A")
    grid_size = params.get("grid_size", "N/A")

    html = """
    <div class="summary">
        <h2>Reconstruction Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <h3>Electrodes Detected</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Total Contacts</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Metal Threshold</h3>
                <p>{} HU</p>
            </div>
            <div class="summary-item">
                <h3>Voxel Size</h3>
                <p>{:.2f} × {:.2f} × {:.2f} mm</p>
            </div>
            <div class="summary-item">
                <h3>Processing Type</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Interface</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Detection Method</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Resolution Parameters</h3>
                <p>xy={}, z={}, grid={} mm</p>
            </div>
            <div class="summary-item">
                <h3>Timestamp</h3>
                <p>{}</p>
            </div>
        </div>

        <div class="electrode-details">
            <h2>Electrode Details</h2>
    """.format(
        num_electrodes,
        total_contacts,
        metadata.get("metal_threshold_HU", "N/A"),
        *metadata.get("voxel_sizes_mm", [0, 0, 0]),
        processing_type,
        interface_type,
        (
            "Manual Seed Placement"
            if detection_method == "manual"
            else (
                "Radial Detection"
                if detection_method == "detect_electrodes_radial"
                else detection_method.replace("_", " ").title()
            )
        ),
        xy_res,
        z_res,
        grid_size,
        (
            timestamp
            if isinstance(timestamp, str)
            else (
                timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(timestamp, "strftime")
                else str(timestamp)[:19]
            )
        ),
    )

    # Add electrode-specific details with intensity profiles
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, electrode in enumerate(data["electrodes"]):
        electrode_type = electrode.get("electrode_type", "Unknown")
        hemisphere = electrode.get("side", "Unknown").capitalize()
        contact_positions = electrode.get("contact_positions", [])
        color = colors[i % len(colors)]

        # Generate slice visualizations early so we can adjust the grid layout
        oriented_slice_b64 = _maybe_generate_oriented_slice(
            electrode, metadata,
        )
        axial_slice_b64 = _maybe_generate_axial_slice(
            electrode, metadata,
        )
        html += f"""
            <div class="electrode-card">
                <h4>Electrode {i+1} - {electrode_type} - {hemisphere}</h4>
                <div class="electrode-content" style="grid-template-columns: 300px 1fr;">
                    <div class="contact-details">
                        <p><strong>Length:</strong> {electrode.get('length_mm', 0):.1f} mm</p>
                        <p><strong>Tip Position:</strong> ({electrode.get('tip_position', [0,0,0])[0]:.1f}, {electrode.get('tip_position', [0,0,0])[1]:.1f}, {electrode.get('tip_position', [0,0,0])[2]:.1f})</p>
                        <p><strong>Contacts:</strong> <span id="contact-count-{i}">{len(contact_positions)}</span></p>
        """

        if contact_positions:
            # Check if we have results from multiple methods
            has_multiple_methods = "contact_detection_results" in electrode

            # The primary method is what was used for reconstruction
            primary_method = params.get("contact_detection_method", "contactAreaCenter")

            if has_multiple_methods:
                html += _generate_multi_method_contact_details(
                    electrode, i, primary_method
                )
            else:
                html += _generate_single_method_contact_details(
                    electrode, contact_positions, primary_method
                )

            # Add 3D coordinates if available (these don't change with method)
            if "contact_positions_3d" in electrode:
                html += """
                        <div class="contact-list" style="margin-top: 15px;">
                            <strong>Contact 3D Coordinates (mm):</strong>
                            <div class="contact-3d-list">
                """
                for j, coords in enumerate(electrode["contact_positions_3d"]):
                    html += f"""
                                <div class="contact-3d-item">
                                    <span>C{j+1}:</span>
                                    <span>({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})</span>
                                </div>
                    """
                html += """
                            </div>
                        </div>
                """

        # Add orientation marker info if available
        orientation = electrode.get("orientation", {})
        if orientation.get("has_markers"):
            markers = orientation.get("markers", {})
            html += """
                        <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">
                            <strong>Orientation Markers</strong>
            """
            for label in ("A", "B"):
                m = markers.get(label)
                if not m:
                    continue
                dist = m.get("distance_from_tip_mm")
                vec = m.get("direction_vector")
                axial_deg = m.get("fitted_angle_axial_deg")

                html += f"""
                            <div style="margin-top: 8px; margin-left: 5px;">
                                <p><strong>Marker {label}</strong></p>
                """
                if dist is not None:
                    html += f'<p style="margin-left:10px;font-size:13px;">Position: {dist:.1f} mm from tip</p>'
                if vec is not None:
                    html += f'<p style="margin-left:10px;font-size:13px;">Vector: ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})</p>'
                if axial_deg is not None:
                    html += f'<p style="margin-left:10px;font-size:13px;">Axial direction: {axial_deg:.1f}&deg;</p>'
                html += """
                            </div>
                """
            html += """
                        </div>
            """

        html += """
                    </div>
        """

        # Right column: rows of visualizations
        html += """
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                        <div style="display: flex; gap: 10px; align-items: start;">
        """

        # Row 1: CT intensities plot + intensity profile
        if oriented_slice_b64:
            html += f"""
                            <div style="flex-shrink: 0;">
                                <img src="data:image/png;base64,{oriented_slice_b64}"
                                     style="height: 456px; width: 180px; border: 1px solid #ddd; border-radius: 5px;">
                            </div>
            """

        if (
            include_intensity_profiles
            and "intensity_profile" in electrode
            and "distance_scale" in electrode
        ):
            plot_id = f"intensity-plot-{i}"
            has_multiple_methods = "contact_detection_results" in electrode

            if has_multiple_methods:
                html += _generate_multi_method_profile_html(
                    electrode, i, color, plot_id, primary_method
                )
            else:
                html += _generate_single_method_profile_html(
                    electrode, i, color, plot_id
                )
        else:
            html += """
                            <div class="intensity-plot" style="display: flex; align-items: center; justify-content: center; flex: 1;">
                                <p style="color: #999;">No intensity profile available</p>
                            </div>
            """

        # Close row 1
        html += """
                        </div>
        """

        # Row 2: Axial slice + marker intensity profiles (if orientation data available)
        orientation = electrode.get("orientation", {})
        markers = orientation.get("markers", {})
        has_axial_or_markers = axial_slice_b64 or markers

        if has_axial_or_markers:
            html += """
                        <div style="display: flex; gap: 10px; align-items: start;">
            """

            if axial_slice_b64:
                html += f"""
                            <div style="position: relative; display: inline-block; flex-shrink: 0;">
                                <img src="data:image/png;base64,{axial_slice_b64}"
                                     style="max-width: 400px; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                                <button class="maximize-btn" title="Click to maximize">
                                    <svg viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                                    </svg>
                                    Maximize
                                </button>
                            </div>
                """

            if markers:
                from .orientation_tab import _build_marker_charts_html

                html += f"""
                            <div style="flex: 1; min-width: 300px;">
                                {_build_marker_charts_html(i, markers, id_suffix="-summary")}
                            </div>
                """

            html += """
                        </div>
            """

        # Close right column
        html += """
                    </div>
        """

        # Close electrode-content div
        html += """
                </div>
        """

        # Close electrode-card div
        html += """
            </div>
        """

    html += """
        </div>
    """

    # Add debug intensity plot if full Pass 2 data is available
    if any("pass2_intensities_full" in e for e in data["electrodes"]):
        html += _generate_debug_intensity_html(data)

    html += """
    </div>
    """

    return html


# Cache for CT data to avoid reloading per electrode
_ct_cache: Dict[str, Any] = {}


def _maybe_generate_oriented_slice(
    electrode: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    """Generate an oriented CT slice for this electrode if possible."""
    if not electrode.get("polynomial"):
        return None

    ct_path = metadata.get("ct_file")
    if not ct_path or not Path(ct_path).exists():
        return None

    try:
        # Load CT data (cached)
        if ct_path not in _ct_cache:
            import nibabel as nib

            ct_nii = nib.load(ct_path)
            _ct_cache[ct_path] = {
                "data": ct_nii.get_fdata(),
                "affine": ct_nii.affine,
            }
        ct = _ct_cache[ct_path]

        from .oriented_slice import generate_oriented_slice_base64

        return generate_oriented_slice_base64(
            ct["data"], ct["affine"], electrode,
        )
    except Exception as e:
        return None


def _maybe_generate_axial_slice(
    electrode: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    """Generate an axial CT slice at the contact region center if possible."""
    if not electrode.get("polynomial") or not electrode.get("contact_positions"):
        return None
    if not electrode.get("orientation", {}).get("markers"):
        return None

    ct_path = metadata.get("ct_file")
    if not ct_path or not Path(ct_path).exists():
        return None

    try:
        if ct_path not in _ct_cache:
            import nibabel as nib

            ct_nii = nib.load(ct_path)
            _ct_cache[ct_path] = {
                "data": ct_nii.get_fdata(),
                "affine": ct_nii.affine,
            }
        ct = _ct_cache[ct_path]

        from .oriented_slice import generate_axial_slice_base64

        return generate_axial_slice_base64(
            ct["data"], ct["affine"], electrode,
        )
    except Exception as e:
        return None


def create_intensity_profile_json(
    electrode: Dict[str, Any],
    electrode_idx: int,
    color: str,
    primary_method: str = "contactAreaCenter",
) -> str:
    """Create JSON data for intensity profile plot with multiple contact detection methods and skeleton deviations."""
    distance = np.array(electrode["distance_scale"])
    intensity = np.array(electrode["intensity_profile"])

    # Check if we have skeleton deviations
    has_skeleton_deviations = "skeleton_deviations_mm" in electrode

    # Main intensity trace
    base_trace = {
        "x": distance.tolist(),
        "y": intensity.tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"Intensity Profile {electrode_idx+1}",
        "line": {"color": color, "width": 2},
        "yaxis": "y",
    }

    # Skeleton deviations trace (if available)
    deviation_trace = None
    if has_skeleton_deviations:
        deviations = np.array(electrode["skeleton_deviations_mm"])
        deviation_trace = {
            "x": distance.tolist(),
            "y": deviations.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Skeleton Deviation",
            "line": {"color": "#9467bd", "width": 2},  # Removed dash for smooth line
            "yaxis": "y2",
            "visible": True,  # Will be toggled by button
        }

    # Check if we have multiple contact detection methods
    has_multiple_methods = "contact_detection_results" in electrode

    if has_multiple_methods:
        return _create_multi_method_profile_json(
            electrode, distance, intensity, base_trace, deviation_trace,
            has_skeleton_deviations, primary_method,
        )
    else:
        return _create_single_method_profile_json(
            electrode, electrode_idx, distance, intensity, base_trace,
            deviation_trace, has_skeleton_deviations,
        )


def generate_summary_stats(data: Dict[str, Any]) -> str:
    """Generate HTML summary statistics section."""
    metadata = data.get("metadata", {})

    # Calculate summary statistics
    num_electrodes = len(data["electrodes"])
    total_contacts = sum(
        len(e.get("contact_positions", [])) for e in data["electrodes"]
    )

    html = """
    <div class="summary">
        <h2>Reconstruction Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <h3>Electrodes Detected</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Total Contacts</h3>
                <p>{}</p>
            </div>
            <div class="summary-item">
                <h3>Metal Threshold</h3>
                <p>{} HU</p>
            </div>
            <div class="summary-item">
                <h3>Voxel Size</h3>
                <p>{:.2f} × {:.2f} × {:.2f} mm</p>
            </div>
        </div>

        <h3 style="margin-top: 20px;">Electrode Details</h3>
    """.format(
        num_electrodes,
        total_contacts,
        metadata.get("metal_threshold_HU", "N/A"),
        *metadata.get("voxel_sizes_mm", [0, 0, 0]),
    )

    # Add electrode-specific details
    for i, electrode in enumerate(data["electrodes"]):
        electrode_type = electrode.get("electrode_type", "Unknown")
        hemisphere = electrode.get("side", "Unknown").capitalize()
        contact_positions = electrode.get("contact_positions", [])

        html += f"""
        <div style="margin-top: 15px; padding: 10px; background-color: white; border-radius: 5px;">
            <strong>Electrode {i+1} - {electrode_type} - {hemisphere}</strong><br>
            Length: {electrode.get('length_mm', 0):.1f} mm<br>
            Tip Position: ({electrode.get('tip_position', [0,0,0])[0]:.1f}, {electrode.get('tip_position', [0,0,0])[1]:.1f}, {electrode.get('tip_position', [0,0,0])[2]:.1f})<br>
            Contacts: {len(contact_positions)}
        """

        if contact_positions:
            html += "<br>Contact Positions (mm from tip): " + ", ".join(
                f"{pos:.2f}" for pos in contact_positions
            )

        html += "</div>"

    html += "</div>"

    return html


def add_contact_comparison(
    fig: go.Figure, electrode: Dict[str, Any], electrode_idx: int, row: int, col: int
):
    """Add contact detection comparison plot."""
    comparison = electrode["contact_comparison"]

    # Plot detected contacts
    if "detected" in comparison:
        detected = comparison["detected"]
        fig.add_trace(
            go.Scatter(
                x=detected["positions"],
                y=detected["intensities"],
                mode="markers",
                name=f"Detected Contacts {electrode_idx+1}",
                marker=dict(color="green", size=10, symbol="circle"),
            ),
            row=row,
            col=col,
        )

    # Plot expected contacts
    if "expected" in comparison:
        expected = comparison["expected"]
        fig.add_trace(
            go.Scatter(
                x=expected["positions"],
                y=expected["intensities"],
                mode="markers",
                name=f"Expected Contacts {electrode_idx+1}",
                marker=dict(color="red", size=10, symbol="x"),
            ),
            row=row,
            col=col,
        )

    # Update axes
    fig.update_xaxes(title_text="Distance from tip (mm)", row=row, col=col)
    fig.update_yaxes(title_text="Intensity (HU)", row=row, col=col)


def add_intensity_profile(
    fig: go.Figure, electrode: Dict[str, Any], electrode_idx: int, row: int, col: int
):
    """Add intensity profile plot for an electrode."""
    distance = np.array(electrode["distance_scale"])
    intensity = np.array(electrode["intensity_profile"])

    # Main intensity profile
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=intensity,
            mode="lines",
            name=f"Intensity Profile {electrode_idx+1}",
            line=dict(color="#1f77b4", width=2),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Add contact markers
    if "contact_positions" in electrode:
        contact_positions = np.array(electrode["contact_positions"])
        contact_intensities = np.interp(contact_positions, distance, intensity)

        fig.add_trace(
            go.Scatter(
                x=contact_positions,
                y=contact_intensities,
                mode="markers+text",
                name=f"Contacts {electrode_idx+1}",
                marker=dict(
                    color="#d62728",
                    size=12,
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                text=[f"C{i+1}" for i in range(len(contact_positions))],
                textposition="top center",
                textfont=dict(size=11, color="#d62728"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add vertical lines for contacts (only for 2D plots)
        for pos in contact_positions:
            # Create a vertical line manually to avoid issues with subplots
            y_range = [intensity.min(), intensity.max()]
            fig.add_trace(
                go.Scatter(
                    x=[pos, pos],
                    y=y_range,
                    mode="lines",
                    line=dict(color="#d62728", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    # Update axes
    fig.update_xaxes(title_text="Distance from tip (mm)", row=row, col=col)
    fig.update_yaxes(title_text="Intensity (HU)", row=row, col=col)


# --- Private helpers ---


def _generate_multi_method_contact_details(electrode, i, primary_method):
    """Generate HTML for multi-method contact position details."""
    contact_positions = electrode.get("contact_positions", [])
    contact_details_id = f"contact-details-{i}"
    primary_positions = contact_positions  # These are from the primary method

    html = f"""
                        <div id="{contact_details_id}">
                        <div style="margin-bottom: 10px; padding: 8px; background-color: #e8f4fd; border-radius: 4px;">
                            <small><strong>Primary Method:</strong> {primary_method} (used for reconstruction)</small>
                        </div>
    """

    # Generate contact details for each method
    for method, result in electrode["contact_detection_results"].items():
        if isinstance(result, dict) and "contact_positions" in result:
            method_positions = result["contact_positions"]
            display_style = "block" if method == primary_method else "none"
            is_primary = method == primary_method

            html += f"""
                        <div class="contact-list" data-method="{method}" data-primary="{str(is_primary).lower()}" style="display: {display_style};">
                            <strong>Contact Positions (mm from tip):</strong>
                            <br><small style="color: #666;">
                                (Method: {method}{' - PRIMARY' if is_primary else ''})
                            </small>
            """

            # Calculate divergence if not primary method
            divergences = []
            if not is_primary and len(method_positions) == len(primary_positions):
                for j, (pos, primary_pos) in enumerate(
                    zip(method_positions, primary_positions)
                ):
                    divergence = pos - primary_pos
                    divergences.append(divergence)
                    html += f"""
                            <div class="contact-item">
                                <span>Contact {j+1}:</span>
                                <span>{pos:.2f} mm</span>
                                <span style="color: {'#d62728' if abs(divergence) > 0.5 else '#666'}; font-size: 11px;">
                                    ({divergence:+.2f} mm)
                                </span>
                            </div>
                    """

                # Calculate and show RMSE
                rmse = np.sqrt(np.mean(np.array(divergences) ** 2))
                html += f"""
                            <div style="margin-top: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;">
                                <strong>RMSE from primary:</strong> {rmse:.3f} mm
                            </div>
                """
            else:
                # Primary method or mismatched number of contacts
                for j, pos in enumerate(method_positions):
                    html += f"""
                            <div class="contact-item">
                                <span>Contact {j+1}:</span>
                                <span>{pos:.2f} mm</span>
                            </div>
                    """

            html += """
                        </div>
            """

    html += """
                        </div>
    """
    return html


def _generate_single_method_contact_details(electrode, contact_positions, primary_method):
    """Generate HTML for single-method contact position details."""
    # Get method display name
    method_display_names = {
        "contactAreaCenter": "Area Center",
        "peak": "Peak",
        "peakWaveCenter": "Peak Wave Center",
    }
    method_display = method_display_names.get(primary_method, primary_method)

    html = f"""
                        <div class="contact-list">
                            <strong>Contact Positions (mm from tip):</strong>
                            <br><small style="color: #666;">
                                (Detection Method: {method_display})
                            </small>
    """
    for j, pos in enumerate(contact_positions):
        html += f"""
                            <div class="contact-item">
                                <span>Contact {j+1}:</span>
                                <span>{pos:.2f} mm</span>
                            </div>
        """
    html += """
                        </div>
    """
    return html


def _generate_multi_method_profile_html(electrode, i, color, plot_id, primary_method):
    """Generate HTML+JS for multi-method intensity profile plot."""
    intensity_data = create_intensity_profile_json(
        electrode, i, color, primary_method
    )

    html = f"""
                    <div style="flex: 1;">
                        <div style="margin-bottom: 10px;">
                            <strong>Contact Detection Method:</strong>
                            <div style="display: inline-block; margin-left: 10px;">
                                <button class="method-btn{' method-btn-active' if primary_method == 'contactAreaCenter' else ''}" data-electrode="{i}" data-method="contactAreaCenter" style="margin-right: 5px;">Area Center</button>
                                <button class="method-btn{' method-btn-active' if primary_method == 'peak' else ''}" data-electrode="{i}" data-method="peak" style="margin-right: 5px;">Peak</button>
                                <button class="method-btn{' method-btn-active' if primary_method == 'peakWaveCenter' else ''}" data-electrode="{i}" data-method="peakWaveCenter">Peak Wave Center</button>
                            </div>
                        </div>
                        <div class="intensity-plot" id="{plot_id}"></div>
                    </div>
    """

    # Add the plot script
    html += f"""
            <script>
                (function() {{
                    var plotData = {intensity_data};
                    var currentMethod = plotData.default_method;
                    var showDeviations = true;

                    // Calculate deviation max for y-axis scale
                    var deviationMax = 0;
                    if (plotData.has_skeleton_deviations) {{
                        for (var method in plotData.methods) {{
                            var traces = plotData.methods[method];
                            for (var j = 0; j < traces.length; j++) {{
                                if (traces[j].name === 'Skeleton Deviation' && traces[j].y) {{
                                    var maxVal = Math.max(...traces[j].y);
                                    deviationMax = Math.max(deviationMax, maxVal);
                                }}
                            }}
                        }}
                        // Set minimum max to 0.5mm if max is less than 0.5
                        deviationMax = Math.max(0.5, deviationMax * 1.1); // Add 10% padding
                    }}

                    var layout = {{
                        title: 'Intensity Profile - ' + currentMethod,
                        xaxis: {{ title: 'Distance from tip (mm)' }},
                        yaxis: {{ title: 'Intensity (HU)', side: 'left' }},
                        showlegend: true,
                        height: 400,
                        margin: {{ l: 60, r: plotData.has_skeleton_deviations ? 80 : 20, t: 40, b: 60 }}
                    }};

                    // Add secondary y-axis for deviations if available
                    if (plotData.has_skeleton_deviations) {{
                        layout.yaxis2 = {{
                            title: 'Deviation (mm)',
                            overlaying: 'y',
                            side: 'right',
                            range: [0, deviationMax],
                            showgrid: false,
                            zeroline: false
                        }};
                    }}

                    // Function to update visibility of deviation traces
                    function updateDeviationVisibility(traces) {{
                        for (var j = 0; j < traces.length; j++) {{
                            if (traces[j].name === 'Skeleton Deviation') {{
                                traces[j].visible = showDeviations;
                            }}
                        }}
                        return traces;
                    }}

                    // Initial plot
                    var initialTraces = updateDeviationVisibility([...plotData.methods[currentMethod]]);
                    Plotly.newPlot('{plot_id}', initialTraces, layout, {{responsive: true}});

                    // Add deviation toggle button if we have deviations
                    if (plotData.has_skeleton_deviations) {{
                        var electrodeCard = document.getElementById('{plot_id}').closest('.electrode-card');
                        var toggleBtn = document.createElement('button');
                        toggleBtn.className = 'method-btn method-btn-active';
                        toggleBtn.style.position = 'absolute';
                        toggleBtn.style.top = '10px';
                        toggleBtn.style.right = '10px';
                        toggleBtn.style.zIndex = '10';
                        toggleBtn.textContent = 'Hide Deviations';
                        toggleBtn.onclick = function() {{
                            showDeviations = !showDeviations;
                            this.textContent = showDeviations ? 'Hide Deviations' : 'Show Deviations';
                            this.classList.toggle('method-btn-active', showDeviations);

                            // Update current plot
                            var updatedTraces = updateDeviationVisibility([...plotData.methods[currentMethod]]);
                            Plotly.react('{plot_id}', updatedTraces, layout);
                        }};
                        electrodeCard.appendChild(toggleBtn);
                    }}

                    // Add button click handlers
                    document.querySelectorAll('[data-electrode="{i}"]').forEach(function(btn) {{
                        btn.addEventListener('click', function() {{
                            var method = this.getAttribute('data-method');
                            if (plotData.methods[method]) {{
                                // Update button states
                                document.querySelectorAll('[data-electrode="{i}"]').forEach(function(b) {{
                                    b.classList.remove('method-btn-active');
                                }});
                                this.classList.add('method-btn-active');

                                // Update current method
                                currentMethod = method;

                                // Update plot
                                layout.title = 'Intensity Profile - ' + method;
                                var updatedTraces = updateDeviationVisibility([...plotData.methods[method]]);
                                Plotly.react('{plot_id}', updatedTraces, layout);

                                // Update contact details
                                var contactDetails = document.getElementById('contact-details-{i}');
                                if (contactDetails) {{
                                    contactDetails.querySelectorAll('.contact-list[data-method]').forEach(function(list) {{
                                        list.style.display = 'none';
                                    }});
                                    var methodList = contactDetails.querySelector('.contact-list[data-method="' + method + '"]');
                                    if (methodList) {{
                                        methodList.style.display = 'block';
                                        // Update contact count (excluding RMSE row)
                                        var contactItems = methodList.querySelectorAll('.contact-item');
                                        var contactCount = contactItems.length;
                                        var countSpan = document.getElementById('contact-count-{i}');
                                        if (countSpan) {{
                                            countSpan.textContent = contactCount;
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }});
                }})();
            </script>
    """
    return html


def _generate_single_method_profile_html(electrode, i, color, plot_id):
    """Generate HTML+JS for single-method intensity profile plot."""
    intensity_data = create_intensity_profile_json(electrode, i, color)

    html = f"""
                    <div style="flex: 1;">
                        <div class="intensity-plot" id="{plot_id}"></div>
                    </div>
    """

    # Add the plot script
    html += f"""
            <script>
                (function() {{
                    var plotData = {intensity_data};
                    var showDeviations = true;

                    // Calculate deviation max for y-axis scale
                    var deviationMax = 0;
                    if (plotData.has_skeleton_deviations) {{
                        var traces = plotData.traces;
                        for (var j = 0; j < traces.length; j++) {{
                            if (traces[j].name === 'Skeleton Deviation' && traces[j].y) {{
                                var maxVal = Math.max(...traces[j].y);
                                deviationMax = Math.max(deviationMax, maxVal);
                            }}
                        }}
                        // Set minimum max to 0.5mm if max is less than 0.5
                        deviationMax = Math.max(0.5, deviationMax * 1.1); // Add 10% padding
                    }}

                    var layout = {{
                        title: 'Intensity Profile',
                        xaxis: {{ title: 'Distance from tip (mm)' }},
                        yaxis: {{ title: 'Intensity (HU)', side: 'left' }},
                        showlegend: true,
                        height: 400,
                        margin: {{ l: 60, r: plotData.has_skeleton_deviations ? 80 : 20, t: 40, b: 60 }}
                    }};

                    // Add secondary y-axis for deviations if available
                    if (plotData.has_skeleton_deviations) {{
                        layout.yaxis2 = {{
                            title: 'Deviation (mm)',
                            overlaying: 'y',
                            side: 'right',
                            range: [0, deviationMax],
                            showgrid: false,
                            zeroline: false
                        }};
                    }}

                    // Function to update visibility of deviation traces
                    function updateDeviationVisibility(traces) {{
                        for (var j = 0; j < traces.length; j++) {{
                            if (traces[j].name === 'Skeleton Deviation') {{
                                traces[j].visible = showDeviations;
                            }}
                        }}
                        return traces;
                    }}

                    // Initial plot
                    var initialTraces = updateDeviationVisibility([...plotData.traces]);
                    Plotly.newPlot('{plot_id}', initialTraces, layout, {{responsive: true}});

                    // Add deviation toggle button if we have deviations
                    if (plotData.has_skeleton_deviations) {{
                        var electrodeCard = document.getElementById('{plot_id}').closest('.electrode-card');
                        var toggleBtn = document.createElement('button');
                        toggleBtn.className = 'method-btn method-btn-active';
                        toggleBtn.style.position = 'absolute';
                        toggleBtn.style.top = '10px';
                        toggleBtn.style.right = '10px';
                        toggleBtn.style.zIndex = '10';
                        toggleBtn.textContent = 'Hide Deviations';
                        toggleBtn.onclick = function() {{
                            showDeviations = !showDeviations;
                            this.textContent = showDeviations ? 'Hide Deviations' : 'Show Deviations';
                            this.classList.toggle('method-btn-active', showDeviations);

                            // Update plot
                            var updatedTraces = updateDeviationVisibility([...plotData.traces]);
                            Plotly.react('{plot_id}', updatedTraces, layout);
                        }};
                        electrodeCard.appendChild(toggleBtn);
                    }}
                }})();
            </script>
    """
    return html


def _generate_debug_intensity_html(data):
    """Generate HTML for debug intensity plots (Pass 2 data)."""
    html = """
        <div class="debug-intensity-section" style="margin-top: 30px;">
            <h2>Debug: Full OOR Intensity Profiles (Including Lookahead)</h2>
            <div class="debug-plot-container" style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
    """

    # Create debug plots for each electrode that has full Pass 2 data
    for i, electrode in enumerate(data["electrodes"]):
        if (
            "pass2_intensities_full" in electrode
            and "pass2_distances_mm_full" in electrode
        ):
            debug_plot_id = f"debug-intensity-plot-{i}"

            html += f"""
                <div style="margin-bottom: 30px;">
                    <h4>Electrode {i+1} - Full Pass 2 Intensity Profile</h4>
                    <div id="{debug_plot_id}" style="width: 100%; height: 400px; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);"></div>
                </div>

                <script>
                    (function() {{
                        var distances = {json.dumps(electrode['pass2_distances_mm_full'])};
                        var intensities = {json.dumps(electrode['pass2_intensities_full'])};
                        var tipThreshold = {electrode.get('pass2_tip_threshold', 1500)};
                        var originalT0Distance = {electrode.get('original_t0_distance_mm', 'null')};

                        var trace = {{
                            x: distances,
                            y: intensities,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'OOR Intensity Profile',
                            line: {{color: '#1f77b4', width: 2}}
                        }};

                        var layout = {{
                            title: 'Pass 2 OOR Intensity Profile (Full Trajectory)<br><sub>Distance is relative to detected tip (where intensity crosses threshold)</sub>',
                            xaxis: {{
                                title: 'Distance from detected tip (mm)<br><sub>Negative = before tip, Positive = after tip</sub>',
                                zeroline: true,
                                zerolinecolor: 'red',
                                zerolinewidth: 3
                            }},
                            yaxis: {{
                                title: 'Intensity (HU)',
                                range: [0, Math.max(...intensities) * 1.1]
                            }},
                            shapes: [
                                {{
                                    type: 'line',
                                    x0: 0,
                                    y0: 0,
                                    x1: 0,
                                    y1: Math.max(...intensities),
                                    line: {{
                                        color: 'red',
                                        width: 3,
                                        dash: 'solid'
                                    }}
                                }},
                                {{
                                    type: 'line',
                                    x0: Math.min(...distances),
                                    y0: tipThreshold,
                                    x1: Math.max(...distances),
                                    y1: tipThreshold,
                                    line: {{
                                        color: 'orange',
                                        width: 2,
                                        dash: 'dash'
                                    }}
                                }}
                            ].concat(originalT0Distance !== null ? [{{
                                type: 'line',
                                x0: originalT0Distance,
                                y0: 0,
                                x1: originalT0Distance,
                                y1: Math.max(...intensities),
                                line: {{
                                    color: 'green',
                                    width: 2,
                                    dash: 'dot'
                                }}
                            }}] : []),
                            annotations: [
                                {{
                                    x: 0,
                                    y: Math.max(...intensities) * 0.9,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Detected Tip<br>(x=0)',
                                    showarrow: true,
                                    arrowhead: 2,
                                    arrowsize: 1,
                                    arrowwidth: 2,
                                    arrowcolor: 'red',
                                    ax: 30,
                                    ay: -30,
                                    bordercolor: 'red',
                                    borderwidth: 2,
                                    borderpad: 4,
                                    bgcolor: 'white',
                                    opacity: 0.9
                                }},
                                {{
                                    x: Math.max(...distances) * 0.7,
                                    y: tipThreshold,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Tip Threshold: ' + tipThreshold.toFixed(0) + ' HU',
                                    showarrow: false,
                                    bordercolor: 'orange',
                                    borderwidth: 1,
                                    borderpad: 4,
                                    bgcolor: 'white',
                                    opacity: 0.9
                                }}
                            ].concat(originalT0Distance !== null ? [{{
                                x: originalT0Distance,
                                y: Math.max(...intensities) * 0.8,
                                xref: 'x',
                                yref: 'y',
                                text: 'Original t=0<br>(' + originalT0Distance.toFixed(1) + ' mm)',
                                showarrow: true,
                                arrowhead: 2,
                                arrowsize: 1,
                                arrowwidth: 2,
                                arrowcolor: 'green',
                                ax: 0,
                                ay: -40,
                                bordercolor: 'green',
                                borderwidth: 2,
                                borderpad: 4,
                                bgcolor: 'white',
                                opacity: 0.9
                            }}] : []),
                            showlegend: true,
                            hovermode: 'closest',
                            margin: {{l: 80, r: 40, t: 60, b: 80}}
                        }};

                        var config = {{
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false
                        }};

                        Plotly.newPlot('{debug_plot_id}', [trace], layout, config);
                    }})();
                </script>
            """

    html += """
            </div>
        </div>
    """

    return html


def _create_multi_method_profile_json(
    electrode, distance, intensity, base_trace, deviation_trace,
    has_skeleton_deviations, primary_method,
):
    """Create JSON for multi-method intensity profile data."""
    method_colors = {
        "contactAreaCenter": "#d62728",  # red
        "peak": "#2ca02c",  # green
        "peakWaveCenter": "#ff7f0e",  # orange
    }

    all_traces = {}
    for method, result in electrode["contact_detection_results"].items():
        if isinstance(result, dict) and "contact_positions" in result:
            method_traces = [base_trace.copy()]

            # Add deviation trace if available
            if deviation_trace:
                method_traces.append(deviation_trace.copy())

            contact_positions = np.array(result["contact_positions"])
            contact_intensities = np.interp(contact_positions, distance, intensity)

            # Add contact markers
            method_traces.append(
                {
                    "x": contact_positions.tolist(),
                    "y": contact_intensities.tolist(),
                    "type": "scatter",
                    "mode": "markers+text",
                    "name": "Contacts",
                    "marker": {
                        "color": method_colors.get(method, "#9467bd"),
                        "size": 10,
                        "symbol": "circle",
                        "line": {"color": "white", "width": 2},
                    },
                    "text": [f"C{i+1}" for i in range(len(contact_positions))],
                    "textposition": "top center",
                    "textfont": {
                        "size": 10,
                        "color": method_colors.get(method, "#9467bd"),
                    },
                    "yaxis": "y",
                }
            )

            # Add vertical lines for contacts
            for pos in contact_positions:
                method_traces.append(
                    {
                        "x": [pos, pos],
                        "y": [intensity.min(), intensity.max()],
                        "type": "scatter",
                        "mode": "lines",
                        "line": {
                            "color": method_colors.get(method, "#9467bd"),
                            "width": 1,
                            "dash": "dash",
                        },
                        "hoverinfo": "skip",
                        "showlegend": False,
                        "yaxis": "y",
                    }
                )

            # Add orientation marker positions (A and B)
            method_traces.extend(
                _get_orientation_marker_traces(electrode, distance, intensity)
            )

            all_traces[method] = method_traces

    return json.dumps(
        {
            "methods": all_traces,
            "default_method": primary_method,
            "has_skeleton_deviations": has_skeleton_deviations,
        }
    )


def _get_orientation_marker_traces(electrode, distance, intensity):
    """Create traces for orientation markers A and B on the intensity profile."""
    traces = []
    orientation = electrode.get("orientation", {})
    if not orientation.get("has_markers"):
        return traces

    markers = orientation.get("markers", {})
    marker_colors = {"A": "#ff7f0e", "B": "#1f77b4"}  # orange for A, blue for B

    for marker_name in ["A", "B"]:
        marker = markers.get(marker_name)
        if marker is None:
            continue
        pos = marker.get("distance_from_tip_mm")
        if pos is None:
            continue

        marker_intensity = float(np.interp(pos, distance, intensity))

        # Marker dot + label
        traces.append(
            {
                "x": [pos],
                "y": [marker_intensity],
                "type": "scatter",
                "mode": "markers+text",
                "name": f"Marker {marker_name}",
                "marker": {
                    "color": marker_colors[marker_name],
                    "size": 12,
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                "text": [f"M{marker_name}"],
                "textposition": "bottom center",
                "textfont": {"size": 10, "color": marker_colors[marker_name]},
                "yaxis": "y",
            }
        )

        # Vertical dashed line
        traces.append(
            {
                "x": [pos, pos],
                "y": [float(intensity.min()), float(intensity.max())],
                "type": "scatter",
                "mode": "lines",
                "line": {
                    "color": marker_colors[marker_name],
                    "width": 1,
                    "dash": "dot",
                },
                "hoverinfo": "skip",
                "showlegend": False,
                "yaxis": "y",
            }
        )

    return traces


def _create_single_method_profile_json(
    electrode, electrode_idx, distance, intensity, base_trace,
    deviation_trace, has_skeleton_deviations,
):
    """Create JSON for single-method intensity profile data."""
    traces = [base_trace]

    # Add deviation trace if available
    if deviation_trace:
        traces.append(deviation_trace)

    if "contact_positions" in electrode:
        contact_positions = np.array(electrode["contact_positions"])
        contact_intensities = np.interp(contact_positions, distance, intensity)

        traces.append(
            {
                "x": contact_positions.tolist(),
                "y": contact_intensities.tolist(),
                "type": "scatter",
                "mode": "markers+text",
                "name": f"Contacts {electrode_idx+1}",
                "marker": {
                    "color": "#d62728",
                    "size": 10,
                    "symbol": "circle",
                    "line": {"color": "white", "width": 2},
                },
                "text": [f"C{i+1}" for i in range(len(contact_positions))],
                "textposition": "top center",
                "textfont": {"size": 10, "color": "#d62728"},
                "yaxis": "y",
            }
        )

        # Add vertical lines for contacts
        for pos in contact_positions:
            traces.append(
                {
                    "x": [pos, pos],
                    "y": [intensity.min(), intensity.max()],
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#d62728", "width": 1, "dash": "dash"},
                    "hoverinfo": "skip",
                    "showlegend": False,
                    "yaxis": "y",
                }
            )

    # Add orientation marker positions (A and B)
    traces.extend(_get_orientation_marker_traces(electrode, distance, intensity))

    return json.dumps(
        {"traces": traces, "has_skeleton_deviations": has_skeleton_deviations}
    )
