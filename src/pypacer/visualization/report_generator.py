"""Generate interactive HTML reports from PyPaCER reconstruction results."""

import base64
import io
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from .isosurface_extraction import extract_electrode_mesh


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
        # Try creating a regular figure instead of subplot
        fig_3d = go.Figure()

        # Add the 3D visualization data directly without row/col
        _add_3d_visualization_direct(fig_3d, data, cached_mesh)

        # Create visibility lists for toggle buttons
        # Default visibility - all traces visible
        default_visible = [True] * len(fig_3d.data)

        # Visibility without electrode metal - hide electrode metal trace
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

        # Visibility without debug polynomials - hide pre-tip polynomials
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

        # Visibility showing only trajectories (hide metal and contacts)
        trajectories_only_visible = []
        for i, trace in enumerate(fig_3d.data):
            if hasattr(trace, "name") and trace.name is not None:
                # Show only electrode trajectories and pre-tip polynomials
                if "Electrode" in trace.name or "Pre-Tip Polynomial" in trace.name:
                    # But not if it's metal or contacts
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
    summary_html = _generate_summary_html_with_profiles(
        data, include_intensity_profiles
    )

    # Generate 3D plot HTML if requested
    plot_3d_html = ""
    if include_3d_visualization:
        # Store output path for skull mesh export
        _add_3d_visualization_direct._output_path = str(output_path)

        # Convert the figure to JSON format for proper rendering
        fig_json = fig_3d.to_json()

        plot_3d_html = f"""
        <div class="plot-3d-container">
            <div id="plot-3d" style="width: 100%; height: 800px;"></div>
        </div>
        <script>
            var plot3dData = {fig_json};
            var plot3dLayout = plot3dData.layout;
            var plot3dTraces = plot3dData.data;
            
            // Force the layout to use full width
            plot3dLayout.autosize = true;
            plot3dLayout.width = undefined;  // Let Plotly calculate based on container
            
            var config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                toImageButtonOptions: {{
                    format: 'png',
                    filename: 'pypacer_3d_visualization',
                    height: 800,
                    width: 1200,
                    scale: 1
                }}
            }};
            
            // Create the plot
            Plotly.newPlot('plot-3d', plot3dTraces, plot3dLayout, config);
            
            // Force an initial resize after a short delay
            setTimeout(function() {{
                Plotly.Plots.resize('plot-3d');
            }}, 100);
        </script>
        """

    # Prepare JSON data for display
    json_formatted = json.dumps(data, indent=2)

    # Create final HTML with tab layout
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PyPaCER Reconstruction Report</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                color: #333;
            }}
            .header p {{
                margin: 5px 0 0 0;
                color: #666;
            }}
            /* Tab styling */
            .tabs {{
                display: flex;
                border-bottom: 2px solid #ddd;
                margin-bottom: 20px;
            }}
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-bottom: none;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
                transition: background-color 0.3s;
            }}
            .tab:hover {{
                background-color: #e9ecef;
            }}
            .tab.active {{
                background-color: white;
                border-bottom: 2px solid white;
                margin-bottom: -2px;
                font-weight: bold;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            /* Original styles */
            .summary {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }}
            .summary h2 {{
                margin-top: 0;
                color: #333;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }}
            .summary-item {{
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .summary-item h3 {{
                margin: 0 0 10px 0;
                color: #007bff;
                font-size: 14px;
            }}
            .summary-item p {{
                margin: 0;
                font-size: 18px;
                font-weight: bold;
            }}
            .plot-section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }}
            .plot-3d-container {{
                width: 100%;
                padding: 0;
                margin: 0;
            }}
            .electrode-details {{
                margin-top: 20px;
            }}
            .electrode-card {{
                margin-bottom: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}
            .electrode-card h4 {{
                margin: 0 0 15px 0;
                color: #333;
                font-size: 18px;
            }}
            .electrode-content {{
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 30px;
            }}
            .contact-details {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .contact-list {{
                margin-top: 10px;
            }}
            .contact-item {{
                display: grid;
                grid-template-columns: 80px 80px auto;
                gap: 10px;
                padding: 5px 0;
                border-bottom: 1px solid #eee;
                align-items: center;
            }}
            .contact-item:last-child {{
                border-bottom: none;
            }}
            .contact-item span:last-child {{
                font-family: 'Courier New', monospace;
                color: #555;
            }}
            .contact-3d-list {{
                margin-top: 8px;
            }}
            .contact-3d-item {{
                display: flex;
                gap: 10px;
                padding: 3px 0;
                font-size: 12px;
            }}
            .contact-3d-item span:first-child {{
                min-width: 30px;
                font-weight: 600;
            }}
            .contact-3d-item span:last-child {{
                font-family: 'Courier New', monospace;
                color: #555;
                white-space: nowrap;
            }}
            .intensity-plot {{
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                min-height: 400px;
                position: relative;
            }}
            .method-btn {{
                padding: 5px 12px;
                border: 1px solid #ddd;
                background-color: white;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s;
            }}
            .method-btn:hover {{
                background-color: #f0f0f0;
            }}
            .method-btn-active {{
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }}
            .method-btn-active:hover {{
                background-color: #0056b3;
            }}
            /* JSON display styling */
            .json-container {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
            }}
            .json-content {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 800px;
                overflow-y: auto;
                border: 1px solid #ddd;
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #666;
                font-size: 12px;
            }}
            /* Fullscreen modal styles */
            .fullscreen-modal {{
                display: none;
                position: fixed;
                z-index: 9999;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(0, 0, 0, 0.95);
            }}
            .fullscreen-modal.active {{
                display: block;
            }}
            .fullscreen-image-container {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                cursor: pointer;
            }}
            .fullscreen-image {{
                max-width: 98vw;
                max-height: 98vh;
                width: auto;
                height: auto;
                object-fit: contain;
                cursor: default;
                box-shadow: 0 0 30px rgba(0,0,0,0.8);
            }}
            .fullscreen-close {{
                position: absolute;
                top: 20px;
                right: 40px;
                color: #fff;
                font-size: 40px;
                font-weight: bold;
                transition: 0.3s;
                cursor: pointer;
                z-index: 1001;
            }}
            .fullscreen-close:hover,
            .fullscreen-close:focus {{
                color: #bbb;
                text-decoration: none;
                cursor: pointer;
            }}
            /* Volume rendering image container with maximize button */
            .volume-image-container {{
                position: relative;
                display: block;
                width: 100%;
                text-align: center;
            }}
            .volume-image-container img {{
                display: inline-block;
                max-width: 100%;
                height: auto;
            }}
            .maximize-btn {{
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px 10px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
                z-index: 10;
            }}
            .maximize-btn:hover {{
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }}
            .maximize-btn svg {{
                width: 16px;
                height: 16px;
                vertical-align: middle;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>PyPaCER Reconstruction Report</h1>
                <p>CT File: {Path(ct_file).name} | Generated: {timestamp}</p>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('summary')">Summary</div>
                <div class="tab" onclick="showTab('visualization3d')">3D Visualization</div>
                {('<div class="tab" onclick="showTab(' + "'volume'" + ')">Volume Rendering</div>') if include_volume_rendering else ""}
                <div class="tab" onclick="showTab('json')">Raw JSON</div>
            </div>
            
            <div id="summary" class="tab-content active">
                {summary_html}
            </div>
            
            <div id="visualization3d" class="tab-content">
                {plot_3d_html}
            </div>
            
            {('<div id="volume" class="tab-content">') if include_volume_rendering else ""}
            {_generate_volume_rendering_html(data) if include_volume_rendering else ""}
            {"</div>" if include_volume_rendering else ""}
            
            <div id="json" class="tab-content">
                <div class="json-container">
                    <h2>Raw Reconstruction Data</h2>
                    <div class="json-content">{json_formatted}</div>
                </div>
            </div>
            
            <div class="footer">
                Generated by PyPaCER | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>
        
        <script>
            function showTab(tabName) {{
                // Hide all tabs
                const tabs = document.querySelectorAll('.tab-content');
                tabs.forEach(tab => {{
                    tab.classList.remove('active');
                }});
                
                // Remove active class from all tab buttons
                const tabButtons = document.querySelectorAll('.tab');
                tabButtons.forEach(button => {{
                    button.classList.remove('active');
                }});
                
                // Show selected tab
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab button
                event.target.classList.add('active');
                
                // If showing the 3D visualization tab, trigger a resize
                if (tabName === 'visualization3d') {{
                    setTimeout(function() {{
                        window.dispatchEvent(new Event('resize'));
                        Plotly.Plots.resize('plot-3d');
                    }}, 100);
                }}
                
                // Volume tab doesn't need special handling as it uses static images
            }}
            
            // Also trigger resize on window resize
            window.addEventListener('resize', function() {{
                if (document.getElementById('visualization3d').classList.contains('active')) {{
                    Plotly.Plots.resize('plot-3d');
                }}
            }});
            
            // Fullscreen image functionality
            let fullscreenModal = null;
            
            function createFullscreenModal() {{
                if (!fullscreenModal) {{
                    fullscreenModal = document.createElement('div');
                    fullscreenModal.className = 'fullscreen-modal';
                    fullscreenModal.innerHTML = `
                        <span class="fullscreen-close">&times;</span>
                        <div class="fullscreen-image-container">
                            <img class="fullscreen-image">
                        </div>
                    `;
                    document.body.appendChild(fullscreenModal);
                    
                    // Close on modal click
                    fullscreenModal.addEventListener('click', function(e) {{
                        if (e.target === fullscreenModal || e.target.className === 'fullscreen-close' || e.target.className === 'fullscreen-image-container') {{
                            closeFullscreen();
                        }}
                    }});
                    
                    // Close on Escape key
                    document.addEventListener('keydown', function(e) {{
                        if (e.key === 'Escape' && fullscreenModal.classList.contains('active')) {{
                            closeFullscreen();
                        }}
                    }});
                }}
                return fullscreenModal;
            }}
            
            function showFullscreen(imgSrc) {{
                const modal = createFullscreenModal();
                const img = modal.querySelector('.fullscreen-image');
                img.src = imgSrc;
                modal.classList.add('active');
                document.body.style.overflow = 'hidden';
            }}
            
            function closeFullscreen() {{
                if (fullscreenModal) {{
                    fullscreenModal.classList.remove('active');
                    document.body.style.overflow = '';
                }}
            }}
            
            // Initialize maximize buttons after page load
            document.addEventListener('DOMContentLoaded', function() {{
                // Add click handlers to all maximize buttons
                document.querySelectorAll('.maximize-btn').forEach(btn => {{
                    btn.addEventListener('click', function(e) {{
                        e.stopPropagation();
                        const img = this.parentElement.querySelector('img');
                        if (img) {{
                            showFullscreen(img.src);
                        }}
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)

    return str(output_path)


def _add_3d_visualization_direct(
    fig: go.Figure, data: Dict[str, Any], cached_mesh: Optional[go.Mesh3d] = None
):
    """Add 3D electrode visualization directly to figure without subplots."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Try to extract metal mesh if CT data is available
    ct_path = data.get("metadata", {}).get("ct_file")
    if ct_path and Path(ct_path).exists():
        # Get output directory from context if available
        output_dir = None
        if hasattr(_add_3d_visualization_direct, "_output_path"):
            output_dir = Path(_add_3d_visualization_direct._output_path).parent

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
        _add_bounding_box_direct(
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
            from ..utils.math_helpers import polyval3

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

        # Plot bounding box
        if "bounding_box" in electrode:
            bbox = electrode["bounding_box"]
            _add_bounding_box_direct(
                fig, bbox["min"], bbox["max"], color=color, name=f"Bbox {i+1}"
            )


def _add_bounding_box_direct(
    fig: go.Figure,
    min_coords: List[float],
    max_coords: List[float],
    color: str,
    name: str,
):
    """Add a wireframe bounding box to the 3D plot without subplots."""
    # Define the 8 corners of the bounding box
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords

    # Define the 12 edges of the box
    edges = [
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


def _add_3d_visualization(fig: go.Figure, data: Dict[str, Any], row: int, col: int):
    """Add 3D electrode visualization to subplot."""
    print(f"[DEBUG] _add_3d_visualization called with row={row}, col={col}")
    print(f"[DEBUG] Number of electrodes in data: {len(data.get('electrodes', []))}")
    print(f"[DEBUG] Figure type: {type(fig)}")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Add CT volume bounding box if available
    if "metadata" in data and "ct_volume_bounding_box" in data["metadata"]:
        ct_bbox = data["metadata"]["ct_volume_bounding_box"]
        _add_bounding_box(
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
            print(
                f"[DEBUG] Electrode {i}: Found trajectory_coordinates with shape {trajectory.shape}"
            )
            print(
                f"[DEBUG] Trajectory X range: {trajectory[:, 0].min():.2f} to {trajectory[:, 0].max():.2f}"
            )
            print(
                f"[DEBUG] Trajectory Y range: {trajectory[:, 1].min():.2f} to {trajectory[:, 1].max():.2f}"
            )
            print(
                f"[DEBUG] Trajectory Z range: {trajectory[:, 2].min():.2f} to {trajectory[:, 2].max():.2f}"
            )

            # Convert to lists to ensure proper serialization
            x_data = trajectory[:, 0].tolist()
            y_data = trajectory[:, 1].tolist()
            z_data = trajectory[:, 2].tolist()
            print(
                f"[DEBUG] Converted trajectory to lists - X: {len(x_data)}, Y: {len(y_data)}, Z: {len(z_data)} points"
            )

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
            print(f"[DEBUG] Added trajectory trace for Electrode {i+1}")
        else:
            print(f"[DEBUG] Electrode {i}: No trajectory_coordinates found")

        # Plot contact positions
        if "contact_positions_3d" in electrode:
            contacts = np.array(electrode["contact_positions_3d"])
            print(
                f"[DEBUG] Electrode {i}: Found contact_positions_3d with shape {contacts.shape}"
            )
            print(
                f"[DEBUG] Contacts X range: {contacts[:, 0].min():.2f} to {contacts[:, 0].max():.2f}"
            )
            print(
                f"[DEBUG] Contacts Y range: {contacts[:, 1].min():.2f} to {contacts[:, 1].max():.2f}"
            )
            print(
                f"[DEBUG] Contacts Z range: {contacts[:, 2].min():.2f} to {contacts[:, 2].max():.2f}"
            )

            # Convert to lists to ensure proper serialization
            x_data = contacts[:, 0].tolist()
            y_data = contacts[:, 1].tolist()
            z_data = contacts[:, 2].tolist()
            print(
                f"[DEBUG] Converted contacts to lists - X: {len(x_data)}, Y: {len(y_data)}, Z: {len(z_data)} points"
            )

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
            print(f"[DEBUG] Added contact trace for Electrode {i+1}")
        else:
            print(f"[DEBUG] Electrode {i}: No contact_positions_3d found")

        # Plot bounding box
        if "bounding_box" in electrode:
            bbox = electrode["bounding_box"]
            _add_bounding_box(
                fig,
                bbox["min"],
                bbox["max"],
                color=color,
                name=f"Bbox {i+1}",
                row=row,
                col=col,
            )

    # Debug: Check figure state after adding all traces
    print(
        f"[DEBUG] Total traces in figure after _add_3d_visualization: {len(fig.data)}"
    )
    for idx, trace in enumerate(fig.data):
        if hasattr(trace, "name"):
            print(f"[DEBUG] Trace {idx}: {trace.name} - Type: {type(trace).__name__}")


def _add_bounding_box(
    fig: go.Figure,
    min_coords: List[float],
    max_coords: List[float],
    color: str,
    name: str,
    row: int,
    col: int,
):
    """Add a wireframe bounding box to the 3D plot."""
    # Define the 8 corners of the bounding box
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords

    # Define the 12 edges of the box
    edges = [
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


def _generate_summary_html_with_profiles(
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

        html += f"""
            <div class="electrode-card">
                <h4>Electrode {i+1} - {electrode_type} - {hemisphere}</h4>
                <div class="electrode-content">
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
                # Create divs for each method's contact positions
                contact_details_id = f"contact-details-{i}"
                primary_positions = (
                    contact_positions  # These are from the primary method
                )

                html += f"""
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
                        if not is_primary and len(method_positions) == len(
                            primary_positions
                        ):
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
            else:
                # Single method - original behavior
                # Get method display name
                method_display_names = {
                    "contactAreaCenter": "Area Center",
                    "peak": "Peak",
                    "peakWaveCenter": "Peak Wave Center",
                }
                method_display = method_display_names.get(
                    primary_method, primary_method
                )

                html += f"""
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

        html += """
                    </div>
        """

        # Add intensity profile if available
        if (
            include_intensity_profiles
            and "intensity_profile" in electrode
            and "distance_scale" in electrode
        ):
            plot_id = f"intensity-plot-{i}"
            has_multiple_methods = "contact_detection_results" in electrode

            if has_multiple_methods:
                # Add method selector buttons
                html += f"""
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
                </div>
            </div>
                """
            else:
                html += f"""
                    <div class="intensity-plot" id="{plot_id}"></div>
                </div>
            </div>
                """

            # Add the plot script
            intensity_data = _create_intensity_profile_json(
                electrode, i, color, primary_method
            )

            if has_multiple_methods:
                # Multi-method plotting with toggle functionality
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
                        var plotContainer = document.getElementById('{plot_id}').parentElement;
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
                        plotContainer.appendChild(toggleBtn);
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
            else:
                # Single method plotting
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
                        var plotContainer = document.getElementById('{plot_id}').parentElement;
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
                        plotContainer.appendChild(toggleBtn);
                    }}
                }})();
            </script>
                """
        else:
            html += """
                    <div class="intensity-plot" style="display: flex; align-items: center; justify-content: center;">
                        <p style="color: #999;">No intensity profile available</p>
                    </div>
                </div>
            </div>
            """

    html += """
        </div>
    """

    # Add debug intensity plot if full Pass 2 data is available
    if any("pass2_intensities_full" in e for e in data["electrodes"]):
        html += """
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

    html += """
    </div>
    """

    return html


def _create_intensity_profile_json(
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
        # Create traces for each method
        all_traces = {}
        method_colors = {
            "contactAreaCenter": "#d62728",  # red
            "peak": "#2ca02c",  # green
            "peakWaveCenter": "#ff7f0e",  # orange
        }

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

                all_traces[method] = method_traces

        return json.dumps(
            {
                "methods": all_traces,
                "default_method": primary_method,
                "has_skeleton_deviations": has_skeleton_deviations,
            }
        )
    else:
        # Single method - original behavior
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

        return json.dumps(
            {"traces": traces, "has_skeleton_deviations": has_skeleton_deviations}
        )


def _generate_summary_stats(data: Dict[str, Any]) -> str:
    """Generate HTML summary statistics section."""
    metadata = data.get("metadata", {})
    params = data.get("reconstruction_parameters", {})

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


def _add_contact_comparison(
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


def _add_intensity_profile(
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


def _generate_volume_rendering_html(data: Dict[str, Any]) -> str:
    """Generate HTML content for volume rendering tab."""
    print("\n[DEBUG] _generate_volume_rendering_html called")

    try:
        # Try to import PyVista
        import nibabel as nib
        import pyvista as pv

        pyvista_available = True
        print("[DEBUG] PyVista imported successfully")
    except ImportError:
        pyvista_available = False
        print("[DEBUG] PyVista not available")

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
    print(f"[DEBUG] CT path from metadata: {ct_path}")

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
        print(f"[DEBUG] Loading CT data from: {ct_path}")
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        affine = ct_nii.affine
        print(
            f"[DEBUG] CT loaded successfully: shape={ct_data.shape}, affine det={np.linalg.det(affine):.2f}"
        )

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
                    # We have both static image and GIF
                    static_image, gif_image = volume_result
                    html += f"""
                    <div class="electrode-card">
                        <h4>Electrode {i+1} - {electrode_type} - {hemisphere}</h4>
                        <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
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
                            <div style="text-align: center;">
                                <p style="margin: 5px 0; font-weight: bold;">Contacts Region (Rotating)</p>
                                <div style="display: inline-block;">
                                    <img src="data:image/gif;base64,{gif_image}" 
                                         style="max-width: 600px; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                                </div>
                            </div>
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
) -> Optional[str]:
    """Generate PyVista volume renderings for a single electrode and return as base64 images."""
    print(
        f"\n[DEBUG] _generate_electrode_volume_rendering for electrode {electrode_idx+1}"
    )

    try:
        import pyvista as pv
        from PIL import Image

        # Extract subvolume
        subvolume, sub_affine, bbox = _extract_electrode_subvolume(
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
        if "contact_positions_3d" in electrode:
            contacts_3d = np.array(electrode["contact_positions_3d"])
            for i, contact in enumerate(contacts_3d):
                sphere = pv.Sphere(center=contact, radius=0.25)
                plotter1.add_mesh(sphere, color="red", opacity=1.0)

        # Add polynomial trajectory
        if "polynomial" in electrode:
            from ..utils.math_helpers import polyval3

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

        # Image 2: Zoomed contacts view with electrode pointing up
        if (
            "contact_positions_3d" in electrode
            and len(electrode["contact_positions_3d"]) >= 2
        ):
            # Extract contact region subvolume
            contacts_subvolume, contacts_sub_affine, contacts_bbox = (
                _extract_contacts_subvolume(ct_data, affine, electrode, padding_mm=2.0)
            )

            if contacts_subvolume is not None:
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
                    from ..utils.math_helpers import polyval3

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

                # Center of contacts
                center_contacts = contacts_3d.mean(axis=0)

                # For contacts view, electrode axis should point up
                # This means the up vector should be the electrode axis
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

        # If we can't create contacts view, just return the full view
        img1 = Image.fromarray(image1)
        buffer = io.BytesIO()
        img1.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return image_base64

    except Exception as e:
        print(
            f"[DEBUG] Error in _generate_electrode_volume_rendering for electrode {electrode_idx+1}: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def _extract_contacts_subvolume(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    padding_mm: float = 10.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Extract subvolume around contact positions only."""
    if "contact_positions_3d" not in electrode:
        return None, None, None

    contacts_3d = np.array(electrode["contact_positions_3d"])

    # Calculate bounding box around contacts with padding
    bbox = {
        "min": contacts_3d.min(axis=0) - padding_mm,
        "max": contacts_3d.max(axis=0) + padding_mm,
        "center": contacts_3d.mean(axis=0),
        "size": contacts_3d.max(axis=0) - contacts_3d.min(axis=0) + 2 * padding_mm,
    }

    # Convert world coordinates to voxel indices
    inv_affine = np.linalg.inv(affine)

    # Transform bbox corners to voxel space
    min_world = np.append(bbox["min"], 1)
    max_world = np.append(bbox["max"], 1)

    min_voxel = inv_affine @ min_world
    max_voxel = inv_affine @ max_world

    # Handle flipped axes by ensuring voxel min < max
    voxel_min = np.minimum(min_voxel[:3], max_voxel[:3])
    voxel_max = np.maximum(min_voxel[:3], max_voxel[:3])

    # Get integer voxel indices (with bounds checking)
    min_idx = np.maximum(0, np.floor(voxel_min).astype(int))
    max_idx = np.minimum(ct_data.shape, np.ceil(voxel_max).astype(int))

    # Extract subvolume
    subvolume_data = ct_data[
        min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2]
    ]

    # Handle negative axes (following native renderer approach)
    subvolume_affine = affine.copy()

    for axis in range(3):
        if affine[axis, axis] < 0:
            # Flip the data along this axis
            subvolume_data = np.flip(subvolume_data, axis=axis)
            # Make spacing positive
            subvolume_affine[axis, axis] = -affine[axis, axis]
            # Adjust origin (accounting for flip)
            subvolume_affine[axis, 3] = affine[axis, 3] + affine[axis, axis] * (
                max_idx[axis] - 1
            )
        else:
            # Positive spacing - just adjust origin
            subvolume_affine[axis, 3] = (
                affine[axis, 3] + affine[axis, axis] * min_idx[axis]
            )

    # IMPORTANT: Adjust origin from voxel center to voxel corner
    # NiBabel affine maps voxel centers, PyVista expects corner at (-0.5, -0.5, -0.5)
    voxel_shift = np.array([-0.5, -0.5, -0.5, 1])
    corner_shift = subvolume_affine @ voxel_shift
    subvolume_affine[:3, 3] = corner_shift[:3]

    return subvolume_data, subvolume_affine, bbox


def _extract_electrode_subvolume(
    ct_data: np.ndarray,
    affine: np.ndarray,
    electrode: Dict[str, Any],
    padding_mm: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Extract subvolume for a single electrode from CT data."""
    print("\n[DEBUG] _extract_electrode_subvolume called")

    # Use the bounding box from the reconstruction JSON if available
    if "bounding_box" in electrode:
        bbox_data = electrode["bounding_box"]
        # Apply padding to the existing bounding box
        bbox = {
            "min": np.array(bbox_data["min"]) - padding_mm,
            "max": np.array(bbox_data["max"]) + padding_mm,
            "center": (np.array(bbox_data["min"]) + np.array(bbox_data["max"])) / 2,
            "size": np.array(bbox_data["max"])
            - np.array(bbox_data["min"])
            + 2 * padding_mm,
        }
    elif "contact_positions_3d" in electrode:
        # Fallback to calculating from contacts if no bounding box is provided
        contacts_3d = np.array(electrode["contact_positions_3d"])

        # Calculate bounding box with padding
        bbox = {
            "min": contacts_3d.min(axis=0) - padding_mm,
            "max": contacts_3d.max(axis=0) + padding_mm,
            "center": contacts_3d.mean(axis=0),
            "size": contacts_3d.max(axis=0) - contacts_3d.min(axis=0) + 2 * padding_mm,
        }
    else:
        return None, None, None

    # Convert world coordinates to voxel indices
    inv_affine = np.linalg.inv(affine)

    # Transform bbox corners to voxel space
    min_world = np.append(bbox["min"], 1)
    max_world = np.append(bbox["max"], 1)

    min_voxel = inv_affine @ min_world
    max_voxel = inv_affine @ max_world

    # Handle flipped axes by ensuring voxel min < max
    voxel_min = np.minimum(min_voxel[:3], max_voxel[:3])
    voxel_max = np.maximum(min_voxel[:3], max_voxel[:3])

    # Get integer voxel indices (with bounds checking)
    min_idx = np.maximum(0, np.floor(voxel_min).astype(int))
    max_idx = np.minimum(ct_data.shape, np.ceil(voxel_max).astype(int))

    # Extract subvolume
    subvolume_data = ct_data[
        min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2]
    ]

    # Handle negative axes (following native renderer approach)
    subvolume_affine = affine.copy()

    for axis in range(3):
        if affine[axis, axis] < 0:
            # Flip the data along this axis
            subvolume_data = np.flip(subvolume_data, axis=axis)
            # Make spacing positive
            subvolume_affine[axis, axis] = -affine[axis, axis]
            # Adjust origin (accounting for flip)
            subvolume_affine[axis, 3] = affine[axis, 3] + affine[axis, axis] * (
                max_idx[axis] - 1
            )
        else:
            # Positive spacing - just adjust origin
            subvolume_affine[axis, 3] = (
                affine[axis, 3] + affine[axis, axis] * min_idx[axis]
            )

    # IMPORTANT: Adjust origin from voxel center to voxel corner
    # NiBabel affine maps voxel centers, PyVista expects corner at (-0.5, -0.5, -0.5)
    voxel_shift = np.array([-0.5, -0.5, -0.5, 1])
    corner_shift = subvolume_affine @ voxel_shift
    subvolume_affine[:3, 3] = corner_shift[:3]

    print(f"[DEBUG] Subvolume shape: {subvolume_data.shape}")
    print(f"[DEBUG] Original affine diagonal: {np.diag(affine[:3, :3])}")
    print(f"[DEBUG] Subvolume affine diagonal: {np.diag(subvolume_affine[:3, :3])}")

    return subvolume_data, subvolume_affine, bbox


# Removed _extract_skull_mesh function for performance reasons


# Removed _extract_electrode_mesh - now imported from isosurface_extraction module


# Removed _save_mesh_as_obj - now in isosurface_extraction module
