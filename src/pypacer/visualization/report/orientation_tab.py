"""Orientation analysis visualizations for the HTML report."""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def generate_orientation_html(data: Dict[str, Any]) -> str:
    """Generate HTML content for the orientation tab.

    Builds per-electrode sections with:
    - Unwrapped marker intensity heatmap
    - Combined marker polar intensity chart (A, B, and inferred C)
    - Contact region polar chart with 6-dir fit overlay and marker directions
    - Contact region 2D Cartesian chart with 6-dir vertical lines
    - Summary sidebar with key orientation parameters
    """
    electrodes = data.get("electrodes", [])
    metadata = data.get("metadata", {})

    html = ""
    for i, electrode in enumerate(electrodes):
        orientation = electrode.get("orientation")
        if not orientation:
            continue
        html += _generate_electrode_orientation_html(electrode, i, orientation, metadata)

    if not html:
        html = """
        <div class="plot-section">
            <h2>Electrode Orientation</h2>
            <p style="color: #666; padding: 40px; text-align: center;">
                No orientation data available.
            </p>
        </div>
        """

    return html


def _generate_electrode_orientation_html(
    electrode: Dict[str, Any],
    electrode_idx: int,
    orientation: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """Generate orientation HTML for a single electrode."""
    electrode_type = electrode.get("electrode_type", "Unknown")
    hemisphere = electrode.get("side", "unknown").capitalize()

    markers = orientation.get("markers", {})

    # --- Header info line ---
    header_info = _build_header_info(markers)

    # --- Marker region heatmap (from CT data) ---
    heatmap_html = _build_marker_heatmap_html(electrode, markers, metadata)

    # --- Marker polar chart ---
    marker_charts_html = _build_marker_charts_html(
        electrode_idx, markers,
    )

    return f"""
    <div class="electrode-card">
        <h4>Electrode {electrode_idx + 1} &mdash; {electrode_type} &mdash; {hemisphere}{header_info}</h4>
        <div style="display: flex; gap: 10px; align-items: start;">
            {heatmap_html}
            <div style="flex: 1;">{marker_charts_html}</div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Header info
# ---------------------------------------------------------------------------

def _build_header_info(
    markers: Dict[str, Any],
) -> str:
    """Build inline header info to append after the electrode title."""
    parts = []

    for label in ("B", "A"):
        m = markers.get(label)
        if not m:
            continue
        fit = m.get("fitted_angle_traj_perp_deg")
        det = m.get("detected_angle_traj_perp_deg")
        angle = fit if fit is not None else det
        if angle is not None:
            parts.append(f"Marker {label}: {angle:.1f}&deg;")

    if not parts:
        return ""

    info = " | ".join(parts)
    return f' <span style="font-weight: normal; font-size: 13px; color: #666;">&mdash; {info}</span>'


# ---------------------------------------------------------------------------
# Marker polar chart
# ---------------------------------------------------------------------------

def _compute_marker_c_angle(fit_b: float, fit_a: float) -> float:
    """Compute Marker C angle — the third 120-degree direction.

    Given fitted B and A (constrained 120 apart), C completes the triplet.
    The triplet is {fit_b, fit_b+120, fit_b+240}.  One of fit_b+120 or
    fit_b+240 matches A; the other is C.
    """
    candidates = [(fit_b + 120) % 360, (fit_b + 240) % 360]
    # Pick the one farthest from A (the other is A itself)
    diffs = [abs((c - fit_a + 180) % 360 - 180) for c in candidates]
    return candidates[0] if diffs[0] > diffs[1] else candidates[1]


def _build_marker_charts_html(
    electrode_idx: int,
    markers: Dict[str, Any],
    id_suffix: str = "",
) -> str:
    """Build a single combined polar chart with markers B, A, and inferred C."""
    marker_data = []  # [(label, marker_dict, profile)]
    for label in ("B", "A"):
        m = markers.get(label)
        if not m:
            continue
        profile = m.get("intensity_profile")
        if not profile:
            continue
        marker_data.append((label, m, profile))

    if not marker_data:
        return ""

    plot_id = f"orient-markers-{electrode_idx}{id_suffix}"
    marker_colors = {"B": "#1f77b4", "A": "#ff7f0e"}
    marker_fill = {"B": "rgba(31,119,180,0.08)", "A": "rgba(255,127,14,0.08)"}
    fit_colors = {"B": "#1f77b4", "A": "#ff7f0e", "C": "#2ca02c"}

    # Compute Marker C angle if we have both fitted directions
    fit_b = markers.get("B", {}).get("fitted_angle_traj_perp_deg")
    fit_a = markers.get("A", {}).get("fitted_angle_traj_perp_deg")
    marker_c_angle = None
    if fit_b is not None and fit_a is not None:
        marker_c_angle = _compute_marker_c_angle(fit_b, fit_a)

    # Build subtitle from both markers + C
    subtitle_parts = []
    for label, m, _ in marker_data:
        dist = m.get("distance_from_tip_mm", 0)
        fit = m.get("fitted_angle_traj_perp_deg")
        part = f"{label} ({dist:.1f} mm)"
        if fit is not None:
            part += f": {fit:.1f}°"
        subtitle_parts.append(part)
    if marker_c_angle is not None:
        subtitle_parts.append(f"C (inferred): {marker_c_angle:.1f}°")
    subtitle = " — ".join(subtitle_parts)

    # Build traces and JS variable declarations
    traces = []
    var_decls = ""
    for label, m, profile in marker_data:
        color = marker_colors[label]
        fill = marker_fill[label]
        fit_color = fit_colors[label]
        det_angle = m.get("detected_angle_traj_perp_deg")
        fit_angle = m.get("fitted_angle_traj_perp_deg")

        var_decls += f"var int_{label} = {json.dumps(profile['mean_intensity'])};\n"
        angle_step = profile.get('angle_step_deg', 0.1)
        var_decls += f"var angles_{label} = Array.from({{length: int_{label}.length}}, (_, i) => i * {angle_step});\n"

        # Intensity profile trace
        traces.append(f"""{{
            r: int_{label},
            theta: angles_{label},
            type: 'scatterpolar',
            mode: 'lines',
            name: 'Marker {label}',
            line: {{ color: '{color}', width: 1.5 }},
            fill: 'toself',
            fillcolor: '{fill}'
        }}""")

        # Detected angle line (dashed, marker colour)
        if det_angle is not None:
            traces.append(f"""{{
            r: [rMin - rPad, rMax + rPad],
            theta: [{det_angle}, {det_angle}],
            type: 'scatterpolar',
            mode: 'lines',
            name: '{label} detected',
            line: {{ color: '{color}', width: 1.5, dash: 'dash' }},
            showlegend: false
        }}""")

        # Fitted angle line (solid, marker colour)
        if fit_angle is not None:
            traces.append(f"""{{
            r: [rMin - rPad, rMax + rPad],
            theta: [{fit_angle}, {fit_angle}],
            type: 'scatterpolar',
            mode: 'lines',
            name: '{label} fitted',
            line: {{ color: '{fit_color}', width: 2.5 }},
            showlegend: true
        }}""")

    # Marker C fitted line (inferred, green)
    if marker_c_angle is not None:
        traces.append(f"""{{
            r: [rMin - rPad, rMax + rPad],
            theta: [{marker_c_angle}, {marker_c_angle}],
            type: 'scatterpolar',
            mode: 'lines',
            name: 'C inferred',
            line: {{ color: '{fit_colors["C"]}', width: 2.5 }},
            showlegend: true
        }}""")

    # Compute global rMin/rMax across all markers
    if len(marker_data) == 1:
        label0 = marker_data[0][0]
        range_js = f"""
            var rMin = Math.min.apply(null, int_{label0});
            var rMax = Math.max.apply(null, int_{label0});
            var rPad = (rMax - rMin) * 0.05;
        """
    else:
        labels = [md[0] for md in marker_data]
        range_js = f"""
            var allInt = int_{labels[0]}.concat(int_{labels[1]});
            var rMin = Math.min.apply(null, allInt);
            var rMax = Math.max.apply(null, allInt);
            var rPad = (rMax - rMin) * 0.05;
        """

    traces_js = "[" + ",".join(traces) + "]"

    return f"""
    <div>
        <div class="intensity-plot" id="{plot_id}" style="min-height:450px;"></div>
        <script>
        (function() {{
            {var_decls}
            {range_js}
            var traces = {traces_js};

            var layout = {{
                title: {{
                    text: 'Marker Intensity Profiles<br><sup>{subtitle}</sup>',
                    font: {{ size: 14 }}
                }},
                polar: {{
                    radialaxis: {{
                        range: [rMin - rPad, rMax + rPad],
                        tickfont: {{ size: 10 }},
                        angle: 90,
                        tickangle: 90
                    }},
                    angularaxis: {{
                        direction: 'counterclockwise',
                        rotation: 90,
                        tickfont: {{ size: 10 }}
                    }}
                }},
                showlegend: true,
                legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center', font: {{ size: 10 }} }},
                margin: {{ l: 40, r: 40, t: 70, b: 30 }},
                height: 480
            }};
            Plotly.newPlot('{plot_id}', traces, layout, {{responsive: true}});
        }})();
        </script>
    </div>
    """


# ---------------------------------------------------------------------------
def _build_marker_heatmap_html(
    electrode: Dict[str, Any],
    markers: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """Generate an unwrapped intensity heatmap of the marker region as a base64 PNG."""
    if not markers:
        return ""

    polynomial = electrode.get("polynomial")
    if not polynomial:
        return ""

    ct_path = metadata.get("ct_file")
    if not ct_path or not Path(ct_path).exists():
        return ""

    # Get marker region bounds from marker positions with padding
    marker_dists = []
    for label in ("B", "A"):
        m = markers.get(label)
        if m and "distance_from_tip_mm" in m:
            marker_dists.append(m["distance_from_tip_mm"])
    if not marker_dists:
        return ""

    region_start = min(marker_dists) - 2.0
    region_end = max(marker_dists) + 2.0
    region_start = max(0.0, region_start)

    try:
        # Load CT data (use same cache as intensity_profiles)
        from .intensity_profiles import _ct_cache

        if ct_path not in _ct_cache:
            import nibabel as nib
            ct_nii = nib.load(ct_path)
            _ct_cache[ct_path] = {
                "data": ct_nii.get_fdata(),
                "affine": ct_nii.affine,
            }
        ct = _ct_cache[ct_path]

        from ...orientation.marker_profile import sample_full_marker_profile
        from ...utils.math_helpers import inv_poly_arc_length_3d

        poly = np.array(polynomial)
        deriv_coeffs = poly[:-1] * np.arange(len(poly) - 1, 0, -1)[:, np.newaxis]

        def _polyval3(coeffs, t):
            result = np.zeros(3)
            for c in coeffs:
                result = result * t + c
            return result

        def trajectory_direction_func(polynomial_arg, distance_mm):
            t = inv_poly_arc_length_3d(polynomial_arg, distance_mm)
            tangent = _polyval3(deriv_coeffs, t)
            return tangent / np.linalg.norm(tangent)

        profile_data = sample_full_marker_profile(
            ct_data=ct["data"],
            affine=ct["affine"],
            electrode_polynomial=poly,
            trajectory_direction_func=trajectory_direction_func,
            distance_start_mm=region_start,
            distance_end_mm=region_end,
            distance_step_mm=0.2,
            radii_mm=[1.25, 1.5, 1.75],
            angle_increment_deg=0.1,
        )

        distances = profile_data["distances"]
        angles_deg = profile_data["angles_deg"]
        intensity_grid = profile_data["intensity_grid"]

        # Get detected and fitted angles for overlays
        marker_angles = []  # [(label, detected, fitted)]
        for label in ("B", "A"):
            m = markers.get(label, {})
            det = m.get("detected_angle_traj_perp_deg")
            fit = m.get("fitted_angle_traj_perp_deg")
            if det is not None or fit is not None:
                marker_angles.append((label, det, fit))

        # Render heatmap to base64 PNG
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.pcolormesh(angles_deg, distances, intensity_grid,
                           cmap="viridis", shading="auto")

        # Marker position lines (white)
        for i, dist in enumerate(marker_dists):
            mlabel = ["B", "A"][i] if i < 2 else f"M{i+1}"
            ax.axhline(dist, color="white", linestyle="-", linewidth=1.5, alpha=0.7,
                       label=f"Marker {mlabel}: {dist:.1f} mm")

        # Detected (dashed) and fitted (solid) angle lines
        line_colors = {"B": "#1f77b4", "A": "#ff7f0e"}
        for label, det, fit in marker_angles:
            c = line_colors.get(label, "#1f77b4")
            if det is not None:
                ax.axvline(det, color=c, linestyle="--", linewidth=1.5, alpha=0.7,
                           label=f"Detected {label}: {det:.1f}°")
            if fit is not None:
                ax.axvline(fit, color=c, linestyle="-", linewidth=2, alpha=0.9,
                           label=f"Fitted {label}: {fit:.1f}°")

        ax.set_xlabel("Angle (°)", fontsize=10)
        ax.set_ylabel("Distance from tip (mm)", fontsize=10)
        ax.set_title("Marker Region Heatmap", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return f"""
            <div style="flex-shrink: 0;">
                <img src="data:image/png;base64,{b64}"
                     style="height: 480px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
        """

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ""
