"""Interactive GUI for electrode detection from CT scans."""

import matplotlib

matplotlib.use("Qt5Agg")  # Use Qt5Agg backend for interactive GUI
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from .._version import __version__ as PYPACER_VERSION
from ..core.cog_trajectory_tracking import CenterOfGravityTracker, TrajectoryPoint
from ..core.contact_detection import detect_contacts
from ..core.polynomial_fitting import fit_polynomial_to_trajectory
from ..core.pypacer import PyPaCER
from ..imaging.preprocessing import detect_metal_artifacts, extract_brain_mask
from ..models.electrode import ELECTRODE_GEOMETRIES, PolynomialElectrodeModel


class PyPaCERGUI:
    """Interactive GUI for electrode detection and visualization."""

    def __init__(
        self,
        ct_path: Optional[Union[str, Path]] = None,
        metal_threshold: float = 2000,
        slice_axis: str = "axial",
        output_dir: Optional[Union[str, Path]] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize electrode detection GUI.

        Args:
            ct_path: Path to CT NIfTI file (optional - can load from GUI)
            metal_threshold: HU threshold for metal detection
            slice_axis: Initial slice view ('axial', 'sagittal', 'coronal')
            output_dir: Custom output directory for saving results (default: <CT_dir>/pypacer/)
            debug_mode: Enable debug mode for additional outputs and verbose logging
        """
        self.ct_path = Path(ct_path) if ct_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.debug_mode = debug_mode
        self.metal_threshold = metal_threshold

        # Initialize data attributes as None - will be set when CT is loaded
        self.pacer = None
        self.ct_data = None
        self.affine = None
        self.voxel_sizes = None

        # GUI state
        self.slice_axis = slice_axis.lower()
        self.axis_map = {"axial": 2, "sagittal": 0, "coronal": 1}
        self.slice_idx = self.axis_map[self.slice_axis]
        self.current_slice = 0  # Will be set when CT is loaded

        # Detection state - initialize before loading CT
        self.electrodes: List[PolynomialElectrodeModel] = []
        self.seed_points: List[Tuple[int, int, int]] = []
        self.cog_trajectories: List[List[TrajectoryPoint]] = (
            []
        )  # Store COG tracking results
        self.refined_trajectories: List[Any] = []  # Store OOR refined trajectories
        self.detection_method: str = (
            "detect_electrodes_radial"  # Track if auto or manual
        )

        # Cached mesh for fast HTML export
        self.electrode_mesh = None
        self.mesh_extraction_thread = None

        # Load CT if provided (after initializing detection state)
        if self.ct_path:
            self._load_ct_file(self.ct_path)
        self.brain_mask: Optional[np.ndarray] = None
        self.metal_mask: Optional[np.ndarray] = None
        self.metal_components: Optional[np.ndarray] = (
            None  # Labeled connected components
        )
        self.num_components: int = 0
        self.show_components: bool = True  # Toggle for component visualization

        # COG tracker - will be initialized when CT is loaded
        self.cog_tracker = None

        # GUI components
        self.fig = None
        self.ax_ct = None
        self.ax_3d = None
        self.controls = {}
        self.slider = None
        self.coord_text = None
        self.result_text = None
        self.contact_info_text = None  # New text for contact information
        self.crosshair_h = None
        self.crosshair_v = None
        self.hover_rect = None

        # Button states
        self.button_enabled = {
            "auto_run": True,  # Initially enabled
            "run_oor": False,
            "save_json": False,
            "save_html": False,
        }
        # Button overlays for visual state
        self.button_overlays = {}

        # Setup GUI
        self._setup_gui()

        # Initialize components if CT is loaded
        if self.ct_data is not None:
            # Initialize COG tracker if not already done
            if self.cog_tracker is None:
                self.cog_tracker = CenterOfGravityTracker(
                    ct_data=self.ct_data,
                    affine=self.affine,
                    metal_threshold=self.metal_threshold,
                    search_radius_mm=5.0,  # Use consistent 5mm physical radius
                    max_direction_change_deg=60.0,
                    min_voxels_per_slice=3,
                )
            self._compute_initial_metal_mask()
            self._update_ct_display()

        # Initialize button states
        self._update_button_states()

    def _setup_gui(self) -> None:
        """Set up the GUI layout and controls."""
        # Create figure with optimized layout
        self.fig = plt.figure(figsize=(20, 11))
        title = (
            f"Electrode Detection - {self.ct_path.name}"
            if self.ct_path
            else "Electrode Detection - No CT Loaded"
        )
        self.fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        # CT slice view (left side, upper) - larger and no labels
        self.ax_ct = self.fig.add_axes([0.02, 0.45, 0.42, 0.48])
        self.ax_ct.set_aspect("equal")
        self.ax_ct.set_title(f"{self.slice_axis.capitalize()} View")

        # 3D visualization (right side, upper) - 50% larger
        self.ax_3d = self.fig.add_axes([0.48, 0.35, 0.50, 0.58], projection="3d")
        self.ax_3d.set_title("Detected Electrodes & COG Trajectories")

        # Intensity profile (right side, lower) - full width
        self.ax_profile = self.fig.add_axes([0.48, 0.10, 0.50, 0.22])
        self.ax_profile.set_title("Intensity Profile Along Trajectory")
        self.ax_profile.set_xlabel("Distance from tip (mm)")
        self.ax_profile.set_ylabel("CT Intensity (HU)")
        self.ax_profile.grid(True, alpha=0.3)

        # Initialize deviation axis variable (toggle moved to main controls)
        self.ax_profile_deviation = None

        # Slice slider - half width and centered under CT view
        slider_width = 0.21  # Half of CT view width
        slider_x = 0.02 + (0.42 - slider_width) / 2  # Center it
        ax_slider = self.fig.add_axes([slider_x, 0.42, slider_width, 0.02])
        n_slices = self.ct_data.shape[self.slice_idx] if self.ct_data is not None else 1
        self.slider = Slider(
            ax_slider,
            f"{self.slice_axis.capitalize()} Slice",
            0,
            n_slices - 1,
            valinit=self.current_slice,
            valstep=1,
            valfmt=f"%d / {n_slices-1}",
            color="steelblue",
        )
        self.slider.on_changed(self._on_slider_change)

        # Voxel info as overlay on CT view (top-left corner)
        self.coord_text = self.ax_ct.text(
            0.02,
            0.98,
            "",
            transform=self.ax_ct.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
        )

        # Results text (on CT axes)
        self.result_text = self.ax_ct.text(
            0.98,
            0.98,
            "",
            transform=self.ax_ct.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
        )

        # Progress/status text (centered on figure)
        self.progress_text = self.fig.text(
            0.5,
            0.5,
            "",
            transform=self.fig.transFigure,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=14,
            weight="bold",
            color="darkblue",
            bbox=dict(boxstyle="round,pad=1", facecolor="yellow", alpha=0.9),
            visible=False,
        )

        # Add controls
        self._setup_controls()

        # Add crosshairs
        self._setup_crosshairs()

        # Connect events
        self._connect_events()

        # Initial display
        self._update_ct_display()
        self._update_3d_display()

    def _setup_controls(self) -> None:
        """Set up control buttons and inputs."""
        # Single row of buttons at bottom
        button_y = 0.02
        button_height = 0.04
        button_spacing = 0.01
        button_x_start = 0.02

        # All buttons in single horizontal row at bottom
        button_width = 0.08
        current_x = button_x_start

        # Load CT button
        ax_load_ct = self.fig.add_axes(
            [current_x, button_y, button_width, button_height]
        )
        self.controls["load_ct"] = Button(ax_load_ct, "Load CT")
        self.controls["load_ct"].on_clicked(self._on_load_ct)
        current_x += button_width + button_spacing

        # Clear button
        ax_clear = self.fig.add_axes([current_x, button_y, button_width, button_height])
        self.controls["clear"] = Button(ax_clear, "Clear All")
        self.controls["clear"].on_clicked(self._on_clear)
        current_x += button_width + button_spacing

        # Auto Run button
        ax_auto = self.fig.add_axes([current_x, button_y, button_width, button_height])
        self.controls["auto_run"] = Button(ax_auto, "Auto Run")
        self.controls["auto_run"].on_clicked(self._on_auto_run)
        current_x += button_width + button_spacing

        # Run OOR button (initially disabled until polynomials fitted)
        ax_oor = self.fig.add_axes([current_x, button_y, button_width, button_height])
        self.controls["run_oor"] = Button(ax_oor, "Run OOR")
        self.controls["run_oor"].on_clicked(self._on_run_oor)

        current_x += button_width + button_spacing

        # Save JSON button (initially disabled)
        ax_save_json = self.fig.add_axes(
            [current_x, button_y, button_width, button_height]
        )
        self.controls["save_json"] = Button(ax_save_json, "Save JSON")
        self.controls["save_json"].on_clicked(self._on_save_json)
        current_x += button_width + button_spacing

        # Save HTML button (initially disabled)
        ax_save_html = self.fig.add_axes(
            [current_x, button_y, button_width, button_height]
        )
        self.controls["save_html"] = Button(ax_save_html, "Save HTML")
        self.controls["save_html"].on_clicked(self._on_save_html)

        # Parameters section (left column, narrower)
        params_x = 0.02
        params_y = 0.08
        params_width = 0.14  # Narrower for 3-column layout
        params_height = 0.32

        # Parameters label
        self.fig.text(
            params_x,
            params_y + params_height,
            "Parameters",
            fontsize=10,
            ha="left",
            weight="bold",
        )

        # Input fields spacing
        input_height = 0.025
        input_spacing = 0.030
        label_width = 0.06  # Narrower labels
        input_width = 0.07  # Narrower inputs
        current_y = params_y + params_height - 0.03

        # Threshold input (HU)
        self.fig.text(
            params_x,
            current_y + input_height / 2,
            "Threshold:",
            fontsize=9,
            va="center",
        )
        ax_threshold = self.fig.add_axes(
            [params_x + label_width + 0.01, current_y, input_width, input_height]
        )
        self.controls["threshold"] = TextBox(
            ax_threshold, "", initial=str(int(self.metal_threshold))
        )
        self.controls["threshold"].on_submit(self._on_threshold_change)
        # Remove label font size setting as we're not using labels
        current_y -= input_spacing

        # Polynomial degree input (for OOR)
        self.fig.text(
            params_x, current_y + input_height / 2, "Poly Deg:", fontsize=9, va="center"
        )
        ax_poly_deg = self.fig.add_axes(
            [params_x + label_width + 0.01, current_y, input_width, input_height]
        )
        self.controls["poly_degree"] = TextBox(ax_poly_deg, "", initial="3")
        self.controls["poly_degree"].on_submit(self._on_poly_degree_change)
        # Remove label font size setting as we're not using labels
        current_y -= input_spacing

        # Angle cutoff input (degrees)
        self.fig.text(
            params_x,
            current_y + input_height / 2,
            "Angle (°):",
            fontsize=9,
            va="center",
        )
        ax_angle_cutoff = self.fig.add_axes(
            [params_x + label_width + 0.01, current_y, input_width, input_height]
        )
        self.controls["angle_cutoff"] = TextBox(ax_angle_cutoff, "", initial="60")
        self.controls["angle_cutoff"].on_submit(self._on_angle_cutoff_change)
        # Remove label font size setting as we're not using labels
        current_y -= input_spacing

        # Search radius input (voxels)
        self.fig.text(
            params_x, current_y + input_height / 2, "Radius:", fontsize=9, va="center"
        )
        ax_search_radius = self.fig.add_axes(
            [params_x + label_width + 0.01, current_y, input_width, input_height]
        )
        self.controls["search_radius"] = TextBox(
            ax_search_radius, "", initial="10"  # Default 10 voxels
        )
        self.controls["search_radius"].on_submit(self._on_search_radius_change)
        # Remove label font size setting as we're not using labels
        current_y -= input_spacing

        # Grid size input (for OOR refinement)
        self.fig.text(
            params_x,
            current_y + input_height / 2,
            "Grid (mm):",
            fontsize=9,
            va="center",
        )
        ax_grid_size = self.fig.add_axes(
            [params_x + label_width + 0.01, current_y, input_width, input_height]
        )
        self.controls["grid_size"] = TextBox(
            ax_grid_size, "", initial="1.5"  # Default 1.5mm
        )
        self.controls["grid_size"].on_submit(self._on_grid_size_change)

        # Output directory display field (read-only, moved to avoid overlap)
        output_dir_y = 0.01
        ax_output_dir = self.fig.add_axes([0.70, output_dir_y, 0.28, 0.015])
        ax_output_dir.axis("off")

        # Determine display directory
        if self.output_dir:
            display_dir = str(self.output_dir)
        else:
            ct_dir = self.ct_path.parent if self.ct_path else Path.cwd()
            display_dir = str(ct_dir / "pypacer")

        # Truncate if too long
        max_len = 45
        if len(display_dir) > max_len:
            display_dir = "..." + display_dir[-(max_len - 3) :]

        self.output_dir_text = ax_output_dir.text(
            0.0,
            0.5,
            f"Output: {display_dir}",
            transform=ax_output_dir.transAxes,
            verticalalignment="center",
            horizontalalignment="left",
            fontsize=7,
            family="monospace",
            color="darkblue",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.3),
        )

        # Options section header
        current_y -= input_spacing
        self.fig.text(
            params_x, current_y, "Options", fontsize=10, ha="left", weight="bold"
        )
        current_y -= 0.025

        # Checkboxes - positioned below parameters in first column
        checkbox_x = params_x
        checkbox_width = params_width

        # Show seed points checkbox
        ax_show_seeds = self.fig.add_axes(
            [checkbox_x, current_y, checkbox_width, 0.025]
        )
        self.controls["show_seeds"] = CheckButtons(
            ax_show_seeds, ["Show Seeds"], [True]
        )
        self.controls["show_seeds"].on_clicked(self._on_show_seeds_change)
        for label in self.controls["show_seeds"].labels:
            label.set_fontsize(9)

        # Show deviation toggle
        current_y -= 0.030
        ax_show_dev = self.fig.add_axes([checkbox_x, current_y, checkbox_width, 0.025])
        self.controls["show_deviation"] = CheckButtons(
            ax_show_dev, ["Show Dev"], [True]
        )
        self.controls["show_deviation"].on_clicked(self._on_show_deviation_change)
        for label in self.controls["show_deviation"].labels:
            label.set_fontsize(9)

        # Show metal components toggle
        current_y -= 0.030
        ax_show_comp = self.fig.add_axes([checkbox_x, current_y, checkbox_width, 0.025])
        self.controls["show_components"] = CheckButtons(
            ax_show_comp, ["Show Components"], [True]
        )
        self.controls["show_components"].on_clicked(self._on_show_components_change)
        for label in self.controls["show_components"].labels:
            label.set_fontsize(9)

        # GPU toggle
        current_y -= 0.030
        ax_use_gpu = self.fig.add_axes([checkbox_x, current_y, checkbox_width, 0.025])
        self.controls["use_gpu"] = CheckButtons(ax_use_gpu, ["Use GPU"], [False])
        self.controls["use_gpu"].on_clicked(self._on_gpu_toggle)
        for label in self.controls["use_gpu"].labels:
            label.set_fontsize(9)

        # Second column - Detection options
        col2_x = params_x + 0.15
        col2_y = params_y + params_height - 0.03
        self.fig.text(
            col2_x,
            col2_y + 0.02,
            "Detection Options",
            fontsize=10,
            ha="left",
            weight="bold",
        )

        # Detection mode
        mode_y = col2_y - 0.03
        self.fig.text(col2_x, mode_y, "Mode:", fontsize=9, ha="left")
        ax_mode = self.fig.add_axes([col2_x, mode_y - 0.08, 0.10, 0.08])
        self.controls["detection_mode"] = RadioButtons(
            ax_mode, ("Fast", "Normal", "High"), active=1
        )
        # Mode already labeled above
        # Initially disable if GPU is on (default is now off)
        if self.controls["use_gpu"].get_status()[0]:
            for label in self.controls["detection_mode"].labels:
                label.set_alpha(0.3)

        # Electrode type
        elec_y = mode_y - 0.12
        self.fig.text(col2_x, elec_y, "Electrode:", fontsize=9, ha="left")
        # Make the axes taller to fit all electrode types
        ax_electrode = self.fig.add_axes([col2_x, elec_y - 0.14, 0.10, 0.14])
        electrode_types = ["Auto"] + list(ELECTRODE_GEOMETRIES.keys())
        # Show all available electrode types
        self.controls["electrode_type"] = RadioButtons(
            ax_electrode, electrode_types, active=0  # Show all types
        )
        # Type already labeled above

        # Third column - Contact detection
        col3_x = col2_x + 0.12
        col3_y = params_y + params_height - 0.03
        self.fig.text(
            col3_x,
            col3_y + 0.02,
            "Contact Detection",
            fontsize=10,
            ha="left",
            weight="bold",
        )

        # Contact method
        contact_y = col3_y - 0.03
        self.fig.text(col3_x, contact_y, "Method:", fontsize=9, ha="left")
        ax_contact_method = self.fig.add_axes([col3_x, contact_y - 0.08, 0.10, 0.08])
        contact_methods = ["Area Center", "Peak", "Peak Wave"]
        self.controls["contact_method"] = RadioButtons(
            ax_contact_method, contact_methods, active=1  # Default to 'Peak'
        )
        self.controls["contact_method"].on_clicked(self._on_contact_method_change)
        # Method already labeled above
        for label in self.controls["contact_method"].labels:
            label.set_fontsize(8)

        # Instructions
        instructions = (
            "Left click: Add seed, auto-track COG & fit poly | Right click: Remove seed | "
            "Scroll: Change slice | Thr=HU threshold, Deg=OOR degree, Ang=angle cutoff°, Rad=search voxels, Grid=OOR grid mm"
        )
        self.fig.text(
            0.50, 0.003, instructions, fontsize=6, ha="center", style="italic"
        )

    def _setup_crosshairs(self) -> None:
        """Set up crosshair overlays."""
        xlim = self.ax_ct.get_xlim() or (0, self.ct_data.shape[0])
        ylim = self.ax_ct.get_ylim() or (0, self.ct_data.shape[1])

        self.crosshair_h = self.ax_ct.plot(
            xlim, [0, 0], "g-", alpha=0.6, linewidth=1, visible=False
        )[0]
        self.crosshair_v = self.ax_ct.plot(
            [0, 0], ylim, "g-", alpha=0.6, linewidth=1, visible=False
        )[0]

    def _connect_events(self) -> None:
        """Connect matplotlib events."""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _precompute_masks(self) -> None:
        """Pre-compute brain and metal masks for full detection (not needed for COG tracking)."""
        if self.brain_mask is None:
            print("Computing brain mask...")
            self.brain_mask = extract_brain_mask(self.ct_data, self.voxel_sizes)

        if self.metal_mask is None:
            print("Computing metal mask...")
            self.metal_mask = detect_metal_artifacts(
                self.ct_data, self.brain_mask, self.metal_threshold
            )
            print(f"Found {np.sum(self.metal_mask)} metal voxels")

        # Compute connected components after metal mask
        if self.metal_mask is not None and self.metal_components is None:
            self._compute_metal_components()

    def _compute_initial_metal_mask(self) -> None:
        """Compute metal mask on startup using current threshold."""
        print(f"Computing metal mask with threshold {self.metal_threshold} HU...")

        # Simple thresholding without brain mask for speed
        self.metal_mask = self.ct_data > self.metal_threshold
        num_metal_voxels = self.metal_mask.sum()
        print(
            f"Found {num_metal_voxels} metal voxels ({100*num_metal_voxels/self.metal_mask.size:.3f}%)"
        )

    def _compute_metal_components(self) -> None:
        """Compute connected components of metal mask."""
        from skimage import measure

        if self.metal_mask is None:
            return

        print("Computing connected components of metal mask...")
        self.metal_components = measure.label(self.metal_mask, connectivity=3)
        self.num_components = self.metal_components.max()

        # Print component sizes
        component_sizes = []
        for i in range(1, self.num_components + 1):
            size = (self.metal_components == i).sum()
            component_sizes.append((i, size))

        component_sizes.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {self.num_components} metal components:")
        for i, (comp_id, size) in enumerate(component_sizes[:10]):  # Show top 10
            print(f"  Component {comp_id}: {size} voxels")

        if len(component_sizes) > 10:
            print(f"  ... and {len(component_sizes) - 10} more smaller components")

    def _ensure_masks_computed(self) -> None:
        """Ensure masks are computed when needed."""
        if self.brain_mask is None or self.metal_mask is None:
            self._precompute_masks()

    def _get_slice_data(self) -> np.ndarray:
        """Get current slice data."""
        if self.slice_axis == "axial":
            return self.ct_data[:, :, self.current_slice].T
        elif self.slice_axis == "sagittal":
            return np.flipud(self.ct_data[self.current_slice, :, :].T)
        else:  # coronal
            return np.flipud(self.ct_data[:, self.current_slice, :].T)

    def _get_aspect_ratio(self) -> float:
        """Calculate aspect ratio for current view."""
        if self.slice_axis == "axial":
            return self.voxel_sizes[1] / self.voxel_sizes[0]
        elif self.slice_axis == "sagittal":
            return self.voxel_sizes[2] / self.voxel_sizes[1]
        else:  # coronal
            return self.voxel_sizes[2] / self.voxel_sizes[0]

    def _display_to_voxel(
        self, x_display: float, y_display: float
    ) -> Tuple[int, int, int]:
        """Convert display coordinates to voxel coordinates."""
        x_display = int(round(x_display))
        y_display = int(round(y_display))

        if self.slice_axis == "axial":
            return (x_display, y_display, self.current_slice)
        elif self.slice_axis == "sagittal":
            return (
                self.current_slice,
                x_display,
                self.ct_data.shape[2] - 1 - y_display,
            )
        else:  # coronal
            return (
                x_display,
                self.current_slice,
                self.ct_data.shape[2] - 1 - y_display,
            )

    def _update_ct_display(self) -> None:
        """Update the CT slice display."""
        self.ax_ct.clear()

        # Check if CT data is loaded
        if self.ct_data is None:
            self.ax_ct.text(
                0.5,
                0.5,
                'No CT loaded\nClick "Load CT" to begin',
                ha="center",
                va="center",
                transform=self.ax_ct.transAxes,
                fontsize=14,
                color="gray",
            )
            self.ax_ct.set_xlim(0, 1)
            self.ax_ct.set_ylim(0, 1)
            self.fig.canvas.draw_idle()
            return

        # Display slice
        slice_data = self._get_slice_data()
        self.ax_ct.imshow(
            slice_data,
            cmap="gray",
            aspect=self._get_aspect_ratio(),
            interpolation="bilinear",
            origin="lower",
        )
        self.ax_ct.set_title(
            f"{self.slice_axis.capitalize()} Slice {self.current_slice}"
        )
        # Remove axis ticks and labels
        self.ax_ct.set_xticks([])
        self.ax_ct.set_yticks([])

        # Overlay metal components if available and enabled
        if self.show_components and self.metal_components is not None:
            self._draw_metal_components()
        # Otherwise show metal mask contour
        elif self.metal_mask is not None:
            metal_slice = self._get_metal_slice()
            if np.any(metal_slice):
                self.ax_ct.contour(
                    metal_slice, levels=[0.5], colors=["red"], alpha=0.8, linewidths=1
                )

        # Draw seed points
        self._draw_seed_points()

        # Draw electrode trajectories
        self._draw_electrode_trajectories()

        # Re-add result text
        self.result_text = self.ax_ct.text(
            0.98,
            0.98,
            self._get_result_text(),
            transform=self.ax_ct.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
        )

        # Re-add voxel info overlay
        self.coord_text = self.ax_ct.text(
            0.02,
            0.98,
            "",
            transform=self.ax_ct.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
        )

        # Add contact info text if contacts have been detected
        if self.electrodes and any(
            len(e.contact_positions) > 0 for e in self.electrodes
        ):
            contact_info = self._get_detailed_contact_info()
            if contact_info:
                self.contact_info_text = self.ax_ct.text(
                    0.02,
                    0.02,
                    contact_info,
                    transform=self.ax_ct.transAxes,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    fontsize=8,
                    family="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9
                    ),
                )

        # Re-setup crosshairs with new limits
        xlim = self.ax_ct.get_xlim()
        ylim = self.ax_ct.get_ylim()
        self.crosshair_h = self.ax_ct.plot(
            xlim, [0, 0], "g-", alpha=0.6, linewidth=1, visible=False
        )[0]
        self.crosshair_v = self.ax_ct.plot(
            [0, 0], ylim, "g-", alpha=0.6, linewidth=1, visible=False
        )[0]

        # Initialize hover rectangle if needed
        self.hover_rect = None

        # Initial draw
        self.fig.canvas.draw_idle()

    def _draw_metal_components(self) -> None:
        """Draw metal components with different colors."""
        if self.metal_components is None:
            return

        # Get component slice
        component_slice = self._get_component_slice()
        if not np.any(component_slice):
            return

        # Create colormap for components
        import matplotlib.pyplot as plt

        # Use a colormap with distinct colors
        cmap = plt.cm.tab20

        # Create overlay for components
        overlay = np.zeros((*component_slice.shape, 4))  # RGBA

        # Color each component
        for comp_id in range(1, min(self.num_components + 1, 21)):  # Limit to 20 colors
            mask = component_slice == comp_id
            if np.any(mask):
                color = cmap(comp_id / 20.0)
                overlay[mask] = color
                overlay[mask, 3] = 0.6  # Set alpha

        # Show overlay
        self.ax_ct.imshow(
            overlay,
            aspect=self._get_aspect_ratio(),
            interpolation="nearest",
            origin="lower",
        )

    def _get_component_slice(self) -> np.ndarray:
        """Get metal component labels for current slice."""
        if self.metal_components is None:
            return np.zeros_like(self._get_slice_data(), dtype=int)

        if self.slice_axis == "axial":
            return self.metal_components[:, :, self.current_slice].T
        elif self.slice_axis == "sagittal":
            return np.flipud(self.metal_components[self.current_slice, :, :].T)
        else:  # coronal
            return np.flipud(self.metal_components[:, self.current_slice, :].T)

    def _get_metal_slice(self) -> np.ndarray:
        """Get metal mask for current slice."""
        if self.metal_mask is None:
            return np.zeros_like(self._get_slice_data())

        if self.slice_axis == "axial":
            return self.metal_mask[:, :, self.current_slice].T
        elif self.slice_axis == "sagittal":
            return np.flipud(self.metal_mask[self.current_slice, :, :].T)
        else:  # coronal
            return np.flipud(self.metal_mask[:, self.current_slice, :].T)

    def _draw_seed_points(self) -> None:
        """Draw seed points on current slice."""
        if not self.seed_points or not self.controls["show_seeds"].get_status()[0]:
            return

        for i, (x, y, z) in enumerate(self.seed_points):
            # Check if seed point is on current slice
            if self.slice_axis == "axial" and z == self.current_slice:
                display_x, display_y = x, y
            elif self.slice_axis == "sagittal" and x == self.current_slice:
                display_x, display_y = y, self.ct_data.shape[2] - 1 - z
            elif self.slice_axis == "coronal" and y == self.current_slice:
                display_x, display_y = x, self.ct_data.shape[2] - 1 - z
            else:
                continue

            # Draw seed point
            circle = Circle(
                (display_x, display_y),
                radius=3,
                facecolor="red",
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            self.ax_ct.add_patch(circle)

            # Add number
            self.ax_ct.text(
                display_x + 5,
                display_y + 5,
                str(i + 1),
                color="red",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    def _draw_electrode_trajectories(self) -> None:
        """Draw electrode trajectories on current slice."""
        if not self.electrodes:
            return

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.electrodes)))

        for i, electrode in enumerate(self.electrodes):
            color = colors[i]

            # Sample trajectory points
            t_values = np.linspace(0, 1, 100)
            trajectory_points = np.array(
                [electrode.get_point_at_parameter(t) for t in t_values]
            )

            # Convert to voxel coordinates
            voxel_coords = []
            for point in trajectory_points:
                point_h = np.append(point, 1)
                voxel_coord = np.linalg.inv(self.affine) @ point_h
                voxel_coords.append(voxel_coord[:3])
            voxel_coords = np.array(voxel_coords)

            # Project onto current slice
            display_points = []
            for vx, vy, vz in voxel_coords:
                if self.slice_axis == "axial":
                    if abs(vz - self.current_slice) < 1:  # Within 1 slice
                        display_points.append([vx, vy])
                elif self.slice_axis == "sagittal":
                    if abs(vx - self.current_slice) < 1:
                        display_points.append([vy, self.ct_data.shape[2] - 1 - vz])
                elif self.slice_axis == "coronal":
                    if abs(vy - self.current_slice) < 1:
                        display_points.append([vx, self.ct_data.shape[2] - 1 - vz])

            if display_points:
                display_points = np.array(display_points)
                self.ax_ct.plot(
                    display_points[:, 0],
                    display_points[:, 1],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    label=f"Electrode {i+1}",
                )

    def _update_3d_display(self) -> None:
        """Update the 3D visualization."""
        self.ax_3d.clear()
        self.ax_3d.set_title("Detected Electrodes & COG Trajectories")

        # Enable fast draw mode for better performance
        if hasattr(self.ax_3d, "_draw_disabled"):
            self.ax_3d._draw_disabled = False

        has_content = False

        # Draw electrodes if available
        if self.electrodes:
            has_content = True
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.electrodes)))

            # First pass: calculate global max deviation for colormap scaling
            global_max_deviation = 0.0
            all_distances_list = []

            for i, electrode in enumerate(self.electrodes):
                if i < len(self.cog_trajectories) and self.cog_trajectories[i]:
                    # Sample polynomial - reduced for performance
                    t_sample = np.linspace(0, 1, 50)  # Reduced from 100
                    poly_points = np.array(
                        [electrode.get_point_at_parameter(t) for t in t_sample]
                    )

                    # Get COG points
                    cog_points = np.array(
                        [p.center_of_gravity for p in self.cog_trajectories[i]]
                    )

                    # Calculate distances
                    distances = np.zeros(len(poly_points))
                    for j, poly_pt in enumerate(poly_points):
                        dists_to_cog = np.linalg.norm(cog_points - poly_pt, axis=1)
                        distances[j] = np.min(dists_to_cog)

                    all_distances_list.append(distances)
                    global_max_deviation = max(global_max_deviation, np.max(distances))
                else:
                    all_distances_list.append(None)

            # Add small margin to max for better visualization
            if global_max_deviation > 0:
                global_max_deviation *= 1.1  # Add 10% margin
            else:
                global_max_deviation = 1.0  # Default if no deviation

            # Store for colorbar
            self._last_max_deviation = global_max_deviation

            # Second pass: draw with adaptive colormap
            for i, electrode in enumerate(self.electrodes):
                color = colors[i]

                # Draw polynomial trajectory with colormap showing deviation from COG
                # Reduce samples for better performance
                t_sample = np.linspace(0, 1, 30)  # Reduced from 100 to 30
                poly_points = np.array(
                    [electrode.get_point_at_parameter(t) for t in t_sample]
                )

                # Use pre-calculated distances
                if all_distances_list[i] is not None and len(all_distances_list[i]) > 0:
                    # Interpolate distances to match reduced sample points
                    original_t = np.linspace(0, 1, len(all_distances_list[i]))
                    distances = np.interp(t_sample, original_t, all_distances_list[i])

                    # Normalize distances using actual range (0 to global_max_deviation)
                    norm_distances = distances / global_max_deviation

                    # Use a colormap (green=good fit, yellow=moderate, red=poor)
                    from matplotlib.cm import (
                        RdYlGn_r,  # Reversed: green for low distance, red for high
                    )

                    colors_array = RdYlGn_r(norm_distances)

                    # Plot polynomial as a single line with average color for performance
                    avg_color = np.mean(colors_array, axis=0)
                    self.ax_3d.plot(
                        poly_points[:, 0],
                        poly_points[:, 1],
                        poly_points[:, 2],
                        color=avg_color,
                        linewidth=3,
                        alpha=0.9,
                    )

                    # Add a text label with fit statistics
                    mean_dist = np.mean(distances)
                    max_dist = np.max(distances)
                    label_text = (
                        f"Poly {i+1} (avg:{mean_dist:.2f}mm, max:{max_dist:.2f}mm)"
                    )

                    # Add to legend by plotting invisible point
                    self.ax_3d.plot(
                        [], [], [], color=color, linewidth=3, label=label_text
                    )

                    # Print statistics
                    print(f"    Polynomial {i+1} fit statistics:")
                    print(f"      Mean deviation: {mean_dist:.2f}mm")
                    print(f"      Max deviation: {max_dist:.2f}mm")
                    print(f"      Min deviation: {np.min(distances):.2f}mm")

                else:
                    # No COG trajectory to compare, just plot normally
                    self.ax_3d.plot(
                        poly_points[:, 0],
                        poly_points[:, 1],
                        poly_points[:, 2],
                        color=color,
                        linewidth=3,
                        alpha=0.9,
                        label=f"Polynomial {i+1}",
                        linestyle="-",
                    )

                # Draw contacts only if they exist
                if len(electrode.contact_positions) > 0:
                    contact_positions = electrode.get_contact_positions_3d()
                    self.ax_3d.scatter(
                        contact_positions[:, 0],
                        contact_positions[:, 1],
                        contact_positions[:, 2],
                        c=[color],
                        s=50,
                        alpha=0.9,
                        marker="o",
                        edgecolors="white",
                        linewidths=1,
                    )

                # Label tip
                tip = electrode.tip_position
                self.ax_3d.text(
                    tip[0],
                    tip[1],
                    tip[2],
                    f"P{i+1}",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

        # Draw COG trajectories if available (show as dots if polynomials exist)
        if self.cog_trajectories:
            has_content = True

            for i, trajectory in enumerate(self.cog_trajectories):
                if not trajectory:
                    continue

                # Extract world coordinates
                points = np.array([p.center_of_gravity for p in trajectory])

                if len(points) > 0:
                    # If we have fitted polynomials, show COG as dots for comparison
                    if self.electrodes:
                        # Show COG points as dots - subsample for performance
                        step = max(1, len(points) // 20)  # Show max 20 points
                        subsampled_points = points[::step]
                        self.ax_3d.scatter(
                            subsampled_points[:, 0],
                            subsampled_points[:, 1],
                            subsampled_points[:, 2],
                            c="gray",
                            s=5,
                            alpha=0.3,
                            label=f"COG Points {i+1}" if i == 0 else "",
                        )
                    else:
                        # No polynomials yet, show full COG trajectory
                        cog_color = plt.cm.viridis(
                            i / max(1, len(self.cog_trajectories) - 1)
                        )
                        self.ax_3d.plot(
                            points[:, 0],
                            points[:, 1],
                            points[:, 2],
                            color=cog_color,
                            linewidth=2,
                            alpha=0.7,
                            linestyle="--",
                            label=f"COG Track {i+1}",
                        )

                        # Mark points with direction changes
                        for point in trajectory:
                            if (
                                point.direction_change_angle is not None
                                and point.direction_change_angle > 20
                            ):
                                self.ax_3d.scatter(
                                    point.center_of_gravity[0],
                                    point.center_of_gravity[1],
                                    point.center_of_gravity[2],
                                    c="orange",
                                    s=30,
                                    marker="x",
                                    alpha=0.8,
                                )

                    # Always mark start and end points
                    self.ax_3d.scatter(
                        points[0, 0],
                        points[0, 1],
                        points[0, 2],
                        c="green",
                        s=60,
                        marker="^",
                        alpha=0.7,
                        edgecolors="white",
                        linewidths=1,
                    )
                    self.ax_3d.scatter(
                        points[-1, 0],
                        points[-1, 1],
                        points[-1, 2],
                        c="red",
                        s=60,
                        marker="v",
                        alpha=0.7,
                        edgecolors="white",
                        linewidths=1,
                    )

        # Draw seed points
        if self.seed_points and self.controls["show_seeds"].get_status()[0]:
            has_content = True
            seed_world_coords = []
            for voxel_coord in self.seed_points:
                voxel_h = np.append(voxel_coord, 1)
                world_coord = (self.affine @ voxel_h)[:3]
                seed_world_coords.append(world_coord)

            if seed_world_coords:
                seed_world_coords = np.array(seed_world_coords)
                self.ax_3d.scatter(
                    seed_world_coords[:, 0],
                    seed_world_coords[:, 1],
                    seed_world_coords[:, 2],
                    c="red",
                    s=100,
                    alpha=0.8,
                    marker="*",
                    edgecolors="white",
                    linewidths=1,
                    label="Seed Points",
                )

        # Show help text if no content
        if not has_content:
            self.ax_3d.text2D(
                0.5,
                0.5,
                'No data to display\nAdd seed points and click "Track COG" or "Detect"',
                transform=self.ax_3d.transAxes,
                ha="center",
                va="center",
            )

        # Setup axes
        self.ax_3d.set_xlabel("X (mm)")
        self.ax_3d.set_ylabel("Y (mm)")
        self.ax_3d.set_zlabel("Z (mm)")
        self.ax_3d.grid(True, alpha=0.3)

        # Add colorbar if we have polynomial fits
        if self.electrodes and self.cog_trajectories:
            # Calculate actual max deviation for colorbar
            actual_max_dev = 0.0
            for i, electrode in enumerate(self.electrodes):
                if i < len(self.cog_trajectories) and self.cog_trajectories[i]:
                    # Get pre-calculated distances if available
                    if hasattr(self, "_last_max_deviation"):
                        actual_max_dev = self._last_max_deviation
                        break

            # If we couldn't get the max, use a default
            if actual_max_dev == 0:
                actual_max_dev = 1.0

            # Create a colorbar to show distance scale
            from matplotlib.cm import RdYlGn_r
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import Normalize

            # Create colorbar axes - check if it already exists
            cbar_ax = None
            for ax in self.fig.axes:
                if hasattr(ax, "_colorbar_axis"):
                    cbar_ax = ax
                    cbar_ax.clear()
                    break

            if cbar_ax is None:
                cbar_ax = self.fig.add_axes([0.92, 0.50, 0.02, 0.25])
                cbar_ax._colorbar_axis = True  # Mark as colorbar axis

            # Use adaptive scale based on actual data
            norm = Normalize(vmin=0, vmax=actual_max_dev)
            cb = ColorbarBase(cbar_ax, cmap=RdYlGn_r, norm=norm, orientation="vertical")
            cb.set_label(
                f"Deviation (mm)\nMax: {actual_max_dev:.2f}mm",
                rotation=270,
                labelpad=20,
            )
            cb.ax.yaxis.set_label_position("right")

            # Format tick labels to show appropriate precision
            if actual_max_dev < 1:
                cb.ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x:.2f}")
                )
            else:
                cb.ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x:.1f}")
                )

        if has_content:
            self.ax_3d.legend(loc="upper right", fontsize=8)

        self.fig.canvas.draw_idle()

        # Update intensity profile
        self._update_intensity_profile()

    def _update_intensity_profile(self) -> None:
        """Update the intensity profile plot."""
        self.ax_profile.clear()

        # Clear secondary axis if it exists
        if self.ax_profile_deviation is not None:
            self.ax_profile_deviation.clear()
            # Hide the axis if toggle is off
            show_dev = (
                self.controls["show_deviation"].get_status()[0]
                if "show_deviation" in self.controls
                else True
            )
            if not show_dev:
                self.ax_profile_deviation.set_visible(False)
            else:
                self.ax_profile_deviation.set_visible(True)
                self.ax_profile_deviation.set_ylabel("Deviation (mm)", color="darkred")
                self.ax_profile_deviation.tick_params(axis="y", labelcolor="darkred")

        # Check if we should show deviation or intensity
        # Show deviation only if we have polynomials and COG trajectories but NO refined trajectories
        show_deviation = bool(
            self.electrodes and self.cog_trajectories and not self.refined_trajectories
        )

        if show_deviation:
            # Show deviation between polynomial and COG trajectory
            self.ax_profile.set_xlabel("Distance along trajectory (mm)")
            self.ax_profile.set_ylabel("Deviation (mm)")
            self.ax_profile.set_title("Polynomial vs COG Trajectory Deviation")
        else:
            # Show intensity profile
            self.ax_profile.set_xlabel("Distance from tip (mm)")
            self.ax_profile.set_ylabel("CT Intensity (HU)")
            self.ax_profile.set_title("Intensity Profile Along Trajectory")

        self.ax_profile.grid(True, alpha=0.3)

        if not self.electrodes and not self.cog_trajectories:
            self.ax_profile.text(
                0.5,
                0.5,
                "No trajectory data available",
                transform=self.ax_profile.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                alpha=0.5,
            )
            self.fig.canvas.draw_idle()
            return

        # If we have both polynomials and COG, show deviation
        if show_deviation:
            self._plot_deviation_profile()
            return

        # Plot intensity profiles for fitted polynomials (only if no COG trajectories)
        if self.electrodes:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.electrodes)))

            for i, electrode in enumerate(self.electrodes):
                color = colors[i]

                # Check if we have refined intensity profile from OOR
                if (
                    hasattr(electrode, "refined_intensity_profile")
                    and electrode.refined_intensity_profile is not None
                ):
                    # Use the refined OOR intensity profile
                    # After re-zeroing, distances already start at 0 (tip position)
                    distances = electrode.refined_distance_scale.copy()
                    intensities = electrode.refined_intensity_profile

                    # No shifting needed - pipeline already handles re-zeroing
                    x_shift = 0.0
                    final_x_shift = 0.0

                    label = f"Electrode {i+1} (OOR refined)"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    # Sample along the polynomial manually
                    n_samples = 200

                    # Check if we have a detected tip from OOR pass 2
                    # This is stored in the refined trajectory if available
                    tip_param = 0.0  # Default: assume tip at t=0
                    x_shift = 0.0  # Amount to shift x-axis

                    if (
                        i < len(self.refined_trajectories)
                        and self.refined_trajectories[i]
                    ):
                        refined = self.refined_trajectories[i]
                        if (
                            hasattr(refined, "pass2_tip_param")
                            and refined.pass2_tip_param is not None
                        ):
                            tip_param = refined.pass2_tip_param
                            # Calculate the distance from t=0 to the detected tip
                            # This will be our x-axis shift
                            from ..utils.math_helpers import poly_arc_length_3d

                            if tip_param > 0:
                                x_shift = poly_arc_length_3d(
                                    electrode.polynomial, 0, tip_param
                                )
                            elif tip_param < 0:
                                x_shift = -poly_arc_length_3d(
                                    electrode.polynomial, tip_param, 0
                                )
                            # Sample from before tip to end
                            t_values = np.linspace(0, 1, n_samples)
                        else:
                            t_values = np.linspace(0, 1, n_samples)
                    else:
                        t_values = np.linspace(0, 1, n_samples)

                    # Get points along trajectory
                    poly_points = np.array(
                        [electrode.get_point_at_parameter(t) for t in t_values]
                    )

                    # Calculate cumulative distance from first sampled point
                    distances = np.zeros(len(t_values))
                    for j in range(1, len(t_values)):
                        distances[j] = distances[j - 1] + np.linalg.norm(
                            poly_points[j] - poly_points[j - 1]
                        )

                    # Sample CT intensities at these points
                    intensities = []
                    for point in poly_points:
                        # Convert world coordinates to voxel coordinates
                        voxel_coord = np.linalg.inv(self.affine) @ np.append(point, 1)
                        voxel_coord = voxel_coord[:3].astype(int)

                        # Check bounds and sample
                        if (
                            0 <= voxel_coord[0] < self.ct_data.shape[0]
                            and 0 <= voxel_coord[1] < self.ct_data.shape[1]
                            and 0 <= voxel_coord[2] < self.ct_data.shape[2]
                        ):
                            intensity = self.ct_data[tuple(voxel_coord)]
                            intensities.append(intensity)
                        else:
                            intensities.append(0)

                    intensities = np.array(intensities)

                    # Find where this profile crosses the tip threshold if available
                    final_x_shift = x_shift  # Default to tip parameter shift
                    if (
                        i < len(self.refined_trajectories)
                        and self.refined_trajectories[i]
                    ):
                        refined = self.refined_trajectories[i]
                        if (
                            hasattr(refined, "pass2_tip_threshold")
                            and refined.pass2_tip_threshold is not None
                        ):
                            tip_threshold = refined.pass2_tip_threshold

                            # Find the first point where intensity crosses the threshold
                            crossing_idx = None
                            for j in range(len(intensities)):
                                if intensities[j] >= tip_threshold:
                                    crossing_idx = j
                                    break

                            if crossing_idx is not None:
                                # Linear interpolation to find exact crossing point
                                if crossing_idx > 0:
                                    # Interpolate between crossing_idx-1 and crossing_idx
                                    y0 = intensities[crossing_idx - 1]
                                    y1 = intensities[crossing_idx]
                                    x0 = distances[crossing_idx - 1]
                                    x1 = distances[crossing_idx]

                                    # Find exact x where y = tip_threshold
                                    if y1 != y0:
                                        x_crossing = x0 + (tip_threshold - y0) * (
                                            x1 - x0
                                        ) / (y1 - y0)
                                    else:
                                        x_crossing = x0
                                else:
                                    x_crossing = distances[0]

                                final_x_shift = x_crossing
                                # Debug: print(f"  Shifted pre-OOR profile {i+1} by {final_x_shift:.2f}mm to align threshold crossing at x=0)")

                    # Apply the shift to align threshold crossing at x=0
                    distances = distances - final_x_shift

                    if final_x_shift != 0.0:
                        label = f"Electrode {i+1} (pre-OOR)"
                    else:
                        label = f"Electrode {i+1} (pre-OOR)"
                    linewidth = 2
                    alpha = 0.9

                # Plot intensity profile
                self.ax_profile.plot(
                    distances,
                    intensities,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label,
                )

                # Add skeleton deviation on secondary y-axis if available and toggle is on
                show_dev = (
                    self.controls["show_deviation"].get_status()[0]
                    if "show_deviation" in self.controls
                    else True
                )
                if (
                    show_dev
                    and hasattr(electrode, "skeleton_deviations_mm")
                    and electrode.skeleton_deviations_mm is not None
                ):
                    # Create secondary y-axis if it doesn't exist
                    if self.ax_profile_deviation is None:
                        self.ax_profile_deviation = self.ax_profile.twinx()
                        self.ax_profile_deviation.set_ylabel(
                            "Deviation (mm)", color="darkred"
                        )
                        self.ax_profile_deviation.tick_params(
                            axis="y", labelcolor="darkred"
                        )
                        self.ax_profile_deviation.spines["right"].set_color("darkred")

                    # Plot deviation with different style
                    # Use the same distance scale as intensity profile (they're aligned)
                    deviation_distances = (
                        electrode.refined_distance_scale.copy()
                    )  # Use original unshifted distances
                    deviations = electrode.skeleton_deviations_mm

                    # Apply the same x-shift as the intensity profile for alignment
                    # The x_shift variable from above contains the shift amount
                    if "x_shift" in locals() and x_shift != 0:
                        deviation_distances = deviation_distances - x_shift

                    # Apply light smoothing for visualization
                    from scipy.ndimage import gaussian_filter1d

                    if len(deviations) > 5:
                        deviations_smooth = gaussian_filter1d(deviations, sigma=1.0)
                    else:
                        deviations_smooth = deviations

                    self.ax_profile_deviation.plot(
                        deviation_distances,
                        deviations_smooth,
                        color="darkred",
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"Deviation {i+1}",
                    )

                    # Mark max deviation
                    max_dev_idx = np.argmax(deviations_smooth)
                    max_dev = deviations_smooth[max_dev_idx]
                    self.ax_profile_deviation.plot(
                        deviation_distances[max_dev_idx],
                        max_dev,
                        marker="x",
                        markersize=6,
                        color="darkred",
                        markeredgewidth=1.5,
                        alpha=0.8,
                    )

                    # Add small annotation for max deviation
                    self.ax_profile_deviation.annotate(
                        f"{max_dev:.3f}mm",
                        xy=(deviation_distances[max_dev_idx], max_dev),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        color="darkred",
                        alpha=0.7,
                    )

                # Add a marker at the threshold crossing point (x=0)
                if i < len(self.refined_trajectories) and self.refined_trajectories[i]:
                    refined = self.refined_trajectories[i]
                    if (
                        hasattr(refined, "pass2_tip_threshold")
                        and refined.pass2_tip_threshold is not None
                    ):
                        # Plot a marker at x=0, y=tip_threshold
                        self.ax_profile.plot(
                            0,
                            refined.pass2_tip_threshold,
                            marker="o",
                            markersize=6,
                            color=color,
                            markeredgecolor="black",
                            markeredgewidth=1,
                            zorder=10,
                        )  # Make sure it's on top

                # Add contact area visualization if using Area Center method
                if (
                    len(electrode.contact_positions) > 0
                    and "contact_method" in self.controls
                    and self.controls["contact_method"].value_selected == "Area Center"
                ):

                    # Get expected electrode geometry
                    from ..models.electrode import ELECTRODE_GEOMETRIES

                    if electrode.electrode_type in ELECTRODE_GEOMETRIES:
                        geometry = ELECTRODE_GEOMETRIES[electrode.electrode_type]

                        # Calculate theoretical electrode extent
                        num_contacts = geometry.num_contacts
                        contact_length = geometry.contact_length_mm
                        contact_spacing = geometry.contact_spacing_mm
                        center_to_center = contact_length + contact_spacing

                        # Total extent from first to last contact center
                        total_extent = (num_contacts - 1) * center_to_center
                        # Add half contact on each end for full artifact region
                        total_artifact_region = total_extent + contact_length

                        # Get the center of the detected contacts
                        contact_center = np.mean(electrode.contact_positions)

                        # Shade the theoretical electrode region
                        region_start = (
                            contact_center - total_artifact_region / 2 - final_x_shift
                        )
                        region_end = (
                            contact_center + total_artifact_region / 2 - final_x_shift
                        )

                        # Add shaded region for theoretical electrode extent
                        self.ax_profile.axvspan(
                            region_start,
                            region_end,
                            alpha=0.15,
                            color="blue",
                            label=f"Expected {electrode.electrode_type} extent ({total_artifact_region:.1f}mm)",
                        )

                        # Add vertical line at detected center
                        center_shifted = contact_center - final_x_shift
                        self.ax_profile.axvline(
                            x=center_shifted,
                            color="green",
                            linestyle="-.",
                            alpha=0.7,
                            linewidth=2,
                            label="Detected center",
                        )

                # Mark contact positions if available (adjust for x-shift)
                if len(electrode.contact_positions) > 0:
                    # The contact positions are in mm from tip, but we've shifted the x-axis
                    # by final_x_shift to align the threshold crossing at x=0
                    # So we need to shift the contact positions by the same amount
                    shifted_contact_positions = (
                        electrode.contact_positions - final_x_shift
                    )

                    for j, contact_pos in enumerate(shifted_contact_positions):
                        # Only plot if within the visible range
                        if distances[0] <= contact_pos <= distances[-1]:
                            # Add label only for first electrode's first contact
                            if i == 0 and j == 0:
                                self.ax_profile.axvline(
                                    x=contact_pos,
                                    color=color,
                                    linestyle="--",
                                    alpha=0.5,
                                    linewidth=1,
                                    label="Contacts",
                                )
                            else:
                                self.ax_profile.axvline(
                                    x=contact_pos,
                                    color=color,
                                    linestyle="--",
                                    alpha=0.5,
                                    linewidth=1,
                                )

        # Also plot COG trajectory intensities if no polynomials fitted yet
        elif self.cog_trajectories:

            cog_colors = plt.cm.viridis(np.linspace(0, 1, len(self.cog_trajectories)))

            from scipy.ndimage import gaussian_filter1d

            for i, trajectory in enumerate(self.cog_trajectories):
                if not trajectory:
                    continue

                color = cog_colors[i]

                # Get points and intensities from COG trajectory
                points = np.array([p.center_of_gravity for p in trajectory])
                intensities = np.array([p.mean_intensity for p in trajectory])

                # Calculate cumulative distance
                distances = np.zeros(len(points))
                for j in range(1, len(points)):
                    distances[j] = distances[j - 1] + np.linalg.norm(
                        points[j] - points[j - 1]
                    )

                # Apply Gaussian smoothing to intensities
                # Sigma of 2 provides moderate smoothing
                if len(intensities) > 5:  # Only smooth if we have enough points
                    intensities_smooth = gaussian_filter1d(intensities, sigma=2)
                else:
                    intensities_smooth = intensities

                # Plot with solid line
                self.ax_profile.plot(
                    distances,
                    intensities_smooth,
                    color=color,
                    linewidth=2.5,
                    alpha=0.9,
                    linestyle="-",  # Solid line
                    label=f"COG Track {i+1}",
                )

        # Add threshold line
        self.ax_profile.axhline(
            y=self.metal_threshold,
            color="red",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
            label=f"Metal threshold ({self.metal_threshold} HU)",
        )

        # Add tip detection threshold if available from OOR
        # Collect unique thresholds and draw each only once
        tip_thresholds = set()
        for i, refined in enumerate(self.refined_trajectories):
            if (
                refined
                and hasattr(refined, "pass2_tip_threshold")
                and refined.pass2_tip_threshold
            ):
                tip_thresholds.add(refined.pass2_tip_threshold)

        # Draw each unique threshold with a label
        for idx, threshold in enumerate(sorted(tip_thresholds)):
            if idx == 0:
                # First threshold gets the main label
                self.ax_profile.axhline(
                    y=threshold,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                    label=f"Tip threshold ({threshold:.0f} HU)",
                )
            else:
                # Additional thresholds get numbered labels
                self.ax_profile.axhline(
                    y=threshold,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                    label=f"Tip threshold {idx+1} ({threshold:.0f} HU)",
                )

        # Add vertical line at x=0 to mark the detected tip position
        if self.refined_trajectories:
            self.ax_profile.axvline(
                x=0,
                color="black",
                linestyle="-",
                alpha=0.3,
                linewidth=1,
                label="Detected tip (x=0)",
            )

        # Set Y limits based on actual data with padding
        if self.electrodes or self.cog_trajectories:
            # Collect all intensity values to determine range
            all_intensities = []

            # From electrodes (refined profiles or manually sampled)
            for i, electrode in enumerate(self.electrodes):
                if (
                    hasattr(electrode, "refined_intensity_profile")
                    and electrode.refined_intensity_profile is not None
                ):
                    all_intensities.extend(electrode.refined_intensity_profile)
                else:
                    # Manually sample intensity values like we do for plotting
                    n_samples = 200
                    t_values = np.linspace(0, 1, n_samples)
                    poly_points = np.array(
                        [electrode.get_point_at_parameter(t) for t in t_values]
                    )

                    for point in poly_points:
                        voxel_coord = np.linalg.inv(self.affine) @ np.append(point, 1)
                        voxel_coord = voxel_coord[:3].astype(int)

                        if (
                            0 <= voxel_coord[0] < self.ct_data.shape[0]
                            and 0 <= voxel_coord[1] < self.ct_data.shape[1]
                            and 0 <= voxel_coord[2] < self.ct_data.shape[2]
                        ):
                            intensity = self.ct_data[tuple(voxel_coord)]
                            all_intensities.append(intensity)

            # From COG trajectories if no electrodes
            if not self.electrodes and self.cog_trajectories:
                for trajectory in self.cog_trajectories:
                    if trajectory:
                        all_intensities.extend([p.mean_intensity for p in trajectory])

            if all_intensities:
                min_intensity = min(all_intensities)
                max_intensity = max(all_intensities)
                # Add 10% padding
                padding = (max_intensity - min_intensity) * 0.1
                self.ax_profile.set_ylim(
                    [min_intensity - padding, max_intensity + padding]
                )
            else:
                # Default range if no data
                self.ax_profile.set_ylim([-1000, 4000])

            # Set y-limits for deviation plot with 0.5mm minimum
            if (
                hasattr(self, "ax_profile_deviation")
                and self.ax_profile_deviation is not None
            ):
                # Find max deviation across all plotted deviations
                max_deviation = 0.0
                for refined in self.refined_trajectories:
                    if (
                        refined
                        and hasattr(refined, "skeleton_deviations_mm")
                        and refined.skeleton_deviations_mm is not None
                    ):
                        deviations = refined.skeleton_deviations_mm
                        if len(deviations) > 0:
                            max_deviation = max(max_deviation, np.max(deviations))

                # Set y-limit with minimum of 0.5mm
                if max_deviation > 0:
                    ylim_max = max(
                        0.5, max_deviation * 1.1
                    )  # At least 0.5mm, or max + 10%
                else:
                    ylim_max = 0.5  # Default to 0.5mm if no data

                self.ax_profile_deviation.set_ylim([0, ylim_max])

            self.ax_profile.legend(loc="best", fontsize=8)

        self.fig.canvas.draw_idle()

    def _plot_deviation_profile(self) -> None:
        """Plot deviation between polynomial fits and COG trajectories."""
        colors = plt.cm.rainbow(
            np.linspace(0, 1, max(len(self.electrodes), len(self.cog_trajectories)))
        )

        from scipy.ndimage import gaussian_filter1d

        for i, (electrode, trajectory) in enumerate(
            zip(self.electrodes, self.cog_trajectories)
        ):
            if not trajectory:
                continue

            color = colors[i]

            # Get COG points
            cog_points = np.array([p.center_of_gravity for p in trajectory])

            # Calculate distances along COG trajectory for x-axis
            distances = np.zeros(len(cog_points))
            for j in range(1, len(cog_points)):
                distances[j] = distances[j - 1] + np.linalg.norm(
                    cog_points[j] - cog_points[j - 1]
                )

            # Sample polynomial densely for nearest point matching
            n_poly_samples = 500  # Dense sampling of polynomial
            t_poly = np.linspace(0, 1, n_poly_samples)
            poly_points = np.array(
                [electrode.get_point_at_parameter(t) for t in t_poly]
            )

            # For each COG point, find the nearest point on the polynomial
            deviations = np.zeros(len(cog_points))
            for j, cog_pt in enumerate(cog_points):
                # Calculate distances from this COG point to all polynomial points
                dists_to_poly = np.linalg.norm(poly_points - cog_pt, axis=1)
                # Find minimum distance
                deviations[j] = np.min(dists_to_poly)

            # Apply smoothing for cleaner visualization
            if len(deviations) > 5:
                deviations_smooth = gaussian_filter1d(deviations, sigma=1.5)
            else:
                deviations_smooth = deviations

            # Plot deviation
            self.ax_profile.plot(
                distances,
                deviations_smooth,
                color=color,
                linewidth=2.5,
                alpha=0.9,
                linestyle="-",
                label=f"Electrode {i+1}",
            )

            # Mark maximum deviation
            max_dev_idx = np.argmax(deviations_smooth)
            max_dev = deviations_smooth[max_dev_idx]
            self.ax_profile.plot(
                distances[max_dev_idx],
                max_dev,
                marker="x",
                markersize=8,
                color=color,
                markeredgewidth=2,
            )

            # Add text annotation for max deviation
            self.ax_profile.annotate(
                f"{max_dev:.2f}mm",
                xy=(distances[max_dev_idx], max_dev),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
                alpha=0.8,
            )

            # Print deviation statistics for verification
            mean_dev = np.mean(deviations)
            min_dev = np.min(deviations)
            max_dev_raw = np.max(deviations)
            print(f"  Deviation plot stats for electrode {i+1}:")
            print(
                f"    Mean: {mean_dev:.3f}mm, Max: {max_dev_raw:.3f}mm, Min: {min_dev:.3f}mm"
            )

        # Add reference lines
        self.ax_profile.axhline(
            y=0, color="black", linestyle="-", alpha=0.3, linewidth=1
        )
        self.ax_profile.axhline(
            y=0.5,
            color="orange",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            label="0.5mm threshold",
        )
        self.ax_profile.axhline(
            y=1.0,
            color="red",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            label="1.0mm threshold",
        )

        # Set reasonable Y limits for deviation
        if self.electrodes:
            all_deviations = []
            for i, (electrode, trajectory) in enumerate(
                zip(self.electrodes, self.cog_trajectories)
            ):
                if trajectory:
                    cog_points = np.array([p.center_of_gravity for p in trajectory])
                    # Use same nearest-point method for consistency
                    n_poly_samples = 500
                    t_poly = np.linspace(0, 1, n_poly_samples)
                    poly_points = np.array(
                        [electrode.get_point_at_parameter(t) for t in t_poly]
                    )

                    for cog_pt in cog_points:
                        dists = np.linalg.norm(poly_points - cog_pt, axis=1)
                        all_deviations.append(np.min(dists))

            if all_deviations:
                max_dev = max(all_deviations)
                self.ax_profile.set_ylim(
                    [0, max(max_dev * 1.1, 0.5)]
                )  # At least 0.5mm range

        self.ax_profile.legend(loc="best", fontsize=8)
        self.fig.canvas.draw_idle()

    def _get_detailed_contact_info(self) -> str:
        """Get compact contact information for display at bottom of CT view."""
        text = ""

        # Check if any electrode has detected contacts
        has_contacts = any(len(e.contact_positions) > 0 for e in self.electrodes)
        if not has_contacts:
            return text

        text = "═══ CONTACT DETECTION RESULTS ═══\n"

        for i, electrode in enumerate(self.electrodes):
            if len(electrode.contact_positions) == 0:
                continue

            # Get 3D positions
            contact_positions_3d = electrode.get_contact_positions_3d()

            text += f"Electrode {i+1} ({electrode.electrode_type}):\n"

            # Create compact table for contacts
            text += "  Contact | Distance | 3D Position (mm)\n"
            text += "  --------|----------|---------------------\n"

            for j, (pos_mm, pos_3d) in enumerate(
                zip(electrode.contact_positions, contact_positions_3d)
            ):
                text += f"  C{j+1:^6} | {pos_mm:7.2f} | ({pos_3d[0]:6.1f}, {pos_3d[1]:6.1f}, {pos_3d[2]:6.1f})\n"

            # Add spacing info
            if len(electrode.contact_positions) > 1:
                spacings = np.diff(electrode.contact_positions)
                text += f"  Spacing: {np.mean(spacings):.2f}±{np.std(spacings):.2f}mm"
                text += f" (range: {np.min(spacings):.2f}-{np.max(spacings):.2f}mm)\n"

            text += "\n"

        return text.rstrip()

    def _get_contact_info_text(self) -> str:
        """Get detailed contact information for detected electrodes."""
        text = ""

        # Check if any electrode has detected contacts
        has_contacts = any(len(e.contact_positions) > 0 for e in self.electrodes)
        if not has_contacts:
            return text

        text += "\n━━━ Contact Info ━━━\n"

        for i, electrode in enumerate(self.electrodes):
            if len(electrode.contact_positions) == 0:
                continue

            text += f"Electrode {i+1} ({electrode.electrode_type}):\n"

            # Get 3D positions of contacts
            contact_positions_3d = electrode.get_contact_positions_3d()

            # Display each contact
            for j, (pos_mm, pos_3d) in enumerate(
                zip(electrode.contact_positions, contact_positions_3d)
            ):
                text += f"  C{j+1}: {pos_mm:.2f}mm from tip\n"
                text += (
                    f"      3D: ({pos_3d[0]:.1f}, {pos_3d[1]:.1f}, {pos_3d[2]:.1f})\n"
                )

            # Calculate and display spacing between contacts
            if len(electrode.contact_positions) > 1:
                text += "  Spacing:\n"
                spacings = np.diff(electrode.contact_positions)
                for j, spacing in enumerate(spacings):
                    text += f"    C{j+1}→C{j+2}: {spacing:.2f}mm\n"

                # Statistics on spacing
                mean_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                text += f"  Mean: {mean_spacing:.2f}±{std_spacing:.2f}mm\n"

            text += "\n"

        return text

    def _get_result_text(self) -> str:
        """Get result summary text with fit statistics and contact information."""
        text = f"Seeds: {len(self.seed_points)}\n"

        # Calculate fit statistics if we have both polynomials and COG trajectories
        if self.electrodes and self.cog_trajectories:
            text += f"Polynomials: {len(self.electrodes)}\n"
            text += "━━━ Fit Statistics ━━━\n"

            all_deviations = []

            for i, electrode in enumerate(self.electrodes):
                if i < len(self.cog_trajectories) and self.cog_trajectories[i]:
                    # Get COG points
                    cog_points = np.array(
                        [p.center_of_gravity for p in self.cog_trajectories[i]]
                    )

                    # Sample polynomial at same density as COG points
                    n_samples = len(cog_points)
                    t_sample = np.linspace(0, 1, n_samples)
                    poly_points = np.array(
                        [electrode.get_point_at_parameter(t) for t in t_sample]
                    )

                    # Calculate point-to-point distances
                    if len(poly_points) == len(cog_points):
                        # Direct comparison (assumes similar parameterization)
                        distances = np.linalg.norm(poly_points - cog_points, axis=1)
                    else:
                        # Find nearest neighbor distances
                        distances = []
                        for poly_pt in poly_points:
                            dists = np.linalg.norm(cog_points - poly_pt, axis=1)
                            distances.append(np.min(dists))
                        distances = np.array(distances)

                    all_deviations.extend(distances)

                    # Statistics for this electrode
                    mean_dev = np.mean(distances)
                    max_dev = np.max(distances)
                    std_dev = np.std(distances)

                    text += f"P{i+1}: μ={mean_dev:.2f} σ={std_dev:.2f} max={max_dev:.2f}mm\n"

            if all_deviations:
                # Overall statistics
                text += "────────────────\n"
                overall_mean = np.mean(all_deviations)
                overall_std = np.std(all_deviations)
                overall_max = np.max(all_deviations)
                percentile_95 = np.percentile(all_deviations, 95)

                text += f"Overall: μ={overall_mean:.2f}mm\n"
                text += f"  σ={overall_std:.2f} max={overall_max:.2f}mm\n"
                text += f"  95%ile={percentile_95:.2f}mm\n"

                # Quality assessment
                if overall_mean < 0.5:
                    quality = "Excellent"
                elif overall_mean < 1.0:
                    quality = "Good"
                elif overall_mean < 2.0:
                    quality = "Fair"
                else:
                    quality = "Poor"
                text += f"Fit Quality: {quality}\n"

        elif self.electrodes:
            text += f"Polynomials: {len(self.electrodes)}\n"
            for i, electrode in enumerate(self.electrodes):
                text += f"P{i+1}: {electrode.electrode_type}\n"
                text += f"  Length: {electrode.length_mm:.1f}mm\n"
        else:
            text += "Polynomials: 0\n"

        if self.cog_trajectories and not self.electrodes:
            text += f"\nCOG Tracks: {len(self.cog_trajectories)}\n"
            for i, trajectory in enumerate(self.cog_trajectories):
                if trajectory:
                    points = np.array([p.center_of_gravity for p in trajectory])
                    if len(points) > 1:
                        diffs = np.diff(points, axis=0)
                        length = np.sum(np.linalg.norm(diffs, axis=1))
                    else:
                        length = 0
                    text += f"Track {i+1}: {len(trajectory)} pts, {length:.1f}mm\n"

        return text

    def _on_slider_change(self, val: int) -> None:
        """Handle slice slider changes."""
        new_slice = int(val)
        if new_slice != self.current_slice:
            self.current_slice = new_slice
            self._update_ct_display()

    def _on_click(self, event) -> None:
        """Handle mouse clicks."""
        if event.inaxes != self.ax_ct:
            return

        if event.xdata is None or event.ydata is None:
            return

        voxel_coord = self._display_to_voxel(event.xdata, event.ydata)

        # Validate coordinates
        if not self._is_valid_voxel(voxel_coord):
            return

        if event.button == 1:  # Left click - add seed point and auto-track
            self.seed_points.append(voxel_coord)
            print(f"Added seed point {len(self.seed_points)}: {voxel_coord}")

            # Mark as manual detection since user placed seed
            self.detection_method = "manual"

            # Automatically run COG tracking on the new seed point
            print("Auto-tracking COG from new seed point...")
            try:
                # Update tracker threshold to match current GUI setting
                self.cog_tracker.metal_threshold = self.metal_threshold

                # Track trajectory from the new seed
                trajectory_points = self.cog_tracker.track_from_seed(
                    seed_voxel=voxel_coord, slice_axis=self.slice_axis
                )

                if trajectory_points:
                    # Always run multi-scale angle detection for comparison
                    max_angle = max(
                        (
                            p.direction_change_angle
                            for p in trajectory_points
                            if p.direction_change_angle is not None
                        ),
                        default=0,
                    )

                    print(f"  Regular detection: max angle = {max_angle:.1f}°")

                    # Always run multi-scale detection to compare results
                    # Use a lower threshold (20°) for multi-scale to catch gradual bends
                    exit_idx = self._detect_skull_exit_multiscale(
                        trajectory_points, angle_threshold_deg=20.0
                    )
                    if exit_idx is not None:
                        print(
                            f"  Multi-scale detection found exit at point {exit_idx} of {len(trajectory_points)}"
                        )
                        # Only truncate if multi-scale found an earlier exit than regular detection
                        if exit_idx < len(trajectory_points):
                            print(
                                "  Truncating trajectory at multi-scale detected exit"
                            )
                            trajectory_points = trajectory_points[:exit_idx]
                    else:
                        print("  Multi-scale detection did not find additional exits")

                    self.cog_trajectories.append(trajectory_points)
                    print(
                        f"  Tracked {len(trajectory_points)} points (after skull exit correction)"
                    )

                    # Get start and end points
                    start_point = trajectory_points[0]
                    end_point = trajectory_points[-1]

                    # Print termination point coordinates
                    print("  Start point (tip):")
                    print(
                        f"    World: X={start_point.center_of_gravity[0]:.1f}, Y={start_point.center_of_gravity[1]:.1f}, Z={start_point.center_of_gravity[2]:.1f} mm"
                    )
                    print(
                        f"    Voxel: ({start_point.center_voxel[0]:.0f}, {start_point.center_voxel[1]:.0f}, {start_point.center_voxel[2]:.0f})"
                    )
                    print("  End point (entry):")
                    print(
                        f"    World: X={end_point.center_of_gravity[0]:.1f}, Y={end_point.center_of_gravity[1]:.1f}, Z={end_point.center_of_gravity[2]:.1f} mm"
                    )
                    print(
                        f"    Voxel: ({end_point.center_voxel[0]:.0f}, {end_point.center_voxel[1]:.0f}, {end_point.center_voxel[2]:.0f})"
                    )

                    # Calculate and print statistics
                    if len(trajectory_points) > 1:
                        points_world = np.array(
                            [p.center_of_gravity for p in trajectory_points]
                        )
                        diffs = np.diff(points_world, axis=0)
                        distances = np.linalg.norm(diffs, axis=1)
                        total_length = np.sum(distances)
                        print(f"  Total length: {total_length:.1f}mm")

                    # Check for significant direction changes
                    max_angle = max(
                        (
                            p.direction_change_angle
                            for p in trajectory_points
                            if p.direction_change_angle is not None
                        ),
                        default=0,
                    )
                    if max_angle > 20:
                        print(
                            f"  Note: Detected direction change of {max_angle:.1f}° (possible skull exit)"
                        )

                    # Automatically fit polynomial with degree 8
                    print("  Auto-fitting polynomial with degree 8...")
                    try:
                        # Convert COG trajectory to world coordinates
                        points_world = np.array(
                            [p.center_of_gravity for p in trajectory_points]
                        )
                        intensities = np.array(
                            [p.mean_intensity for p in trajectory_points]
                        )

                        # Fit polynomial with degree 8
                        from ..core.polynomial_fitting import (
                            fit_polynomial_to_trajectory,
                        )

                        poly_result = fit_polynomial_to_trajectory(
                            points_world, degree=8, weights=intensities
                        )

                        if poly_result:
                            # Refine skull exit detection using polynomial
                            print(
                                "    Refining skull exit detection along polynomial..."
                            )
                            refined_exit_idx, refined_exit_dist = (
                                self.cog_tracker.refine_skull_exit_detection(
                                    trajectory_points,
                                    poly_result.polynomial,
                                    step_size_mm=0.5,  # Check every 0.5mm for finer resolution
                                )
                            )

                            # If refined exit is significantly different, re-track
                            if (
                                refined_exit_idx is not None
                                and refined_exit_idx < len(trajectory_points) - 5
                            ):
                                original_length = len(trajectory_points)
                                print(
                                    f"    Refined exit found at point {refined_exit_idx} (was {original_length})"
                                )

                                # Truncate trajectory at refined exit point
                                trajectory_points = trajectory_points[
                                    : refined_exit_idx + 1
                                ]
                                self.cog_trajectories[-1] = (
                                    trajectory_points  # Update stored trajectory
                                )

                                # Refit polynomial with truncated trajectory
                                points_world = np.array(
                                    [p.center_of_gravity for p in trajectory_points]
                                )
                                intensities = np.array(
                                    [p.mean_intensity for p in trajectory_points]
                                )

                                poly_result = fit_polynomial_to_trajectory(
                                    points_world, degree=8, weights=intensities
                                )

                                print(
                                    f"    Refitted polynomial with {len(trajectory_points)} points"
                                )
                            else:
                                # Make sure we're using the latest trajectory points
                                points_world = np.array(
                                    [p.center_of_gravity for p in trajectory_points]
                                )
                                intensities = np.array(
                                    [p.mean_intensity for p in trajectory_points]
                                )

                            # Create electrode model for the newly fitted polynomial
                            from ..models.electrode import PolynomialElectrodeModel

                            # Calculate bounding box
                            min_coords = points_world.min(axis=0)
                            max_coords = points_world.max(axis=0)
                            bounding_box = (min_coords, max_coords)

                            electrode = PolynomialElectrodeModel(
                                polynomial=poly_result.polynomial,
                                electrode_type="Medtronic 3389",  # Default type
                                contact_positions=np.array([]),  # No contacts yet
                                intensity_profile=intensities,
                                distance_scale=poly_result.distance_scale_mm,
                                bounding_box=bounding_box,
                            )

                            # Update electrodes list (append if new trajectory)
                            if len(self.electrodes) < len(self.cog_trajectories):
                                self.electrodes.append(electrode)
                            else:
                                # Replace existing electrode for this trajectory
                                self.electrodes[len(self.cog_trajectories) - 1] = (
                                    electrode
                                )

                            print("    Polynomial fitted successfully (degree 8)")
                            print(
                                f"    Arc length: {poly_result.total_length_mm:.1f}mm"
                            )

                            # Enable OOR button now that we have polynomials
                            self._update_button_states()
                    except Exception as poly_e:
                        print(f"    Auto polynomial fitting failed: {poly_e}")
                else:
                    print("  No trajectory found from this seed point")

            except Exception as e:
                print(f"  Auto COG tracking failed: {e}")

        elif (
            event.button == 3
        ):  # Right click - remove nearest seed point and its trajectory
            if self.seed_points:
                # Find nearest seed point
                distances = []
                for seed in self.seed_points:
                    dist = np.linalg.norm(np.array(voxel_coord) - np.array(seed))
                    distances.append(dist)

                nearest_idx = np.argmin(distances)
                removed_seed = self.seed_points.pop(nearest_idx)

                # Also remove corresponding COG trajectory if it exists
                if nearest_idx < len(self.cog_trajectories):
                    self.cog_trajectories.pop(nearest_idx)
                    print(f"Removed seed point and trajectory: {removed_seed}")
                else:
                    print(f"Removed seed point: {removed_seed}")

        self._update_ct_display()
        self._update_3d_display()

    def _on_motion(self, event) -> None:
        """Handle mouse motion."""
        if event.inaxes != self.ax_ct:
            if self.crosshair_h:
                self.crosshair_h.set_visible(False)
                self.crosshair_v.set_visible(False)
            if self.coord_text:
                self.coord_text.set_text("")
            if self.hover_rect:
                self.hover_rect.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        if event.xdata is None or event.ydata is None:
            return

        # Skip if no CT data loaded
        if self.ct_data is None:
            return

        # Update crosshairs
        if self.crosshair_h and self.crosshair_v:
            self.crosshair_h.set_ydata([event.ydata, event.ydata])
            self.crosshair_v.set_xdata([event.xdata, event.xdata])
            self.crosshair_h.set_visible(True)
            self.crosshair_v.set_visible(True)

        # Update coordinate display
        voxel_coord = self._display_to_voxel(event.xdata, event.ydata)
        if self._is_valid_voxel(voxel_coord):
            world_coord = (self.affine @ np.append(voxel_coord, 1))[:3]
            intensity = self.ct_data[voxel_coord]

            # Check if in metal region
            is_metal = (
                self.metal_mask[voxel_coord] if self.metal_mask is not None else False
            )
            metal_text = " [METAL]" if is_metal else ""

            coord_text = f"Voxel: ({voxel_coord[0]:3d}, {voxel_coord[1]:3d}, {voxel_coord[2]:3d})\n"
            coord_text += f"World: ({world_coord[0]:7.1f}, {world_coord[1]:7.1f}, {world_coord[2]:7.1f}) mm\n"
            coord_text += f"Intensity: {intensity:.0f} HU{metal_text}\n"
            coord_text += f"Threshold: {self.metal_threshold} HU"

            if self.coord_text:
                self.coord_text.set_text(coord_text)

        # Show intensity preview rectangle
        self._update_hover_preview(event.xdata, event.ydata)

        # Only update the CT display area, not the whole figure
        self.ax_ct.figure.canvas.draw_idle()

    def _update_hover_preview(self, x: float, y: float) -> None:
        """Update hover preview rectangle."""
        voxel_coord = self._display_to_voxel(x, y)
        if not self._is_valid_voxel(voxel_coord):
            return

        # Remove old rectangle
        if self.hover_rect:
            self.hover_rect.remove()
            self.hover_rect = None

        # Use COG tracker search radius (matches the actual search area)
        if self.cog_tracker is None:
            return
        search_radius = self.cog_tracker.search_radius

        # Draw the rectangle centered at the mouse position
        rect_size = search_radius * 2  # Diameter of search area
        rect_left = x - search_radius
        rect_bottom = y - search_radius

        # Now check all pixels that will be inside this rectangle
        # Convert rectangle bounds to integer pixel indices
        i_min = int(max(0, np.floor(rect_left)))
        i_max = int(min(self.ct_data.shape[0] - 1, np.ceil(rect_left + rect_size)))
        j_min = int(max(0, np.floor(rect_bottom)))
        j_max = int(min(self.ct_data.shape[1] - 1, np.ceil(rect_bottom + rect_size)))

        # Extract the region based on display rectangle bounds
        i, j, k = voxel_coord
        if self.slice_axis == "axial":
            region = self.ct_data[i_min : i_max + 1, j_min : j_max + 1, k]
        elif self.slice_axis == "sagittal":
            region = self.ct_data[
                k,
                i_min : i_max + 1,
                self.ct_data.shape[2] - 1 - j_max : self.ct_data.shape[2] - j_min,
            ]
        else:  # coronal
            region = self.ct_data[
                i_min : i_max + 1,
                k,
                self.ct_data.shape[2] - 1 - j_max : self.ct_data.shape[2] - j_min,
            ]

        # Box is green if any pixel in the region is metal
        has_metal = np.any(region >= self.metal_threshold) if region.size > 0 else False
        color = "green" if has_metal else "red"

        # Create the rectangle at the exact position we checked
        self.hover_rect = Rectangle(
            (rect_left, rect_bottom),
            rect_size,
            rect_size,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            linestyle="-",
            alpha=0.8,
        )
        self.ax_ct.add_patch(self.hover_rect)

    def _on_scroll(self, event) -> None:
        """Handle scroll wheel events."""
        if event.inaxes != self.ax_ct:
            return

        # Skip if no CT data loaded
        if self.ct_data is None:
            return

        if event.button == "up":
            self.current_slice = min(
                self.current_slice + 1, self.ct_data.shape[self.slice_idx] - 1
            )
        elif event.button == "down":
            self.current_slice = max(self.current_slice - 1, 0)

        self.slider.set_val(self.current_slice)

    def _on_clear(self, _event) -> None:
        """Handle clear button."""
        self.seed_points.clear()
        self.electrodes.clear()
        self.cog_trajectories.clear()
        self.refined_trajectories.clear()  # Also clear refined trajectories
        print("Cleared all seed points, electrodes, and trajectories")

        # Re-enable auto run button
        self.button_enabled["auto_run"] = True

        self._update_ct_display()
        self._update_3d_display()
        self._update_intensity_profile()  # Force update of 2D plot to reset it
        self._update_button_states()  # Update button states after clearing

    def _on_load_ct(self, _event) -> None:
        """Handle load CT button."""
        # Try to use matplotlib's file dialog first
        try:
            # Qt backend is available, try to use file dialog
            try:
                from matplotlib import pyplot as plt

                # Try to get the Qt application
                from matplotlib.backends.qt_compat import QtWidgets

                file_dialog = QtWidgets.QFileDialog()
                file_path, _ = file_dialog.getOpenFileName(
                    None,
                    "Select CT NIfTI file",
                    "",
                    "NIfTI files (*.nii *.nii.gz);;All files (*.*)",
                )

                if file_path:
                    self._load_ct_file(Path(file_path))
                return
            except:
                pass
        except:
            pass

        # Fallback to tkinter if available
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()  # Hide the main window

            file_path = filedialog.askopenfilename(
                title="Select CT NIfTI file",
                filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")],
            )

            root.destroy()

            if file_path:
                self._load_ct_file(Path(file_path))
        except ImportError:
            # Last resort - use a simple text input dialog
            from matplotlib.widgets import TextBox

            # Create a small figure for file path input
            fig_dialog = plt.figure(figsize=(8, 2))
            fig_dialog.suptitle("Enter CT File Path", fontsize=12)

            ax_text = fig_dialog.add_axes([0.1, 0.5, 0.8, 0.3])
            text_box = TextBox(ax_text, "Path:", initial="")

            def submit(text):
                if text and Path(text).exists():
                    self._load_ct_file(Path(text))
                    plt.close(fig_dialog)
                else:
                    print(f"File not found: {text}")

            text_box.on_submit(submit)

            # Add instructions
            fig_dialog.text(
                0.5,
                0.2,
                "Enter full path to .nii or .nii.gz file and press Enter",
                ha="center",
                fontsize=10,
                color="gray",
            )

            plt.show()

    def _load_ct_file(self, ct_path: Path) -> None:
        """Load a CT file and initialize all necessary components."""
        try:
            print(f"\nLoading CT file: {ct_path}")

            # Clear any existing data
            self.seed_points.clear()
            self.electrodes.clear()
            self.cog_trajectories.clear()
            self.refined_trajectories.clear()
            self.detection_method = "detect_electrodes_radial"  # Reset to default

            # Update path and load data
            self.ct_path = ct_path
            self.pacer = PyPaCER(ct_path, metal_threshold=self.metal_threshold)

            # Update data references
            self.ct_data = self.pacer.ct_data
            self.affine = self.pacer.affine
            self.voxel_sizes = self.pacer.voxel_sizes

            # Update current slice to center
            self.current_slice = self.ct_data.shape[self.slice_idx] // 2

            # Initialize COG tracker with new data
            self.cog_tracker = CenterOfGravityTracker(
                ct_data=self.ct_data,
                affine=self.affine,
                metal_threshold=self.metal_threshold,
                search_radius_mm=5.0,  # Use consistent 5mm physical radius
                max_direction_change_deg=30.0,
                min_voxels_per_slice=3,
            )

            # Compute metal mask
            self._compute_initial_metal_mask()

            # Update GUI title if fig exists
            if hasattr(self, "fig") and self.fig:
                self.fig.suptitle(
                    f"Electrode Detection - {ct_path.name}",
                    fontsize=14,
                    fontweight="bold",
                )

                # Update slider range
                n_slices = self.ct_data.shape[self.slice_idx]
                self.slider.set_val(self.current_slice)
                self.slider.ax.clear()
                self.slider.__init__(
                    self.slider.ax,
                    f"{self.slice_axis.capitalize()} Slice",
                    0,
                    n_slices - 1,
                    valinit=self.current_slice,
                    valstep=1,
                    valfmt=f"%d / {n_slices-1}",
                    color="steelblue",
                )
                self.slider.on_changed(self._on_slider_change)

            # Update displays if GUI is initialized
            if hasattr(self, "ax_ct") and self.ax_ct:
                self._update_ct_display()
                self._update_3d_display()
                self._update_intensity_profile()
                self._update_button_states()

            print(f"CT loaded successfully: {self.ct_data.shape}")
            print(f"Voxel sizes: {self.voxel_sizes} mm")

            # Start background mesh extraction
            self._start_background_mesh_extraction()

        except Exception as e:
            print(f"Error loading CT file: {e}")
            import traceback

            traceback.print_exc()

    def _start_background_mesh_extraction(self) -> None:
        """Start mesh extraction in background thread."""
        # Cancel any existing thread
        if self.mesh_extraction_thread and self.mesh_extraction_thread.is_alive():
            print("Previous mesh extraction still running, skipping new extraction")
            return

        def extract_mesh_background():
            """Extract mesh in background thread."""
            try:
                print("Starting background mesh extraction...")
                from ..visualization.isosurface_extraction import extract_electrode_mesh

                # Use the current CT path
                ct_path_str = str(self.ct_path)

                # Extract mesh without electrode data (just the metal artifacts)
                self.electrode_mesh = extract_electrode_mesh(
                    ct_path_str, output_dir=None, electrode_data=None
                )

                if self.electrode_mesh:
                    print("Background mesh extraction completed successfully")
                else:
                    print("Background mesh extraction returned None")

            except Exception as e:
                print(f"Background mesh extraction failed: {e}")
                self.electrode_mesh = None

        # Start thread
        self.mesh_extraction_thread = threading.Thread(
            target=extract_mesh_background, daemon=True
        )
        self.mesh_extraction_thread.start()

    def _on_auto_run(self, _event) -> None:
        """Handle auto run button - automatically detect electrodes and run analysis."""
        # Check if button is enabled
        if not self.button_enabled.get("auto_run", False):
            return

        # Mark as radial detection since using auto run
        self.detection_method = "detect_electrodes_radial"

        print("\n=== Starting automatic electrode detection ===")

        try:
            # Show initial progress
            self._show_progress("Initializing...")

            # Clear any existing data
            self.seed_points.clear()
            self.electrodes.clear()
            self.cog_trajectories.clear()
            self.refined_trajectories.clear()
            self.detection_method = "detect_electrodes_radial"  # Reset to default

            # Update display to show cleared state
            self._update_ct_display()
            self._update_3d_display()
            self._update_intensity_profile()

            # Find electrode seed points using radial search
            self._show_progress("Searching for electrodes...")
            seed_points = self._find_electrode_seeds()

            if not seed_points:
                print("No electrode artifacts found in search regions")
                self._hide_progress()
                return

            print(f"\nFound {len(seed_points)} potential electrode locations")
            self.seed_points.extend(seed_points)

            # Update display to show seed points
            self._update_ct_display()

            # Track COG for each seed point using the same logic as manual clicks
            print("\nTracking center of gravity trajectories...")
            for i, seed in enumerate(seed_points):
                print(
                    f"\nProcessing electrode {i+1}/{len(seed_points)} at seed {seed}..."
                )
                self._show_progress(f"Tracking COG {i+1}/{len(seed_points)}...")

                # Use the exact same logic as manual click
                try:
                    # Update tracker threshold to match current GUI setting
                    self.cog_tracker.metal_threshold = self.metal_threshold

                    # Track trajectory from the seed
                    trajectory_points = self.cog_tracker.track_from_seed(
                        seed_voxel=seed, slice_axis=self.slice_axis
                    )

                    if trajectory_points:
                        # Always run multi-scale angle detection for comparison
                        max_angle = max(
                            (
                                p.direction_change_angle
                                for p in trajectory_points
                                if p.direction_change_angle is not None
                            ),
                            default=0,
                        )

                        print(f"  Regular detection: max angle = {max_angle:.1f}°")

                        # Always run multi-scale detection to compare results
                        # Use a lower threshold (20°) for multi-scale to catch gradual bends
                        exit_idx = self._detect_skull_exit_multiscale(
                            trajectory_points, angle_threshold_deg=20.0
                        )
                        if exit_idx is not None:
                            print(
                                f"  Multi-scale detection found exit at point {exit_idx} of {len(trajectory_points)}"
                            )
                            # Only truncate if multi-scale found an earlier exit than regular detection
                            if exit_idx < len(trajectory_points):
                                print(
                                    "  Truncating trajectory at multi-scale detected exit"
                                )
                                trajectory_points = trajectory_points[:exit_idx]
                        else:
                            print(
                                "  Multi-scale detection did not find additional exits"
                            )

                        self.cog_trajectories.append(trajectory_points)
                        print(
                            f"  Tracked {len(trajectory_points)} points (after skull exit correction)"
                        )

                        # Update displays to show COG trajectory
                        self._update_ct_display()
                        self._update_3d_display()

                        # Get trajectory info
                        start_point = trajectory_points[0]
                        end_point = trajectory_points[-1]
                    print(f"  Start point (tip): {start_point.center_voxel}")
                    print(f"  End point (entry): {end_point.center_voxel}")

                    # Calculate total length
                    if len(trajectory_points) > 1:
                        points_world = np.array(
                            [p.center_of_gravity for p in trajectory_points]
                        )
                        diffs = np.diff(points_world, axis=0)
                        distances = np.linalg.norm(diffs, axis=1)
                        total_length = np.sum(distances)
                        print(f"  Total length: {total_length:.1f}mm")

                        # Automatically fit polynomial with degree 8
                        print("  Auto-fitting polynomial with degree 8...")
                        self._show_progress(
                            f"Fitting polynomial {i+1}/{len(seed_points)}..."
                        )
                        try:
                            # Convert COG trajectory to world coordinates
                            points_world = np.array(
                                [p.center_of_gravity for p in trajectory_points]
                            )
                            intensities = np.array(
                                [p.mean_intensity for p in trajectory_points]
                            )

                            # Fit polynomial with degree 8
                            from ..core.polynomial_fitting import (
                                fit_polynomial_to_trajectory,
                            )

                            poly_result = fit_polynomial_to_trajectory(
                                points_world, degree=8, weights=intensities
                            )

                            if poly_result:
                                # Create electrode model for the newly fitted polynomial
                                from ..models.electrode import PolynomialElectrodeModel

                                # Calculate bounding box
                                min_coords = points_world.min(axis=0)
                                max_coords = points_world.max(axis=0)
                                bounding_box = (min_coords, max_coords)

                                electrode = PolynomialElectrodeModel(
                                    polynomial=poly_result.polynomial,
                                    electrode_type="Medtronic 3389",  # Default type
                                    contact_positions=np.array([]),  # No contacts yet
                                    intensity_profile=intensities,
                                    distance_scale=poly_result.distance_scale_mm,
                                    bounding_box=bounding_box,
                                )

                                self.electrodes.append(electrode)

                                print("    Polynomial fitted successfully (degree 8)")
                                print(
                                    f"    Arc length: {poly_result.total_length_mm:.1f}mm"
                                )

                                # Update displays to show fitted polynomial
                                self._update_ct_display()
                                self._update_3d_display()
                                self._update_intensity_profile()
                        except Exception as poly_e:
                            print(f"    Auto polynomial fitting failed: {poly_e}")
                    else:
                        print(f"  No trajectory found from seed {seed}")

                except Exception as e:
                    print(f"  Failed to process seed {seed}: {e}")
                    import traceback

                    traceback.print_exc()

            print(f"\nSuccessfully tracked {len(self.cog_trajectories)} electrodes")

            # Update displays
            self._update_ct_display()
            self._update_3d_display()
            self._update_button_states()

            # Run OOR refinement if trajectories were found
            if self.electrodes:
                print("\nRunning OOR refinement...")
                self._show_progress("Running OOR refinement...")
                self._on_run_oor(None)

            # Hide progress when done
            self._hide_progress()

        except Exception as e:
            print(f"Auto run failed: {e}")
            import traceback

            traceback.print_exc()
            self._hide_progress()

    def _find_electrode_seeds(self) -> List[Tuple[int, int, int]]:
        """Find electrode seed points using radial search from brain center."""
        import numpy as np
        from scipy.ndimage import center_of_mass, label

        # Get brain center (use image center as approximation)
        center_voxel = np.array(self.ct_data.shape) // 2
        print(f"Starting search from center: {center_voxel}")

        # Search radii in mm
        search_radii_mm = [30, 50, 70, 90]

        # Convert mm to voxels
        mean_voxel_size = np.mean(self.voxel_sizes)

        all_seed_points = []
        previous_mask = np.zeros_like(self.metal_mask, dtype=bool)

        for radius_mm in search_radii_mm:
            radius_voxels = int(radius_mm / mean_voxel_size)
            print(f"\nSearching at radius {radius_mm}mm ({radius_voxels} voxels)...")

            # Create spherical mask for current radius
            current_mask = self._create_spherical_mask(center_voxel, radius_voxels)

            # Find NEW metal voxels in this shell (exclude previously searched regions)
            shell_mask = current_mask & ~previous_mask
            metal_in_shell = self.metal_mask & shell_mask
            num_metal = metal_in_shell.sum()

            if num_metal == 0:
                print(f"  No new metal voxels found in shell at {radius_mm}mm")
                previous_mask = current_mask
                continue

            print(f"  Found {num_metal} metal voxels in shell")

            # Find connected components in the shell
            labeled, num_components = label(metal_in_shell)
            print(f"  Found {num_components} connected components in shell")

            # Get center of mass for each component
            shell_seeds = []
            for comp_id in range(1, num_components + 1):
                comp_mask = labeled == comp_id
                comp_size = comp_mask.sum()

                # Skip very small components (likely noise)
                if comp_size < 10:
                    continue

                # Get center of mass
                com = center_of_mass(comp_mask)
                com_voxel = tuple(int(round(c)) for c in com)

                # Check if this point is actually metal
                if self.ct_data[com_voxel] > self.metal_threshold:
                    # Check if this seed is far enough from existing seeds
                    is_new = True
                    for existing_seed in all_seed_points:
                        dist = np.linalg.norm(
                            np.array(com_voxel) - np.array(existing_seed)
                        )
                        if (
                            dist < 20
                        ):  # If within 20 voxels of existing seed, consider it the same electrode
                            is_new = False
                            break

                    if is_new:
                        print(
                            f"    Component {comp_id}: {comp_size} voxels, seed at {com_voxel}"
                        )
                        shell_seeds.append(com_voxel)

            all_seed_points.extend(shell_seeds)
            previous_mask = current_mask

            # Continue searching even if we found electrodes
            if len(all_seed_points) >= 2:
                print(
                    f"\nFound {len(all_seed_points)} electrodes total, stopping search"
                )
                break

        print(f"\nTotal electrode seeds found: {len(all_seed_points)}")
        return all_seed_points

    def _create_spherical_mask(self, center: np.ndarray, radius: int) -> np.ndarray:
        """Create a spherical mask around a center point."""
        # Create coordinate grids
        x, y, z = np.ogrid[
            : self.ct_data.shape[0], : self.ct_data.shape[1], : self.ct_data.shape[2]
        ]

        # Calculate distance from center
        dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2

        # Create spherical mask
        mask = dist_sq <= radius**2

        return mask

    def _show_progress(self, message: str) -> None:
        """Show progress message overlay."""
        self.progress_text.set_text(message)
        self.progress_text.set_visible(True)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()  # Force update

    def _hide_progress(self) -> None:
        """Hide progress message overlay."""
        self.progress_text.set_visible(False)
        self.fig.canvas.draw_idle()

    def _detect_skull_exit_multiscale(
        self, trajectory_points: List, angle_threshold_deg: float = 25.0
    ) -> Optional[int]:
        """
        Simple multi-scale angle detection for skull exit.
        Checks angles over different window sizes to catch gradual bends.

        Args:
            trajectory_points: List of trajectory points
            angle_threshold_deg: Angle threshold in degrees

        Returns:
            Index of likely skull exit point, or None if no exit detected
        """
        if len(trajectory_points) < 20:  # Need enough points for analysis
            return None

        try:
            # Convert to numpy arrays
            points_world = np.array([p.center_of_gravity for p in trajectory_points])

            # Multi-scale angle analysis
            # Look at angles over different window sizes
            window_sizes = [5, 10, 15]  # Look at angles over 5, 10, and 15 points

            max_angles_by_scale = {}
            exit_candidates = []

            # Check each window size
            for window in window_sizes:
                if window >= len(points_world) // 2:
                    continue  # Skip if window too large

                max_angle = 0
                max_angle_idx = None

                for i in range(window, len(points_world) - window):
                    # Vector from i-window to i
                    v1 = points_world[i] - points_world[i - window]
                    # Vector from i to i+window
                    v2 = points_world[i + window] - points_world[i]

                    # Normalize
                    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
                    v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

                    # Calculate angle
                    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))

                    # Track maximum angle at this scale
                    if angle_deg > max_angle:
                        max_angle = angle_deg
                        max_angle_idx = i

                    # Check if angle exceeds threshold
                    if angle_deg > angle_threshold_deg:
                        exit_candidates.append((i, angle_deg, window))

                max_angles_by_scale[window] = (max_angle, max_angle_idx)

            # Report findings
            print("\n  Multi-scale angle detection results:")
            for window, (max_angle, idx) in max_angles_by_scale.items():
                if idx is not None:
                    print(
                        f"    Window {window} points: max angle {max_angle:.1f}° at point {idx}"
                    )

            # Return the earliest exit candidate
            if exit_candidates:
                exit_candidates.sort(key=lambda x: x[0])  # Sort by index
                exit_idx, angle, window = exit_candidates[0]
                print(
                    f"  --> Detected {angle:.1f}° bend at point {exit_idx} (window={window} points)"
                )
                return exit_idx
            else:
                print(f"  --> No angles exceeded threshold of {angle_threshold_deg}°")

            return None

        except Exception as e:
            print(f"Error in multi-scale angle detection: {e}")
            return None

    def _on_track_cog(self, _event) -> None:
        """Handle COG tracking button."""
        if not self.seed_points:
            print("No seed points defined. Click on metal regions first.")
            return

        print(f"Starting COG tracking from {len(self.seed_points)} seed points...")

        # Clear previous trajectories
        self.cog_trajectories.clear()

        # Track from each seed point
        for i, seed_voxel in enumerate(self.seed_points):
            print(
                f"\nTracking from seed point {i+1}/{len(self.seed_points)}: {seed_voxel}"
            )

            try:
                # Update tracker threshold to match current GUI setting
                self.cog_tracker.metal_threshold = self.metal_threshold

                # Track trajectory from seed
                trajectory_points = self.cog_tracker.track_from_seed(
                    seed_voxel=seed_voxel, slice_axis=self.slice_axis
                )

                if trajectory_points:
                    self.cog_trajectories.append(trajectory_points)
                    print(f"  Tracked {len(trajectory_points)} points")

                    # Get start and end points
                    start_point = trajectory_points[0]
                    end_point = trajectory_points[-1]

                    # Print termination point coordinates
                    print("  Start point (tip):")
                    print(
                        f"    World: X={start_point.center_of_gravity[0]:.1f}, Y={start_point.center_of_gravity[1]:.1f}, Z={start_point.center_of_gravity[2]:.1f} mm"
                    )
                    print(
                        f"    Voxel: ({start_point.center_voxel[0]:.0f}, {start_point.center_voxel[1]:.0f}, {start_point.center_voxel[2]:.0f})"
                    )
                    print("  End point (entry):")
                    print(
                        f"    World: X={end_point.center_of_gravity[0]:.1f}, Y={end_point.center_of_gravity[1]:.1f}, Z={end_point.center_of_gravity[2]:.1f} mm"
                    )
                    print(
                        f"    Voxel: ({end_point.center_voxel[0]:.0f}, {end_point.center_voxel[1]:.0f}, {end_point.center_voxel[2]:.0f})"
                    )

                    # Print statistics
                    total_voxels = sum(p.num_voxels for p in trajectory_points)
                    mean_intensity = np.mean(
                        [p.mean_intensity for p in trajectory_points]
                    )

                    # Calculate total length
                    if len(trajectory_points) > 1:
                        points_world = np.array(
                            [p.center_of_gravity for p in trajectory_points]
                        )
                        diffs = np.diff(points_world, axis=0)
                        distances = np.linalg.norm(diffs, axis=1)
                        total_length = np.sum(distances)
                    else:
                        total_length = 0

                    print(f"  Total length: {total_length:.1f}mm")
                    print(f"  Total voxels: {total_voxels}")
                    print(f"  Mean intensity: {mean_intensity:.0f} HU")

                    # Check for direction changes
                    max_angle_change = 0
                    for point in trajectory_points:
                        if point.direction_change_angle is not None:
                            max_angle_change = max(
                                max_angle_change, point.direction_change_angle
                            )

                    if max_angle_change > 0:
                        print(f"  Max direction change: {max_angle_change:.1f}°")
                else:
                    print(f"  No trajectory found from seed {i+1}")

            except Exception as e:
                print(f"  COG tracking failed for seed {i+1}: {e}")
                import traceback

                traceback.print_exc()

        print(
            f"\nCOG tracking complete. Found {len(self.cog_trajectories)} trajectories."
        )

        # Update displays
        self._update_ct_display()
        self._update_3d_display()

    def _update_button_states(self, skip_auto_contact_detection=False) -> None:
        """Update button enable/disable states based on current data."""
        # Auto Run button - disable if pipeline has started
        has_started = bool(self.seed_points) or bool(self.electrodes)
        self.button_enabled["auto_run"] = not has_started
        self._set_button_appearance("auto_run", not has_started)

        # OOR button - enable if we have polynomials
        has_polynomials = bool(self.electrodes)
        self.button_enabled["run_oor"] = has_polynomials
        self._set_button_appearance("run_oor", has_polynomials)

        # Save buttons - enable if we have detected contacts
        has_contacts = False
        if self.electrodes:
            has_contacts = any(len(e.contact_positions) > 0 for e in self.electrodes)

        self.button_enabled["save_json"] = has_contacts
        self.button_enabled["save_html"] = has_contacts
        self._set_button_appearance("save_json", has_contacts)
        self._set_button_appearance("save_html", has_contacts)

    def _set_button_appearance(self, button_name: str, enabled: bool) -> None:
        """Set button appearance based on enabled state."""
        if button_name not in self.controls:
            return

        button = self.controls[button_name]

        # Only style buttons that have enable/disable states
        if button_name in ["run_oor", "save_json", "save_html", "auto_run"]:
            if enabled:
                # Enabled state
                button.label.set_color("darkgreen")
                button.label.set_weight("bold")
            else:
                # Disabled state
                button.label.set_color("darkred")
                button.label.set_weight("normal")

    def _run_contact_detection(self) -> None:
        """Run contact detection on refined trajectories."""
        if not self.refined_trajectories:
            print("No refined trajectories available. Run OOR first.")
            return

        # Get selected contact detection method
        method_map = {
            "Area Center": "contactAreaCenter",
            "Peak": "peak",
            "Peak Wave": "peakWaveCenter",
        }
        selected_method = self.controls["contact_method"].value_selected
        detection_method = method_map.get(selected_method, "contactAreaCenter")

        # Show progress message
        self._show_progress(f"Detecting contacts using method: {detection_method}")

        # Run contact detection on each refined trajectory

        for i, (electrode, refined) in enumerate(
            zip(self.electrodes, self.refined_trajectories)
        ):
            if not refined:
                continue

            self._show_progress(
                f"Detecting contacts for electrode {i+1}/{len(self.electrodes)}..."
            )
            print(f"\nDetecting contacts for electrode {i+1}...")

            try:
                # Detect contacts using the selected method
                contacts = detect_contacts(
                    refined_model=refined,
                    method=detection_method,
                    electrode_type=electrode.electrode_type,
                    display_profile=False,  # Don't show matplotlib plots
                    limit_search_mm=20.0,
                    run_all_methods=False,  # Just run selected method
                )

                if contacts is not None and len(contacts) > 0:
                    electrode.contact_positions = contacts
                    print(f"  Detected {len(contacts)} contacts")
                    for j, pos in enumerate(contacts):
                        print(f"    Contact {j+1}: {pos:.2f}mm from tip")

                    # Run orientation detection and electrode type classification
                    try:
                        orientation_data, classified_type = (
                            self.pacer._run_orientation_detection(
                                electrode,
                                electrode_idx=i,
                            )
                        )
                        if orientation_data is not None:
                            electrode.orientation_data = orientation_data
                            print(f"  Orientation: markers detected")
                        electrode.electrode_type = classified_type
                        print(f"  Electrode type: {classified_type}")
                    except Exception as e:
                        print(f"  Orientation detection error: {e}")

                    # Hemisphere detection for tip and entry positions
                    try:
                        electrode.tip_hemisphere = (
                            self.pacer._determine_hemisphere(
                                electrode.tip_position
                            )
                        )
                        electrode.entry_hemisphere = (
                            self.pacer._determine_hemisphere(
                                electrode.entry_position
                            )
                        )
                        print(
                            f"  Hemisphere: tip={electrode.tip_hemisphere}, "
                            f"entry={electrode.entry_hemisphere}"
                        )
                    except Exception as e:
                        print(f"  Hemisphere detection error: {e}")
                else:
                    print("  No contacts detected")
                    electrode.contact_positions = np.array([])

            except Exception as e:
                print(f"  Contact detection error: {e}")
                electrode.contact_positions = np.array([])

        self._show_progress("Contact detection complete")
        self._hide_progress()
        self._update_3d_display()
        self._update_ct_display()  # Update CT display to show contact info
        self._update_button_states()  # Update save button state

        # Autosave reconstruction data after contact detection
        self._autosave_reconstruction_data()

    def _on_contact_method_change(self, label) -> None:
        """Handle contact method change - auto-run detection if OOR has been run."""
        print(f"Contact detection method changed to: {label}")

        # If we have refined trajectories, re-run contact detection
        if self.refined_trajectories:
            self._run_contact_detection()

    def _on_detect(self, _event) -> None:
        """Handle detect button."""
        if not self.seed_points:
            print("No seed points defined. Click on metal regions first.")
            return

        print(
            f"Starting full electrode detection from {len(self.seed_points)} seed points..."
        )

        # Ensure masks are computed for full detection pipeline
        self._ensure_masks_computed()

        # Get detection parameters
        mode = self.controls["detection_mode"].value_selected
        electrode_type_label = self.controls["electrode_type"].value_selected
        electrode_type = (
            None if electrode_type_label == "Auto" else electrode_type_label
        )

        try:
            # Convert seed points to world coordinates for detection
            world_seed_points = []
            for voxel_coord in self.seed_points:
                world_coord = (self.affine @ np.append(voxel_coord, 1))[:3]
                world_seed_points.append(world_coord)

            # Set detection parameters based on mode
            if mode == "Fast":
                self.electrodes = self.pacer.detect_electrodes_fast(
                    electrode_type=electrode_type, auto_save_json=False
                )
            elif mode == "High":
                self.electrodes = self.pacer.detect_electrodes(
                    electrode_type=electrode_type,
                    xy_resolution=0.05,  # High resolution
                    z_resolution=0.01,
                    grid_size=2.0,
                    auto_save_json=False,
                )
            else:  # Normal
                self.electrodes = self.pacer.detect_electrodes(
                    electrode_type=electrode_type, auto_save_json=False
                )

            print(f"Detection complete. Found {len(self.electrodes)} electrodes.")

        except Exception as e:
            print(f"Detection failed: {e}")
            import traceback

            traceback.print_exc()

        self._update_ct_display()
        self._update_3d_display()

    def _prepare_reconstruction_data(self) -> dict:
        """Prepare reconstruction data for saving."""
        # Calculate CT volume bounding box
        ct_shape = self.ct_data.shape
        voxel_min = np.array([0, 0, 0, 1])
        voxel_max = np.array([ct_shape[0] - 1, ct_shape[1] - 1, ct_shape[2] - 1, 1])
        world_min = (self.affine @ voxel_min)[:3]
        world_max = (self.affine @ voxel_max)[:3]
        ct_bbox_min = np.minimum(world_min, world_max)
        ct_bbox_max = np.maximum(world_min, world_max)

        # Use the tracked detection method
        method = self.detection_method

        # Map GUI contact method names to CLI names
        contact_method_map = {
            "Area Center": "contactAreaCenter",
            "Peak": "peak",
            "Peak Wave": "peakWaveCenter",
        }
        gui_contact_method = (
            self.controls["contact_method"].value_selected
            if "contact_method" in self.controls
            else "Area Center"
        )
        contact_detection_method = contact_method_map.get(
            gui_contact_method, "contactAreaCenter"
        )

        # Get resolution parameters from detection mode or use defaults
        detection_mode = (
            self.controls["detection_mode"].value_selected
            if "detection_mode" in self.controls
            else "Normal"
        )
        if detection_mode == "Fast":
            xy_resolution = 0.3
            z_resolution = 0.1
            grid_size = 1.0
        elif detection_mode == "High":
            xy_resolution = 0.05
            z_resolution = 0.01
            grid_size = 2.0
        else:  # Normal
            xy_resolution = 0.1
            z_resolution = 0.025
            grid_size = 1.5

        # Override grid_size with user input if available
        try:
            grid_size = float(self.controls["grid_size"].text)
        except (ValueError, AttributeError, KeyError):
            pass  # Keep default

        # Prepare reconstruction parameters
        parameters = {
            "method": method,
            "contact_detection_method": contact_detection_method,
            "electrode_type": (
                self.controls["electrode_type"].value_selected
                if "electrode_type" in self.controls
                else None
            ),
            "final_degree": (
                int(self.controls["poly_degree"].text)
                if "poly_degree" in self.controls
                else 3
            ),
            "xy_resolution": xy_resolution,
            "z_resolution": z_resolution,
            "grid_size": grid_size,
            "search_radii_mm": [
                float(self.cog_tracker.search_radius * min(self.voxel_sizes))
            ],  # Convert to mm
            "metal_threshold": self.metal_threshold,
            "refinement_threshold": 800,
            "processing_type": (
                "GPU" if self.controls["use_gpu"].get_status()[0] else "CPU"
            ),
            "interface": "GUI",
        }

        # Prepare comprehensive data using PyPaCER format
        data = {
            "metadata": {
                "ct_file": str(self.ct_path.resolve()),
                "timestamp": datetime.now().isoformat(),
                "pypacer_version": PYPACER_VERSION,
                "voxel_sizes_mm": self.voxel_sizes.tolist(),
                "metal_threshold_HU": self.metal_threshold,
                "num_electrodes_detected": len(self.electrodes),
                "ct_volume_shape": list(ct_shape),
                "ct_volume_bounding_box": {
                    "min": ct_bbox_min.tolist(),
                    "max": ct_bbox_max.tolist(),
                },
            },
            "reconstruction_parameters": parameters,
            "seed_points": {
                "voxel": self.seed_points,
                "world": [
                    (self.affine @ np.append(voxel_coord, 1))[:3].tolist()
                    for voxel_coord in self.seed_points
                ],
            },
            "electrodes": [],
        }

        # Add electrode data with additional details
        for i, electrode in enumerate(self.electrodes):
            electrode_data = electrode.to_dict()
            electrode_data["electrode_index"] = i

            # Add 3D contact positions
            if len(electrode.contact_positions) > 0:
                electrode_data["contact_positions_3d"] = (
                    electrode.get_contact_positions_3d().tolist()
                )

            # Add trajectory coordinates (sampled)
            n_points = 100
            t_values = np.linspace(0, 1, n_points)
            trajectory_points = [
                electrode.get_point_at_parameter(t).tolist() for t in t_values
            ]
            electrode_data["trajectory_coordinates"] = trajectory_points

            # Add COG trajectory if available
            if i < len(self.cog_trajectories) and self.cog_trajectories[i]:
                cog_points = [
                    p.center_of_gravity.tolist() for p in self.cog_trajectories[i]
                ]
                electrode_data["cog_trajectory"] = {
                    "points": cog_points,
                    "num_points": len(cog_points),
                }

            # Add refinement info if available
            if i < len(self.refined_trajectories) and self.refined_trajectories[i]:
                refined = self.refined_trajectories[i]
                electrode_data["refinement_info"] = {
                    "refined": True,
                    "total_length_mm": (
                        refined.total_length_mm
                        if hasattr(refined, "total_length_mm")
                        else None
                    ),
                    "tip_detected": hasattr(refined, "pass2_tip_param"),
                    "tip_threshold_HU": (
                        refined.pass2_tip_threshold
                        if hasattr(refined, "pass2_tip_threshold")
                        else None
                    ),
                    "tip_param": (
                        refined.pass2_tip_param
                        if hasattr(refined, "pass2_tip_param")
                        else None
                    ),
                }

                # Add skeleton deviation statistics if available
                if (
                    hasattr(electrode, "skeleton_deviations_mm")
                    and electrode.skeleton_deviations_mm is not None
                ):
                    electrode_data["refinement_info"]["skeleton_deviation_stats"] = {
                        "mean_mm": float(np.mean(electrode.skeleton_deviations_mm)),
                        "max_mm": float(np.max(electrode.skeleton_deviations_mm)),
                        "min_mm": float(np.min(electrode.skeleton_deviations_mm)),
                    }

            data["electrodes"].append(electrode_data)

        # Add debug information if debug mode is enabled
        if self.debug_mode:
            data["debug_info"] = {
                "output_directory": (
                    str(self.output_dir) if self.output_dir else "default"
                ),
                "num_seed_points": len(self.seed_points),
                "seed_points": [list(sp) for sp in self.seed_points],
                "cog_trajectories_count": len(self.cog_trajectories),
                "refined_trajectories_count": len(self.refined_trajectories),
                "metal_components_found": self.num_components,
                "processing_stages": {
                    "cog_tracking_complete": len(self.cog_trajectories) > 0,
                    "oor_refinement_complete": len(self.refined_trajectories) > 0,
                    "contact_detection_complete": any(
                        len(e.contact_positions) > 0 for e in self.electrodes
                    ),
                },
                "gui_settings": {
                    "initial_slice_axis": self.slice_axis,
                    "metal_threshold_HU": self.metal_threshold,
                    "search_radius_voxels": (
                        self.cog_tracker.search_radius
                        if hasattr(self.cog_tracker, "search_radius")
                        else 10
                    ),
                },
            }

            # Add intensity profiles if available and in debug mode
            if self.debug_mode:
                for i, electrode in enumerate(self.electrodes):
                    if (
                        hasattr(electrode, "intensity_profile")
                        and electrode.intensity_profile is not None
                    ):
                        if "electrodes" in data and i < len(data["electrodes"]):
                            data["electrodes"][i]["debug_intensity_profile"] = {
                                "available": True,
                                "length": len(electrode.intensity_profile),
                                "has_distance_scale": hasattr(
                                    electrode, "distance_scale"
                                )
                                and electrode.distance_scale is not None,
                            }

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            """Convert numpy types to Python types recursively."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        # Convert all numpy types in the data structure
        return convert_types(data)

    def _on_save_json(self, _event) -> None:
        """Handle save JSON button."""
        # Check if button is enabled
        if not self.button_enabled.get("save_json", False):
            return

        # Check if contacts have been detected
        if not self.electrodes or not any(
            len(e.contact_positions) > 0 for e in self.electrodes
        ):
            print("Cannot save: No contacts detected.")
            return

        try:
            # Prepare data first
            data = self._prepare_reconstruction_data()

            # Determine output directory
            if self.output_dir:
                output_dir = self.output_dir
            else:
                # Default: Create pypacer directory in CT data location
                ct_dir = self.ct_path.parent if hasattr(self, "ct_path") else Path.cwd()
                output_dir = ct_dir / "pypacer"

            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"electrode_reconstruction_{timestamp}.json"

            # Save data
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            # Save minified version (core data only, no large profile arrays)
            profile_keys = {
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
            for electrode_data in data["electrodes"]:
                mini_electrode = {
                    k: v
                    for k, v in electrode_data.items()
                    if k not in profile_keys
                }
                # Strip marker intensity profiles from orientation data
                if "orientation" in mini_electrode:
                    orient = json.loads(
                        json.dumps(mini_electrode["orientation"])
                    )
                    for marker in orient.get("markers", {}).values():
                        marker.pop("intensity_profile", None)
                    if "contact_intensity_profile" in orient:
                        orient["contact_intensity_profile"].pop(
                            "intensity", None
                        )
                    mini_electrode["orientation"] = orient
                mini_data["electrodes"].append(mini_electrode)

            mini_path = json_path.with_stem(json_path.stem + "_mini")
            with open(mini_path, "w") as f:
                json.dump(mini_data, f, indent=2)

            print("\n" + "=" * 60)
            print("JSON SAVED")
            print(f"Location: {json_path}")
            print(f"Mini JSON: {mini_path}")
            print("=" * 60)

        except Exception as e:
            print(f"Error saving JSON: {e}")

    def _autosave_reconstruction_data(self) -> None:
        """Autosave reconstruction data to a hidden file after contact detection."""
        # Check if contacts have been detected
        if not self.electrodes or not any(
            len(e.contact_positions) > 0 for e in self.electrodes
        ):
            return

        try:
            # Prepare data first
            data = self._prepare_reconstruction_data()

            # Always save in the same directory as the loaded NIfTI file
            ct_dir = self.ct_path.parent if hasattr(self, "ct_path") else Path.cwd()

            # Save to hidden autosave file in CT directory
            autosave_path = ct_dir / ".pypacer_autosave.json"

            # Save data
            with open(autosave_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\nAutosaved reconstruction data to: {autosave_path}")

        except Exception as e:
            print(f"Warning: Failed to autosave reconstruction data: {e}")

    def _on_save_html(self, _event) -> None:
        """Handle save HTML button."""
        # Check if button is enabled
        if not self.button_enabled.get("save_html", False):
            return

        # Check if contacts have been detected
        if not self.electrodes or not any(
            len(e.contact_positions) > 0 for e in self.electrodes
        ):
            print("Cannot save: No contacts detected.")
            return

        try:
            import tempfile

            # Show processing indicator
            self.progress_text.set_text("Generating HTML report...")
            self.progress_text.set_visible(True)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # Prepare data
            data = self._prepare_reconstruction_data()

            # Check if we should wait for background mesh extraction
            if (
                self.electrode_mesh is None
                and self.mesh_extraction_thread
                and self.mesh_extraction_thread.is_alive()
            ):
                print("Waiting for background mesh extraction to complete...")
                self.mesh_extraction_thread.join(timeout=5.0)  # Wait up to 5 seconds

            # Determine output directory
            if self.output_dir:
                output_dir = self.output_dir
            else:
                # Default: Create pypacer directory in CT data location
                ct_dir = self.ct_path.parent if hasattr(self, "ct_path") else Path.cwd()
                output_dir = ct_dir / "pypacer"

            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = output_dir / f"reconstruction_report_{timestamp}.html"

            # Update progress
            self.progress_text.set_text("Saving data and generating report...")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # Save to temporary JSON for HTML generation
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_json_path = tmp.name

            # Generate HTML report
            try:
                from ..visualization.report import generate_html_report

                generate_html_report(
                    reconstruction_json_path=tmp_json_path,
                    output_path=str(html_path),
                    cached_mesh=self.electrode_mesh,  # Pass pre-computed mesh
                )

                # Hide progress indicator and show success
                self.progress_text.set_visible(False)

                print("\n" + "=" * 60)
                print("HTML REPORT SAVED")
                print(f"Location: {html_path}")
                print("=" * 60)

                # Briefly show success message
                self.progress_text.set_text(f"HTML saved: {html_path.name}")
                self.progress_text.set_visible(True)
                self.fig.canvas.draw_idle()

                # Schedule hiding the message after 3 seconds
                import threading

                def hide_message():
                    import time

                    time.sleep(3)
                    self.progress_text.set_visible(False)
                    self.fig.canvas.draw_idle()

                threading.Thread(target=hide_message, daemon=True).start()

            except Exception as e:
                # Hide progress indicator on error
                self.progress_text.set_visible(False)
                self.fig.canvas.draw_idle()
                print(f"Error generating HTML report: {e}")
            finally:
                # Clean up temporary JSON
                Path(tmp_json_path).unlink(missing_ok=True)

        except Exception as e:
            # Hide progress indicator on error
            self.progress_text.set_visible(False)
            self.fig.canvas.draw_idle()
            print(f"Error saving HTML: {e}")

    def _on_threshold_change(self, text: str) -> None:
        """Handle threshold change."""
        try:
            new_threshold = float(text)
            if new_threshold > 0:
                self.metal_threshold = new_threshold
                self.pacer.metal_threshold = new_threshold
                self.cog_tracker.metal_threshold = (
                    new_threshold  # Update COG tracker too
                )
                print(f"Threshold changed to: {new_threshold} HU")
                # Invalidate metal mask and components (will be recomputed if needed)
                self.metal_mask = None
                self.metal_components = None
                self.num_components = 0
                # Recompute only metal mask with new threshold
                self._compute_initial_metal_mask()
                self._update_ct_display()
        except ValueError:
            print(f"Invalid threshold value: {text}")

    def _on_show_seeds_change(self, _label) -> None:
        """Handle show seeds checkbox."""
        self._update_ct_display()

    def _on_show_components_change(self, _label) -> None:
        """Handle show components checkbox."""
        self.show_components = self.controls["show_components"].get_status()[0]
        self._update_ct_display()

    def _on_show_deviation_change(self, _label) -> None:
        """Handle show deviation checkbox."""
        self._update_intensity_profile()

    def _on_poly_degree_change(self, text: str) -> None:
        """Handle polynomial degree change (for OOR final degree)."""
        try:
            new_degree = int(text)
            if 1 <= new_degree <= 8:
                print(f"OOR polynomial degree set to: {new_degree}")
            else:
                print("Polynomial degree must be between 1 and 8")
                self.controls["poly_degree"].set_val("3")
        except ValueError:
            print(f"Invalid polynomial degree: {text}")
            self.controls["poly_degree"].set_val("3")

    def _on_angle_cutoff_change(self, text: str) -> None:
        """Handle angle cutoff input changes for COG tracking."""
        try:
            angle = float(text)
            if 0 <= angle <= 90:
                # Update the max_direction_change in radians
                self.cog_tracker.max_direction_change = np.radians(angle)
                print(f"COG angle cutoff set to {angle}°")
            else:
                print(f"Invalid angle {angle}. Must be between 0 and 90.")
                self.controls["angle_cutoff"].set_val("60")
        except ValueError:
            print(f"Invalid angle value: {text}")
            self.controls["angle_cutoff"].set_val("60")

    def _on_search_radius_change(self, text: str) -> None:
        """Handle search radius input changes (in voxels)."""
        try:
            radius = int(text)
            if 1 <= radius <= 20:
                self.cog_tracker.search_radius = radius
                radius_mm = radius * np.mean(self.voxel_sizes)
                print(f"Search radius set to {radius} voxels (~{radius_mm:.1f}mm)")
                # Update the display to show the new search box size
                self._update_ct_display()
            else:
                print(f"Invalid radius {radius}. Must be between 1 and 20 voxels.")
                self.controls["search_radius"].set_val("10")
        except ValueError:
            print(f"Invalid radius value: {text}")
            self.controls["search_radius"].set_val("10")

    def _on_grid_size_change(self, text: str) -> None:
        """Handle grid size input changes for OOR refinement."""
        try:
            grid_size = float(text)
            if 0.5 <= grid_size <= 5.0:
                print(f"OOR grid size set to {grid_size}mm")
                # Grid size is used when running OOR, no immediate action needed
            else:
                print(f"Invalid grid size {grid_size}. Must be between 0.5 and 5.0 mm.")
                self.controls["grid_size"].set_val("1.5")
        except ValueError:
            print("Invalid grid size. Must be a number.")
            self.controls["grid_size"].set_val("1.5")

    def _on_gpu_toggle(self, label: str) -> None:
        """Handle GPU toggle checkbox."""
        use_gpu = self.controls["use_gpu"].get_status()[0]
        print(f"GPU mode: {'Enabled' if use_gpu else 'Disabled'}")

        # Enable/disable detection mode based on GPU toggle
        if use_gpu:
            # Disable mode selection when GPU is on
            for label in self.controls["detection_mode"].labels:
                label.set_alpha(0.3)
        else:
            # Enable mode selection when GPU is off
            for label in self.controls["detection_mode"].labels:
                label.set_alpha(1.0)

        self.fig.canvas.draw_idle()

    def _on_auto_fit_polynomial(self, _event) -> None:
        """Automatically find the best polynomial degree by testing all degrees."""
        if not self.cog_trajectories:
            print("No COG trajectories available. Track COG first.")
            return

        print("=" * 60)
        print("AUTO FIT: Testing polynomial degrees 1-8...")
        print("=" * 60)

        best_results = []  # Store best degree for each trajectory

        # Get electrode type setting
        electrode_type_label = self.controls["electrode_type"].value_selected
        electrode_type = (
            None if electrode_type_label == "Auto" else electrode_type_label
        )

        # Test each trajectory
        for traj_idx, trajectory in enumerate(self.cog_trajectories):
            if not trajectory:
                continue

            # Extract points
            points_world = np.array([p.center_of_gravity for p in trajectory])
            intensities = np.array([p.mean_intensity for p in trajectory])
            n_points = len(points_world)

            print(f"\nTrajectory {traj_idx+1}: Testing degrees 1-{min(8, n_points-1)}")
            print("-" * 40)

            degree_results = []

            # Test each degree
            for degree in range(1, min(9, n_points)):  # Max degree is n_points - 1
                try:
                    # Fit polynomial
                    trajectory_model = fit_polynomial_to_trajectory(
                        points_world, degree=degree, weights=intensities
                    )

                    # Calculate fit quality
                    # Use the polynomial evaluation from the model
                    from ..utils.math_helpers import polyval3

                    t_sample = np.linspace(0, 1, n_points)
                    poly_points = np.array(
                        [polyval3(trajectory_model.polynomial, t) for t in t_sample]
                    )

                    # Calculate deviations
                    if len(poly_points) == len(points_world):
                        deviations = np.linalg.norm(poly_points - points_world, axis=1)
                    else:
                        deviations = []
                        for poly_pt in poly_points:
                            dists = np.linalg.norm(points_world - poly_pt, axis=1)
                            deviations.append(np.min(dists))
                        deviations = np.array(deviations)

                    mean_dev = np.mean(deviations)
                    max_dev = np.max(deviations)
                    std_dev = np.std(deviations)

                    # Calculate AIC (Akaike Information Criterion) for model selection
                    # AIC = measure of fit quality + penalty for complexity
                    #
                    # Why use AIC?
                    # - Higher degree polynomials will ALWAYS fit better (lower error)
                    # - But they may overfit, capturing noise instead of true trajectory
                    # - AIC penalizes complex models to prevent overfitting
                    #
                    # AIC = n * log(MSE) + 2 * k
                    # where: n = number of data points
                    #        MSE = mean squared error (fit quality)
                    #        k = number of parameters in model
                    #
                    # For polynomials: k = (degree + 1) * 3 dimensions
                    # Example: degree 3 has 4 coefficients × 3 dims = 12 parameters
                    #          degree 5 has 6 coefficients × 3 dims = 18 parameters
                    #
                    # Lower AIC = better balance of accuracy and simplicity
                    n = len(deviations)
                    mse = np.mean(deviations**2)
                    k = (degree + 1) * 3  # Number of parameters

                    if mse > 0:
                        aic = n * np.log(mse) + 2 * k
                    else:
                        aic = -np.inf  # Perfect fit (unlikely in practice)

                    degree_results.append(
                        {
                            "degree": degree,
                            "mean_dev": mean_dev,
                            "max_dev": max_dev,
                            "std_dev": std_dev,
                            "aic": aic,
                            "model": trajectory_model,
                        }
                    )

                    print(
                        f"  Degree {degree}: μ={mean_dev:.3f}mm, σ={std_dev:.3f}mm, max={max_dev:.3f}mm, AIC={aic:.1f}"
                    )

                except Exception as e:
                    print(f"  Degree {degree}: Failed - {e}")

            if degree_results:
                # Find best degree using AIC
                best_result = min(degree_results, key=lambda x: x["aic"])

                # Also check if a lower degree is nearly as good (within 5% higher AIC)
                # Prefer simpler models when performance is similar
                for result in sorted(degree_results, key=lambda x: x["degree"]):
                    if result["aic"] <= best_result["aic"] * 1.05:  # Within 5% of best
                        best_result = result
                        break

                best_results.append(best_result)

                print(f"\n  ★ Best fit: Degree {best_result['degree']}")
                print(f"    Mean deviation: {best_result['mean_dev']:.3f}mm")
                print(f"    AIC score: {best_result['aic']:.1f}")

        # Apply best fits
        if best_results:
            print("\n" + "=" * 60)
            print("APPLYING BEST FITS")
            print("=" * 60)

            # Clear existing electrodes
            self.electrodes.clear()

            # Create electrode models with best degrees
            for i, best in enumerate(best_results):
                try:
                    # Calculate bounding box
                    traj_idx = i  # Assumes same order
                    if traj_idx < len(self.cog_trajectories):
                        points_world = np.array(
                            [
                                p.center_of_gravity
                                for p in self.cog_trajectories[traj_idx]
                            ]
                        )
                        intensities = np.array(
                            [p.mean_intensity for p in self.cog_trajectories[traj_idx]]
                        )

                        min_coords = points_world.min(axis=0)
                        max_coords = points_world.max(axis=0)
                        bounding_box = (min_coords, max_coords)

                        # Create electrode model
                        electrode_type_to_use = (
                            electrode_type if electrode_type else "Medtronic 3389"
                        )

                        electrode = PolynomialElectrodeModel(
                            polynomial=best["model"].polynomial,
                            electrode_type=electrode_type_to_use,
                            contact_positions=np.array([]),  # No contacts for debugging
                            intensity_profile=intensities,
                            distance_scale=best["model"].distance_scale_mm,
                            bounding_box=bounding_box,
                        )

                        self.electrodes.append(electrode)
                        print(
                            f"Applied degree {best['degree']} polynomial to trajectory {i+1}"
                        )

                        # Update degree display to show the best degree found
                        if i == 0:  # Update display with first trajectory's best degree
                            self.controls["poly_degree"].set_val(str(best["degree"]))

                except Exception as e:
                    print(f"Failed to apply best fit for trajectory {i+1}: {e}")

            print(
                f"\nAuto-fit complete. Created {len(self.electrodes)} electrode models."
            )

            # Update displays
            self._update_ct_display()
            self._update_3d_display()
        else:
            print("No valid polynomial fits found.")

    def _on_run_oor(self, _event) -> None:
        """Run Orthogonal Optimal Resampling refinement on fitted polynomials."""
        # Check if button is enabled
        if not self.button_enabled.get("run_oor", False):
            return

        if not self.electrodes:
            print("No fitted polynomials available. Fit polynomials first.")
            return

        # Get the user-specified polynomial degree for OOR
        try:
            oor_poly_degree = int(self.controls["poly_degree"].text)
        except (ValueError, AttributeError):
            oor_poly_degree = 3  # Default to 3 if not valid

        # Get the user-specified grid size for OOR
        try:
            oor_grid_size = float(self.controls["grid_size"].text)
        except (ValueError, AttributeError):
            oor_grid_size = 1.5  # Default to 1.5mm if not valid

        # Get detection mode to determine OOR resolution settings
        mode = self.controls["detection_mode"].value_selected

        # Set resolution parameters based on mode
        if mode == "Fast":
            xy_resolution = 0.3
            z_resolution = 0.1
            # Use user-specified grid_size or default to 1.0mm for fast mode
            if oor_grid_size == 1.5:  # If using default, switch to fast default
                oor_grid_size = 1.0
        elif mode == "High":
            xy_resolution = 0.05
            z_resolution = 0.01
            # Use user-specified grid_size or default to 2.0mm for high mode
            if oor_grid_size == 1.5:  # If using default, switch to high default
                oor_grid_size = 2.0
        else:  # Normal
            xy_resolution = 0.1
            z_resolution = 0.025
            # Keep user-specified grid_size or default 1.5mm

        print(
            f"OOR parameters: mode={mode}, polynomial degree={oor_poly_degree}, grid size={oor_grid_size}mm"
        )
        print(
            f"                xy_resolution={xy_resolution}mm, z_resolution={z_resolution}mm"
        )

        # Show progress
        self._show_progress("Starting OOR refinement...")

        print("=" * 60)
        print("Running OOR (Orthogonal Optimal Resampling) refinement...")
        print(f"Mode: {mode}")
        print(f"GUI polynomial degree setting: {oor_poly_degree}")
        print(
            "Auto degree selection: ENABLED (will select best degree for bottom 20mm)"
        )
        print("=" * 60)

        # Check GPU preference and availability
        use_gpu = self.controls["use_gpu"].get_status()[0]
        from ..gpu.gpu_utils import pytorch_gpu_available

        gpu_available = pytorch_gpu_available()

        # Determine which refinement function to use
        if use_gpu and not gpu_available:
            error_msg = """
            ============================================================
            ERROR: GPU acceleration is not available!

            PyTorch CUDA is required for OOR refinement but was not detected.
            Possible reasons:
            1. No NVIDIA GPU present in the system
            2. PyTorch is installed without CUDA support
            3. CUDA drivers are not properly installed
            4. GPU is not accessible (e.g., in use by another process)

            To use CPU instead (slower), please uncheck 'Use GPU' in the GUI
            and run OOR refinement again.

            To check your PyTorch installation:
              python -c "import torch; print(torch.cuda.is_available())"
            ============================================================
            """
            print(error_msg)
            self._show_progress("ERROR: GPU not available - stopping")
            return
        elif use_gpu and gpu_available:
            from ..gpu.refinement_gpu import refine_electrode_trajectory_gpu

            use_gpu_refinement = True
            print("GPU acceleration available (PyTorch CUDA) - using GPU for OOR")
            self._show_progress("GPU acceleration available")
        else:
            from ..core.refinement import refine_electrode_trajectory

            use_gpu_refinement = False
            print("Using CPU for OOR refinement (GPU disabled)")
            self._show_progress("Using CPU for OOR refinement")

        # Clear previous refined trajectories
        self.refined_trajectories.clear()

        for i, electrode in enumerate(self.electrodes):
            print(f"\nRefining electrode {i+1}...")
            self._show_progress(f"OOR refinement {i+1}/{len(self.electrodes)}...")

            try:
                # Extract point cloud around polynomial trajectory
                # We'll use either COG points or extract from CT

                if i < len(self.cog_trajectories) and self.cog_trajectories[i]:
                    # Use COG points as initial point cloud
                    print(f"  Using {len(self.cog_trajectories[i])} COG points")
                    self._show_progress(
                        f"Using {len(self.cog_trajectories[i])} COG points for electrode {i+1}"
                    )
                    points_world = np.array(
                        [p.center_of_gravity for p in self.cog_trajectories[i]]
                    )
                    intensities = np.array(
                        [p.mean_intensity for p in self.cog_trajectories[i]]
                    )
                else:
                    # Extract points in cylinder around polynomial
                    print("  Extracting point cloud around polynomial...")
                    self._show_progress(
                        f"Extracting point cloud for electrode {i+1}..."
                    )
                    points_world, intensities = self._extract_cylinder_points(
                        electrode.polynomial, radius_mm=3.5
                    )
                    print(f"  Extracted {len(points_world)} points")
                    self._show_progress(f"Extracted {len(points_world)} points")

                if len(points_world) < 10:
                    print("  Insufficient points for refinement")
                    continue

                # Create initial trajectory object compatible with refinement
                from ..core.trajectory_fit import InitialTrajectory

                self._show_progress(f"Preparing trajectory for electrode {i+1}...")
                initial_traj = InitialTrajectory(
                    polynomial=electrode.polynomial,
                    skeleton=points_world,  # Use points as skeleton
                    total_length_mm=electrode.length_mm,
                    degree=electrode.polynomial.shape[0] - 1,
                )

                # Run refinement with parameters optimized for GUI usage
                print("  Running OOR refinement...")

                # GPU is required - no CPU fallback
                # Determine debug output directory if in debug mode
                debug_output_dir = None
                if self.debug_mode:
                    if self.output_dir:
                        debug_output_dir = str(self.output_dir)
                    else:
                        # Use default pypacer directory
                        ct_dir = self.ct_path.parent
                        debug_output_dir = str(ct_dir / "pypacer")

                self._show_progress(
                    f"Running {'GPU' if use_gpu_refinement else 'CPU'} refinement for electrode {i+1}..."
                )

                if use_gpu_refinement:
                    refined = refine_electrode_trajectory_gpu(
                        initial_traj,
                        points_world,
                        intensities,
                        self.ct_data,
                        self.affine,
                        final_degree=oor_poly_degree,  # Use user-specified degree
                        xy_resolution=xy_resolution,  # Mode-dependent resolution
                        z_resolution=z_resolution,  # Mode-dependent resolution along trajectory
                        grid_size=oor_grid_size,  # Mode-dependent grid size
                        use_gpu=True,
                        refinement_threshold=800,  # Lower threshold for refinement
                        refinement_radius_mm=3.5,
                        use_subvolume=True,  # Use subvolume for faster processing
                        auto_select_degree=True,  # Auto-select best degree based on bottom 20mm
                        contact_region_mm=20.0,  # Evaluate fit quality in bottom 20mm
                        electrode_idx=i,  # Pass electrode index
                        debug_output_dir=debug_output_dir,  # Pass debug directory if in debug mode
                    )
                else:
                    refined = refine_electrode_trajectory(
                        initial_traj,
                        points_world,
                        intensities,
                        self.ct_data,
                        self.affine,
                        final_degree=oor_poly_degree,  # Use user-specified degree
                        xy_resolution=xy_resolution,  # Mode-dependent resolution
                        z_resolution=z_resolution,  # Mode-dependent resolution along trajectory
                        grid_size=oor_grid_size,  # Mode-dependent grid size
                        refinement_threshold=800,  # Lower threshold for refinement
                        refinement_radius_mm=3.5,
                        auto_select_degree=True,  # Auto-select best degree based on bottom 20mm
                        contact_region_mm=20.0,  # Evaluate fit quality in bottom 20mm
                        electrode_idx=i,  # Pass electrode index
                        debug_output_dir=debug_output_dir,  # Pass debug directory if in debug mode
                    )

                self.refined_trajectories.append(refined)

                print("  Refinement complete:")
                if hasattr(refined, "total_length_mm"):
                    print(f"    Final length: {refined.total_length_mm:.1f}mm")
                print(f"    Intensity profile points: {len(refined.intensity_profile)}")
                print(f"    Distance scale points: {len(refined.distance_scale_mm)}")

                # Show progress for completion
                self._show_progress(f"Electrode {i+1} refinement complete")

                # Update button states but skip auto contact detection during OOR loop
                self._update_button_states(skip_auto_contact_detection=True)

                # Update electrode model with refined polynomial and intensity profile
                electrode.polynomial = refined.polynomial

                # Update the main intensity profile and distance scale attributes
                # This ensures they are saved correctly in the JSON export
                electrode.intensity_profile = refined.intensity_profile
                electrode.distance_scale = refined.distance_scale_mm

                # Recalculate bounding box based on refined trajectory
                # The original bounding box was from COG points, which doesn't match the refined polynomial
                from ..utils.math_helpers import polyval3

                t_values = np.linspace(0.0, 1.0, 200)
                refined_trajectory = np.array(
                    [polyval3(refined.polynomial, t) for t in t_values]
                )
                min_coords = refined_trajectory.min(axis=0)
                max_coords = refined_trajectory.max(axis=0)
                electrode.bounding_box = (min_coords, max_coords)

                # Also store as refined attributes for GUI visualization
                electrode.refined_intensity_profile = refined.intensity_profile
                electrode.refined_distance_scale = refined.distance_scale_mm

                # Store polynomial before tip detection if available (debug mode)
                if (
                    hasattr(refined, "polynomial_before_tip_detection")
                    and refined.polynomial_before_tip_detection is not None
                ):
                    electrode.polynomial_before_tip_detection = (
                        refined.polynomial_before_tip_detection
                    )

                # Store full Pass 2 data if available (debug mode)
                if (
                    hasattr(refined, "pass2_intensities_full")
                    and refined.pass2_intensities_full is not None
                ):
                    electrode.pass2_intensities_full = refined.pass2_intensities_full
                if (
                    hasattr(refined, "pass2_distances_mm_full")
                    and refined.pass2_distances_mm_full is not None
                ):
                    electrode.pass2_distances_mm_full = refined.pass2_distances_mm_full
                if (
                    hasattr(refined, "pass2_tip_threshold")
                    and refined.pass2_tip_threshold is not None
                ):
                    electrode.pass2_tip_threshold = refined.pass2_tip_threshold
                if (
                    hasattr(refined, "original_t0_distance_mm")
                    and refined.original_t0_distance_mm is not None
                ):
                    electrode.original_t0_distance_mm = refined.original_t0_distance_mm

                # Store skeleton deviation data if available
                if (
                    hasattr(refined, "skeleton_deviations_mm")
                    and refined.skeleton_deviations_mm is not None
                ):
                    electrode.skeleton_deviations_mm = refined.skeleton_deviations_mm
                    # Note: deviations are aligned with distance_scale, no need for separate distances
                    print(
                        f"    Stored skeleton deviations: {len(refined.skeleton_deviations_mm)} points"
                    )

                # Initialize empty contact positions (will be filled by auto-detection)
                electrode.contact_positions = np.array([])
                print("    Ready for contact detection (will run automatically)")

                # Update displays to show refined trajectory
                self._update_3d_display()
                self._update_intensity_profile()

            except Exception as e:
                print(f"  Refinement failed: {e}")
                import traceback

                traceback.print_exc()
                self.refined_trajectories.append(None)

        print(
            f"\nOOR refinement complete. Refined {len([r for r in self.refined_trajectories if r])} electrodes."
        )
        self._show_progress(
            f"OOR complete - refined {len([r for r in self.refined_trajectories if r])} electrodes"
        )

        # Automatically run contact detection after OOR
        print("\nAutomatically detecting contacts...")
        self._run_contact_detection()

        # Update button states after OOR and contact detection
        self._update_button_states(skip_auto_contact_detection=True)

        # Update displays to show refined trajectories
        self._update_ct_display()
        self._update_3d_display()
        # Explicitly update intensity profile to ensure refined data is shown
        self._update_intensity_profile()

    def _extract_cylinder_points(
        self, polynomial: np.ndarray, radius_mm: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract points within a cylinder around the polynomial trajectory.

        Args:
            polynomial: Polynomial coefficients
            radius_mm: Radius of cylinder in mm

        Returns:
            Tuple of (points_world, intensities)
        """
        from ..utils.math_helpers import polyval3

        # Sample polynomial at regular intervals
        n_samples = 100
        t_values = np.linspace(0, 1, n_samples)
        trajectory_points = np.array([polyval3(polynomial, t) for t in t_values])

        # Convert to voxel coordinates
        affine_inv = np.linalg.inv(self.affine)
        trajectory_voxels = []
        for tp in trajectory_points:
            tv = (affine_inv @ np.append(tp, 1))[:3]
            trajectory_voxels.append(tv)
        trajectory_voxels = np.array(trajectory_voxels)

        # Calculate radius in voxels
        radius_voxels = int(np.ceil(radius_mm / np.min(self.voxel_sizes)))

        # Extract voxels within cylinder
        points_voxel = []
        intensities = []

        for tv in trajectory_voxels:
            # Get neighborhood around trajectory point
            i, j, k = tv.astype(int)

            for di in range(-radius_voxels, radius_voxels + 1):
                for dj in range(-radius_voxels, radius_voxels + 1):
                    for dk in range(-radius_voxels, radius_voxels + 1):
                        vi, vj, vk = i + di, j + dj, k + dk

                        # Check bounds
                        if (
                            0 <= vi < self.ct_data.shape[0]
                            and 0 <= vj < self.ct_data.shape[1]
                            and 0 <= vk < self.ct_data.shape[2]
                        ):

                            # Check if within cylinder radius
                            dist_sq = di**2 + dj**2 + dk**2
                            if dist_sq <= radius_voxels**2:
                                intensity = self.ct_data[vi, vj, vk]

                                # Only include metal voxels
                                if intensity >= self.metal_threshold:
                                    points_voxel.append([vi, vj, vk])
                                    intensities.append(intensity)

        if not points_voxel:
            return np.array([]), np.array([])

        # Remove duplicates
        points_voxel = np.unique(np.array(points_voxel), axis=0)

        # Convert to world coordinates
        points_world = []
        final_intensities = []
        for pv in points_voxel:
            pw = (self.affine @ np.append(pv, 1))[:3]
            points_world.append(pw)
            final_intensities.append(self.ct_data[tuple(pv.astype(int))])

        return np.array(points_world), np.array(final_intensities)

    def _on_fit_polynomial(self, _event) -> None:
        """Fit polynomials to COG trajectories and create electrode models."""
        if not self.cog_trajectories:
            print("No COG trajectories available. Track COG first.")
            return

        # Get polynomial degree
        try:
            poly_degree = int(self.controls["poly_degree"].text)
        except ValueError:
            poly_degree = 3

        print(
            f"Fitting degree-{poly_degree} polynomials to {len(self.cog_trajectories)} trajectories..."
        )

        # Clear existing electrodes
        self.electrodes.clear()

        # Get electrode type setting for the model
        electrode_type_label = self.controls["electrode_type"].value_selected
        electrode_type = (
            None if electrode_type_label == "Auto" else electrode_type_label
        )

        # Process each COG trajectory
        for i, trajectory in enumerate(self.cog_trajectories):
            if not trajectory or len(trajectory) < poly_degree + 1:
                print(
                    f"  Trajectory {i+1}: Insufficient points ({len(trajectory) if trajectory else 0}) for degree-{poly_degree} polynomial"
                )
                continue

            try:
                # Extract points and create intensity array
                points_world = np.array([p.center_of_gravity for p in trajectory])
                intensities = np.array([p.mean_intensity for p in trajectory])

                print(f"\n  Processing trajectory {i+1}: {len(points_world)} points")

                # Fit polynomial directly to COG trajectory points
                trajectory_model = fit_polynomial_to_trajectory(
                    points_world,
                    degree=min(
                        poly_degree, len(points_world) - 1
                    ),  # Ensure degree is valid
                    weights=intensities,
                )

                print(f"    Fitted degree-{trajectory_model.degree} polynomial")
                print(
                    f"    Trajectory length: {trajectory_model.total_length_mm:.1f}mm"
                )

                # Skip refinement and contact detection for debugging
                # Just create a simple electrode model with the polynomial

                # Calculate bounding box
                min_coords = points_world.min(axis=0)
                max_coords = points_world.max(axis=0)
                bounding_box = (min_coords, max_coords)

                # Create simple electrode model without contacts for visualization
                # Use a valid electrode type or default
                electrode_type_to_use = (
                    electrode_type if electrode_type else "Medtronic 3389"
                )

                # Use empty contact positions for now (debugging only)
                electrode = PolynomialElectrodeModel(
                    polynomial=trajectory_model.polynomial,
                    electrode_type=electrode_type_to_use,  # Use valid type
                    contact_positions=np.array([]),  # No contacts for now
                    intensity_profile=intensities,
                    distance_scale=trajectory_model.distance_scale_mm,
                    bounding_box=bounding_box,
                )

                self.electrodes.append(electrode)
                print(f"    Successfully created polynomial fit {i+1}")

            except Exception as e:
                print(f"  Failed to process trajectory {i+1}: {e}")
                import traceback

                traceback.print_exc()

        print(
            f"\nPolynomial fitting complete. Created {len(self.electrodes)} electrode models."
        )

        # Update displays
        self._update_ct_display()
        self._update_3d_display()

    def _is_valid_voxel(self, voxel_coord: Tuple[int, int, int]) -> bool:
        """Check if voxel coordinates are within bounds."""
        if self.ct_data is None:
            return False
        i, j, k = voxel_coord
        return (
            0 <= i < self.ct_data.shape[0]
            and 0 <= j < self.ct_data.shape[1]
            and 0 <= k < self.ct_data.shape[2]
        )

    def run(self) -> List[PolynomialElectrodeModel]:
        """
        Run the interactive electrode detector.

        Returns:
            List of detected electrode models
        """
        plt.show()
        return self.electrodes


def main():
    """Main function for testing the GUI."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pypacer_gui.py <ct_nifti_path>")
        return

    ct_path = sys.argv[1]
    gui = PyPaCERGUI(ct_path)
    electrodes = gui.run()

    print(f"Final results: {len(electrodes)} electrodes detected")
    for i, electrode in enumerate(electrodes):
        print(
            f"  Electrode {i+1}: {electrode.electrode_type}, "
            f"{electrode.length_mm:.1f}mm, {len(electrode.contact_positions)} contacts"
        )


if __name__ == "__main__":
    main()
