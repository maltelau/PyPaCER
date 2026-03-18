"""Main PaCER class for electrode reconstruction."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend

from .._version import __version__ as PYPACER_VERSION
from ..imaging.preprocessing import (
    detect_metal_artifacts,
    extract_brain_mask,
    filter_metal_components,
)
from ..models.electrode import PolynomialElectrodeModel
from ..orientation import (
    classify_electrode_type,
    detect_directional_markers,
    determine_marker_orientation,
    fit_constrained_marker_directions,
)
from ..orientation import angle_to_vector, validate_marker_pair
from ..utils.math_helpers import inv_poly_arc_length_3d
from .contact_detection import detect_contacts
from .electrode_detection import extract_electrode_pointclouds
from .refinement import refine_electrode_trajectory
from .trajectory_fit import fit_initial_trajectory


class PyPaCER:
    """
    Main class for DBS electrode reconstruction from CT images.

    Attributes:
        ct_image: NIfTI CT image object
        brain_mask: Binary brain mask
        metal_threshold: HU threshold for metal detection
        electrodes: List of detected electrode models
    """

    def __init__(
        self,
        ct_path: Union[str, Path],
        brain_mask: Optional[Union[str, Path, np.ndarray]] = None,
        metal_threshold: float = 2000,
        use_gpu: bool = False,
        debug_output_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize PaCER with CT image.

        Args:
            ct_path: Path to CT NIfTI file
            brain_mask: Optional brain mask (auto-detected if None)
            metal_threshold: Hounsfield unit threshold for metal
            use_gpu: Enable GPU acceleration if available
            debug_output_dir: Optional directory for debug outputs (e.g., orthogonal grids)
            output_dir: Optional directory for saving results (JSON, reports)
        """
        self.ct_path = Path(ct_path)
        self.ct_image = nib.load(str(self.ct_path))
        self.ct_data = self.ct_image.get_fdata()
        self.affine = self.ct_image.affine
        self.voxel_sizes = np.array(self.ct_image.header.get_zooms()[:3])

        # Validate slice thickness
        if self.voxel_sizes.max() > 1.0:
            print(
                f"Warning: Slice thickness {self.voxel_sizes.max():.2f}mm > 1mm. "
                "Contact detection may be unreliable."
            )

        self.metal_threshold = metal_threshold
        self.brain_mask = brain_mask
        self.use_gpu = use_gpu
        self.debug_output_dir = str(debug_output_dir) if debug_output_dir else None
        self.output_dir = str(output_dir) if output_dir else None
        self.electrodes: List[PolynomialElectrodeModel] = []

        # Clean stale debug files from previous runs
        if self.debug_output_dir:
            debug_subdir = Path(self.debug_output_dir) / "debug"
            if debug_subdir.exists():
                import shutil

                shutil.rmtree(debug_subdir)

    def detect_electrodes(
        self,
        contact_detection_method: str = "contactAreaCenter",
        electrode_type: Optional[str] = None,
        final_degree: int = 3,
        display_profiles: bool = False,
        xy_resolution: float = 0.1,
        z_resolution: float = 0.025,
        grid_size: float = 1.5,
        auto_save_json: bool = True,
        min_electrode_length_mm: float = 40.0,
        refinement_threshold: Optional[
            float
        ] = 800,  # Lower threshold for refinement, None to disable
        detection_method: str = "radial_search",  # New parameter
        search_radii_mm: List[float] = None,
        orientation_params: Optional[Dict[str, Any]] = None,
    ) -> List[PolynomialElectrodeModel]:
        """
        Run complete electrode detection pipeline.

        Args:
            contact_detection_method: Method for contact detection
                ('peak', 'peakWaveCenter', 'contactAreaCenter')
            electrode_type: Force specific electrode type or auto-detect
            final_degree: Polynomial degree for final model
            display_profiles: Show intensity profiles during processing
            xy_resolution: Resolution for orthogonal grid sampling in mm (default 0.1)
            z_resolution: Resolution along trajectory in mm (default 0.025)
            grid_size: Size of orthogonal sampling grid in mm (default 1.5)
            auto_save_json: Automatically save reconstruction results to JSON in CT directory (default True)
            min_electrode_length_mm: Minimum electrode length in mm to keep (default 40.0)
            refinement_threshold: HU threshold for refinement (default 800)
            detection_method: Method for finding electrodes - 'radial_search' (default), 'brain_mask_auto', or 'brain_mask_custom'
            search_radii_mm: Radii for radial search method (default: [30, 40, 50] mm)
            orientation_params: Optional dict to override orientation detection
                defaults. See _run_orientation_detection for supported keys.

        Returns:
            List of detected electrode models
        """
        print(f"Processing {self.ct_path.name}")
        print(f"Voxel size: {self.voxel_sizes}")
        print(f"Detection method: {detection_method}")

        # Handle different detection methods
        if detection_method == "radial_search":
            # For radial search, redirect to detect_electrodes_auto which uses the same approach
            # but doesn't accept custom resolution parameters
            return self.detect_electrodes_auto(
                contact_detection_method=contact_detection_method,
                electrode_type=electrode_type,
                auto_save_json=auto_save_json,
                search_radii_mm=search_radii_mm,
                max_electrodes=2,
                verbose=True,
                orientation_params=orientation_params,
            )

        elif detection_method == "brain_mask_auto":
            # Step 1: Extract brain mask automatically
            print("Extracting brain mask...")
            self.brain_mask = extract_brain_mask(self.ct_data, self.voxel_sizes)
            print(
                f"Brain mask shape: {self.brain_mask.shape}, covering {np.sum(self.brain_mask)} voxels"
            )

        elif detection_method == "brain_mask_custom":
            # Step 1: Use provided brain mask
            if self.brain_mask is None:
                raise ValueError(
                    "Custom brain mask method requires brain_mask to be provided during initialization"
                )

            if isinstance(self.brain_mask, (str, Path)):
                mask_path = str(self.brain_mask)
                mask_img = nib.load(mask_path)
                self.brain_mask = mask_img.get_fdata().astype(bool)
                print(f"Loaded brain mask from {mask_path}")

            # Validate brain mask
            if not isinstance(self.brain_mask, np.ndarray):
                self.brain_mask = np.array(self.brain_mask)

            print(
                f"Using custom brain mask: {self.brain_mask.shape}, covering {np.sum(self.brain_mask)} voxels"
            )

        else:
            raise ValueError(
                f"Unknown detection method: {detection_method}. Use 'radial_search', 'brain_mask_auto', or 'brain_mask_custom'"
            )

        # For brain mask methods, continue with original pipeline
        # Step 2: Detect metal artifacts
        print(f"Detecting metal artifacts (threshold={self.metal_threshold} HU)...")
        metal_mask = detect_metal_artifacts(
            self.ct_data, self.brain_mask, self.metal_threshold
        )

        # Step 3: Filter metal components
        print("Filtering metal components...")
        labeled_metal, num_electrodes = filter_metal_components(
            metal_mask, self.ct_data, self.voxel_sizes
        )

        # Step 4: Extract electrode point clouds
        print("Extracting electrode point clouds...")
        point_clouds = extract_electrode_pointclouds(
            labeled_metal,
            self.ct_data,
            self.voxel_sizes,
            self.affine,
            min_length_mm=min_electrode_length_mm,
        )

        print(f"Found {len(point_clouds)} potential electrodes")

        # Step 5: Process each electrode
        print("Processing electrodes...")
        self.electrodes = []
        for i, pc in enumerate(point_clouds):
            print(f"\n--- Processing electrode {i+1}/{len(point_clouds)} ---")

            # Initial trajectory fit (high degree)
            initial_trajectory = fit_initial_trajectory(
                pc.points_world, pc.intensities, degree=8
            )

            # Trajectory refinement
            refined_model = refine_electrode_trajectory(
                initial_trajectory,
                pc.points_world,
                pc.intensities,
                self.ct_data,
                self.affine,
                final_degree=final_degree,
                xy_resolution=xy_resolution,
                z_resolution=z_resolution,
                grid_size=grid_size,
                electrode_idx=i if self.debug_output_dir else None,
                debug_output_dir=self.debug_output_dir,
                refinement_threshold=refinement_threshold,
            )

            # Contact detection
            contacts = detect_contacts(
                refined_model,
                method=contact_detection_method,
                electrode_type=electrode_type,
                display_profile=display_profiles,
                run_all_methods=(self.debug_output_dir is not None),
            )

            # Create final electrode model
            # Get electrode type from contact detection or use default
            detected_electrode_type = getattr(refined_model, "electrode_type", None)
            if detected_electrode_type is None:
                detected_electrode_type = (
                    electrode_type or "Medtronic 3389/B33005"
                )  # Default to 0.5mm spacing types

            # Calculate bounding box from point cloud
            min_coords = pc.points_world.min(axis=0)
            max_coords = pc.points_world.max(axis=0)
            bounding_box = (min_coords, max_coords)

            electrode = PolynomialElectrodeModel(
                polynomial=refined_model.polynomial,
                electrode_type=detected_electrode_type,
                contact_positions=contacts,
                intensity_profile=refined_model.intensity_profile,
                distance_scale=refined_model.distance_scale_mm,
                bounding_box=bounding_box,
                skeleton_deviations_mm=(
                    refined_model.skeleton_deviations_mm
                    if hasattr(refined_model, "skeleton_deviations_mm")
                    else None
                ),
                polynomial_before_tip_detection=(
                    refined_model.polynomial_before_tip_detection
                    if hasattr(refined_model, "polynomial_before_tip_detection")
                    else None
                ),
            )

            # Store contact detection results if available (debug mode)
            if hasattr(refined_model, "contact_detection_results"):
                electrode.contact_detection_results = (
                    refined_model.contact_detection_results
                )

            # Store full Pass 2 debug data if available
            if hasattr(refined_model, "pass2_intensities_full"):
                electrode.pass2_intensities_full = refined_model.pass2_intensities_full
            if hasattr(refined_model, "pass2_distances_mm_full"):
                electrode.pass2_distances_mm_full = (
                    refined_model.pass2_distances_mm_full
                )
            if hasattr(refined_model, "pass2_tip_threshold"):
                electrode.pass2_tip_threshold = refined_model.pass2_tip_threshold
            if hasattr(refined_model, "original_t0_distance_mm"):
                electrode.original_t0_distance_mm = (
                    refined_model.original_t0_distance_mm
                )

            # Orientation detection and electrode type classification
            orientation_data, classified_type = self._run_orientation_detection(
                electrode,
                electrode_idx=i,
                orientation_params=orientation_params,
            )
            if orientation_data is not None:
                electrode.orientation_data = orientation_data
            electrode.electrode_type = classified_type

            # Hemisphere detection for tip and entry positions
            electrode.tip_hemisphere = self._determine_hemisphere(
                electrode.tip_position
            )
            electrode.entry_hemisphere = self._determine_hemisphere(
                electrode.entry_position
            )

            self.electrodes.append(electrode)

            # Save combined intensity profile plot if debug output is enabled
            if self.debug_output_dir:
                self._save_combined_intensity_plot(
                    refined_model, electrode, i, self.debug_output_dir
                )

            print(f"  Electrode {i+1} successfully added to results")

        print(f"\nAll electrodes processed. Total: {len(self.electrodes)}")

        # Auto-save results if requested
        if auto_save_json and self.electrodes:
            print("Saving reconstruction results...")
            # Collect all parameters used for reconstruction
            parameters = {
                "contact_detection_method": contact_detection_method,
                "electrode_type": electrode_type,
                "final_degree": final_degree,
                "xy_resolution": xy_resolution,
                "z_resolution": z_resolution,
                "grid_size": grid_size,
                "display_profiles": display_profiles,
                "use_gpu": self.use_gpu,
                "debug_output_enabled": self.debug_output_dir is not None,
            }

            # Save reconstruction results
            json_path = self._save_reconstruction_json(parameters)
            print("Reconstruction results saved successfully")

            # Save intensity profile plots
            for i, electrode in enumerate(self.electrodes):
                if (
                    electrode.intensity_profile is not None
                    and electrode.distance_scale is not None
                ):
                    self._save_intensity_profile_plot(
                        electrode.intensity_profile,
                        electrode.distance_scale,
                        i,
                        output_dir,
                        electrode_model=electrode,
                    )

        print(
            f"detect_electrodes completed. Returning {len(self.electrodes)} electrodes"
        )
        return self.electrodes

    def visualize(self, electrode_idx: Optional[int] = None):
        """Visualize detected electrodes in 3D."""
        if not self.electrodes:
            print("No electrodes detected yet. Run detect_electrodes() first.")
            return

        from ..visualization.electrode_renderer import visualize_electrodes

        if electrode_idx is not None:
            visualize_electrodes(
                [self.electrodes[electrode_idx]], self.ct_data, self.affine
            )
        else:
            visualize_electrodes(self.electrodes, self.ct_data, self.affine)

    def export_results(self, output_path: Union[str, Path], format: str = "json"):
        """
        Export electrode models to file.

        Args:
            output_path: Output file path
            format: Export format ('json', 'hdf5', 'mat')
        """
        output_path = Path(output_path)

        if format == "json":
            data = {
                "electrodes": [e.to_dict() for e in self.electrodes],
                "ct_file": str(self.ct_path.resolve()),
                "voxel_sizes": self.voxel_sizes.tolist(),
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "hdf5":
            import h5py

            with h5py.File(output_path, "w") as f:
                for i, electrode in enumerate(self.electrodes):
                    grp = f.create_group(f"electrode_{i}")
                    electrode.to_hdf5(grp)

        elif format == "mat":
            from scipy.io import savemat

            data = {
                "electrodes": [e.to_matlab_struct() for e in self.electrodes],
                "ct_file": str(self.ct_path.resolve()),
                "voxel_sizes": self.voxel_sizes,
            }
            savemat(output_path, data)

        print(f"Results exported to {output_path}")

    def _run_orientation_detection(
        self,
        electrode: PolynomialElectrodeModel,
        electrode_idx: int = None,
        orientation_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Run orientation detection on an electrode after contact detection.

        Detects directional markers, classifies electrode type, and if markers
        are present, determines their orientation angles.

        Args:
            electrode: Electrode model with contact_positions, distance_scale,
                intensity_profile, and skeleton_deviations_mm populated.
            electrode_idx: Electrode index for debug output file naming.
            orientation_params: Optional dict of parameters to override defaults
                for the orientation detection pipeline. Supported keys:

                Marker detection (detect_directional_markers):
                    marker_offset_mm (float): Distance above last contact where
                        marker search begins. Default: 2.5
                    max_distance_mm (float): Maximum distance from tip for
                        marker search region. Default: 20.0
                    deviation_threshold (float): Minimum skeleton deviation for
                        peak detection. Default: 0.08
                    min_peak_distance_mm (float): Minimum distance between
                        detected peaks. Default: 2.0
                    expected_num_peaks (int): Expected number of marker peaks.
                        Default: 2

                Orientation analysis (determine_marker_orientation):
                    radii_mm (list[float]): Circular sampling radii around
                        trajectory. Default: [1.25, 1.5, 1.75]
                    angle_increment_deg (float): Angular step for intensity
                        sampling. Default: 0.1
                    smoothing_window (int): Window size for profile smoothing.
                        Default: 5
                    check_for_bias (bool): Check for bias from opposite marker.
                        Default: True
                    bias_opposite_peak_threshold (float): Threshold for bias
                        detection. Default: 0.7

                Marker pair validation (validate_marker_pair):
                    min_separation_deg (float): Minimum valid angular separation
                        between markers. Default: 120.0
                    max_separation_deg (float): Maximum valid angular separation
                        between markers. Default: 150.0

                Constrained fitting (fit_constrained_marker_directions):
                    angular_constraint_deg (float): Fixed angular separation
                        for fitted marker directions. Default: 120.0

        Returns:
            Tuple of (orientation_data_dict or None, classified_electrode_type).
            On failure, returns (None, electrode.electrode_type).
        """
        if orientation_params is None:
            orientation_params = {}
        original_type = electrode.electrode_type

        try:
            # Need distance_scale, intensity_profile, and contact_positions
            if (
                electrode.distance_scale is None
                or electrode.intensity_profile is None
                or electrode.contact_positions is None
                or len(electrode.contact_positions) == 0
            ):
                print("  Orientation detection skipped: missing required data")
                return None, original_type

            # Step 1: Detect directional markers
            marker_detection_kwargs = {}
            for key in (
                "marker_offset_mm",
                "max_distance_mm",
                "deviation_threshold",
                "min_peak_distance_mm",
                "expected_num_peaks",
            ):
                if key in orientation_params:
                    marker_detection_kwargs[key] = orientation_params[key]

            marker_result = detect_directional_markers(
                distance_scale=electrode.distance_scale,
                intensity_profile=electrode.intensity_profile,
                skeleton_deviations=electrode.skeleton_deviations_mm,
                contact_positions=electrode.contact_positions.tolist(),
                **marker_detection_kwargs,
            )
            print(
                f"  Marker detection: {len(marker_result.marker_peak_locations)} peak(s) found "
                f"via {marker_result.detection_method} (confidence={marker_result.confidence:.2f})"
            )

            # Step 2: Classify electrode type based on contact spacing + markers
            classified_type = classify_electrode_type(
                contact_positions=electrode.contact_positions.tolist(),
                marker_detection_result=marker_result,
            )

            # Build orientation data dict (cast numpy types to Python natives for JSON)
            orientation_data = {
                "has_markers": bool(marker_result.has_markers),
                "marker_detection_confidence": float(marker_result.confidence),
                "detection_method": str(marker_result.detection_method),
                "classified_electrode_type": str(classified_type),
            }

            # Step 3: If markers detected, determine their orientation
            if marker_result.has_markers and len(marker_result.marker_peak_locations) > 0:
                markers_info = {}
                marker_orientations = []

                # Sort peaks: B is closer to contacts (lower distance), A is farther
                sorted_peaks = sorted(marker_result.marker_peak_locations)
                labels = ["B", "A"] if len(sorted_peaks) >= 2 else ["B"]

                for idx, (label, peak_mm) in enumerate(
                    zip(labels, sorted_peaks[: len(labels)])
                ):
                    # Get trajectory direction at marker location
                    t_marker = inv_poly_arc_length_3d(electrode.polynomial, peak_mm)
                    direction = electrode.get_tangent_at(t_marker)

                    # Determine marker orientation
                    orientation_kwargs = {}
                    for key in (
                        "radii_mm",
                        "angle_increment_deg",
                        "smoothing_window",
                        "check_for_bias",
                        "bias_opposite_peak_threshold",
                    ):
                        if key in orientation_params:
                            orientation_kwargs[key] = orientation_params[key]

                    orient_result = determine_marker_orientation(
                        ct_data=self.ct_data,
                        affine=self.affine,
                        electrode_polynomial=electrode.polynomial,
                        marker_location_mm=peak_mm,
                        trajectory_direction=direction,
                        **orientation_kwargs,
                    )
                    marker_orientations.append(orient_result)
                    print(
                        f"    Marker {label} at {peak_mm:.1f}mm: "
                        f"angle={orient_result.peak_angle_deg:.1f}\u00b0, "
                        f"confidence={orient_result.confidence:.2f}"
                        f"{', bias detected' if orient_result.analysis_metadata.get('bias_detected') else ''}"
                    )

                    # Compute 3D position at marker location
                    marker_position = electrode.get_point_at_parameter(t_marker)

                    markers_info[label] = {
                        "distance_from_tip_mm": float(peak_mm),
                        "detected_angle_traj_perp_deg": float(orient_result.peak_angle_deg),
                        "detection_confidence": float(orient_result.confidence),
                        "position_xyz": marker_position.tolist(),
                        "direction_vector": orient_result.orientation_vector_world.tolist(),
                        # Intensity profile for orientation tab charts
                        "intensity_profile": {
                            "angle_step_deg": float(orient_result.sampling_result.angles_deg[1] - orient_result.sampling_result.angles_deg[0]),
                            "mean_intensity": orient_result.sampling_result.mean_intensity_by_angle.tolist(),
                        },
                    }

                # If two markers, validate pair and fit constrained directions
                fitted_angles_debug = None
                if len(marker_orientations) == 2:
                    validate_kwargs = {}
                    for key in ("min_separation_deg", "max_separation_deg"):
                        if key in orientation_params:
                            validate_kwargs[key] = orientation_params[key]

                    is_valid, angular_sep = validate_marker_pair(
                        marker_orientations[0],
                        marker_orientations[1],
                        **validate_kwargs,
                    )
                    orientation_data["marker_pair_valid"] = bool(is_valid)
                    orientation_data["marker_pair_angular_separation_deg"] = float(
                        angular_sep
                    )
                    valid_str = "valid" if is_valid else "INVALID"
                    print(f"    Pair separation: {angular_sep:.1f}\u00b0 ({valid_str})")

                    # Fit constrained directions (120 deg separation)
                    fit_kwargs = {}
                    if "angular_constraint_deg" in orientation_params:
                        fit_kwargs["angular_constraint_deg"] = orientation_params[
                            "angular_constraint_deg"
                        ]

                    fitted_b, fitted_a = fit_constrained_marker_directions(
                        marker_orientations[0],
                        marker_orientations[1],
                        **fit_kwargs,
                    )
                    markers_info["B"]["fitted_angle_traj_perp_deg"] = float(fitted_b)
                    markers_info["A"]["fitted_angle_traj_perp_deg"] = float(fitted_a)
                    fitted_angles_debug = [fitted_b, fitted_a]
                    print(
                        f"    Constrained fit (120\u00b0): B={fitted_b:.1f}\u00b0, A={fitted_a:.1f}\u00b0"
                    )

                    # Update direction vectors to use fitted angles
                    for label, fitted_angle, orient_result in zip(
                        labels, [fitted_b, fitted_a], marker_orientations
                    ):
                        fitted_vec = angle_to_vector(
                            fitted_angle,
                            orient_result.sampling_result.normal_vector,
                            self.affine,
                        )
                        markers_info[label]["direction_vector"] = fitted_vec.tolist()

                # Save combined marker orientation debug visualization
                if self.debug_output_dir and electrode_idx is not None:
                    from pathlib import Path

                    from ..orientation.visualization import (
                        visualize_marker_orientations,
                    )

                    debug_dir = Path(self.debug_output_dir) / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)

                    # Collect positions and directions for all markers
                    marker_positions_debug = []
                    marker_directions_debug = []
                    for peak_mm_d in sorted_peaks[: len(labels)]:
                        t_d = inv_poly_arc_length_3d(electrode.polynomial, peak_mm_d)
                        marker_positions_debug.append(
                            electrode.get_point_at_parameter(t_d)
                        )
                        marker_directions_debug.append(
                            electrode.get_tangent_at(t_d)
                        )

                    visualize_marker_orientations(
                        ct_data=self.ct_data,
                        affine=self.affine,
                        marker_positions=marker_positions_debug,
                        marker_directions=marker_directions_debug,
                        orientation_results=marker_orientations,
                        labels=labels,
                        electrode_idx=electrode_idx,
                        fitted_angles=fitted_angles_debug,
                        output_path=debug_dir
                        / f"electrode_{electrode_idx}_marker_orientations.png",
                    )

                orientation_data["markers"] = markers_info
                print(
                    f"  Orientation: markers detected (confidence={marker_result.confidence:.2f}), "
                    f"classified as {classified_type}"
                )

            else:
                print(f"  Orientation: no markers detected, classified as {classified_type}")

            return orientation_data, classified_type

        except Exception as e:
            print(f"  Warning: Orientation detection failed: {e}")
            if self.debug_output_dir:
                import traceback

                traceback.print_exc()
            return None, original_type

    def _determine_hemisphere(self, point_world: np.ndarray) -> str:
        """
        Determine which hemisphere a world-coordinate point is in.

        NIfTI world coordinates follow the RAS convention where the X axis
        is the Left-Right axis with positive values pointing Right.

        Args:
            point_world: 3D point in world coordinates.

        Returns:
            "left" or "right"
        """
        return "right" if point_world[0] >= 0 else "left"

    def _save_intensity_profile_plot(
        self,
        intensity_profile: np.ndarray,
        distance_scale: np.ndarray,
        electrode_idx: int,
        output_dir: Path,
        electrode_model: Optional["PolynomialElectrodeModel"] = None,
    ) -> Path:
        """
        Save intensity profile plots as PNG.

        Args:
            intensity_profile: Intensity values along electrode
            distance_scale: Distance from tip in mm
            electrode_idx: Electrode index
            output_dir: Directory to save plot
            electrode_model: Optional electrode model with contact positions

        Returns:
            Path to saved plot file
        """
        # This plot has been removed - returning None
        return None

    def _save_combined_intensity_plot(
        self,
        refined_model,
        electrode: PolynomialElectrodeModel,
        electrode_idx: int,
        output_dir: str,
    ):
        """Save combined pass 2 and pass 3 intensity profile plot with contact markers using Plotly."""
        try:
            from pathlib import Path

            import numpy as np
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create subplots with custom spacing
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    "<b>Pass 2: Full Trajectory Intensity Profile</b>",
                    "<b>Pass 3: Contact Region Detail with Detected Contacts</b>",
                ),
                vertical_spacing=0.12,
                row_heights=[0.5, 0.5],
            )

            # Color scheme
            primary_color = "#1f77b4"
            contact_color = "#d62728"
            tip_color = "#2ca02c"
            threshold_color = "#ff7f0e"

            # Top plot: Pass 2 intensity profile (full trajectory) - only available in GPU version
            if (
                hasattr(refined_model, "pass2_intensities")
                and refined_model.pass2_intensities is not None
            ):
                # Add intensity trace
                fig.add_trace(
                    go.Scatter(
                        x=refined_model.pass2_distances_mm,
                        y=refined_model.pass2_intensities,
                        mode="lines",
                        name="Intensity Profile",
                        line=dict(color=primary_color, width=2),
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

                # Add detected tip marker
                fig.add_vline(
                    x=0,
                    line=dict(color=tip_color, width=2, dash="solid"),
                    annotation_text="Detected Tip",
                    annotation_position="top right",
                    row=1,
                    col=1,
                )

                # Add tip threshold line
                if refined_model.pass2_tip_threshold is not None:
                    fig.add_hline(
                        y=refined_model.pass2_tip_threshold,
                        line=dict(color=threshold_color, width=2, dash="dash"),
                        annotation_text=f"Tip Threshold ({refined_model.pass2_tip_threshold:.0f} HU)",
                        annotation_position="top right",
                        row=1,
                        col=1,
                    )

            # Bottom plot: Contact region detail
            if (
                electrode.contact_positions is not None
                and len(electrode.contact_positions) > 0
            ):
                min_contact = min(electrode.contact_positions)
                max_contact = max(electrode.contact_positions)
                zoom_min = max(0, min_contact - 5)
                zoom_max = min(refined_model.distance_scale_mm.max(), max_contact + 5)
            else:
                zoom_min = 0
                zoom_max = min(20, refined_model.distance_scale_mm.max())

            # Find indices for zoom region
            zoom_mask = (refined_model.distance_scale_mm >= zoom_min) & (
                refined_model.distance_scale_mm <= zoom_max
            )
            zoom_distance = refined_model.distance_scale_mm[zoom_mask]
            zoom_intensity = refined_model.intensity_profile[zoom_mask]

            if len(zoom_distance) > 0:
                # Add intensity trace for contact region
                fig.add_trace(
                    go.Scatter(
                        x=zoom_distance,
                        y=zoom_intensity,
                        mode="lines",
                        name="Contact Region Intensity",
                        line=dict(color=primary_color, width=3),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

                # Add contact markers
                if electrode.contact_positions is not None:
                    contact_x = []
                    contact_y = []
                    contact_labels = []

                    for i, contact_pos in enumerate(electrode.contact_positions):
                        if zoom_min <= contact_pos <= zoom_max:
                            contact_intensity = np.interp(
                                contact_pos, zoom_distance, zoom_intensity
                            )
                            contact_x.append(contact_pos)
                            contact_y.append(contact_intensity)
                            contact_labels.append(f"C{i+1}")

                            # Add vertical line for each contact
                            fig.add_vline(
                                x=contact_pos,
                                line=dict(color=contact_color, width=1.5, dash="dash"),
                                row=2,
                                col=1,
                            )

                    # Add contact markers
                    if contact_x:
                        fig.add_trace(
                            go.Scatter(
                                x=contact_x,
                                y=contact_y,
                                mode="markers+text",
                                name="Detected Contacts",
                                marker=dict(
                                    color=contact_color,
                                    size=12,
                                    symbol="circle",
                                    line=dict(color="white", width=2),
                                ),
                                text=contact_labels,
                                textposition="top center",
                                textfont=dict(size=12, color=contact_color),
                                showlegend=True,
                            ),
                            row=2,
                            col=1,
                        )

                # Set y-axis range with padding
                y_margin = (zoom_intensity.max() - zoom_intensity.min()) * 0.15
                y_min = zoom_intensity.min() - y_margin
                y_max = zoom_intensity.max() + y_margin + 100

                fig.update_yaxes(range=[y_min, y_max], row=2, col=1)
                fig.update_xaxes(range=[zoom_min, zoom_max], row=2, col=1)

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"<b>Electrode {electrode_idx + 1} - Intensity Profile Analysis</b><br>"
                    + f'<sub>Type: {electrode.electrode_type if electrode.electrode_type else "Unknown"}</sub>',
                    font=dict(size=18),
                    x=0.5,
                    xanchor="center",
                ),
                height=800,
                width=1000,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                template="plotly_white",
                hovermode="x unified",
            )

            # Update axes labels
            fig.update_xaxes(
                title_text="Distance from tip (mm)",
                title_font=dict(size=14),
                row=2,
                col=1,
            )
            fig.update_xaxes(
                title_text="Distance from tip (mm)",
                title_font=dict(size=14),
                row=1,
                col=1,
            )
            fig.update_yaxes(
                title_text="Intensity (HU)", title_font=dict(size=14), row=1, col=1
            )
            fig.update_yaxes(
                title_text="Intensity (HU)", title_font=dict(size=14), row=2, col=1
            )

            # Add grid
            fig.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
            )
            fig.update_yaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
            )

            # No longer saving individual files - plots are embedded in the main HTML report

        except Exception as e:
            print(f"  Warning: Failed to save combined intensity profile plot - {e}")

    def _save_reconstruction_json(
        self, parameters: Dict[str, Any], output_filename: Optional[str] = None
    ) -> Path:
        """
        Save reconstruction results with parameters to JSON file.

        Args:
            parameters: Dictionary of reconstruction parameters used
            output_filename: Optional custom filename (auto-generated if None)

        Returns:
            Path to saved JSON file
        """
        # Create output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"pypacer_{timestamp}.json"

        # Determine output directory
        if self.output_dir:
            output_dir = Path(self.output_dir)
        elif self.debug_output_dir:
            output_dir = Path(self.debug_output_dir)
        else:
            output_dir = self.ct_path.parent

        # Create full path
        output_path = output_dir / output_filename

        # Ensure it has .json extension
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")

        # Calculate CT volume bounding box in world coordinates
        ct_shape = self.ct_data.shape
        # Get corners of the CT volume in voxel space
        voxel_min = np.array([0, 0, 0, 1])
        voxel_max = np.array([ct_shape[0] - 1, ct_shape[1] - 1, ct_shape[2] - 1, 1])
        # Transform to world coordinates
        world_min = (self.affine @ voxel_min)[:3]
        world_max = (self.affine @ voxel_max)[:3]
        # Ensure min/max are correct (affine might flip axes)
        ct_bbox_min = np.minimum(world_min, world_max)
        ct_bbox_max = np.maximum(world_min, world_max)

        # Prepare comprehensive data
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
            "electrodes": [],
        }

        # Add electrode data with additional details
        for i, electrode in enumerate(self.electrodes):
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

            data["electrodes"].append(electrode_data)

        # Save to file
        with open(output_path, "w") as f:
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
                k: v for k, v in electrode_data.items() if k not in profile_keys
            }
            # Strip marker intensity profiles from orientation data
            if "orientation" in mini_electrode:
                orient = json.loads(json.dumps(mini_electrode["orientation"]))
                for marker in orient.get("markers", {}).values():
                    marker.pop("intensity_profile", None)
                if "contact_intensity_profile" in orient:
                    orient["contact_intensity_profile"].pop("intensity", None)
                mini_electrode["orientation"] = orient
            mini_data["electrodes"].append(mini_electrode)

        mini_path = output_path.with_stem(output_path.stem + "_mini")
        with open(mini_path, "w") as f:
            json.dump(mini_data, f, indent=2)

        print(f"Reconstruction results saved to: {output_path}")
        print(f"Minified results saved to: {mini_path}")
        return output_path

    def detect_electrodes_auto(
        self,
        contact_detection_method: str = "contactAreaCenter",
        electrode_type: Optional[str] = None,
        auto_save_json: bool = True,
        search_radii_mm: List[float] = None,
        max_electrodes: int = 2,
        verbose: bool = True,
        orientation_params: Optional[Dict[str, Any]] = None,
    ) -> List[PolynomialElectrodeModel]:
        """
        Detect electrodes using the GUI auto run pipeline.

        This method mirrors the exact workflow from the GUI's auto run button:
        1. Find electrode seeds using radial search from brain center
        2. Track COG trajectories from each seed point
        3. Fit initial polynomials (degree 8) to trajectories
        4. Run OOR refinement with automatic degree selection
        5. Detect contacts

        Args:
            contact_detection_method: Method for contact detection
                ('peak', 'peakWaveCenter', 'contactAreaCenter')
            electrode_type: Force specific electrode type or auto-detect
            auto_save_json: Automatically save reconstruction results
            search_radii_mm: Radii to search for electrodes (default: [30, 40, 50])
            max_electrodes: Expected maximum number of electrodes (default: 2)
            verbose: Print progress messages
            orientation_params: Optional dict to override orientation detection
                defaults. See _run_orientation_detection for supported keys.

        Returns:
            List of detected electrode models
        """
        if search_radii_mm is None:
            # Use more conservative default radii (30-50mm typical for DBS electrodes)
            search_radii_mm = [30, 40, 50]

        if verbose:
            print("\n=== Starting automatic electrode detection ===")

        # Import required modules

        from ..models.electrode import PolynomialElectrodeModel
        from .cog_trajectory_tracking import CenterOfGravityTracker
        from .contact_detection import detect_contacts
        from .polynomial_fitting import fit_polynomial_to_trajectory
        from .refinement import refine_electrode_trajectory

        # Step 1: Detect metal artifacts
        if verbose:
            print(f"Detecting metal artifacts (threshold={self.metal_threshold} HU)...")
        metal_mask = self.ct_data > self.metal_threshold

        # Step 2: Find electrode seeds using radial search
        if verbose:
            print("Finding electrode seeds using radial search...")
        seed_points = self._find_electrode_seeds_radial(
            metal_mask, search_radii_mm, max_electrodes, verbose
        )

        if not seed_points:
            if verbose:
                print("No electrode artifacts found in search regions")
            return []

        if verbose:
            print(f"Found {len(seed_points)} potential electrode locations")

        # Step 3: Initialize COG tracker
        cog_tracker = CenterOfGravityTracker(
            ct_data=self.ct_data,
            affine=self.affine,
            metal_threshold=self.metal_threshold,
            search_radius_mm=5.0,  # Use consistent 5mm physical radius
            max_direction_change_deg=60.0,  # Match GUI threshold
            min_voxels_per_slice=3,
        )

        # Step 4: Track COG trajectories and fit polynomials
        self.electrodes = []
        cog_trajectories = []

        for i, seed in enumerate(seed_points):
            if verbose:
                print(
                    f"\nProcessing electrode {i+1}/{len(seed_points)} at seed {seed}..."
                )

            try:
                # Track trajectory from seed
                trajectory_points = cog_tracker.track_from_seed(
                    seed_voxel=seed, slice_axis="axial"  # Default to axial
                )

                if not trajectory_points:
                    if verbose:
                        print(f"  No trajectory found from seed {seed}")
                    continue

                # Check for skull exit using multi-scale detection
                exit_idx = self._detect_skull_exit_multiscale(
                    trajectory_points, angle_threshold_deg=25.0
                )
                if exit_idx is not None:
                    if verbose:
                        print(
                            f"  Truncating trajectory at skull exit (point {exit_idx} of {len(trajectory_points)})"
                        )
                    trajectory_points = trajectory_points[:exit_idx]

                cog_trajectories.append(trajectory_points)

                if verbose:
                    print(f"  Tracked {len(trajectory_points)} points")

                # Convert COG trajectory to world coordinates
                points_world = np.array(
                    [p.center_of_gravity for p in trajectory_points]
                )
                intensities = np.array([p.mean_intensity for p in trajectory_points])

                # Calculate trajectory length
                if len(points_world) > 1:
                    diffs = np.diff(points_world, axis=0)
                    distances = np.linalg.norm(diffs, axis=1)
                    total_length = np.sum(distances)
                    if verbose:
                        print(f"  Total length: {total_length:.1f}mm")

                    # Skip if too short
                    if total_length < 40.0:  # Minimum electrode length
                        if verbose:
                            print(
                                f"  Skipping: trajectory too short ({total_length:.1f}mm < 40mm)"
                            )
                        continue

                # Step 5: Fit initial polynomial (degree 8)
                if verbose:
                    print("  Fitting initial polynomial (degree 8)...")

                poly_result = fit_polynomial_to_trajectory(
                    points_world, degree=8, weights=intensities
                )

                if not poly_result:
                    if verbose:
                        print("  Failed to fit polynomial")
                    continue

                # Calculate bounding box
                min_coords = points_world.min(axis=0)
                max_coords = points_world.max(axis=0)
                bounding_box = (min_coords, max_coords)

                # Create initial electrode model
                initial_electrode = PolynomialElectrodeModel(
                    polynomial=poly_result.polynomial,
                    electrode_type=electrode_type or "Medtronic 3389/B33005",
                    contact_positions=np.array([]),
                    intensity_profile=intensities,
                    distance_scale=poly_result.distance_scale_mm,
                    bounding_box=bounding_box,
                )
                # Store the distance scale as an attribute for compatibility
                initial_electrode.distance_scale_mm = poly_result.distance_scale_mm
                initial_electrode.total_length_mm = poly_result.total_length_mm

                # Step 6: Run OOR refinement with automatic degree selection
                if verbose:
                    print("  Running OOR refinement with automatic degree selection...")

                try:
                    # Check if GPU should be used
                    if self.use_gpu:
                        import torch

                        if not torch.cuda.is_available():
                            raise RuntimeError(
                                "GPU processing requested (use_gpu=True) but no CUDA GPU available. "
                                "Set use_gpu=False to use CPU instead."
                            )
                        from ..gpu.refinement_gpu import refine_electrode_trajectory_gpu

                        refined_model = refine_electrode_trajectory_gpu(
                            initial_electrode,
                            points_world,
                            intensities,
                            self.ct_data,
                            self.affine,
                            final_degree=3,  # Will be overridden by auto_select_degree
                            xy_resolution=0.2,  # Match GUI settings
                            z_resolution=0.025,
                            grid_size=2.0,  # Match GUI settings
                            electrode_idx=i if self.debug_output_dir else None,
                            debug_output_dir=self.debug_output_dir,
                            refinement_threshold=800,
                            refinement_radius_mm=3.5,
                            use_subvolume=True,  # Use subvolume for faster processing
                            auto_select_degree=True,  # Auto-select best degree based on bottom 20mm
                            contact_region_mm=20.0,  # Evaluate fit quality in bottom 20mm
                        )
                        if verbose:
                            print("    GPU refinement completed")
                    else:
                        # CPU refinement with automatic degree selection
                        refined_model = refine_electrode_trajectory(
                            initial_electrode,
                            points_world,
                            intensities,
                            self.ct_data,
                            self.affine,
                            final_degree=3,  # Will be overridden by auto_select_degree
                            xy_resolution=0.2,  # Match GUI settings
                            z_resolution=0.025,
                            grid_size=2.0,  # Match GUI settings
                            electrode_idx=i if self.debug_output_dir else None,
                            debug_output_dir=self.debug_output_dir,
                            refinement_threshold=800,
                            refinement_radius_mm=3.5,
                            auto_select_degree=True,  # Auto-select best degree based on bottom 20mm
                            contact_region_mm=20.0,  # Evaluate fit quality in bottom 20mm
                        )
                        if verbose:
                            print("    CPU refinement completed")

                    # Get the refined degree if available
                    if hasattr(refined_model, "polynomial_degree"):
                        if verbose:
                            print(
                                f"    Auto-selected polynomial degree: {refined_model.polynomial_degree}"
                            )

                except Exception as e:
                    # Critical failure - stop processing this electrode
                    print(
                        f"\n    *** ERROR: OOR refinement failed for electrode {i+1} ***"
                    )
                    print(f"    Error: {e}")
                    print(
                        "    This electrode will be skipped - refinement is required for accurate results"
                    )
                    if verbose:
                        import traceback

                        traceback.print_exc()
                    # Skip this electrode completely - don't add unrefined results
                    continue

                # Step 7: Detect contacts
                if verbose:
                    print("  Detecting contacts...")

                contacts = detect_contacts(
                    refined_model,
                    method=contact_detection_method,
                    electrode_type=electrode_type,
                    display_profile=False,
                    limit_search_mm=20.0,
                    run_all_methods=False,
                )

                if contacts is not None and len(contacts) > 0:
                    refined_model.contact_positions = contacts
                    if verbose:
                        print(f"    Detected {len(contacts)} contacts")
                else:
                    refined_model.contact_positions = np.array([])
                    if verbose:
                        print("    No contacts detected")

                # Create final electrode model
                # Check if refined_model has distance_scale_mm attribute
                if hasattr(refined_model, "distance_scale_mm"):
                    distance_scale = refined_model.distance_scale_mm
                elif hasattr(refined_model, "distance_scale"):
                    distance_scale = refined_model.distance_scale
                else:
                    distance_scale = initial_electrode.distance_scale

                final_electrode = PolynomialElectrodeModel(
                    polynomial=refined_model.polynomial,
                    electrode_type=refined_model.electrode_type,
                    contact_positions=refined_model.contact_positions,
                    intensity_profile=refined_model.intensity_profile,
                    distance_scale=distance_scale,
                    bounding_box=bounding_box,
                    skeleton_deviations_mm=(
                        refined_model.skeleton_deviations_mm
                        if hasattr(refined_model, "skeleton_deviations_mm")
                        else None
                    ),
                    polynomial_before_tip_detection=(
                        refined_model.polynomial_before_tip_detection
                        if hasattr(refined_model, "polynomial_before_tip_detection")
                        else None
                    ),
                )
                # Store additional attributes for compatibility and report generation
                final_electrode.distance_scale_mm = distance_scale

                # Store refined trajectory info that's needed for visualization
                if hasattr(refined_model, "pass2_tip_threshold"):
                    final_electrode.pass2_tip_threshold = (
                        refined_model.pass2_tip_threshold
                    )
                if hasattr(refined_model, "skeleton_deviations_mm"):
                    final_electrode.skeleton_deviations_mm = (
                        refined_model.skeleton_deviations_mm
                    )
                if hasattr(refined_model, "pass2_tip_param"):
                    final_electrode.pass2_tip_param = refined_model.pass2_tip_param
                if hasattr(refined_model, "pass2_intensities_full"):
                    final_electrode.pass2_intensities_full = (
                        refined_model.pass2_intensities_full
                    )
                if hasattr(refined_model, "pass2_distances_mm_full"):
                    final_electrode.pass2_distances_mm_full = (
                        refined_model.pass2_distances_mm_full
                    )
                if hasattr(refined_model, "original_t0_distance_mm"):
                    final_electrode.original_t0_distance_mm = (
                        refined_model.original_t0_distance_mm
                    )

                # Store as refined attributes for GUI visualization compatibility
                final_electrode.refined_intensity_profile = (
                    refined_model.intensity_profile
                )
                final_electrode.refined_distance_scale = distance_scale

                # Orientation detection and electrode type classification
                orientation_data, classified_type = self._run_orientation_detection(
                    final_electrode,
                    electrode_idx=i,
                    orientation_params=orientation_params,
                )
                if orientation_data is not None:
                    final_electrode.orientation_data = orientation_data
                final_electrode.electrode_type = classified_type

                # Hemisphere detection for tip and entry positions
                final_electrode.tip_hemisphere = self._determine_hemisphere(
                    final_electrode.tip_position
                )
                final_electrode.entry_hemisphere = self._determine_hemisphere(
                    final_electrode.entry_position
                )

                self.electrodes.append(final_electrode)
                if verbose:
                    print(f"  Successfully added electrode {i+1}")

            except Exception as e:
                if verbose:
                    print(f"  Failed to process seed {seed}: {e}")
                    import traceback

                    traceback.print_exc()

        if verbose:
            print(f"\nSuccessfully detected {len(self.electrodes)} electrode(s)")

        # Auto-save results if requested
        if auto_save_json and self.electrodes:
            if verbose:
                print("Saving reconstruction results...")

            # Collect all parameters used
            parameters = {
                "method": "detect_electrodes_auto",
                "contact_detection_method": contact_detection_method,
                "electrode_type": electrode_type,
                "search_radii_mm": search_radii_mm,
                "metal_threshold": self.metal_threshold,
                "use_gpu": self.use_gpu,
                "debug_output_enabled": self.debug_output_dir is not None,
            }

            # Save reconstruction results
            json_path = self._save_reconstruction_json(parameters)
            if verbose:
                print("Reconstruction results saved successfully")

        return self.electrodes

    def _find_electrode_seeds_radial(
        self,
        metal_mask: np.ndarray,
        search_radii_mm: List[float],
        max_electrodes: int = 2,
        verbose: bool = True,
    ) -> List[Tuple[int, int, int]]:
        """
        Find electrode seed points using radial search from brain center.
        Stops searching when likely electrodes are found.

        Args:
            metal_mask: Binary mask of metal voxels
            search_radii_mm: List of search radii in mm
            max_electrodes: Expected maximum number of electrodes (default: 2)
            verbose: Print progress

        Returns:
            List of seed point coordinates (voxel indices)
        """
        from scipy.ndimage import center_of_mass, label

        # Get brain center (use image center as approximation)
        center_voxel = np.array(self.ct_data.shape) // 2
        if verbose:
            print(f"  Starting search from center: {center_voxel}")

        # Convert mm to voxels
        mean_voxel_size = np.mean(self.voxel_sizes)

        # First, label ALL connected metal components in the volume
        all_metal_labeled, num_total_components = label(metal_mask)
        if verbose:
            print(f"  Total metal components in volume: {num_total_components}")

        all_seed_points = []
        found_component_ids = set()  # Track which component IDs we've already found
        previous_mask = np.zeros_like(metal_mask, dtype=bool)

        # Expected number of electrodes
        expected_electrodes = max_electrodes
        min_component_size = 100  # Minimum voxels for a valid electrode component

        for radius_mm in search_radii_mm:
            radius_voxels = int(radius_mm / mean_voxel_size)
            if verbose:
                print(
                    f"\n  Searching at radius {radius_mm}mm ({radius_voxels} voxels)..."
                )

            # Create spherical mask for current radius
            current_mask = self._create_spherical_mask(
                center_voxel, radius_voxels, metal_mask.shape
            )

            # Find NEW metal voxels in this shell (exclude previously searched regions)
            shell_mask = current_mask & ~previous_mask
            metal_in_shell = metal_mask & shell_mask
            num_metal = metal_in_shell.sum()

            if num_metal == 0:
                if verbose:
                    print("    No new metal voxels found in shell")
                previous_mask = current_mask
                continue

            if verbose:
                print(f"    Found {num_metal} metal voxels in shell")

            # Find connected components in this shell
            labeled_metal, num_components = label(metal_in_shell)

            if verbose:
                print(f"    Found {num_components} connected components in shell")

            # Process components in this shell
            new_seeds_this_radius = []
            for comp_id in range(1, num_components + 1):
                component_mask = labeled_metal == comp_id
                component_size = component_mask.sum()

                # Skip small components (likely noise)
                if component_size < min_component_size:
                    continue

                # Find which global component this shell component belongs to
                # Get any point from this shell component
                component_points = np.argwhere(component_mask)
                if len(component_points) == 0:
                    continue

                # Check what global component ID this belongs to
                sample_point = tuple(component_points[0])
                global_comp_id = all_metal_labeled[sample_point]

                # Get center of mass of the shell component
                com = center_of_mass(component_mask)
                seed_point = tuple(int(round(c)) for c in com)

                # Verify the seed point is in metal
                if metal_mask[seed_point]:
                    # Check if this seed is far enough from existing seeds
                    is_new = True
                    min_distance_voxels = 20  # Minimum distance between electrode seeds

                    for existing_seed in all_seed_points:
                        dist = np.linalg.norm(
                            np.array(seed_point) - np.array(existing_seed)
                        )
                        if dist < min_distance_voxels:
                            is_new = False
                            if verbose:
                                print(
                                    f"      Component {comp_id}: {component_size} voxels - skipping (too close to existing seed, dist={dist:.1f} voxels)"
                                )
                            break

                    if is_new:
                        all_seed_points.append(seed_point)
                        new_seeds_this_radius.append(seed_point)
                        found_component_ids.add(global_comp_id)

                        if verbose:
                            # Count total voxels in this global component
                            total_voxels = np.sum(all_metal_labeled == global_comp_id)
                            print(
                                f"      Component {comp_id}: {component_size} voxels, seed at {seed_point}"
                            )
                            print(
                                f"        Global component ID: {global_comp_id} (total {total_voxels} voxels)"
                            )

            # If we found seeds at this radius
            if new_seeds_this_radius:
                if verbose:
                    print(f"\n  Found {len(all_seed_points)} electrodes total")

                # Stop searching if we've found the expected number of electrodes
                # or if we found at least 2 at this radius (likely bilateral electrodes)
                if (
                    len(all_seed_points) >= expected_electrodes
                    or len(new_seeds_this_radius) >= 2
                ):
                    if verbose:
                        print("  Found expected electrodes, stopping search")
                    break

            previous_mask = current_mask

        # Final filtering: if we have too many seeds, keep only the largest components
        if (
            len(all_seed_points) > expected_electrodes * 2
        ):  # More than 4 is likely too many
            if verbose:
                print(
                    f"\n  Too many seeds ({len(all_seed_points)}), filtering to keep largest {expected_electrodes * 2}"
                )

            # Sort by component size and keep the largest
            component_sizes = [mask.sum() for mask in all_component_masks]
            sorted_indices = np.argsort(component_sizes)[::-1]  # Descending order

            # Keep only the largest components
            all_seed_points = [
                all_seed_points[i] for i in sorted_indices[: expected_electrodes * 2]
            ]
            if verbose:
                print(f"  Kept {len(all_seed_points)} largest components")

        if verbose:
            print(f"\n  Total electrode seeds found: {len(all_seed_points)}")

        return all_seed_points

    def _create_spherical_mask(
        self, center: np.ndarray, radius_voxels: int, shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Create a spherical mask centered at given point.

        Args:
            center: Center point in voxel coordinates
            radius_voxels: Radius in voxels
            shape: Shape of output mask

        Returns:
            Binary mask with sphere
        """
        # Create coordinate grids
        x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]

        # Calculate distance from center
        dist_from_center = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )

        # Create mask
        mask = dist_from_center <= radius_voxels

        return mask

    def _detect_skull_exit_multiscale(
        self, trajectory_points: List, angle_threshold_deg: float = 25.0
    ) -> Optional[int]:
        """
        Detect skull exit point using multi-scale angle analysis.

        Args:
            trajectory_points: List of trajectory points
            angle_threshold_deg: Angle threshold in degrees

        Returns:
            Index of skull exit point or None
        """
        # Need sufficient points for reliable analysis
        if len(trajectory_points) < 20:
            return None

        # Get world coordinates
        points = np.array([p.center_of_gravity for p in trajectory_points])

        # Use larger window sizes for more stable angle detection
        # Smaller windows can detect noise as sharp angles
        window_sizes = [5, 10, 15]  # Match GUI implementation

        for window in window_sizes:
            if window >= len(points) // 2:
                continue  # Skip if window too large

            for i in range(window, len(points) - window):
                # Get vectors before and after
                vec_before = points[i] - points[i - window]
                vec_after = points[i + window] - points[i]

                # Normalize
                norm_before = np.linalg.norm(vec_before)
                norm_after = np.linalg.norm(vec_after)

                if norm_before > 0 and norm_after > 0:
                    vec_before = vec_before / norm_before
                    vec_after = vec_after / norm_after

                    # Calculate angle
                    cos_angle = np.clip(np.dot(vec_before, vec_after), -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))

                    if angle_deg > angle_threshold_deg:
                        if self.debug_output_dir:  # Only print in debug mode
                            print(
                                f"    Multi-scale angle detection (window={window} points):"
                            )
                            print(f"      Detected {angle_deg:.1f}° bend at point {i}")
                        return i

        return None

    def detect_electrodes_fast(
        self,
        contact_detection_method: str = "contactAreaCenter",
        electrode_type: Optional[str] = None,
        auto_save_json: bool = True,
        min_electrode_length_mm: float = 40.0,
        refinement_threshold: Optional[
            float
        ] = 800,  # Lower threshold for refinement, None to disable
    ) -> List[PolynomialElectrodeModel]:
        """
        Fast electrode detection with reduced quality settings.

        Approximately 5-10x faster than default settings.

        Args:
            contact_detection_method: Detection method
            electrode_type: Force specific electrode type
            auto_save_json: Automatically save reconstruction results to JSON in CT directory (default True)
            min_electrode_length_mm: Minimum electrode length in mm to keep (default 40.0)

        Returns:
            List of detected electrode models
        """
        return self.detect_electrodes(
            contact_detection_method=contact_detection_method,
            electrode_type=electrode_type,
            final_degree=3,
            xy_resolution=0.3,  # 3x coarser
            z_resolution=0.1,  # 4x coarser
            grid_size=1.0,  # Smaller grid
            display_profiles=False,
            auto_save_json=auto_save_json,
            min_electrode_length_mm=min_electrode_length_mm,
            refinement_threshold=refinement_threshold,
        )

    def detect_electrodes_radial(
        self,
        contact_detection_method: str = "contactAreaCenter",
        electrode_type: Optional[str] = None,
        final_degree: int = 3,
        xy_resolution: float = 0.1,
        z_resolution: float = 0.025,
        grid_size: float = 1.5,
        auto_save_json: bool = True,
        min_electrode_length_mm: float = 40.0,
        refinement_threshold: Optional[float] = 800,
        search_radii_mm: List[float] = None,
        max_electrodes: int = 4,
        verbose: bool = True,
        orientation_params: Optional[Dict[str, Any]] = None,
    ) -> List[PolynomialElectrodeModel]:
        """
        Detect electrodes using radial search with configurable parameters.

        This method combines the radial search approach from detect_electrodes_auto
        with the ability to specify custom resolution parameters for fast/normal/high quality modes.

        Args:
            contact_detection_method: Method for contact detection
            electrode_type: Force specific electrode type or auto-detect
            final_degree: Final polynomial degree for trajectory
            xy_resolution: XY resolution for refinement in mm
            z_resolution: Z resolution for refinement in mm
            grid_size: Grid size for refinement in mm
            auto_save_json: Automatically save reconstruction results
            min_electrode_length_mm: Minimum electrode length in mm
            refinement_threshold: HU threshold for refinement
            search_radii_mm: Radii to search for electrodes (default: [30, 40, 50])
            max_electrodes: Maximum number of electrodes to detect
            verbose: Print progress messages
            orientation_params: Optional dict to override orientation detection
                defaults. See _run_orientation_detection for supported keys.

        Returns:
            List of detected electrode models
        """
        if search_radii_mm is None:
            search_radii_mm = [30, 40, 50]

        print(
            "\n=== Starting radial search electrode detection (detect_electrodes_radial) ==="
        )
        print(
            f"Resolution parameters: xy={xy_resolution}mm, z={z_resolution}mm, grid={grid_size}mm"
        )

        # Import required modules

        from .cog_trajectory_tracking import CenterOfGravityTracker
        from .polynomial_fitting import fit_polynomial_to_trajectory

        # Step 1: Detect metal artifacts
        if verbose:
            print(f"Detecting metal artifacts (threshold={self.metal_threshold} HU)...")
        metal_mask = self.ct_data > self.metal_threshold

        # Step 2: Find electrode seeds using radial search
        if verbose:
            print("Finding electrode seeds using radial search...")
        seed_points = self._find_electrode_seeds_radial(
            metal_mask, search_radii_mm, max_electrodes, verbose
        )

        if not seed_points:
            if verbose:
                print("No electrode seeds found")
            self.electrodes = []
            return self.electrodes

        if verbose:
            print(f"Found {len(seed_points)} potential electrode locations\n")

        # Step 3: Initialize COG tracker
        cog_tracker = CenterOfGravityTracker(
            ct_data=self.ct_data,
            affine=self.affine,
            metal_threshold=self.metal_threshold,
            search_radius_mm=5.0,
            max_direction_change_deg=60.0,
            min_voxels_per_slice=3,
        )

        # Step 4: Track COG trajectories and fit polynomials
        self.electrodes = []
        cog_trajectories = []

        for i, seed in enumerate(seed_points):
            if verbose:
                print(
                    f"Processing electrode {i+1}/{len(seed_points)} at seed {seed}..."
                )

            try:
                # Track trajectory from seed
                trajectory_points = cog_tracker.track_from_seed(
                    seed_voxel=seed, slice_axis="axial"
                )

                # Check minimum length
                if len(trajectory_points) < 10:
                    if verbose:
                        print(
                            f"  Skipping: trajectory too short ({len(trajectory_points)} points)"
                        )
                    continue

                # Check for skull exit using multi-scale detection
                exit_idx = self._detect_skull_exit_multiscale(
                    trajectory_points, angle_threshold_deg=25.0
                )
                if exit_idx is not None:
                    if verbose:
                        print(
                            f"  Truncating trajectory at skull exit (point {exit_idx} of {len(trajectory_points)})"
                        )
                    trajectory_points = trajectory_points[:exit_idx]

                if verbose:
                    print(
                        f"  Tracked {len(trajectory_points)} points (after skull exit correction)"
                    )

                # Extract world coordinates from trajectory points
                points_array = np.array(
                    [p.center_of_gravity for p in trajectory_points]
                )

                # Calculate trajectory length
                if len(points_array) > 1:
                    diffs = np.diff(points_array, axis=0)
                    distances = np.linalg.norm(diffs, axis=1)
                    total_length = np.sum(distances)
                else:
                    total_length = 0.0

                if verbose:
                    print(f"  Total length: {total_length:.1f}mm")

                if total_length < min_electrode_length_mm:
                    if verbose:
                        print(
                            f"  Skipping: trajectory too short ({total_length:.1f}mm < {min_electrode_length_mm}mm)"
                        )
                    continue

                # Get intensities from trajectory points
                intensities = np.array([p.mean_intensity for p in trajectory_points])

                # Fit initial trajectory
                if verbose:
                    print("  Fitting initial polynomial (degree 8)...")
                poly_result = fit_polynomial_to_trajectory(
                    points_array, degree=8, weights=intensities
                )

                if not poly_result:
                    if verbose:
                        print("  Failed to fit polynomial")
                    continue

                # Calculate bounding box
                min_coords = points_array.min(axis=0)
                max_coords = points_array.max(axis=0)
                bounding_box = (min_coords, max_coords)

                # Create initial electrode model
                initial_electrode = PolynomialElectrodeModel(
                    polynomial=poly_result.polynomial,
                    electrode_type=electrode_type or "Medtronic 3389/B33005",
                    contact_positions=np.array([]),
                    intensity_profile=intensities,
                    distance_scale=poly_result.distance_scale_mm,
                    bounding_box=bounding_box,
                )
                # Store the distance scale as an attribute for compatibility
                initial_electrode.distance_scale_mm = poly_result.distance_scale_mm
                initial_electrode.total_length_mm = poly_result.total_length_mm

                # Refine trajectory with specified parameters
                if verbose:
                    print(
                        f"  Running refinement (xy_res={xy_resolution}mm, z_res={z_resolution}mm)..."
                    )
                refined_model = refine_electrode_trajectory(
                    initial_electrode,
                    points_array,
                    intensities,
                    self.ct_data,
                    self.affine,
                    final_degree=final_degree,
                    xy_resolution=xy_resolution,
                    z_resolution=z_resolution,
                    grid_size=grid_size,
                    refinement_threshold=refinement_threshold,
                    electrode_idx=i if self.debug_output_dir else None,
                    debug_output_dir=self.debug_output_dir,
                    auto_select_degree=True,  # Auto-select best degree based on bottom 20mm
                    contact_region_mm=20.0,  # Evaluate fit quality in bottom 20mm
                )

                # Detect contacts
                if verbose:
                    print("  Detecting contacts...")
                contacts = detect_contacts(
                    refined_model,
                    method=contact_detection_method,
                    electrode_type=electrode_type,
                    display_profile=False,
                    run_all_methods=(self.debug_output_dir is not None),
                )

                # Create electrode model
                detected_electrode_type = getattr(refined_model, "electrode_type", None)
                if detected_electrode_type is None:
                    detected_electrode_type = electrode_type or "Medtronic 3389/B33005"

                # Calculate bounding box
                min_coords = points_array.min(axis=0)
                max_coords = points_array.max(axis=0)
                bounding_box = (min_coords, max_coords)

                electrode = PolynomialElectrodeModel(
                    polynomial=refined_model.polynomial,
                    electrode_type=detected_electrode_type,
                    contact_positions=contacts,
                    intensity_profile=refined_model.intensity_profile,
                    distance_scale=refined_model.distance_scale_mm,
                    bounding_box=bounding_box,
                    skeleton_deviations_mm=(
                        refined_model.skeleton_deviations_mm
                        if hasattr(refined_model, "skeleton_deviations_mm")
                        else None
                    ),
                    polynomial_before_tip_detection=(
                        refined_model.polynomial_before_tip_detection
                        if hasattr(refined_model, "polynomial_before_tip_detection")
                        else None
                    ),
                )

                # Store contact detection results if available (debug mode)
                if hasattr(refined_model, "contact_detection_results"):
                    electrode.contact_detection_results = (
                        refined_model.contact_detection_results
                    )

                # Store full Pass 2 debug data if available
                if hasattr(refined_model, "pass2_intensities_full"):
                    electrode.pass2_intensities_full = (
                        refined_model.pass2_intensities_full
                    )
                if hasattr(refined_model, "pass2_distances_mm_full"):
                    electrode.pass2_distances_mm_full = (
                        refined_model.pass2_distances_mm_full
                    )
                if hasattr(refined_model, "pass2_tip_threshold"):
                    electrode.pass2_tip_threshold = refined_model.pass2_tip_threshold
                if hasattr(refined_model, "original_t0_distance_mm"):
                    electrode.original_t0_distance_mm = (
                        refined_model.original_t0_distance_mm
                    )

                # Orientation detection and electrode type classification
                orientation_data, classified_type = self._run_orientation_detection(
                    electrode,
                    electrode_idx=i,
                    orientation_params=orientation_params,
                )
                if orientation_data is not None:
                    electrode.orientation_data = orientation_data
                electrode.electrode_type = classified_type

                # Hemisphere detection for tip and entry positions
                electrode.tip_hemisphere = self._determine_hemisphere(
                    electrode.tip_position
                )
                electrode.entry_hemisphere = self._determine_hemisphere(
                    electrode.entry_position
                )

                self.electrodes.append(electrode)
                cog_trajectories.append(
                    {
                        "seed": seed,
                        "trajectory": trajectory_points,
                        "polynomial": initial_electrode.polynomial,
                    }
                )

                # Save combined intensity profile plot if debug output is enabled
                if self.debug_output_dir:
                    self._save_combined_intensity_plot(
                        refined_model, electrode, i, self.debug_output_dir
                    )

                if verbose:
                    print(f"  ✓ Successfully processed electrode {i+1}")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to process electrode {i+1}: {str(e)}")
                if self.debug_output_dir:
                    import traceback

                    traceback.print_exc()

        if verbose:
            print(f"\n✓ Successfully detected {len(self.electrodes)} electrode(s)")

        # Save results if requested
        if auto_save_json and self.electrodes:
            parameters = {
                "method": "detect_electrodes_radial",
                "contact_detection_method": contact_detection_method,
                "electrode_type": electrode_type,
                "final_degree": final_degree,
                "xy_resolution": xy_resolution,
                "z_resolution": z_resolution,
                "grid_size": grid_size,
                "search_radii_mm": search_radii_mm,
                "metal_threshold": self.metal_threshold,
                "refinement_threshold": refinement_threshold,
                "processing_type": "CPU",
                "interface": "CLI",  # Will be overridden by GUI if called from there
            }

            self._save_reconstruction_json(parameters)

        return self.electrodes
