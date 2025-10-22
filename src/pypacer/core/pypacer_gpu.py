"""GPU-accelerated PyPaCER class for electrode reconstruction."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np

from ..gpu.gpu_utils import GPUMemoryManager, gpu_available
from ..gpu.refinement_gpu import refine_electrode_trajectory_gpu
from ..models.electrode import PolynomialElectrodeModel
from .contact_detection import detect_contacts
from .contact_detection_comparison import compare_contact_detection_methods
from .electrode_detection import extract_electrode_pointclouds
from .pypacer import PyPaCER
from .trajectory_fit import fit_initial_trajectory


class PyPaCER_GPU(PyPaCER):
    """
    GPU-accelerated version of PyPaCER for faster electrode reconstruction.

    This class inherits from PyPaCER and uses GPU acceleration (via PyTorch)
    for trajectory refinement and OOR processing. Preprocessing steps use
    CPU for now. Falls back to CPU methods when GPU is not available.
    """

    def __init__(
        self,
        ct_path: Union[str, Path],
        brain_mask: Optional[Union[str, Path, np.ndarray]] = None,
        metal_threshold: float = 2000,
        max_gpu_memory_mb: int = 2048,
        debug_output_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize GPU-accelerated PyPaCER.

        Args:
            ct_path: Path to CT NIfTI file
            brain_mask: Optional brain mask (auto-detected if None)
            metal_threshold: Hounsfield unit threshold for metal
            max_gpu_memory_mb: Maximum GPU memory to use in MB
            debug_output_dir: Optional directory for debug outputs (e.g., orthogonal grids)
            output_dir: Directory for saving results (JSON, reports)
        """
        # Initialize base class with use_gpu=True
        super().__init__(
            ct_path=ct_path,
            brain_mask=brain_mask,
            metal_threshold=metal_threshold,
            use_gpu=True,
            debug_output_dir=debug_output_dir,
            output_dir=output_dir,
        )

        # GPU-specific initialization
        self.gpu_available = gpu_available()
        self.gpu_memory_manager = GPUMemoryManager(max_gpu_memory_mb)
        self.output_dir = output_dir  # Store output directory

        if self.gpu_available:
            print(f"GPU acceleration enabled (max memory: {max_gpu_memory_mb}MB)")
            # Print GPU info
            try:
                import torch

                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    print(f"Using GPU: {device_name}")
                    capability = torch.cuda.get_device_capability(0)
                    print(f"Compute capability: {capability[0]}.{capability[1]}")
            except:
                pass
        else:
            print("GPU not available, falling back to CPU processing")

    def detect_electrodes_radial(
        self,
        contact_detection_method: str = "contactAreaCenter",
        electrode_type: Optional[str] = None,
        final_degree: int = 3,
        display_profiles: bool = False,
        xy_resolution: float = 0.1,
        z_resolution: float = 0.025,
        grid_size: float = 2.0,
        auto_save_json: bool = True,
        min_electrode_length_mm: float = 40.0,
        refinement_threshold: Optional[float] = 800,
        refinement_radius_mm: float = 3.5,
        debug_z_axis_scale: float = 8.0,
        use_subvolume: bool = True,
        search_radii_mm: List[float] = [30, 40, 50],
        max_electrodes: int = 4,
        verbose: bool = True,
    ) -> List[PolynomialElectrodeModel]:
        """
        GPU-accelerated electrode detection using radial search.

        This method searches for electrodes by looking at increasing distances
        from the brain center, which is more robust than brain mask methods.

        Args:
            Same as detect_electrodes plus:
            search_radii_mm: List of radii to search at (in mm)
            max_electrodes: Maximum number of electrodes to detect
            verbose: Print progress messages

        Returns:
            List of detected electrode models
        """
        print("Starting GPU-accelerated radial search electrode detection")
        print(f"Processing {self.ct_path.name}")
        print(f"Voxel size: {self.voxel_sizes}")

        overall_start = time.time()

        # Import required modules

        from ..core.cog_trajectory_tracking import CenterOfGravityTracker
        from ..core.contact_detection import detect_contacts
        from ..core.contact_detection_comparison import (
            compare_contact_detection_methods,
        )
        from ..core.polynomial_fitting import fit_polynomial_to_trajectory

        # Step 1: Detect metal artifacts
        if verbose:
            print(f"Detecting metal artifacts (threshold={self.metal_threshold} HU)...")
        metal_mask = self.ct_data > self.metal_threshold

        # Step 2: Find electrode seeds using radial search
        if verbose:
            print("Finding electrode seeds using radial search...")
        # Import parent's methods if not available
        if not hasattr(self, "_find_electrode_seeds_radial"):
            from ..core.pypacer import PyPaCER

            # Bind parent's methods to this instance
            self._find_electrode_seeds_radial = (
                PyPaCER._find_electrode_seeds_radial.__get__(self, type(self))
            )
            self._detect_skull_exit_multiscale = (
                PyPaCER._detect_skull_exit_multiscale.__get__(self, type(self))
            )

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
        process_start = time.time()
        self.electrodes = []

        for i, seed in enumerate(seed_points):
            electrode_start = time.time()
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
                points_world = np.array(
                    [p.center_of_gravity for p in trajectory_points]
                )

                # Calculate trajectory length
                if len(points_world) > 1:
                    diffs = np.diff(points_world, axis=0)
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
                initial_trajectory = PolynomialElectrodeModel(
                    polynomial=poly_result.polynomial,
                    electrode_type=electrode_type or "Medtronic 3389/B33005",
                    contact_positions=np.array([]),
                    intensity_profile=intensities,
                    distance_scale=poly_result.distance_scale_mm,
                    bounding_box=bounding_box,
                )
                # Store the distance scale as an attribute for compatibility
                initial_trajectory.distance_scale_mm = poly_result.distance_scale_mm
                initial_trajectory.total_length_mm = poly_result.total_length_mm

                # GPU-accelerated trajectory refinement
                if verbose:
                    print("  Running GPU-accelerated refinement...")
                try:
                    if not self.gpu_available:
                        error_msg = """
                        PyPaCER_GPU requires GPU acceleration but no GPU was detected!
                        
                        Use the regular PyPaCER class for CPU processing, or ensure:
                        1. PyTorch is installed with CUDA support
                        2. NVIDIA GPU and drivers are properly installed
                        3. GPU is not in use by another process
                        
                        To check: python -c "import torch; print(torch.cuda.is_available())"
                        """
                        raise RuntimeError(error_msg)

                    refined_model = refine_electrode_trajectory_gpu(
                        initial_trajectory,
                        points_world,
                        intensities,
                        self.ct_data,
                        self.affine,
                        final_degree=final_degree,
                        xy_resolution=xy_resolution,
                        z_resolution=z_resolution,
                        grid_size=grid_size,
                        use_gpu=True,
                        electrode_idx=i if self.debug_output_dir else None,
                        debug_output_dir=self.debug_output_dir,
                        refinement_threshold=refinement_threshold,
                        refinement_radius_mm=refinement_radius_mm,
                        debug_z_axis_scale=debug_z_axis_scale,
                        use_subvolume=use_subvolume,
                        auto_select_degree=True,  # Auto-select best degree based on bottom 20mm
                        contact_region_mm=20.0,  # Evaluate fit quality in bottom 20mm
                    )
                except Exception as e:
                    print(f"GPU refinement failed: {e}")
                    raise

                # Contact detection
                if verbose:
                    print("  Detecting contacts...")
                if contact_detection_method == "comparison":
                    comparison_result = compare_contact_detection_methods(
                        refined_model,
                        electrode_type=electrode_type,
                        save_plots=True,
                        output_dir=self.debug_output_dir,
                    )
                    contacts = comparison_result.method_results[
                        "contactAreaCenter"
                    ].contact_positions
                    refined_model.contact_comparison = comparison_result
                else:
                    contacts = detect_contacts(
                        refined_model,
                        method=contact_detection_method,
                        electrode_type=electrode_type,
                        display_profile=display_profiles,
                        run_all_methods=(self.debug_output_dir is not None),
                    )

                # Create final electrode model
                detected_electrode_type = getattr(refined_model, "electrode_type", None)
                if detected_electrode_type is None:
                    detected_electrode_type = electrode_type or "Medtronic 3389/B33005"

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

                # Store comparison results if available
                if hasattr(refined_model, "contact_comparison"):
                    electrode.contact_comparison = refined_model.contact_comparison

                # Store contact detection results if available
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

                self.electrodes.append(electrode)

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

        print(f"\nElectrode processing: {time.time() - process_start:.2f}s")
        print(f"Total GPU processing time: {time.time() - overall_start:.2f}s")

        # Clear GPU memory cache
        if self.gpu_available:
            self.gpu_memory_manager.clear_cache()
            mem_info = self.gpu_memory_manager.get_memory_info()
            if mem_info.get("available", False):
                print(f"GPU memory allocated: {mem_info.get('allocated_mb', 0):.1f}MB")

        # Auto-save results if requested
        if auto_save_json and self.electrodes:
            parameters = {
                "contact_detection_method": contact_detection_method,
                "electrode_type": electrode_type,
                "final_degree": final_degree,
                "xy_resolution": xy_resolution,
                "z_resolution": z_resolution,
                "grid_size": grid_size,
                "display_profiles": display_profiles,
                "use_gpu": True,
                "gpu_available": self.gpu_available,
                "debug_output_enabled": self.debug_output_dir is not None,
                "detection_method": "radial_search",
                "search_radii_mm": search_radii_mm,
                "processing_type": "GPU",
                "interface": "CLI",  # Will be overridden by GUI if called from there
            }

            # Save reconstruction results
            json_path = self._save_reconstruction_json(parameters)

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
                        (
                            Path(self.output_dir)
                            if self.output_dir
                            else self.ct_path.parent
                        ),
                        electrode_model=electrode,
                    )

        return self.electrodes

    def detect_electrodes(
        self,
        contact_detection_method: str = "contactAreaCenter",
        electrode_type: Optional[str] = None,
        final_degree: int = 3,
        display_profiles: bool = False,
        xy_resolution: float = 0.1,
        z_resolution: float = 0.025,
        grid_size: float = 2.0,
        auto_save_json: bool = True,
        min_electrode_length_mm: float = 40.0,
        refinement_threshold: Optional[
            float
        ] = 800,  # Lower threshold for refinement, None to disable
        refinement_radius_mm: float = 3.5,  # Radius around trajectory for refinement
        debug_z_axis_scale: float = 8.0,  # Scale factor for z-axis in debug NIfTI
        use_subvolume: bool = True,  # Use memory-efficient subvolume interpolation
    ) -> List[PolynomialElectrodeModel]:
        """
        GPU-accelerated electrode detection pipeline.

        Overrides parent method to use GPU acceleration for:
        - Metal artifact detection
        - Trajectory refinement
        - Mathematical operations

        Args:
            Same as parent class

        Returns:
            List of detected electrode models
        """
        print("Starting GPU-accelerated electrode detection")
        print(f"Processing {self.ct_path.name}")
        print(f"Voxel size: {self.voxel_sizes}")

        overall_start = time.time()

        # Step 1: Extract brain mask if needed
        if self.brain_mask is None:
            print("Extracting brain mask...")
            mask_start = time.time()
            # Always use CPU version for brain mask extraction for now
            from ..imaging.preprocessing import extract_brain_mask

            self.brain_mask = extract_brain_mask(self.ct_data, self.voxel_sizes)
            print(f"  Brain mask extraction: {time.time() - mask_start:.2f}s")
        elif isinstance(self.brain_mask, (str, Path)):
            mask_path = str(self.brain_mask)
            mask_img = nib.load(mask_path)
            self.brain_mask = mask_img.get_fdata().astype(bool)
            print(f"Loaded brain mask from {mask_path}")

        # Validate brain mask
        if not isinstance(self.brain_mask, np.ndarray):
            self.brain_mask = np.array(self.brain_mask)

        print(
            f"Brain mask shape: {self.brain_mask.shape}, covering {np.sum(self.brain_mask)} voxels"
        )

        # Step 2: Detect metal artifacts (GPU-accelerated)
        print(f"Detecting metal artifacts (threshold={self.metal_threshold} HU)...")
        metal_start = time.time()

        # Use CPU preprocessing for metal detection (GPU preprocessing used CuPy which is removed)
        from ..imaging.preprocessing import detect_metal_artifacts

        metal_mask = detect_metal_artifacts(
            self.ct_data, self.brain_mask, self.metal_threshold
        )

        print(f"  Metal detection: {time.time() - metal_start:.2f}s")

        # Step 3: Filter metal components (GPU-accelerated)
        print("Filtering metal components...")
        filter_start = time.time()

        # Use CPU preprocessing for component filtering (GPU preprocessing used CuPy which is removed)
        from ..imaging.preprocessing import filter_metal_components

        labeled_metal, num_electrodes = filter_metal_components(
            metal_mask, self.ct_data, self.voxel_sizes
        )

        print(f"  Component filtering: {time.time() - filter_start:.2f}s")

        # Step 4: Extract electrode point clouds
        print("Extracting electrode point clouds...")
        extract_start = time.time()
        point_clouds = extract_electrode_pointclouds(
            labeled_metal,
            self.ct_data,
            self.voxel_sizes,
            self.affine,
            min_length_mm=min_electrode_length_mm,
        )
        print(f"  Point cloud extraction: {time.time() - extract_start:.2f}s")

        print(f"Found {len(point_clouds)} potential electrodes")

        # Step 5: Process each electrode
        process_start = time.time()

        # Sequential processing is optimal for single GPU
        print("Using sequential GPU processing")
        self.electrodes = []

        for i, pc in enumerate(point_clouds):
            print(f"\n--- Processing electrode {i+1}/{len(point_clouds)} ---")
            electrode_start = time.time()

            # Initial trajectory fit
            initial_trajectory = fit_initial_trajectory(
                pc.points_world, pc.intensities, degree=8
            )

            # GPU-accelerated trajectory refinement (required for PyPaCER_GPU)
            try:
                if not self.gpu_available:
                    error_msg = """
                    PyPaCER_GPU requires GPU acceleration but no GPU was detected!
                    
                    Use the regular PyPaCER class for CPU processing, or ensure:
                    1. PyTorch is installed with CUDA support
                    2. NVIDIA GPU and drivers are properly installed
                    3. GPU is not in use by another process
                    
                    To check: python -c "import torch; print(torch.cuda.is_available())"
                    """
                    raise RuntimeError(error_msg)

                refined_model = refine_electrode_trajectory_gpu(
                    initial_trajectory,
                    pc.points_world,
                    pc.intensities,
                    self.ct_data,
                    self.affine,
                    final_degree=final_degree,
                    xy_resolution=xy_resolution,
                    z_resolution=z_resolution,
                    grid_size=grid_size,
                    use_gpu=True,
                    electrode_idx=i if self.debug_output_dir else None,
                    debug_output_dir=self.debug_output_dir,
                    refinement_threshold=refinement_threshold,
                    refinement_radius_mm=refinement_radius_mm,
                    debug_z_axis_scale=debug_z_axis_scale,
                    use_subvolume=use_subvolume,
                )
            except Exception as e:
                print(f"GPU refinement failed: {e}")
                raise  # Re-raise the exception to see the full error

            # Contact detection - run comparison if requested
            if contact_detection_method == "comparison":
                # Run both methods and compare
                comparison_result = compare_contact_detection_methods(
                    refined_model,
                    electrode_type=electrode_type,
                    save_plots=True,
                    output_dir=self.debug_output_dir,
                )

                # Use contactAreaCenter as the default for the model
                contacts = comparison_result.method_results[
                    "contactAreaCenter"
                ].contact_positions

                # Store comparison results in refined model for later access
                refined_model.contact_comparison = comparison_result
            else:
                # Single method detection
                contacts = detect_contacts(
                    refined_model,
                    method=contact_detection_method,
                    electrode_type=electrode_type,
                    display_profile=display_profiles,
                    run_all_methods=(self.debug_output_dir is not None),
                )

            # Create final electrode model
            detected_electrode_type = getattr(refined_model, "electrode_type", None)
            if detected_electrode_type is None:
                detected_electrode_type = electrode_type or "Medtronic 3389/B33005"

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

            # Store comparison results if available
            if hasattr(refined_model, "contact_comparison"):
                electrode.contact_comparison = refined_model.contact_comparison

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

            self.electrodes.append(electrode)

            # Save combined intensity profile plot if debug output is enabled
            if self.debug_output_dir:
                self._save_combined_intensity_plot(
                    refined_model, electrode, i, self.debug_output_dir
                )
            print(
                f"  Electrode {i+1} processing time: {time.time() - electrode_start:.2f}s"
            )

        print(f"\nElectrode processing: {time.time() - process_start:.2f}s")
        print(f"Total GPU processing time: {time.time() - overall_start:.2f}s")

        # Clear GPU memory cache
        if self.gpu_available:
            self.gpu_memory_manager.clear_cache()
            mem_info = self.gpu_memory_manager.get_memory_info()
            if mem_info.get("available", False):
                print(f"GPU memory allocated: {mem_info.get('allocated_mb', 0):.1f}MB")

        # Auto-save results if requested
        if auto_save_json and self.electrodes:
            parameters = {
                "contact_detection_method": contact_detection_method,
                "electrode_type": electrode_type,
                "final_degree": final_degree,
                "xy_resolution": xy_resolution,
                "z_resolution": z_resolution,
                "grid_size": grid_size,
                "display_profiles": display_profiles,
                "use_gpu": True,
                "gpu_available": self.gpu_available,
                "debug_output_enabled": self.debug_output_dir is not None,
            }

            # Save reconstruction results
            json_path = self._save_reconstruction_json(parameters)

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
                        (
                            Path(self.output_dir)
                            if self.output_dir
                            else self.ct_path.parent
                        ),
                        electrode_model=electrode,
                    )

        return self.electrodes

    def benchmark_gpu_speedup(self) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance on current dataset.

        Returns:
            Dictionary with timing information
        """
        print("\nBenchmarking GPU vs CPU performance...")

        # GPU timing
        gpu_start = time.time()
        gpu_electrodes = self.detect_electrodes(auto_save_json=False)
        gpu_time = time.time() - gpu_start

        # Temporarily disable GPU
        self.use_gpu = False
        original_gpu_available = self.gpu_available
        self.gpu_available = False

        # CPU timing
        cpu_start = time.time()
        cpu_electrodes = self.detect_electrodes(auto_save_json=False)
        cpu_time = time.time() - cpu_start

        # Restore GPU settings
        self.use_gpu = True
        self.gpu_available = original_gpu_available

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        results = {
            "gpu_time": gpu_time,
            "cpu_time": cpu_time,
            "speedup": speedup,
            "gpu_electrodes": len(gpu_electrodes),
            "cpu_electrodes": len(cpu_electrodes),
        }

        print("\nBenchmark Results:")
        print(f"  GPU time: {gpu_time:.2f}s")
        print(f"  CPU time: {cpu_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        return results

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

            # Top plot: Pass 2 intensity profile (full trajectory)
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
