#!/usr/bin/env python3
"""
PyPaCER GPU - GPU-accelerated electrode reconstruction from CT scans.

Usage:
    pypacer_gpu <ct_file> [options]

Example:
    pypacer_gpu ct_scan.nii.gz --metal-threshold 1800
    pypacer_gpu ct_scan.nii.gz --output-dir output/ --no-report
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pypacer.core.pypacer_gpu import PyPaCER_GPU
from pypacer.gpu.gpu_utils import gpu_available


def main():
    """Main entry point for the GPU-accelerated PyPaCER CLI."""
    parser = argparse.ArgumentParser(
        description="GPU-accelerated DBS electrode reconstruction from CT scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with radial search (default)
  pypacer_gpu ct_scan.nii.gz
  
  # Use automatic brain mask extraction
  pypacer_gpu ct_scan.nii.gz --brain-mask
  
  # Use custom brain mask file
  pypacer_gpu ct_scan.nii.gz --brain-mask brain_mask.nii.gz
  
  # Custom search radii for radial search
  pypacer_gpu ct_scan.nii.gz --search-radii 25 35 45 55
  
  # Specify output directory
  pypacer_gpu ct_scan.nii.gz --output-dir results/
  
  # Generate HTML report
  pypacer_gpu ct_scan.nii.gz --html
  
  # Custom metal threshold
  pypacer_gpu ct_scan.nii.gz --metal-threshold 1500
  
  # Specify electrode type
  pypacer_gpu ct_scan.nii.gz --electrode-type "Medtronic 3389"
        """,
    )

    # Required arguments
    parser.add_argument("ct_file", type=str, help="Path to the CT NIfTI file")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as CT file)",
    )

    # Optional arguments
    parser.add_argument(
        "--brain-mask",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help='Use brain mask method. Provide path or use "auto" (pypacer_gpu ct.nii.gz --brain-mask)',
    )

    parser.add_argument(
        "--search-radii",
        type=float,
        nargs="+",
        default=[30, 40, 50],
        help="Search radii for radial search in mm (default: 30 40 50)",
    )

    parser.add_argument(
        "--metal-threshold",
        type=float,
        default=2000,
        help="Hounsfield unit threshold for metal detection (default: 2000)",
    )

    parser.add_argument(
        "--contact-method",
        type=str,
        default="contactAreaCenter",
        choices=["contactAreaCenter", "peak", "peakWaveCenter", "comparison"],
        help="Contact detection method (default: contactAreaCenter)",
    )

    parser.add_argument(
        "--electrode-type",
        type=str,
        default=None,
        help='Force specific electrode type (e.g., "Medtronic 3389")',
    )

    parser.add_argument(
        "--xy-resolution",
        type=float,
        default=None,
        help="Orthogonal grid resolution in mm (default: 0.1)",
    )

    parser.add_argument(
        "--z-resolution",
        type=float,
        default=None,
        help="Along-trajectory resolution in mm (default: 0.025)",
    )

    parser.add_argument(
        "--grid-size",
        type=float,
        default=2.0,
        help="Sampling grid size in mm (default: 2.0)",
    )

    parser.add_argument(
        "--min-length",
        type=float,
        default=40.0,
        help="Minimum electrode length in mm (default: 40.0)",
    )

    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        default=2048,
        help="Maximum GPU memory to use in MB (default: 2048)",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report with reconstruction results",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug outputs (orthogonal grids, all contact detection methods)",
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="Run GPU vs CPU benchmark comparison"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate input file
    ct_path = Path(args.ct_file)
    if not ct_path.exists():
        print(f"Error: CT file not found: {ct_path}", file=sys.stderr)
        sys.exit(1)

    if ct_path.suffix not in [".nii", ".gz"]:
        print(
            f"Error: CT file must be NIfTI format (.nii or .nii.gz): {ct_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = ct_path.parent

    # Initialize log file variable
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Set up logging if debug is enabled
    if args.debug:
        # Create a logger that captures all output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = output_dir / f"pypacer_gpu_{timestamp}.log"

        # Configure logging to capture stdout/stderr
        class TeeOutput:
            def __init__(self, file_handle, original_stream):
                self.file = file_handle
                self.terminal = original_stream

            def write(self, message):
                self.terminal.write(message)
                self.file.write(message)
                self.file.flush()

            def flush(self):
                self.terminal.flush()
                self.file.flush()

            def __getattr__(self, name):
                # Delegate all other attributes to the original stream
                return getattr(self.terminal, name)

        # Open log file and redirect stdout/stderr
        log_file = open(log_filename, "w")
        sys.stdout = TeeOutput(log_file, sys.stdout)
        sys.stderr = TeeOutput(log_file, sys.stderr)

        print(f"Debug mode enabled - logging output to: {log_filename}")

    # Check GPU availability
    if not gpu_available():
        print(
            "Warning: GPU not available. Consider using the standard pypacer CLI for CPU processing."
        )
        response = input("Continue with CPU fallback? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    # Set resolution parameters
    xy_resolution = args.xy_resolution or 0.1
    z_resolution = args.z_resolution or 0.025

    # Initialize GPU-accelerated PaCER
    try:
        if args.verbose:
            print("Initializing GPU-accelerated PaCER...")
            print(f"  CT file: {ct_path}")
            print(f"  Output directory: {output_dir}")
            print(f"  Metal threshold: {args.metal_threshold} HU")
            print(f"  Resolution: {xy_resolution}mm (xy), {z_resolution}mm (z)")

        pacer = PyPaCER_GPU(
            ct_path=str(ct_path),
            brain_mask=args.brain_mask,
            metal_threshold=args.metal_threshold,
            max_gpu_memory_mb=args.max_gpu_memory,
            debug_output_dir=str(output_dir) if args.debug else None,
            output_dir=str(output_dir),  # Always pass output directory
        )

        # Run benchmark if requested
        if args.benchmark:
            print("\nRunning GPU vs CPU benchmark...")
            benchmark_results = pacer.benchmark_gpu_speedup()

            # Save benchmark results
            benchmark_file = output_dir / "benchmark_results.json"
            with open(benchmark_file, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"Benchmark results saved to: {benchmark_file}")
            return

        # Run electrode detection
        start_time = time.time()

        # Choose detection method based on brain mask argument
        if args.brain_mask is not None:
            # Brain mask method requested
            if args.verbose:
                print("Using brain mask detection method")
            electrodes = pacer.detect_electrodes(
                contact_detection_method=args.contact_method,
                electrode_type=args.electrode_type,
                xy_resolution=xy_resolution,
                z_resolution=z_resolution,
                grid_size=args.grid_size,
                min_electrode_length_mm=args.min_length,
                auto_save_json=True,
            )
        else:
            # Default to radial search
            if args.verbose:
                print("Using radial search detection method")
            electrodes = pacer.detect_electrodes_radial(
                contact_detection_method=args.contact_method,
                electrode_type=args.electrode_type,
                xy_resolution=xy_resolution,
                z_resolution=z_resolution,
                grid_size=args.grid_size,
                min_electrode_length_mm=args.min_length,
                search_radii_mm=args.search_radii,
                auto_save_json=True,
            )

        elapsed_time = time.time() - start_time

        print(f"\nReconstruction completed in {elapsed_time:.2f} seconds")
        print(f"Detected {len(electrodes)} electrode(s)")

        # Print summary
        for i, electrode in enumerate(electrodes):
            print(f"\nElectrode {i+1}:")
            print(f"  Type: {electrode.electrode_type}")
            print(f"  Contacts: {len(electrode.contact_positions)}")
            if args.verbose and electrode.contact_positions is not None:
                print(
                    f"  Contact positions (mm from tip): {electrode.contact_positions}"
                )
            if hasattr(electrode, "orientation_data") and electrode.orientation_data:
                od = electrode.orientation_data
                has_markers = od.get("has_markers", False)
                if has_markers and "markers" in od:
                    markers = od["markers"]
                    print("  Orientation:")
                    for label in ("B", "A"):
                        if label in markers:
                            m = markers[label]
                            detected = m.get("detected_angle_traj_perp_deg")
                            fitted = m.get("fitted_angle_traj_perp_deg")
                            loc = m.get("location_mm")
                            conf = m.get("confidence")
                            parts = [f"Marker {label}:"]
                            if loc is not None:
                                parts.append(f"{loc:.1f}mm")
                            if detected is not None:
                                parts.append(f"detected {detected:.1f}\u00b0")
                            if fitted is not None:
                                parts.append(f"fitted {fitted:.1f}\u00b0")
                            if conf is not None:
                                parts.append(f"(conf {conf:.2f})")
                            print(f"    {' '.join(parts)}")
                    sep = od.get("marker_pair_angular_separation_deg")
                    valid = od.get("marker_pair_valid")
                    if sep is not None:
                        status = "valid" if valid else "invalid"
                        print(f"    Separation: {sep:.1f}\u00b0 ({status})")

        # Generate HTML report if requested
        if args.html and electrodes:
            print("\nGenerating HTML report...")
            try:
                from pypacer._version import __version__ as PYPACER_VERSION
                from pypacer.visualization.report import (
                    generate_html_report_from_data,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = output_dir / f"pypacer_report_{timestamp}.html"

                # Prepare metadata
                metadata = {
                    "ct_file": str(ct_path),
                    "timestamp": datetime.now().isoformat(),
                    "pypacer_version": PYPACER_VERSION,
                    "voxel_sizes_mm": pacer.voxel_sizes.tolist(),
                    "metal_threshold_HU": args.metal_threshold,
                    "num_electrodes_detected": len(electrodes),
                    "use_gpu": True,
                }

                # Prepare reconstruction parameters (matching what's saved in JSON)
                reconstruction_parameters = {
                    "method": (
                        "detect_electrodes"
                        if args.brain_mask
                        else "detect_electrodes_radial"
                    ),
                    "detection_method": (
                        "brain_mask" if args.brain_mask else "radial_search"
                    ),
                    "contact_detection_method": args.contact_method,
                    "electrode_type": args.electrode_type,
                    "xy_resolution": xy_resolution,
                    "z_resolution": z_resolution,
                    "grid_size": args.grid_size,
                    "final_degree": 3,
                    "metal_threshold": args.metal_threshold,
                    "processing_type": "GPU",
                    "interface": "CLI",
                    "display_profiles": False,
                    "use_gpu": True,
                    "gpu_available": pacer.gpu_available,
                    "debug_output_enabled": args.debug,
                    "search_radii_mm": (
                        args.search_radii if not args.brain_mask else None
                    ),
                }

                # Generate report directly from electrode data
                generate_html_report_from_data(
                    electrodes=electrodes,
                    metadata=metadata,
                    reconstruction_parameters=reconstruction_parameters,
                    output_path=str(report_path),
                )
                print(f"HTML report generated: {report_path}")
            except Exception as e:
                print(f"Warning: Failed to generate HTML report: {e}")

        print(f"\nAll results saved to: {output_dir}")

    except Exception as e:
        print(f"Error during reconstruction: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up logging
        if log_file is not None:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
