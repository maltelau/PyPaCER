#!/usr/bin/env python3
"""
PyPaCER - Electrode reconstruction from CT scans.

Usage:
    pypacer <ct_file> [options]

Example:
    pypacer ct_scan.nii.gz --output-dir output/
    pypacer ct_scan.nii.gz --fast
    pypacer ct_scan.nii.gz --detection-method brain_mask_auto
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pypacer.core.pypacer import PyPaCER


def main(argv=None):
    """Main entry point for the PyPaCER CLI."""
    parser = argparse.ArgumentParser(
        description="DBS electrode reconstruction from CT scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection Methods:
  radial_search (default)   - Searches outward from brain center (most robust)
  brain_mask_auto          - Automatic brain extraction + metal detection
  brain_mask_custom        - Use provided brain mask + metal detection

Examples:
  # Basic usage with radial search (default)
  pypacer ct_scan.nii.gz
  
  # Fast mode with reduced quality
  pypacer ct_scan.nii.gz --fast
  
  # Use automatic brain mask method
  pypacer ct_scan.nii.gz --brain-mask
  
  # Custom brain mask
  pypacer ct_scan.nii.gz --brain-mask brain_mask.nii.gz
  
  # Specify electrode type and output
  pypacer ct_scan.nii.gz --electrode-type "Medtronic 3389" --output-dir results/
        """,
    )

    # Required arguments
    parser.add_argument("ct_file", type=str, help="Path to the CT NIfTI file")

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as CT file)",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "mat", "hdf5"],
        help="Output format for results (default: json)",
    )

    # Detection method
    parser.add_argument(
        "--detection-method",
        type=str,
        default="radial_search",
        choices=["radial_search", "brain_mask_auto", "brain_mask_custom"],
        help="Method for finding electrodes (default: radial_search)",
    )

    parser.add_argument(
        "--brain-mask",
        nargs="?",
        const="auto",
        default=None,
        help="Use brain mask method. If no file provided, uses automatic extraction. If file provided, uses that mask.",
    )

    parser.add_argument(
        "--search-radii",
        type=float,
        nargs="+",
        default=None,
        help="Search radii in mm for radial_search method (default: 30 40 50)",
    )

    # Processing parameters
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
        choices=["contactAreaCenter", "peak", "peakWaveCenter"],
        help="Contact detection method (default: contactAreaCenter)",
    )

    parser.add_argument(
        "--electrode-type",
        type=str,
        default=None,
        help='Force specific electrode type (e.g., "Medtronic 3389")',
    )

    # Quality settings
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode with reduced quality settings"
    )

    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="High quality mode with finer resolution",
    )

    parser.add_argument(
        "--xy-resolution",
        type=float,
        default=None,
        help="XY resolution for refinement in mm",
    )

    parser.add_argument(
        "--z-resolution",
        type=float,
        default=None,
        help="Z resolution for refinement in mm",
    )

    # Other options
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save results to file"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug outputs (orthogonal grids, all contact detection methods)",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report with reconstruction results",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    ct_path = Path(args.ct_file)
    if not ct_path.exists():
        print(f"Error: CT file not found: {ct_path}")
        return 1

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
        log_filename = output_dir / f"pypacer_{timestamp}.log"

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

    # Handle brain mask argument to set detection method
    if args.brain_mask is not None:
        if args.brain_mask == "auto":
            # --brain-mask with no file: use auto
            args.detection_method = "brain_mask_auto"
            args.brain_mask = None
        else:
            # --brain-mask with file: use custom
            args.detection_method = "brain_mask_custom"
            # Validate the file exists
            mask_path = Path(args.brain_mask)
            if not mask_path.exists():
                print(f"Error: Brain mask file not found: {mask_path}")
                return 1

    # Start processing
    print(f"\n{'='*60}")
    print("PyPaCER - Electrode Reconstruction")
    print(f"{'='*60}")
    print(f"CT File: {ct_path}")
    print(f"Detection Method: {args.detection_method}")
    print(f"Output Directory: {output_dir}")

    if args.fast:
        print("Mode: Fast (reduced quality)")
    elif args.high_quality:
        print("Mode: High Quality")
    else:
        print("Mode: Normal")

    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        # Initialize PyPaCER
        pypacer = PyPaCER(
            ct_path=str(ct_path),
            brain_mask=args.brain_mask,
            metal_threshold=args.metal_threshold,
            debug_output_dir=output_dir if args.debug else None,
            output_dir=output_dir,
        )

        # Set resolution parameters based on mode
        if args.fast:
            # Fast mode - reduced quality for speed
            xy_res = args.xy_resolution or 0.3  # 3x coarser than normal
            z_res = args.z_resolution or 0.1  # 4x coarser than normal
            grid_size = 1.0  # Smaller grid
        elif args.high_quality:
            # High quality mode - finer resolution
            xy_res = args.xy_resolution or 0.05  # 2x finer than normal
            z_res = args.z_resolution or 0.01  # 2.5x finer than normal
            grid_size = 2.0  # Standard grid
        else:
            # Normal mode - balanced settings
            xy_res = args.xy_resolution or 0.1
            z_res = args.z_resolution or 0.025
            grid_size = 1.5  # Standard grid

        # Detect electrodes with appropriate parameters
        if args.detection_method == "radial_search":
            # Use detect_electrodes_radial which accepts resolution parameters
            electrodes = pypacer.detect_electrodes_radial(
                contact_detection_method=args.contact_method,
                electrode_type=args.electrode_type,
                xy_resolution=xy_res,
                z_resolution=z_res,
                grid_size=grid_size,
                auto_save_json=not args.no_save,
                search_radii_mm=args.search_radii,
                max_electrodes=4,
                verbose=True,  # Always show progress for radial search
            )
        else:
            # Use detect_electrodes for brain mask methods
            electrodes = pypacer.detect_electrodes(
                contact_detection_method=args.contact_method,
                electrode_type=args.electrode_type,
                xy_resolution=xy_res,
                z_resolution=z_res,
                grid_size=grid_size,
                auto_save_json=not args.no_save,
                detection_method=args.detection_method,
                search_radii_mm=args.search_radii,
            )

        elapsed_time = time.time() - start_time

        # Print results
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Processing time: {elapsed_time:.1f}s")
        print(f"Electrodes detected: {len(electrodes)}")

        for i, electrode in enumerate(electrodes):
            print(f"\nElectrode {i+1}:")
            print(f"  Type: {electrode.electrode_type}")

            # Get trajectory length
            total_length = electrode.length_mm
            print(f"  Trajectory length: {total_length:.1f}mm")

            # Get tip and entry points
            tip_point = electrode.get_point_at_parameter(0.0)
            entry_point = electrode.get_point_at_parameter(1.0)
            print(
                f"  Tip point: [{tip_point[0]:.2f}, {tip_point[1]:.2f}, {tip_point[2]:.2f}]"
            )
            print(
                f"  Entry point: [{entry_point[0]:.2f}, {entry_point[1]:.2f}, {entry_point[2]:.2f}]"
            )

            if (
                electrode.contact_positions is not None
                and len(electrode.contact_positions) > 0
            ):
                print(f"  Contacts: {len(electrode.contact_positions)}")
                print("  Contact positions:")

                # Get 3D positions for each contact
                contact_positions_3d = electrode.get_contact_positions_3d()

                for j, (pos_mm, pos_3d) in enumerate(
                    zip(electrode.contact_positions, contact_positions_3d)
                ):
                    print(f"    Contact {j+1}: {pos_mm:.2f}mm from tip")
                    print(
                        f"      Coordinates: [{pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f}]"
                    )
            else:
                print("  Contacts: Failed to detect")

        # Save results if requested
        if not args.no_save and args.output_format != "json":  # JSON already saved
            output_file = (
                output_dir / f"{ct_path.stem}_reconstruction.{args.output_format}"
            )
            pypacer.export_results(output_file, format=args.output_format)
            print(f"\nResults saved to: {output_file}")

        # Generate HTML report if requested
        if args.html and electrodes:
            print("\nGenerating HTML report...")
            try:
                from pypacer._version import __version__ as PYPACER_VERSION
                from pypacer.visualization.report_generator import (
                    generate_html_report_from_data,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = output_dir / f"pypacer_report_{timestamp}.html"

                # Prepare metadata
                metadata = {
                    "ct_file": str(ct_path),
                    "timestamp": datetime.now().isoformat(),
                    "pypacer_version": PYPACER_VERSION,
                    "voxel_sizes_mm": pypacer.voxel_sizes.tolist(),
                    "metal_threshold_HU": args.metal_threshold,
                    "num_electrodes_detected": len(electrodes),
                }

                # Prepare reconstruction parameters (matching what's saved in JSON)
                reconstruction_parameters = {
                    "method": (
                        "detect_electrodes_radial"
                        if args.detection_method == "radial_search"
                        else "detect_electrodes"
                    ),
                    "detection_method": args.detection_method,
                    "contact_detection_method": args.contact_method,
                    "electrode_type": args.electrode_type,
                    "xy_resolution": xy_res,
                    "z_resolution": z_res,
                    "grid_size": grid_size,
                    "final_degree": 3,
                    "metal_threshold": args.metal_threshold,
                    "refinement_threshold": 800,
                    "processing_type": "CPU",
                    "interface": "CLI",
                    "display_profiles": False,
                    "use_gpu": False,
                    "debug_output_enabled": args.debug,
                }
                if args.detection_method == "radial_search":
                    reconstruction_parameters["search_radii_mm"] = [30, 40, 50]

                # Generate report directly from electrode data
                generate_html_report_from_data(
                    electrodes=electrodes,
                    metadata=metadata,
                    reconstruction_parameters=reconstruction_parameters,
                    output_path=str(report_path),
                )
                print(f"HTML report saved to: {report_path}")
            except Exception as e:
                print(f"Failed to generate HTML report: {e}")

        print("\n✓ Processing completed successfully")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        # Clean up logging
        if log_file is not None:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
