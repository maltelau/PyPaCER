"""Command line interface for electrode detection GUI."""

import argparse
import sys
from pathlib import Path

from ..gui.pypacer_gui import PyPaCERGUI


def main():
    """Main function for electrode detection GUI CLI."""
    parser = argparse.ArgumentParser(
        description="Interactive electrode detection GUI for PyPaCER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run python -m pypacer.cli.pypacer_gui patient_ct.nii.gz
  
  # With custom threshold
  uv run python -m pypacer.cli.pypacer_gui patient_ct.nii.gz --threshold 1500
  
  # Different slice view
  uv run python -m pypacer.cli.pypacer_gui patient_ct.nii.gz --axis sagittal

Usage Instructions:
  - Left click: Add seed point for electrode detection
  - Right click: Remove nearest seed point
  - Scroll wheel: Navigate through slices
  - Hover: Preview intensity and metal detection
  - Green rectangle: Metal threshold exceeded (good seed point)
  - Red rectangle: Below metal threshold
  
Detection Workflow:
  1. Navigate to slices showing electrode artifacts
  2. Click on high-intensity electrode regions to add seed points
  3. Adjust threshold if needed (default 2000 HU)
  4. Select detection mode (Fast/Normal/High Quality)
  5. Choose electrode type or leave as Auto
  6. Click "Detect" to run electrode reconstruction
  7. Results are shown in 3D view and can be saved to JSON
        """,
    )

    parser.add_argument(
        "ct_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to CT NIfTI file (optional - can load from GUI)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=2000,
        help="Metal detection threshold in Hounsfield Units (default: 2000)",
    )

    parser.add_argument(
        "--axis",
        choices=["axial", "sagittal", "coronal"],
        default="axial",
        help="Initial slice view orientation (default: axial)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for saving JSON and HTML reports (default: <CT_dir>/pypacer/)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for additional outputs and verbose logging",
    )

    args = parser.parse_args()

    # Validate CT file if provided
    ct_path = None
    if args.ct_path:
        ct_path = Path(args.ct_path)
        if not ct_path.exists():
            print(f"Error: CT file not found: {ct_path}")
            sys.exit(1)

        if ct_path.suffix.lower() not in [".nii", ".gz"]:
            print(
                f"Warning: Expected NIfTI file (.nii or .nii.gz), got: {ct_path.suffix}"
            )

    # Validate and create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    print("Starting electrode detection GUI...")
    if ct_path:
        print(f"CT file: {ct_path}")
    else:
        print("No CT file provided - use Load CT button in GUI")
    print(f"Metal threshold: {args.threshold} HU")
    print(f"Initial view: {args.axis}")
    if args.debug:
        print("Debug mode: ENABLED")
    print()

    try:
        # Initialize and run GUI
        gui = PyPaCERGUI(
            ct_path=ct_path,
            metal_threshold=args.threshold,
            slice_axis=args.axis,
            output_dir=output_dir,
            debug_mode=args.debug,
        )

        print("GUI initialized successfully. Starting interactive session...")
        electrodes = gui.run()

        # Summary
        print("\nFinal Results:")
        print(f"  Electrodes detected: {len(electrodes)}")
        for i, electrode in enumerate(electrodes):
            print(f"    Electrode {i+1}: {electrode.electrode_type}")
            print(f"      Length: {electrode.length_mm:.1f}mm")
            print(f"      Contacts: {len(electrode.contact_positions)}")
            print(
                f"      Tip: ({electrode.tip_position[0]:.1f}, "
                f"{electrode.tip_position[1]:.1f}, {electrode.tip_position[2]:.1f}) mm"
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
