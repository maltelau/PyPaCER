#!/usr/bin/env python3
"""
Run electrode detection with automatic orientation analysis.

After reconstructing electrodes from a post-operative CT, PyPaCER automatically
detects directional markers and classifies each electrode as directional
(Medtronic B33005/B33015) or non-directional (Medtronic 3389/3387).

For directional electrodes, marker orientation angles are determined and
stored in the output JSON alongside hemisphere labels for the tip and entry
positions.

Orientation detection parameters can be overridden via the orientation_params
dict passed to detect_electrodes(). See examples below.

Usage:
    python run_orientation_detection.py /path/to/ct_image.nii.gz

Example:
    python run_orientation_detection.py patient_ct_postop.nii.gz --output-dir results/
"""

import argparse
import json
import sys
from pathlib import Path

from pypacer import PyPaCER


def main():
    parser = argparse.ArgumentParser(
        description="Run electrode detection with orientation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (all defaults)
  python run_orientation_detection.py ct_postop.nii.gz

  # Specify output directory
  python run_orientation_detection.py ct_postop.nii.gz --output-dir results/

  # Use a brain mask for electrode detection
  python run_orientation_detection.py ct_postop.nii.gz --brain-mask mask.nii.gz

  # Override orientation detection parameters
  python run_orientation_detection.py ct_postop.nii.gz \\
      --orientation-radii 1.0 1.25 1.5 \\
      --orientation-angle-step 0.5 \\
      --marker-deviation-threshold 0.10
""",
    )
    parser.add_argument("ct_image", type=str, help="Path to post-operative CT NIfTI file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: same directory as CT)",
    )
    parser.add_argument(
        "--brain-mask",
        type=str,
        default=None,
        help="Optional brain mask NIfTI file",
    )

    # Orientation detection overrides
    orientation_group = parser.add_argument_group(
        "Orientation detection overrides",
        "Override default parameters for directional marker detection and orientation analysis.",
    )
    orientation_group.add_argument(
        "--orientation-radii",
        type=float,
        nargs="+",
        default=None,
        help="Circular sampling radii in mm (default: 1.25 1.5 1.75)",
    )
    orientation_group.add_argument(
        "--orientation-angle-step",
        type=float,
        default=None,
        help="Angular step in degrees for orientation sampling (default: 0.1)",
    )
    orientation_group.add_argument(
        "--orientation-smoothing",
        type=int,
        default=None,
        help="Smoothing window size for orientation profiles (default: 5)",
    )
    orientation_group.add_argument(
        "--marker-deviation-threshold",
        type=float,
        default=None,
        help="Minimum skeleton deviation for marker peak detection (default: 0.08)",
    )
    orientation_group.add_argument(
        "--marker-offset-mm",
        type=float,
        default=None,
        help="Distance above last contact where marker search begins (default: 2.5)",
    )
    orientation_group.add_argument(
        "--marker-max-distance-mm",
        type=float,
        default=None,
        help="Maximum distance from tip for marker search region (default: 20.0)",
    )
    orientation_group.add_argument(
        "--marker-min-separation",
        type=float,
        default=None,
        help="Minimum valid angular separation between markers in degrees (default: 120)",
    )
    orientation_group.add_argument(
        "--marker-max-separation",
        type=float,
        default=None,
        help="Maximum valid angular separation between markers in degrees (default: 150)",
    )
    orientation_group.add_argument(
        "--angular-constraint",
        type=float,
        default=None,
        help="Fixed angular separation for fitted marker directions in degrees (default: 120)",
    )

    # Debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (orientation visualisations, polar plots, etc.)",
    )

    args = parser.parse_args()

    ct_path = Path(args.ct_image)
    if not ct_path.exists():
        print(f"Error: CT file not found: {ct_path}")
        sys.exit(1)

    # Build orientation_params from CLI arguments
    # All supported keys documented in PyPaCER._run_orientation_detection()
    orientation_params = {}
    if args.orientation_radii is not None:
        orientation_params["radii_mm"] = args.orientation_radii
    if args.orientation_angle_step is not None:
        orientation_params["angle_increment_deg"] = args.orientation_angle_step
    if args.orientation_smoothing is not None:
        orientation_params["smoothing_window"] = args.orientation_smoothing
    if args.marker_deviation_threshold is not None:
        orientation_params["deviation_threshold"] = args.marker_deviation_threshold
    if args.marker_offset_mm is not None:
        orientation_params["marker_offset_mm"] = args.marker_offset_mm
    if args.marker_max_distance_mm is not None:
        orientation_params["max_distance_mm"] = args.marker_max_distance_mm
    if args.marker_min_separation is not None:
        orientation_params["min_separation_deg"] = args.marker_min_separation
    if args.marker_max_separation is not None:
        orientation_params["max_separation_deg"] = args.marker_max_separation
    if args.angular_constraint is not None:
        orientation_params["angular_constraint_deg"] = args.angular_constraint

    # Resolve output directory (needed for debug output path)
    output_dir = args.output_dir if args.output_dir else str(ct_path.parent)

    # Initialize PyPaCER
    pacer = PyPaCER(
        ct_path=str(ct_path),
        brain_mask=args.brain_mask,
        output_dir=output_dir,
        debug_output_dir=output_dir if args.debug else None,
    )

    # Run electrode detection with optional orientation parameter overrides.
    #
    # The orientation_params dict is forwarded to the internal orientation
    # detection pipeline. Any keys not provided use their defaults.
    #
    # Python API examples (without CLI):
    #
    #   # Default orientation detection (no overrides)
    #   electrodes = pacer.detect_electrodes()
    #
    #   # Custom sampling radii and angular step
    #   electrodes = pacer.detect_electrodes(
    #       orientation_params={
    #           "radii_mm": [1.0, 1.25, 1.5],
    #           "angle_increment_deg": 0.5,
    #       }
    #   )
    #
    #   # Relax marker pair validation for unusual electrode geometries
    #   electrodes = pacer.detect_electrodes(
    #       orientation_params={
    #           "min_separation_deg": 100.0,
    #           "max_separation_deg": 160.0,
    #       }
    #   )
    #
    #   # Increase deviation threshold to reduce false marker detections
    #   electrodes = pacer.detect_electrodes(
    #       orientation_params={
    #           "deviation_threshold": 0.12,
    #           "marker_offset_mm": 3.0,
    #       }
    #   )
    #
    #   # Also works with detect_electrodes_auto and detect_electrodes_radial:
    #   electrodes = pacer.detect_electrodes_auto(
    #       orientation_params={"radii_mm": [1.0, 1.5, 2.0]}
    #   )
    #
    electrodes = pacer.detect_electrodes(
        orientation_params=orientation_params or None,
    )

    if not electrodes:
        print("\nNo electrodes detected.")
        sys.exit(0)

    # Print results summary
    print("\n" + "=" * 60)
    print("ORIENTATION DETECTION RESULTS")
    print("=" * 60)

    for i, electrode in enumerate(electrodes):
        print(f"\nElectrode {i + 1}: {electrode.electrode_type}")
        print(f"  Contacts: {electrode.contact_positions}")
        print(f"  Tip position: {electrode.tip_position}")
        print(f"  Entry position: {electrode.entry_position}")

        if electrode.orientation_data:
            od = electrode.orientation_data
            print(f"  Classified type: {od['classified_electrode_type']}")
            print(f"  Detection method: {od['detection_method']}")
            print(f"  Detection confidence: {od['marker_detection_confidence']:.2f}")

            if od["has_markers"] and "markers" in od:
                print("  Directional markers detected:")
                for label, marker in od["markers"].items():
                    angle = marker["detected_angle_traj_perp_deg"]
                    fitted = marker.get("fitted_angle_traj_perp_deg")
                    conf = marker["detection_confidence"]
                    dist = marker["distance_from_tip_mm"]
                    print(
                        f"    Marker {label}: {dist:.1f}mm from tip, "
                        f"angle={angle:.1f} deg"
                        + (f" (fitted: {fitted:.1f} deg)" if fitted is not None else "")
                        + f", confidence={conf:.2f}"
                    )

                if "marker_pair_valid" in od:
                    sep = od["marker_pair_angular_separation_deg"]
                    valid = od["marker_pair_valid"]
                    print(
                        f"    Marker pair: {sep:.1f} deg separation "
                        f"({'valid' if valid else 'INVALID'})"
                    )
            else:
                print("  Non-directional electrode (no markers detected)")
        else:
            print("  Orientation detection: not available")

    print()


if __name__ == "__main__":
    main()
