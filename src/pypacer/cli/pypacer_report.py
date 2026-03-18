#!/usr/bin/env python3
"""
PyPaCER Report Generator - Create interactive HTML reports from reconstruction results.

Usage:
    pypacer_report <reconstruction_json> [options]

Example:
    pypacer_report reconstruction_20240101_120000.json
    pypacer_report reconstruction.json --output my_report.html --no-3d
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pypacer.visualization.report import generate_html_report


def main():
    """Main entry point for the report generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML reports from PyPaCER reconstruction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report with default settings
  pypacer_report reconstruction.json
  
  # Specify output file
  pypacer_report reconstruction.json --output report.html
  
  # Generate report without 3D visualization
  pypacer_report reconstruction.json --no-3d
  
  # Generate minimal report (no intensity profiles or 3D)
  pypacer_report reconstruction.json --no-intensity --no-3d
        """,
    )

    # Required arguments
    parser.add_argument(
        "reconstruction_json", type=str, help="Path to the reconstruction JSON file"
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (auto-generated if not specified)",
    )

    parser.add_argument(
        "--no-intensity",
        action="store_true",
        help="Exclude intensity profile plots from report",
    )

    parser.add_argument(
        "--no-3d", action="store_true", help="Exclude 3D visualization from report"
    )

    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Exclude contact detection comparison if available",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.reconstruction_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.suffix == ".json":
        print(f"Error: Input file must be a JSON file: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate JSON content
    try:
        with open(input_path) as f:
            data = json.load(f)

        # Check for required fields
        if "electrodes" not in data:
            print(
                "Error: Invalid reconstruction JSON - missing 'electrodes' field",
                file=sys.stderr,
            )
            sys.exit(1)

        # Check if this is a minified JSON (missing data needed for reports)
        if "_mini" in input_path.stem:
            print(
                "Error: Cannot generate report from minified JSON (*_mini.json).\n"
                "       The mini JSON contains only core results (contacts, orientation)\n"
                "       and is missing intensity profiles and trajectory data needed for\n"
                "       the HTML report. Use the full JSON file instead.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate that electrodes have required data for report generation
        required_fields = ["polynomial", "contact_positions"]
        for i, electrode in enumerate(data["electrodes"]):
            missing = [f for f in required_fields if f not in electrode]
            if missing:
                print(
                    f"Error: Electrode {i+1} is missing required fields: {', '.join(missing)}.\n"
                    f"       This JSON may not contain enough data for report generation.",
                    file=sys.stderr,
                )
                sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate report
    try:
        if args.verbose:
            print(f"Generating report from: {input_path}")
            print(f"  Electrodes found: {len(data['electrodes'])}")
            if args.output:
                print(f"  Output path: {args.output}")

        output_path = generate_html_report(
            reconstruction_json_path=str(input_path),
            output_path=args.output,
            include_intensity_profiles=not args.no_intensity,
            include_3d_visualization=not args.no_3d,
            include_contact_comparison=not args.no_comparison,
        )

        print(f"Report generated successfully: {output_path}")

        # Print summary information
        if args.verbose:
            print("\nReport contents:")
            if not args.no_3d:
                print("  ✓ 3D electrode visualization")
            if not args.no_intensity:
                print("  ✓ Intensity profile plots")
            if not args.no_comparison:
                print("  ✓ Contact detection comparison (if available)")

    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
