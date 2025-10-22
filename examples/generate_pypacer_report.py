#!/usr/bin/env python3
"""
Generate PyPaCER HTML report from reconstruction results.

This is a simple wrapper around the PyPaCER report generator CLI tool that creates
an HTML report with 3D volume renderings and rotating GIF animations.

Usage:
    python generate_pypacer_report.py reconstruction.json [--output report.html]

Example:
    python generate_pypacer_report.py results/patient001_reconstruction.json --output electrode_viz.html
"""

import argparse
import sys
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Generate PyPaCER HTML report from reconstruction JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (generates report in same directory as JSON)
  python generate_pypacer_report.py reconstruction.json

  # Specify output file
  python generate_pypacer_report.py reconstruction.json --output my_report.html

  # Full path example
  python generate_pypacer_report.py results/patient001_reconstruction.json --output reports/patient001_viz.html

Notes:
  - This script uses the PyPaCER report generator (pypacer-report command)
  - The reconstruction JSON must include the CT file path in its metadata
  - Requires PyVista for 3D rendering (install with: pip install pyvista)
  - Generates both static views and rotating GIF animations
  - All visualizations are embedded in the HTML (no external dependencies)
        """
    )

    parser.add_argument(
        'reconstruction_json',
        type=str,
        help='Path to PyPaCER reconstruction JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HTML file path (default: auto-generated in same directory as JSON)'
    )

    args = parser.parse_args()

    # Validate input
    reconstruction_json = Path(args.reconstruction_json)

    if not reconstruction_json.exists():
        print(f"Error: Reconstruction file not found: {reconstruction_json}")
        sys.exit(1)

    # Build command
    cmd = ['pypacer-report', str(reconstruction_json)]

    if args.output:
        cmd.extend(['--output', args.output])

    # Print info
    print("=" * 60)
    print("PyPaCER Report Generator")
    print("=" * 60)
    print(f"Reconstruction: {reconstruction_json}")
    if args.output:
        print(f"Output: {args.output}")
    print()
    print("Running pypacer-report...")
    print()

    # Run the report generator
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✓ PyPaCER report generated successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\nError: Report generation failed with exit code {e.returncode}")
        print("\nMake sure:")
        print("  1. PyPaCER is installed (pip install pypacer)")
        print("  2. PyVista is installed (pip install pyvista)")
        print("  3. The reconstruction JSON contains a valid CT file path")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: pypacer-report command not found")
        print("\nMake sure PyPaCER is installed:")
        print("  pip install pypacer")
        print("  # or")
        print("  uv pip install -e .")
        sys.exit(1)


if __name__ == '__main__':
    main()
