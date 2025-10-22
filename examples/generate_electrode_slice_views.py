#!/usr/bin/env python3
"""
Generate orthogonal slice view images along electrode trajectories.

For each electrode, this script generates 2 PNG images showing orthogonal planes
along the electrode axis, with all contacts visible and marked with crosshairs.

Usage:
    python generate_electrode_slice_views.py reconstruction.json ct_image.nii.gz [--output-dir slices/]

Example:
    python generate_electrode_slice_views.py results/reconstruction.json data/ct.nii.gz --output-dir visualizations/
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    parser = argparse.ArgumentParser(
        description='Generate orthogonal cross-section images along electrodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (saves to current directory)
  python generate_electrode_slice_views.py reconstruction.json ct_scan.nii.gz

  # Specify output directory
  python generate_electrode_slice_views.py reconstruction.json ct_scan.nii.gz --output-dir slices/

  # Adjust plane dimensions
  python generate_electrode_slice_views.py reconstruction.json ct_scan.nii.gz --width 40.0 --height 60.0

  # Use jet colormap
  python generate_electrode_slice_views.py reconstruction.json ct_scan.nii.gz --colormap jet

  # Use viridis colormap with custom window/level
  python generate_electrode_slice_views.py reconstruction.json ct_scan.nii.gz --cmap viridis --clim 500 2500

Notes:
  - Generates 2 orthogonal planes per electrode (lateral and frontal views)
  - Both planes span the entire electrode length showing all contacts
  - Circular crosshairs mark each contact position
  - Output: PNG files named electrode_<N>_lateral.png and electrode_<N>_frontal.png
        """
    )

    parser.add_argument('reconstruction_json', type=str,
                       help='Path to PyPaCER reconstruction JSON file')
    parser.add_argument('ct_file', type=str,
                       help='Path to CT NIfTI file')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                       help='Output directory for PNG files (default: current directory)')
    parser.add_argument('--width', type=float, default=32.0,
                       help='Width of plane perpendicular to electrode in mm (default: 32.0)')
    parser.add_argument('--height', type=float, default=48.0,
                       help='Height of plane along electrode in mm (default: 48.0)')
    parser.add_argument('--resolution', type=float, default=0.1,
                       help='Sampling resolution in mm (default: 0.1)')
    parser.add_argument('--crosshair-radius', type=float, default=0.1,
                       help='Radius of crosshair circles in mm (default: 0.1)')
    parser.add_argument('--clim', type=float, nargs=2, default=[0, 3000],
                       help='Color limits for CT display in HU (default: 0 3000)')
    parser.add_argument('--colormap', '--cmap', type=str, default='gray',
                       help='Colormap for CT display (default: gray). Options: gray, jet, viridis, hot, bone, etc.')

    args = parser.parse_args()

    # Validate inputs
    reconstruction_json = Path(args.reconstruction_json)
    ct_path = Path(args.ct_file)
    output_dir = Path(args.output_dir)

    if not reconstruction_json.exists():
        print(f"Error: Reconstruction file not found: {reconstruction_json}")
        sys.exit(1)

    if not ct_path.exists():
        print(f"Error: CT file not found: {ct_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("PyPaCER Electrode Cross-Section Generator")
    print("=" * 60)
    print(f"Loading reconstruction from: {reconstruction_json}")
    print(f"Loading CT data from: {ct_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        import nibabel as nib
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install nibabel scipy matplotlib")
        sys.exit(1)

    # Import PyPaCER visualization utilities
    try:
        from pypacer.visualization import generate_orthogonal_slice_views
    except ImportError as e:
        print(f"Error: Could not import PyPaCER visualization utilities: {e}")
        print("\nMake sure PyPaCER is installed or you're running from the project directory.")
        sys.exit(1)

    # Load reconstruction
    with open(reconstruction_json, 'r') as f:
        data = json.load(f)

    # Load CT data
    ct_nii = nib.load(str(ct_path))
    ct_data = ct_nii.get_fdata()
    affine = ct_nii.affine

    print(f"CT shape: {ct_data.shape}")
    print(f"Number of electrodes: {len(data['electrodes'])}")
    print()

    # Process each electrode
    total_images = 0
    for elec_idx, electrode in enumerate(data['electrodes']):
        electrode_type = electrode.get('electrode_type', 'Unknown')
        print(f"Processing Electrode {elec_idx + 1} ({electrode_type})...")

        # Get contact positions
        if 'contact_positions_3d' not in electrode:
            print(f"  Warning: Missing contact positions, skipping.")
            continue

        contact_positions_3d = np.array(electrode['contact_positions_3d'])
        print(f"  Found {len(contact_positions_3d)} contacts")

        # Generate orthogonal slice views using utility function
        print(f"  Generating slice views...")
        views = generate_orthogonal_slice_views(
            ct_data,
            affine,
            contact_positions_3d,
            width=args.width,
            height=args.height,
            resolution=args.resolution
        )

        # Process each view (lateral and frontal)
        for plane_name in ['lateral', 'frontal']:
            print(f"  Rendering {plane_name} view...", end=' ')

            view_data = views[plane_name]
            intensity_map = view_data['intensity_map']
            extent = view_data['extent']
            contact_coords = view_data['contact_coords']

            # Create figure
            fig, ax = plt.subplots(figsize=(6, 10), dpi=150)

            # Display CT data
            im = ax.imshow(
                intensity_map,
                extent=[extent['u_min'], extent['u_max'], extent['v_min'], extent['v_max']],
                origin='lower',
                cmap=args.colormap,
                vmin=args.clim[0],
                vmax=args.clim[1],
                aspect='auto'
            )

            # Add crosshairs for each contact
            for contact_idx, (u_coord, v_coord) in enumerate(contact_coords):
                # Add crosshair circle
                crosshair_circle = Circle(
                    (u_coord, v_coord),
                    args.crosshair_radius,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.5
                )
                ax.add_patch(crosshair_circle)

                # Add contact label (further away from contact)
                ax.text(
                    u_coord + args.crosshair_radius + 1.5,
                    v_coord,
                    f'C{contact_idx + 1}',
                    color='red',
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='center'
                )

            # Labels and title
            ax.set_xlabel('Distance (mm)', fontsize=12)
            ax.set_ylabel('Distance along electrode (mm)', fontsize=12)
            ax.set_title(
                f'Electrode {elec_idx + 1} - {plane_name.capitalize()} View\n'
                f'{electrode_type}',
                fontsize=14,
                fontweight='bold'
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('HU', fontsize=11)

            # Grid
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

            # Add center line along electrode
            ax.axvline(0, color='yellow', linewidth=0.5, linestyle='--', alpha=0.5)

            # Save figure
            output_filename = f'electrode_{elec_idx + 1:02d}_{plane_name}.png'
            output_path = output_dir / output_filename
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            print(f"saved to {output_filename}")
            total_images += 1

    print()
    print("=" * 60)
    print(f"✓ Generated {total_images} cross-section images")
    print(f"  Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
