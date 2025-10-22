#!/usr/bin/env python3
"""
Transform electrode contact positions to template space using ANTs transforms.

This script reads a PyPaCER reconstruction JSON file and applies ANTs transformations
to map the contact positions from native space to template/atlas space.

Note: Points are transformed in the OPPOSITE direction of images, so we use the
inverse transform (as documented in antsApplyTransformsToPoints).

Usage:
    python transform_contacts_ants.py reconstruction.json output0GenericAffine.mat [--output transformed.json]

Example:
    python transform_contacts_ants.py results/reconstruction.json ants/output0GenericAffine.mat --output results/reconstruction_template.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Transform electrode contacts to template space using ANTs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (saves to reconstruction_transformed.json)
  python transform_contacts_ants.py reconstruction.json output0GenericAffine.mat

  # Specify output file
  python transform_contacts_ants.py reconstruction.json output0GenericAffine.mat --output transformed.json

  # With multiple transforms (applied in reverse order, like ANTs)
  python transform_contacts_ants.py reconstruction.json output0GenericAffine.mat output1Warp.nii.gz --output transformed.json

  # Chain transforms from Image A -> B -> C (electrodes in A, want them in C)
  # Have: transform A->B, transform C->B
  # Use inverse for A->B (index 0), forward for C->B (index 1)
  python transform_contacts_ants.py reconstruction.json transform_AtoB.mat transform_CtoB.mat --forward-for 1 --output transformed.json

Notes:
  - Transforms are applied in REVERSE order (last transform first)
  - Affine transforms (.mat) use INVERSE by default (correct for point transforms)
  - Deformation fields (.nii.gz) are automatically used in FORWARD direction
    - If you need inverse deformation, use the InverseWarp file (e.g., output1InverseWarp.nii.gz)
  - Use --forward-for to manually override specific transforms (rarely needed)
  - Supports both affine (.mat) and deformation field (.nii.gz) transforms
  - Requires antspyx: pip install antspyx
        """
    )

    parser.add_argument('reconstruction_json', type=str,
                       help='Path to PyPaCER reconstruction JSON file')
    parser.add_argument('transforms', type=str, nargs='+',
                       help='ANTs transform file(s) - can be .mat (affine) or .nii.gz (warp)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSON file (default: <input>_transformed.json)')
    parser.add_argument('--inverse', action='store_true',
                       help='Use forward transform instead of inverse for ALL transforms (advanced use only)')
    parser.add_argument('--forward-for', type=int, nargs='+', metavar='INDEX',
                       help='Indices of transforms to use in FORWARD direction (0-based). '
                            'Example: --forward-for 1 means use forward for the 2nd transform. '
                            'By default, all transforms use inverse (for point transforms).')

    args = parser.parse_args()

    # Validate inputs
    reconstruction_json = Path(args.reconstruction_json)

    if not reconstruction_json.exists():
        print(f"Error: Reconstruction file not found: {reconstruction_json}")
        sys.exit(1)

    # Check all transform files exist
    transform_files = [Path(t) for t in args.transforms]
    for tf in transform_files:
        if not tf.exists():
            print(f"Error: Transform file not found: {tf}")
            sys.exit(1)

    # Set output path
    if args.output:
        output_json = Path(args.output)
    else:
        output_json = reconstruction_json.parent / f"{reconstruction_json.stem}_transformed.json"

    # Import transformation utilities
    try:
        from pypacer.utils.coordinate_transforms import transform_reconstruction_contacts
    except ImportError as e:
        print(f"Error: Could not import PyPaCER utilities: {e}")
        print("\nMake sure PyPaCER is installed or you're running from the project directory.")
        sys.exit(1)

    # Load reconstruction data
    print("=" * 60)
    print("PyPaCER Contact Transform to Template Space")
    print("=" * 60)
    print(f"Loading reconstruction from: {reconstruction_json}")

    with open(reconstruction_json, 'r') as f:
        data = json.load(f)

    print(f"Number of electrodes: {len(data['electrodes'])}")
    print(f"Transform files: {[str(t) for t in transform_files]}")

    # Build per-transform inverse flags
    if args.forward_for is not None:
        # Check for conflicts
        if args.inverse:
            print("Error: Cannot use both --inverse and --forward-for")
            sys.exit(1)

        # Validate indices
        for idx in args.forward_for:
            if idx < 0 or idx >= len(transform_files):
                print(f"Error: Invalid transform index {idx}. Must be 0-{len(transform_files)-1}")
                sys.exit(1)

        # Build list: True (inverse) by default, False (forward) for specified indices
        use_inverse_list = [True] * len(transform_files)
        for idx in args.forward_for:
            use_inverse_list[idx] = False

        print(f"Transform directions: {['INVERSE' if inv else 'FORWARD' for inv in use_inverse_list]}")
        use_inverse_param = use_inverse_list
    else:
        # Use same setting for all transforms (old behavior)
        use_inverse_param = not args.inverse  # Default is to use inverse
        direction = "INVERSE" if use_inverse_param else "FORWARD"
        print(f"Transform direction: {direction} (all transforms)")

    print()

    # Transform all contacts using utility function
    try:
        data, total_contacts_transformed = transform_reconstruction_contacts(
            data,
            transform_files,
            use_inverse=use_inverse_param
        )

        # Print summary for each electrode
        for elec_idx, electrode in enumerate(data['electrodes']):
            electrode_type = electrode.get('electrode_type', 'Unknown')
            print(f"Electrode {elec_idx + 1} ({electrode_type}):")

            if 'contact_positions_3d' in electrode:
                num_contacts = len(electrode['contact_positions_3d'])
                print(f"  ✓ Transformed {num_contacts} contacts")

                # Show first contact as example
                if 'contact_positions_3d_native' in electrode:
                    native = electrode['contact_positions_3d_native'][0]
                    transformed = electrode['contact_positions_3d'][0]
                    print(f"    Example: Contact 1 native: {native}")
                    print(f"             Contact 1 template: {transformed}")
            print()

    except Exception as e:
        print(f"Error transforming contacts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save transformed reconstruction
    print()
    print(f"Saving transformed reconstruction to: {output_json}")

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print()
    print("=" * 60)
    print(f"✓ Transformed {total_contacts_transformed} contacts from {len(data['electrodes'])} electrodes")
    print(f"  Output: {output_json.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
