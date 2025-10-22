"""
Coordinate transformation utilities for PyPaCER.

This module provides functions for transforming electrode coordinates between
different coordinate systems (RAS/LPS) and applying spatial transformations
(e.g., ANTs registration transforms).
"""

from pathlib import Path

import numpy as np


def ras_to_lps(points):
    """
    Convert points from RAS to LPS coordinate system.

    RAS (Right-Anterior-Superior) is used by NIfTI and neuroimaging software.
    LPS (Left-Posterior-Superior) is used by ITK and ANTs.

    Args:
        points: Nx3 array of points in RAS coordinates

    Returns:
        Nx3 array of points in LPS coordinates
    """
    points = np.atleast_2d(points)
    points_lps = points.copy()
    points_lps[:, 0] = -points[:, 0]  # R -> L (flip X)
    points_lps[:, 1] = -points[:, 1]  # A -> P (flip Y)
    # Z stays the same (S -> S)
    return points_lps


def lps_to_ras(points):
    """
    Convert points from LPS to RAS coordinate system.

    LPS (Left-Posterior-Superior) is used by ITK and ANTs.
    RAS (Right-Anterior-Superior) is used by NIfTI and neuroimaging software.

    Args:
        points: Nx3 array of points in LPS coordinates

    Returns:
        Nx3 array of points in RAS coordinates
    """
    points = np.atleast_2d(points)
    points_ras = points.copy()
    points_ras[:, 0] = -points[:, 0]  # L -> R (flip X)
    points_ras[:, 1] = -points[:, 1]  # P -> A (flip Y)
    # Z stays the same (S -> S)
    return points_ras


def apply_ants_transforms_to_points(
    points, transform_files, use_inverse=True, input_coordinate_system="RAS"
):
    """
    Apply ANTs transformations to points with automatic coordinate system handling.

    This function handles the coordinate system conversion between RAS (used by
    PyPaCER/NIfTI) and LPS (used by ANTs/ITK) automatically.

    Args:
        points: Nx3 array of points in physical coordinates
        transform_files: List of paths to ANTs transform files (.mat or .nii.gz)
        use_inverse: Whether to use inverse transform. Can be:
            - bool: Apply same setting to all transforms (default True for points)
            - list of bool: Per-transform inverse flags (must match length of transform_files)
            Note: Deformation fields (.nii, .nii.gz, .nrrd) are automatically detected and
            will NOT be inverted regardless of this setting. Use InverseWarp files explicitly.
        input_coordinate_system: 'RAS' or 'LPS' (default 'RAS')

    Returns:
        Nx3 array of transformed points in the same coordinate system as input

    Raises:
        ImportError: If antspyx or pandas is not installed
        ValueError: If points are not Nx3 or coordinate system is invalid

    Example:
        >>> contacts = np.array([[10, 20, 30], [11, 21, 31]])
        >>> transforms = ['output0GenericAffine.mat']
        >>> transformed = apply_ants_transforms_to_points(contacts, transforms)

        >>> # Use inverse for first transform, forward for second
        >>> transformed = apply_ants_transforms_to_points(
        ...     contacts,
        ...     ['transform_AtoB.mat', 'transform_CtoB.mat'],
        ...     use_inverse=[True, False]
        ... )
    """
    try:
        import ants
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            f"Missing required package: {e}. "
            "Install with: pip install antspyx pandas"
        )

    # Validate inputs
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got shape {points.shape}")

    if input_coordinate_system not in ["RAS", "LPS"]:
        raise ValueError(
            f"Coordinate system must be 'RAS' or 'LPS', got {input_coordinate_system}"
        )

    # Convert to LPS if input is RAS (ANTs expects LPS)
    if input_coordinate_system == "RAS":
        points_lps = ras_to_lps(points)
    else:
        points_lps = points.copy()

    # Create DataFrame for ANTs
    points_df = pd.DataFrame(points_lps, columns=["x", "y", "z"])

    # Convert transform files to strings
    transform_files = [Path(f) for f in transform_files]
    transforms_str = [str(f) for f in transform_files]

    # Validate transform files exist
    for tf in transform_files:
        if not tf.exists():
            raise FileNotFoundError(f"Transform file not found: {tf}")

    # Handle use_inverse parameter - convert to list if needed
    if isinstance(use_inverse, bool):
        # Same inverse setting for all transforms
        which_to_invert = [use_inverse] * len(transforms_str)
    elif isinstance(use_inverse, (list, tuple)):
        # Per-transform inverse settings
        if len(use_inverse) != len(transforms_str):
            raise ValueError(
                f"Length of use_inverse ({len(use_inverse)}) must match "
                f"number of transforms ({len(transforms_str)})"
            )
        which_to_invert = list(use_inverse)
    else:
        raise TypeError(
            f"use_inverse must be bool or list of bool, got {type(use_inverse)}"
        )

    # Auto-detect deformation fields and disable inversion
    # Deformation fields (.nii, .nii.gz, .nrrd) cannot be inverted by ANTs
    # User should provide the appropriate warp (forward or inverse) directly
    for i, tf in enumerate(transform_files):
        tf_lower = str(tf).lower()
        is_deformation = any(
            tf_lower.endswith(ext) for ext in [".nii", ".nii.gz", ".nrrd"]
        )

        if is_deformation and which_to_invert[i]:
            # Warn and disable inversion for deformation fields
            import warnings

            warnings.warn(
                f"Transform {i} ({tf.name}) appears to be a deformation field. "
                f"Deformation fields cannot be inverted - using forward direction. "
                f"If you need the inverse transformation, use the InverseWarp file instead.",
                UserWarning,
            )
            which_to_invert[i] = False

    # Apply transforms
    # Note: transforms are applied in REVERSE order (ANTs convention)
    transformed_df = ants.apply_transforms_to_points(
        dim=3,
        points=points_df,
        transformlist=transforms_str,
        whichtoinvert=which_to_invert,
    )

    # Convert back to numpy array
    transformed_points_lps = transformed_df[["x", "y", "z"]].values

    # Convert back to original coordinate system if needed
    if input_coordinate_system == "RAS":
        transformed_points = lps_to_ras(transformed_points_lps)
    else:
        transformed_points = transformed_points_lps

    return transformed_points


def transform_reconstruction_contacts(
    reconstruction_data, transform_files, use_inverse=True
):
    """
    Transform all electrode contacts in a PyPaCER reconstruction using ANTs transforms.

    This function modifies the reconstruction data in-place, adding transformed
    coordinates and preserving original coordinates with '_native' suffix.

    Args:
        reconstruction_data: Dictionary containing PyPaCER reconstruction data
        transform_files: List of paths to ANTs transform files
        use_inverse: Whether to use inverse transform. Can be:
            - bool: Apply same setting to all transforms (default True for points)
            - list of bool: Per-transform inverse flags (must match length of transform_files)

    Returns:
        Tuple of (updated reconstruction data, total_contacts_transformed)

    Example:
        >>> with open('reconstruction.json') as f:
        ...     data = json.load(f)
        >>> transform_reconstruction_contacts(data, ['transform.mat'])
        >>> # data now contains transformed coordinates

        >>> # Chain transforms A->B->C: use inverse for A->B, forward for C->B
        >>> transform_reconstruction_contacts(
        ...     data,
        ...     ['transform_AtoB.mat', 'transform_CtoB.mat'],
        ...     use_inverse=[True, False]
        ... )
    """
    total_contacts = 0

    for electrode in reconstruction_data.get("electrodes", []):
        # Transform contact positions
        if "contact_positions_3d" in electrode:
            contacts = np.array(electrode["contact_positions_3d"])
            transformed = apply_ants_transforms_to_points(
                contacts, transform_files, use_inverse=use_inverse
            )

            # Store original and transformed
            electrode["contact_positions_3d_native"] = electrode["contact_positions_3d"]
            electrode["contact_positions_3d"] = transformed.tolist()
            total_contacts += len(contacts)

        # Transform tip position
        if "tip_position" in electrode:
            tip = np.array(electrode["tip_position"]).reshape(1, 3)
            transformed_tip = apply_ants_transforms_to_points(
                tip, transform_files, use_inverse=use_inverse
            )
            electrode["tip_position_native"] = electrode["tip_position"]
            electrode["tip_position"] = transformed_tip[0].tolist()

        # Transform entry position
        if "entry_position" in electrode:
            entry = np.array(electrode["entry_position"]).reshape(1, 3)
            transformed_entry = apply_ants_transforms_to_points(
                entry, transform_files, use_inverse=use_inverse
            )
            electrode["entry_position_native"] = electrode["entry_position"]
            electrode["entry_position"] = transformed_entry[0].tolist()

    # Update metadata
    if "metadata" not in reconstruction_data:
        reconstruction_data["metadata"] = {}

    reconstruction_data["metadata"]["transformed_to_template"] = True
    reconstruction_data["metadata"]["transform_files"] = [
        str(f) for f in transform_files
    ]
    reconstruction_data["metadata"]["transform_inverse_used"] = use_inverse

    return reconstruction_data, total_contacts
