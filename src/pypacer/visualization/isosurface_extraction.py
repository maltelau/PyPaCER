"""Simplified isosurface extraction utilities for 3D visualization of CT data."""

import time
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go


def _debug_print(message: str, start_time: Optional[float] = None):
    """Print debug message with timestamp."""
    pass


def extract_electrode_mesh(
    ct_path: str,
    output_dir: Optional[str] = None,
    electrode_data: Optional[Dict[str, Any]] = None,
) -> Optional[go.Mesh3d]:
    """Extract electrode metal artifacts mesh from CT data using HU thresholding.

    Args:
        ct_path: Path to the CT NIfTI file
        output_dir: Optional directory to save the mesh as OBJ file
        electrode_data: Optional electrode data dict containing contact_positions_3d

    Returns:
        Plotly Mesh3d object or None if extraction fails
    """
    try:
        import nibabel as nib
        from scipy import ndimage
        from skimage import measure

        start_time = time.time()
        _debug_print(f"Starting electrode mesh extraction from: {ct_path}", start_time)

        # Load CT data
        _debug_print("Loading CT data...", start_time)
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        affine = ct_nii.affine

        _debug_print(
            f"CT shape: {ct_data.shape}, range: [{ct_data.min():.1f}, {ct_data.max():.1f}] HU",
            start_time,
        )

        # Apply metal threshold
        metal_threshold = 1700
        _debug_print(f"Applying threshold at {metal_threshold} HU", start_time)
        metal_mask = ct_data > metal_threshold

        num_metal_voxels = metal_mask.sum()
        _debug_print(
            f"Metal voxels: {num_metal_voxels} ({100*num_metal_voxels/metal_mask.size:.3f}%)",
            start_time,
        )

        if num_metal_voxels == 0:
            _debug_print("No metal voxels found!", start_time)
            return None

        # Find connected components
        _debug_print("Finding connected components...", start_time)
        labeled = measure.label(metal_mask)
        num_components = labeled.max()
        _debug_print(f"Found {num_components} connected components", start_time)

        # Find largest component (simple approach)
        _debug_print("Selecting largest component...", start_time)
        component_sizes = []
        for label_id in range(1, num_components + 1):
            size = (labeled == label_id).sum()
            component_sizes.append((size, label_id))

        if not component_sizes:
            _debug_print("No components found!", start_time)
            return None

        component_sizes.sort(reverse=True)
        largest_label = component_sizes[0][1]
        _debug_print(
            f"Largest component: label={largest_label}, size={component_sizes[0][0]} voxels",
            start_time,
        )

        # Keep only largest component
        final_mask = labeled == largest_label

        # Optional: mild smoothing
        _debug_print("Applying Gaussian smoothing...", start_time)
        mask_smooth = ndimage.gaussian_filter(final_mask.astype(float), sigma=1.0)
        final_mask = mask_smooth > 0.3

        # Run marching cubes
        _debug_print("Running marching cubes...", start_time)
        verts, faces, _, _ = measure.marching_cubes(final_mask, level=0.5)
        _debug_print(
            f"Generated mesh: {len(verts)} vertices, {len(faces)} faces", start_time
        )

        # Fix face orientation
        faces = faces[:, [0, 2, 1]]

        # Transform to world coordinates
        _debug_print("Transforming to world coordinates...", start_time)
        verts_homo = np.hstack([verts, np.ones((len(verts), 1))])
        verts_world = (affine @ verts_homo.T).T[:, :3]

        _debug_print(
            f"Mesh bounds - X: [{verts_world[:, 0].min():.1f}, {verts_world[:, 0].max():.1f}], "
            f"Y: [{verts_world[:, 1].min():.1f}, {verts_world[:, 1].max():.1f}], "
            f"Z: [{verts_world[:, 2].min():.1f}, {verts_world[:, 2].max():.1f}]",
            start_time,
        )

        # Create plotly mesh
        _debug_print("Creating Plotly mesh...", start_time)
        x_verts, y_verts, z_verts = verts_world.T
        i_faces, j_faces, k_faces = faces.T

        electrode_mesh = go.Mesh3d(
            x=x_verts.tolist(),
            y=y_verts.tolist(),
            z=z_verts.tolist(),
            i=i_faces.tolist(),
            j=j_faces.tolist(),
            k=k_faces.tolist(),
            opacity=0.6,
            color="limegreen",
            name="Electrode Metal",
            showscale=False,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2, roughness=0.5),
            lightposition=dict(x=1000, y=1000, z=1000),
        )

        _debug_print(
            f"Electrode mesh extraction completed in {time.time() - start_time:.2f}s",
            start_time,
        )
        return electrode_mesh

    except Exception as e:
        _debug_print(
            f"Exception: {type(e).__name__}: {str(e)}",
            start_time if "start_time" in locals() else None,
        )
        import traceback

        traceback.print_exc()
        return None
