"""PyPaCER visualization module for generating reports and plots."""

from .electrode_sliceview import (
    compute_orthogonal_vectors,
    create_volume_interpolator,
    generate_orthogonal_slice_views,
    generate_slice_plane,
    get_world_to_voxel_transform,
    project_contacts_to_plane,
)
from .report_generator import generate_html_report

__all__ = [
    "generate_html_report",
    "generate_orthogonal_slice_views",
    "generate_slice_plane",
    "project_contacts_to_plane",
    "compute_orthogonal_vectors",
    "create_volume_interpolator",
    "get_world_to_voxel_transform",
]
