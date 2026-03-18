"""Polynomial electrode model implementation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.math_helpers import inv_poly_arc_length_3d, poly_arc_length_3d, polyval3


@dataclass
class ElectrodeGeometry:
    """Electrode geometry specifications."""

    name: str
    diameter_mm: float
    contact_length_mm: float
    contact_spacing_mm: float
    num_contacts: int
    tip_length_mm: float

    @property
    def contact_centers_mm(self) -> np.ndarray:
        """Get contact center positions from tip."""
        centers = []
        pos = self.tip_length_mm
        for i in range(self.num_contacts):
            pos += self.contact_length_mm / 2
            centers.append(pos)
            pos += self.contact_length_mm / 2 + self.contact_spacing_mm
        return np.array(centers)


def load_electrode_types() -> Dict[str, ElectrodeGeometry]:
    """Load electrode types from electrode_types.json file."""
    # Get the path to electrode_types.json in the same directory as this file
    json_path = Path(__file__).parent / "electrode_types.json"

    with open(json_path) as f:
        data = json.load(f)

    electrode_geometries = {}

    # Currently only support Medtronic electrodes
    if "medtronic" in data["dbs_leads"]:
        for lead in data["dbs_leads"]["medtronic"]["leads"]:
            # Create key as "Medtronic {model}"
            key = f"Medtronic {lead['model']}"

            # Create ElectrodeGeometry object
            electrode_geometries[key] = ElectrodeGeometry(
                name=key,
                diameter_mm=lead["diameter_mm"],
                contact_length_mm=lead["contact_length_mm"],
                contact_spacing_mm=lead["contact_spacing_mm"],
                num_contacts=lead["total_contacts"],
                tip_length_mm=1.5,  # Standard tip length for Medtronic
            )

    return electrode_geometries


# Load electrode geometries from JSON file
ELECTRODE_GEOMETRIES = load_electrode_types()

# List of supported electrode types for validation
SUPPORTED_ELECTRODE_TYPES = list(ELECTRODE_GEOMETRIES.keys())


class PolynomialElectrodeModel:
    """
    Represents a DBS electrode as a 3D polynomial curve.

    The polynomial maps from parameter t in [0,1] to 3D space,
    where t=0 is the tip and t=1 is the entry point.
    """

    def __init__(
        self,
        polynomial: np.ndarray,
        electrode_type: str = "Medtronic 3389",
        contact_positions: Optional[np.ndarray] = None,
        active_contact: Optional[int] = None,
        intensity_profile: Optional[np.ndarray] = None,
        distance_scale: Optional[np.ndarray] = None,
        bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        skeleton_deviations_mm: Optional[np.ndarray] = None,
        polynomial_before_tip_detection: Optional[np.ndarray] = None,
        orientation_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize electrode model.

        Args:
            polynomial: Polynomial coefficients (degree+1 x 3)
            electrode_type: Type of electrode
            contact_positions: Detected contact positions in mm from tip
            active_contact: Index of active contact (0-based)
            intensity_profile: Optional intensity values along electrode
            distance_scale: Optional distance values in mm from tip
            bounding_box: Optional (min_coords, max_coords) bounding box in world coordinates
            skeleton_deviations_mm: Optional deviation values between skeleton and polynomial (aligned with distance_scale)
            orientation_data: Optional orientation detection results (marker locations, angles, etc.)
        """
        self.polynomial = polynomial

        # Store the electrode type (may be combined, e.g. "Medtronic 3389/B33005")
        self.electrode_type = electrode_type

        # Get geometry - handle combined types
        if electrode_type in ELECTRODE_GEOMETRIES:
            self.geometry = ELECTRODE_GEOMETRIES[electrode_type]
        else:
            # Handle combined types like "Medtronic 3389/B33005"
            electrode_types = [t.strip() for t in electrode_type.split("/")]
            geometry_found = False

            for et in electrode_types:
                if et in ELECTRODE_GEOMETRIES:
                    self.geometry = ELECTRODE_GEOMETRIES[et]
                    geometry_found = True
                    break

            if not geometry_found:
                raise ValueError(
                    f"Unsupported electrode type: {electrode_type}. "
                    f"Supported types are: {', '.join(SUPPORTED_ELECTRODE_TYPES)}"
                )

        # Use detected positions if available, otherwise use geometry
        if contact_positions is not None:
            self.contact_positions = contact_positions
        else:
            self.contact_positions = self.geometry.contact_centers_mm

        self.active_contact = active_contact
        self.intensity_profile = intensity_profile
        self.distance_scale = distance_scale
        self.bounding_box = bounding_box
        self.skeleton_deviations_mm = skeleton_deviations_mm
        self.polynomial_before_tip_detection = polynomial_before_tip_detection
        self.orientation_data = orientation_data
        self.tip_hemisphere = None
        self.entry_hemisphere = None

        # Cache frequently used values
        self._length = None
        self._skeleton = None

    @property
    def length_mm(self) -> float:
        """Total length of electrode trajectory."""
        if self._length is None:
            self._length = poly_arc_length_3d(self.polynomial, 0, 1)
        return self._length

    @property
    def skeleton(self) -> np.ndarray:
        """Get dense sampling of electrode trajectory."""
        if self._skeleton is None:
            t = np.linspace(0, 1, 1000)
            self._skeleton = polyval3(self.polynomial, t)
        return self._skeleton

    @property
    def tip_position(self) -> np.ndarray:
        """Get electrode tip position (t=0)."""
        return polyval3(self.polynomial, 0)

    @property
    def entry_position(self) -> np.ndarray:
        """Get electrode entry position (t=1)."""
        return polyval3(self.polynomial, 1)

    def get_contact_positions_3d(self) -> np.ndarray:
        """Get 3D positions of all contacts."""
        positions_3d = []
        for pos_mm in self.contact_positions:
            t = inv_poly_arc_length_3d(self.polynomial, pos_mm)
            positions_3d.append(polyval3(self.polynomial, t))
        return np.array(positions_3d)

    def get_active_contact_position(self) -> Optional[np.ndarray]:
        """Get 3D position of active contact."""
        if self.active_contact is None:
            return None
        pos_mm = self.contact_positions[self.active_contact]
        t = inv_poly_arc_length_3d(self.polynomial, pos_mm)
        return polyval3(self.polynomial, t)

    def get_point_at_parameter(self, t: float) -> np.ndarray:
        """Get 3D point at parameter t."""
        return polyval3(self.polynomial, t)

    def get_tangent_at(self, t: float) -> np.ndarray:
        """Get tangent vector at parameter t."""
        # Derivative of polynomial
        deriv_coeffs = (
            self.polynomial[:-1]
            * np.arange(len(self.polynomial) - 1, 0, -1)[:, np.newaxis]
        )
        tangent = polyval3(deriv_coeffs, t)
        return tangent / np.linalg.norm(tangent)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to electrode."""
        distances = np.linalg.norm(self.skeleton - point, axis=1)
        return np.min(distances)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "polynomial": self.polynomial.tolist(),
            "electrode_type": self.electrode_type,
            "contact_positions": self.contact_positions.tolist(),
            "active_contact": self.active_contact,
            "length_mm": self.length_mm,
            "tip_position": self.tip_position.tolist(),
            "entry_position": self.entry_position.tolist(),
        }
        # Include intensity profile if available
        if self.intensity_profile is not None:
            data["intensity_profile"] = self.intensity_profile.tolist()
        if self.distance_scale is not None:
            data["distance_scale"] = self.distance_scale.tolist()
        # Include deviation data if available (uses same distance_scale for alignment)
        if self.skeleton_deviations_mm is not None:
            data["skeleton_deviations_mm"] = self.skeleton_deviations_mm.tolist()
        # Include bounding box if available
        if self.bounding_box is not None:
            data["bounding_box"] = {
                "min": self.bounding_box[0].tolist(),
                "max": self.bounding_box[1].tolist(),
            }
        # Include polynomial before tip detection if available (debug mode)
        if self.polynomial_before_tip_detection is not None:
            data["polynomial_before_tip_detection"] = (
                self.polynomial_before_tip_detection.tolist()
            )
        # Include full Pass 2 data if available (debug mode)
        if (
            hasattr(self, "pass2_intensities_full")
            and self.pass2_intensities_full is not None
        ):
            data["pass2_intensities_full"] = self.pass2_intensities_full.tolist()
        if (
            hasattr(self, "pass2_distances_mm_full")
            and self.pass2_distances_mm_full is not None
        ):
            data["pass2_distances_mm_full"] = self.pass2_distances_mm_full.tolist()
        if (
            hasattr(self, "pass2_tip_threshold")
            and self.pass2_tip_threshold is not None
        ):
            data["pass2_tip_threshold"] = self.pass2_tip_threshold
        if hasattr(self, "pass2_tip_param") and self.pass2_tip_param is not None:
            data["pass2_tip_param"] = self.pass2_tip_param
        if (
            hasattr(self, "original_t0_distance_mm")
            and self.original_t0_distance_mm is not None
        ):
            data["original_t0_distance_mm"] = self.original_t0_distance_mm
        # Include refined profiles if available (for deviation plotting)
        if (
            hasattr(self, "refined_intensity_profile")
            and self.refined_intensity_profile is not None
        ):
            data["refined_intensity_profile"] = self.refined_intensity_profile.tolist()
        if (
            hasattr(self, "refined_distance_scale")
            and self.refined_distance_scale is not None
        ):
            data["refined_distance_scale"] = self.refined_distance_scale.tolist()
        # Include contact detection results if available (debug mode)
        if hasattr(self, "contact_detection_results"):
            data["contact_detection_results"] = self.contact_detection_results
        # Include orientation data if available
        if self.orientation_data is not None:
            data["orientation"] = self.orientation_data
        # Include hemisphere labels if available
        if self.tip_hemisphere is not None:
            data["tip_hemisphere"] = self.tip_hemisphere
            data["side"] = self.tip_hemisphere  # Add 'side' for HTML report compatibility
        if self.entry_hemisphere is not None:
            data["entry_hemisphere"] = self.entry_hemisphere
        return data

    def to_matlab_struct(self) -> Dict[str, Any]:
        """Convert to MATLAB-compatible structure."""
        return {
            "r3polynomial": self.polynomial.T,  # MATLAB expects 3xN
            "electrodeInfo": {
                "string": self.electrode_type,
                "diameterMm": self.geometry.diameter_mm,
                "ringContactCentersMm": self.contact_positions,
            },
            "activeContact": (
                self.active_contact + 1 if self.active_contact is not None else None
            ),  # MATLAB 1-indexed
            "apprTotalLengthMm": self.length_mm,
        }

    def to_hdf5(self, group):
        """Save to HDF5 group."""
        group.create_dataset("polynomial", data=self.polynomial)
        group.attrs["electrode_type"] = self.electrode_type
        group.create_dataset("contact_positions", data=self.contact_positions)
        if self.active_contact is not None:
            group.attrs["active_contact"] = self.active_contact

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolynomialElectrodeModel":
        """Create from dictionary."""
        electrode = cls(
            polynomial=np.array(data["polynomial"]),
            electrode_type=data["electrode_type"],
            contact_positions=np.array(data["contact_positions"]),
            active_contact=data.get("active_contact"),
            intensity_profile=(
                np.array(data["intensity_profile"])
                if "intensity_profile" in data
                else None
            ),
            distance_scale=(
                np.array(data["distance_scale"]) if "distance_scale" in data else None
            ),
            skeleton_deviations_mm=(
                np.array(data["skeleton_deviations_mm"])
                if "skeleton_deviations_mm" in data
                else None
            ),
            orientation_data=data.get("orientation"),
        )
        electrode.tip_hemisphere = data.get("tip_hemisphere")
        electrode.entry_hemisphere = data.get("entry_hemisphere")
        return electrode

    def __repr__(self) -> str:
        return (
            f"PolynomialElectrodeModel(type={self.electrode_type}, "
            f"length={self.length_mm:.1f}mm, "
            f"contacts={len(self.contact_positions)})"
        )
