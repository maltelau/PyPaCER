# Directional Electrode Orientation Detection

This module detects and determines the orientation of directional DBS electrodes (Medtronic B33015 and B33005 Sensight leads). It is integrated into the main PyPaCER pipeline and runs automatically after contact detection.

## Overview

Directional DBS electrodes have radiopaque markers that enable current steering. These markers:
- Are located proximal to the contact region (typically 2–10mm above the **last contact** furthest from tip)
- Appear as two distinct peaks in trajectory deviation profiles
- Point in different directions (separated by ~120°) perpendicular to the electrode trajectory

## Pipeline

Orientation detection runs as part of `PyPaCER.detect_electrodes()` (and the `_auto` / `_radial` variants). After contacts are detected for each electrode, the pipeline calls `_run_orientation_detection()` which executes these steps:

1. **Marker detection** — `detect_directional_markers()` searches the region above the last contact for deviation peaks that indicate radiopaque markers.
2. **Electrode classification** — `classify_electrode_type()` uses contact spacing and marker presence to label the electrode (e.g. Medtronic B33005, B33015, 3389, 3387).
3. **Orientation determination** — For each detected marker peak, `determine_marker_orientation()` samples a circular intensity profile perpendicular to the trajectory and finds the peak direction.
4. **Pair validation** — When two markers are found, `validate_marker_pair()` checks that their angular separation falls within the expected range (120°–150°).
5. **Constrained fitting** — `fit_constrained_marker_directions()` adjusts both marker angles so they are exactly 120° apart (the true physical separation), minimising the total angular adjustment.

Results are stored on `electrode.orientation_data` and serialised to the output JSON.

## Modules

### `marker_detection.py`

Detects whether an electrode has directional markers by analysing the region from `last_contact + marker_offset_mm` to `max_distance_mm`:

- **Deviation peaks** (preferred): Finds peaks in polynomial fit deviations using `scipy.signal.find_peaks`.
- **Intensity peaks** (fallback): Uses within-region intensity variation when deviation data yields no peaks.

### `circular_sampling.py`

Samples CT intensities in circular patterns perpendicular to the electrode trajectory:

- **Multiple radii**: Intensity is sampled at several distances from the trajectory centre and averaged.
- **Angular resolution**: Configurable angle increments.
- **Flexible planes**: Can sample in trajectory-perpendicular plane (default) or axial plane.

### `orientation_analysis.py`

Core analysis functions:

- `determine_marker_orientation()` — Samples circular intensity, smooths the profile, finds the peak via intensity-weighted centre, checks for trajectory bias, and computes a confidence score.
- `validate_marker_pair()` — Checks angular separation between two marker orientations.
- `fit_constrained_marker_directions()` — Adjusts marker angles to enforce a fixed angular constraint.

### `marker_profile.py`

Samples circular intensity profiles along the marker region of the trajectory for debug visualisation (heatmaps, contour plots).

### `visualization.py`, `visualization_debug.py`, `visualization_profile.py`

Visualisation tools including polar plots, orthogonal plane views, marker pair comparisons, and marker region profile plots. Generated when debug output is enabled.

## Configurable Parameters

All parameters below can be overridden by passing an `orientation_params` dict to `detect_electrodes()`, `detect_electrodes_auto()`, or `detect_electrodes_radial()`. Only keys that are provided are overridden; everything else keeps its default.

```python
electrodes = pacer.detect_electrodes(
    orientation_params={
        "radii_mm": [1.0, 1.25, 1.5],
        "angle_increment_deg": 0.5,
        "deviation_threshold": 0.10,
    }
)
```

### Marker Detection (`detect_directional_markers`)

| Parameter | Default | Description |
|---|---|---|
| `marker_offset_mm` | 2.5 | Distance (mm) above the last contact where the marker search region begins. Skips the contact zone to avoid false detections from contact artifacts. |
| `max_distance_mm` | 20.0 | Maximum distance from tip (mm) that caps the far end of the marker search region. Prevents searching too far up the shaft where signal degrades. |
| `deviation_threshold` | 0.08 | Minimum skeleton deviation value for a peak to be considered a marker candidate. Higher values reduce false positives but may miss subtle markers. |
| `min_peak_distance_mm` | 2.0 | Minimum spacing (mm) between detected deviation peaks, passed to `scipy.signal.find_peaks(distance=...)`. Prevents a single physical marker from producing two peaks. The two Medtronic markers are ~3mm apart, so 2.0 is a sensible floor. |
| `expected_num_peaks` | 2 | Expected number of marker peaks. Used to select the top-N peaks when more candidates are found. |

### Orientation Analysis (`determine_marker_orientation`)

| Parameter | Default | Description |
|---|---|---|
| `radii_mm` | [1.25, 1.5, 1.75] | Circular sampling radii (mm) around the trajectory centre at each marker location. Intensity is sampled at each radius and averaged. Note: the low-level `sample_circular_intensity` function has its own default of [0.5, 0.75, 1.0], but when called from the pipeline via `determine_marker_orientation` the default is [1.25, 1.5, 1.75]. |
| `angle_increment_deg` | 0.1 | Angular step (degrees) for circular intensity sampling. Smaller values give finer resolution at the cost of more computation. The low-level `sample_circular_intensity` default is 5.0°, but the pipeline uses 0.1°. |
| `smoothing_window` | 5 | Window size (in samples) for smoothing the circular intensity profile before peak detection. |
| `check_for_bias` | True | Whether to check for trajectory bias — a significant intensity peak at ~180° opposite the main peak, which indicates the polynomial trajectory centre may be offset from the true electrode centre. When bias is detected the confidence score is penalised (×0.7) but the detected angle is not changed. |
| `bias_opposite_peak_threshold` | 0.7 | Ratio threshold for bias detection. If intensity at the opposite angle (180° from peak) exceeds this fraction of the peak intensity, bias is flagged. |

### Pair Validation (`validate_marker_pair`)

| Parameter | Default | Description |
|---|---|---|
| `min_separation_deg` | 120.0 | Minimum valid angular separation (degrees) between two marker orientations. For Medtronic leads the true physical separation is 120°. |
| `max_separation_deg` | 150.0 | Maximum valid angular separation (degrees). CT artifact spread can make the detected separation appear wider than 120°, so the valid range extends to 150°. |

### Constrained Fitting (`fit_constrained_marker_directions`)

| Parameter | Default | Description |
|---|---|---|
| `angular_constraint_deg` | 120.0 | Fixed angular separation (degrees) enforced between the two fitted marker directions. The fitting adjusts both angles to minimise total angular change while maintaining exactly this separation. |

## CLI Access

The main CLI and the example script expose the most commonly tuned parameters:

```bash
# Main CLI
pypacer ct_scan.nii.gz \
    --orientation-radii 1.0 1.25 1.5 \
    --orientation-angle-step 0.5 \
    --orientation-smoothing 7 \
    --marker-deviation-threshold 0.10 \
    --marker-min-separation 110 \
    --marker-max-separation 160

# Example script (examples/run_orientation_detection.py)
python examples/run_orientation_detection.py ct_scan.nii.gz \
    --orientation-radii 1.0 1.25 1.5 \
    --orientation-angle-step 0.5 \
    --marker-offset-mm 3.0 \
    --marker-max-distance-mm 18.0 \
    --angular-constraint 120
```

## Programmatic Usage

```python
import json
import nibabel as nib
import numpy as np
from pypacer.orientation import (
    detect_directional_markers,
    classify_electrode_type,
    determine_marker_orientation,
    fit_constrained_marker_directions,
)
from pypacer.orientation import validate_marker_pair

# Load reconstruction data
with open('reconstruction.json') as f:
    data = json.load(f)

electrode = data['electrodes'][0]
distance_scale = np.array(electrode['refined_distance_scale'])
intensity_profile = np.array(electrode['refined_intensity_profile'])
skeleton_deviations = np.array(electrode['skeleton_deviations_mm'])
contact_positions = electrode['contact_positions']
polynomial = np.array(electrode['polynomial'])

# Step 1: Detect markers
marker_result = detect_directional_markers(
    distance_scale=distance_scale,
    intensity_profile=intensity_profile,
    skeleton_deviations=skeleton_deviations,
    contact_positions=contact_positions,
    marker_offset_mm=2.5,
    deviation_threshold=0.08,
)

print(f"Markers detected: {marker_result.has_markers}")
print(f"Confidence: {marker_result.confidence:.2f}")
print(f"Peak locations: {marker_result.marker_peak_locations}")

# Step 2: Classify electrode type
electrode_type = classify_electrode_type(contact_positions, marker_result)
print(f"Electrode type: {electrode_type}")

# Step 3: Determine orientation for each marker
if marker_result.has_markers:
    ct_img = nib.load('ct_image.nii.gz')
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine

    orientation_results = []
    for marker_location in marker_result.marker_peak_locations:
        # In practice you would compute the trajectory direction from the
        # polynomial at this location. Here we use a placeholder.
        trajectory_direction = np.array([0.0, 0.0, 1.0])

        orientation_result = determine_marker_orientation(
            ct_data=ct_data,
            affine=affine,
            electrode_polynomial=polynomial,
            marker_location_mm=marker_location,
            trajectory_direction=trajectory_direction,
            radii_mm=[1.25, 1.5, 1.75],
            angle_increment_deg=0.1,
        )

        print(f"Marker at {marker_location:.1f}mm: {orientation_result.peak_angle_deg:.1f}°")
        print(f"  Confidence: {orientation_result.confidence:.2f}")
        print(f"  Bias detected: {orientation_result.analysis_metadata['bias_detected']}")
        orientation_results.append(orientation_result)

    # Step 4: Validate marker pair
    if len(orientation_results) == 2:
        is_valid, separation = validate_marker_pair(
            orientation_results[0],
            orientation_results[1],
            min_separation_deg=120.0,
            max_separation_deg=150.0,
        )
        print(f"\nAngular separation: {separation:.1f}° (valid: {is_valid})")

        # Step 5: Fit constrained directions
        fitted_b, fitted_a = fit_constrained_marker_directions(
            orientation_results[0],
            orientation_results[1],
            angular_constraint_deg=120.0,
        )
        print(f"Fitted angles: B={fitted_b:.1f}°, A={fitted_a:.1f}°")
```

## Implementation Notes

### Coordinate Systems

The module handles transformations between:
- **Voxel coordinates**: CT array indices
- **World coordinates**: Physical RAS/LPS coordinates
- **Local trajectory coordinates**: Plane perpendicular to electrode

### Polynomial Evaluation

Polynomials are stored with coefficients in descending power order:
- `P(t) = c[0]*t^n + c[1]*t^(n-1) + ... + c[n]`
- Parameter `t` ranges from 0 (tip) to 1 (entry point)
- Distance mapping is computed by sampling trajectory and calculating arc length

### Trajectory Bias Detection

The polynomial fit may be biased toward the markers (higher intensity). This can result in:
- Trajectory centre offset from true electrode centre
- Two peaks appearing at ~180° apart in circular sampling

When `check_for_bias` is enabled (default), `_check_trajectory_bias()` compares the intensity at the opposite angle (180° from the main peak) to the peak intensity. If the ratio exceeds `bias_opposite_peak_threshold` (default 0.7), bias is flagged. This **penalises the confidence score** (×0.7) but does **not** alter the detected angle.

## Parameter Tuning

### For Different CT Quality

**High-quality CT** (slice thickness ≤ 0.5mm):
- Use smaller radii: `radii_mm=[0.75, 1.0, 1.25]`
- Default angle step (0.1°) is appropriate

**Lower-quality CT** (slice thickness > 0.5mm):
- Use larger radii: `radii_mm=[1.5, 1.75, 2.0]`
- Consider increasing smoothing: `smoothing_window=7`

### Reducing False Marker Detections

- Increase `deviation_threshold` (e.g. 0.10 or 0.12) to require more prominent peaks
- Increase `marker_offset_mm` to skip more of the contact region

### Relaxing Pair Validation

For non-standard electrode geometries or heavily artefacted scans:
- Widen the valid range: `min_separation_deg=100.0, max_separation_deg=160.0`

## Troubleshooting

### Low Confidence Scores
- **Check bias detection**: If `bias_detected=True` in metadata, confidence is penalised. Inspect the polar plot to see if there is a genuine opposite peak.
- **Increase sampling radii**: May need larger radii if markers are farther from centre.
- **Inspect smoothing**: Too much smoothing can flatten the peak; too little retains noise.

### Invalid Marker Pair Separation
- Expected separation is 120°–150° for Medtronic electrodes.
- **Check individual confidences**: Low confidence indicates poor detection for one or both markers.
- **Visualise orthogonal planes**: Inspect actual marker positions with debug output enabled.

### No Markers Detected
- **Inspect deviation data**: Ensure `skeleton_deviations_mm` is available in the reconstruction.
- **Lower deviation threshold**: Try `deviation_threshold=0.05` to catch subtle markers.
- **Adjust region bounds**: Try different `marker_offset_mm` or `max_distance_mm`.

## Output Files

When debug output is enabled, the pipeline generates per-electrode visualisations:

1. **`electrode_N_marker_orientations.png`**: Combined marker orientation polar plots with detected and fitted angles.
2. **`electrode_N_full_profile.png`**: 3D contour plot of marker region intensity (when debug enabled).


