# PyPaCER Examples

This directory contains example scripts demonstrating common PyPaCER workflows for working with DBS electrode reconstructions.

## Available Scripts

### 1. `generate_electrode_slice_views.py`

Generate orthogonal 2D cross-section images along electrode trajectories.

**Purpose:** Visualize electrodes in CT scans by creating cross-sectional views perpendicular to and along the electrode axis. Useful for quality control and verification of electrode placement.

**Outputs:** PNG images showing lateral and frontal views of each electrode with contact positions marked.

**Usage:**
```bash
python generate_electrode_slice_views.py reconstruction.json ct_scan.nii.gz --output-dir slices/
```

**Options:**
- `--width`, `--height`: Adjust plane dimensions (mm)
- `--colormap`: Change CT display colormap (gray, jet, viridis, etc.)
- `--clim`: Set CT window/level (Hounsfield units)

---

### 2. `transform_contacts_ants.py`

Transform electrode contact positions between image spaces using ANTs registration transforms.

**Purpose:** Map electrode coordinates from native/patient space to standard template/atlas space, or chain multiple transformations (e.g., postop CT → preop CT → T1w → MNI template).

**Outputs:** JSON file with transformed contact positions (preserves original coordinates as `*_native` fields).

**Usage:**
```bash
# Simple single transform
python transform_contacts_ants.py reconstruction.json transform.mat --output transformed.json

# Chain multiple transforms (A → B → C → D)
python transform_contacts_ants.py reconstruction.json \
  transform_AtoB.mat \
  transform_BtoC.mat \
  transform_CtoD_affine.mat \
  transform_CtoD_InverseWarp.nii.gz \
  --output electrodes_in_D.json
```

**Important Notes:**
- **Affine transforms** (.mat): Automatically inverted (correct for point transforms)
- **Deformation fields** (.nii.gz): Automatically used in forward direction (cannot be inverted)
  - Use `output1InverseWarp.nii.gz` if you need the inverse deformation
- Transforms are applied in **reverse order** (ANTs convention)
- Requires: `pip install antspyx pandas`

**Advanced Usage:**
```bash
# Manual control with --forward-for (rarely needed)
python transform_contacts_ants.py reconstruction.json \
  transform1.mat transform2.mat \
  --forward-for 1 \
  --output transformed.json
```

---

### 3. `generate_pypacer_report.py`

Generate an interactive HTML report with 3D visualizations of electrode reconstructions.

**Purpose:** Create publication-ready visualizations with volume renderings and rotating GIF animations of reconstructed electrodes.

**Outputs:** HTML file with embedded 3D visualizations and animations.

**Usage:**
```bash
python generate_pypacer_report.py reconstruction.json --output report.html
```

**Options:**
- `--output`: Specify output HTML file path
- Wraps the `pypacer_report` CLI tool for convenience

---

## Common Workflows

### Workflow 1: Basic Quality Control
```bash
# 1. Generate slice views for visual inspection
python generate_electrode_slice_views.py reconstruction.json ct.nii.gz -o slices/

# 2. Create HTML report for detailed review
python generate_pypacer_report.py reconstruction.json -o report.html
```

### Workflow 2: Transform to Template Space
```bash
# Step 1: Perform ANTs registrations
# CT postop → CT preop (rigid)
antsRegistrationSyN.sh -d 3 -t r -f ct_preop.nii.gz -m ct_postop.nii.gz -o ct_post_to_pre_

# CT preop → T1w (rigid)
antsRegistrationSyN.sh -d 3 -t r -f t1w.nii.gz -m ct_preop.nii.gz -o ct_pre_to_t1w_

# T1w → Template (nonlinear)
antsRegistrationSyN.sh -d 3 -t s -f template.nii.gz -m t1w.nii.gz -o t1w_to_template_

# Step 2: Transform electrode contacts through the full chain
# reconstruction.json is in CT postop space (where electrodes were reconstructed)
python transform_contacts_ants.py \
  reconstruction.json \
  ct_post_to_pre_0GenericAffine.mat \
  ct_pre_to_t1w_0GenericAffine.mat \
  t1w_to_template_0GenericAffine.mat \
  t1w_to_template_1InverseWarp.nii.gz \
  --output reconstruction_template.json

# Step 3: Generate report in template space
python generate_pypacer_report.py reconstruction_template.json -o template_report.html
```

### Workflow 3: Multi-Modal Registration Chain
```bash
# Chain: postop CT → preop CT → T1w → MNI template
python transform_contacts_ants.py \
  reconstruction.json \
  ct_post_to_ct_pre.mat \
  ct_pre_to_t1w.mat \
  t1w_to_mni_affine.mat \
  t1w_to_mni_InverseWarp.nii.gz \
  --output reconstruction_mni.json
```

---

## Input File Formats

- **Reconstruction JSON**: Output from PyPaCER CLI (`pypacer` command)
- **CT/MRI Images**: NIfTI format (`.nii` or `.nii.gz`)
- **ANTs Transforms**:
  - Affine: `.mat` files (e.g., `output0GenericAffine.mat`)
  - Deformation: `.nii.gz` files (e.g., `output1Warp.nii.gz`, `output1InverseWarp.nii.gz`)

---

## Requirements

All scripts require PyPaCER to be installed:
```bash
pip install pypacer
```

All necessary dependencies (`antspyx`, `pandas`, `nibabel`, `scipy`, `matplotlib`) are automatically installed with PyPaCER.

---

## Getting Help

For detailed options on any script:
```bash
python script_name.py --help
```

For PyPaCER documentation and support:
- GitHub: https://github.com/mvpetersen/PyPaCER
- Issues: https://github.com/mvpetersen/PyPaCER/issues
