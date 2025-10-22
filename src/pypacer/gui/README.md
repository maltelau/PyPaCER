# PyPaCER Electrode Detection GUI

![PyPaCER GUI Screenshot](docs/images/pypacer-gui-screenshot.png)

## Overview

The PyPaCER Electrode Detection GUI provides an intuitive, interactive interface for detecting and reconstructing deep brain stimulation (DBS) electrodes from post-operative CT scans. This GUI combines automated detection algorithms with manual seed point selection for optimal accuracy and user control.

## Key Features

### 🖼️ **Split View Interface**
- **Left Panel**: Interactive CT slice visualization with real-time controls
- **Right Panel**: 3D electrode and trajectory visualization
- **Bottom Panel**: Comprehensive controls and coordinate information

### 🎯 **Interactive Seed Point Selection**
- Click-based seed point placement on metal artifacts
- Visual threshold preview with color-coded rectangles
- Real-time metal artifact detection overlay
- Support for multiple electrodes

### ⚡ **Multiple Detection Modes**
- **Fast**: 5-10x faster processing (reduced quality)
- **Normal**: Balanced speed and accuracy (recommended)
- **High Quality**: Maximum accuracy (slower processing)

### 📊 **Real-time Visualization**
- Live coordinate display (voxel and world coordinates)
- Intensity values and metal detection status
- Interactive crosshairs and navigation
- Color-coded electrode trajectories and contacts

### 💾 **Comprehensive Output**
- JSON export with full reconstruction data
- Electrode trajectories and contact positions
- Metadata and processing parameters
- Compatible with PyPaCER visualization tools

## Quick Start

### Command Line Usage

```bash
# Basic usage
uv run python -m pypacer.cli.electrode_gui patient_ct.nii.gz

# With custom threshold (useful for different scanners)
uv run python -m pypacer.cli.electrode_gui patient_ct.nii.gz --threshold 1500

# Start with different slice view
uv run python -m pypacer.cli.electrode_gui patient_ct.nii.gz --axis sagittal
```

### Python API

```python
from pypacer.gui import ElectrodeDetectionGUI

# Initialize and run GUI
gui = ElectrodeDetectionGUI(
    ct_path="path/to/ct_scan.nii.gz",
    metal_threshold=2000,
    slice_axis='axial'
)

# Run interactive session
electrodes = gui.run()

# Access results
print(f"Detected {len(electrodes)} electrodes")
for electrode in electrodes:
    print(f"Type: {electrode.electrode_type}")
    print(f"Length: {electrode.length_mm:.1f}mm")
```

### Demo Script

```bash
# Run comprehensive demo with instructions
uv run python examples/electrode_gui_demo.py path/to/ct_scan.nii.gz
```

## User Interface

### Mouse Controls
- **Left Click**: Add seed point for electrode detection
- **Right Click**: Remove nearest seed point
- **Scroll Wheel**: Navigate through CT slices
- **Mouse Hover**: Display coordinates and intensity values

### Visual Indicators
- **🟢 Green Rectangle**: Intensity above metal threshold (good seed point)
- **🔴 Red Rectangle**: Intensity below metal threshold
- **Red Contours**: Detected metal artifacts
- **⭐ Red Stars**: Seed points (3D view)
- **Colored Lines**: Detected electrode trajectories
- **⚫ Circles**: Electrode contact positions

### Control Panel
- **Clear All**: Remove all seed points and results
- **Detect**: Run electrode reconstruction pipeline
- **Save**: Export results to JSON format
- **Threshold**: Adjust metal detection threshold
- **Detection Mode**: Choose quality vs. speed
- **Electrode Type**: Select specific model or auto-detect
- **Show Seeds**: Toggle seed point visibility

## Detection Workflow

### 1. 📥 **Data Loading**
- Automatic CT data loading and validation
- Brain mask generation
- Metal artifact detection with configurable threshold

### 2. 🎯 **Seed Point Selection**
- Navigate to slices showing electrode artifacts
- Click on bright, high-intensity electrode regions
- Visual feedback shows metal detection status
- Add multiple seed points for multiple electrodes

### 3. ⚙️ **Parameter Configuration**
- Choose detection mode (Fast/Normal/High Quality)
- Select electrode type or use auto-detection
- Adjust metal threshold if needed
- Enable auto-save for results

### 4. 🔍 **Electrode Detection**
- Click "Detect" to run reconstruction
- Real-time progress monitoring
- Automatic electrode model generation
- Contact detection and positioning

### 5. 📊 **Results Visualization**
- 3D trajectory display with contact positions
- Electrode type and geometry information
- Length measurements and statistics
- Export capabilities to JSON format


## Output Format

Results are saved in comprehensive JSON format:

```json
{
  "metadata": {
    "ct_file": "path/to/ct_scan.nii.gz",
    "timestamp": "2024-01-15T10:30:00",
    "voxel_sizes_mm": [0.5, 0.5, 1.0],
    "metal_threshold_HU": 2000,
    "num_electrodes_detected": 2
  },
  "seed_points_voxel": [[120, 145, 67], [180, 155, 72]],
  "seed_points_world": [[-15.2, 8.5, -2.1], [12.8, 6.2, 2.5]],
  "electrodes": [
    {
      "electrode_type": "Medtronic 3389",
      "length_mm": 45.2,
      "contact_positions": [3.0, 5.5, 8.0, 10.5],
      "tip_position": [-15.1, 8.4, -2.2],
      "entry_position": [-12.5, 15.8, 32.1],
      "polynomial": [...],
      "contact_positions_3d": [...]
    }
  ]
}
```

## Integration

### PyPaCER Pipeline
- Seamless integration with core PyPaCER algorithms
- Access to all preprocessing and detection methods
- Compatible with existing visualization tools
- Support for GPU acceleration when available


## Troubleshooting

### Common Issues

#### 🚫 **No Electrodes Detected**
- **Cause**: Metal threshold too high
- **Solution**: Lower threshold (try 1500-1800 HU)
- **Check**: Ensure seed points are on bright artifacts

#### 📏 **Poor Trajectory Fitting**
- **Cause**: Insufficient artifacts or high noise
- **Solution**: Use High Quality mode, add more seed points
- **Check**: CT slice thickness ≤1mm recommended

#### 💻 **GUI Performance Issues**
- **Cause**: Large volumes or limited memory
- **Solution**: Use Fast mode initially, close other apps
- **Check**: 8GB RAM recommended for large datasets

### System Requirements

- **Python**: 3.9+ with PyPaCER dependencies
- **Memory**: 4GB minimum, 8GB recommended
- **Display**: 1920x1080 minimum resolution
- **Dependencies**: matplotlib, numpy, scipy, nibabel

## Advanced Usage

### Custom Processing

```python
# Access underlying PaCER instance
pacer = gui.pacer

# Run with custom parameters
electrodes = pacer.detect_electrodes(
    xy_resolution=0.05,
    z_resolution=0.01,
    grid_size=2.0,
    contact_detection_method='contactAreaCenter'
)
```

## Development

🧠 **For questions or support**, please refer to the main PyPaCER documentation or create an issue in the repository.