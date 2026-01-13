# NightUAV-Sim Dataset Generation Tutorial

A comprehensive guide for generating synthetic UAV imagery using Unreal Engine 5.3, CitySample, and custom flight planning tools.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Flight Path Planning](#3-flight-path-planning)
4. [Sequence Generation in UE5](#4-sequence-generation-in-ue5)
5. [Rendering Configuration](#5-rendering-configuration)
6. [Batch Rendering](#6-batch-rendering)

---

## 1. Prerequisites

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Unreal Engine | 5.3+ | Rendering engine |
| CitySample | Latest | Urban environment assets |
| Python | 3.9+ | Flight planning scripts |
| Movie Render Queue | Built-in | High-quality rendering |

### Python Dependencies

```bash
pip install numpy pandas shapely matplotlib pydantic
```

### Project Structure

```
YourProject/
├── Content/
│   ├── Python/
│   │   ├── OrthoPlanning.py        # Nadir flight planning
│   │   ├── obliquePlanning.py      # Oblique flight planning
│   │   ├── utils_ortho_sequencer.py    # Sequence generator (Nadir)
│   │   ├── utils_oblique_sequencer.py  # Sequence generator (Oblique)
│   │   ├── utils.py                # Utility functions
│   │   ├── utils_actor.py          # Actor utilities
│   │   ├── pydantic_model.py       # Data models
│   │   └── GLOBAL_VARS.py          # Global variables
│   └── Sequences/
│       └── NightTime/              # Generated sequences
└── Plugins/
    └── MatrixCityPlugin/           # CitySample integration
```

---

## 2. Environment Setup

### 2.1 CitySample Configuration

1. Download and install CitySample from Epic Games Launcher
2. Create a new project or migrate CitySample content to your project
3. Open the main city level

### 2.2 Enable Python Scripting in UE5

1. Go to **Edit → Plugins**
2. Search for "Python" and enable **Python Editor Script Plugin**
3. Restart the editor
4. Configure Python path: **Edit → Project Settings → Python → Additional Paths**
   - Add: `{ProjectDir}/Content/Python`

### 2.3 Create Required Data Model

Create `pydantic_model.py` in your Python folder:

```python
from pydantic import BaseModel
from typing import Tuple

class SequenceKey(BaseModel):
    frame: int
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
```

---

## 3. Flight Path Planning

### 3.1 Understanding Coordinate System

NightUAV-Sim uses centimeter units (UE5 default):

| Parameter | Description | Unit |
|-----------|-------------|------|
| X, Y | Horizontal position | cm |
| Z | Altitude (base_z + H) | cm |
| Yaw | Heading angle | degrees |
| Pitch | Tilt angle | degrees |
| Roll | Bank angle | degrees |

### 3.2 Nadir (Orthophoto) Planning

For vertical downward-facing imagery:

```bash
python OrthoPlanning.py \
    --use_default_boundary \
    --H 25000 \
    --front 0.80 \
    --side 0.70 \
    --csv drone_nadir.csv \
    --plot
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--H` | 25000 | Flight height in cm (250m) |
| `--front` | 0.80 | Forward overlap (80%) |
| `--side` | 0.70 | Side overlap (70%) |
| `--base_z` | 0 | Ground elevation in cm |
| `--f` | 3.5 | Focal length in cm |
| `--sw` | 3.6 | Sensor width in cm |
| `--sh` | 2.4 | Sensor height in cm |
| `--Nx` | 6000 | Image width in pixels |
| `--Ny` | 4000 | Image height in pixels |

**Output CSV Format (Nadir):**

```csv
X,Y,Z
-91500.0,16150.0,25000.0
-91500.0,22150.0,25000.0
...
```

### 3.3 Oblique (Five-Direction) Planning

For 3D reconstruction with tilted cameras:

```bash
python obliquePlanning.py \
    --use_default_boundary \
    --H 25000 \
    --oblique \
    --gimbal_pitch -45 \
    --auto_rotate \
    --csv drone_oblique.csv \
    --plot
```

**Additional Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--oblique` | False | Enable 5-direction mode |
| `--gimbal_pitch` | -45 | Camera tilt angle |
| `--auto_rotate` | False | Optimize flight heading |

**Five Directions Generated:**

| Direction | Pitch | Yaw Offset | Description |
|-----------|-------|------------|-------------|
| Nadir | -90° | 0° | Vertical downward |
| Forward | -45° | 0° | Tilted forward |
| Backward | -45° | 180° | Tilted backward |
| Right | -45° | 90° | Tilted right |
| Left | -45° | 270° | Tilted left |

**Output CSV Format (Oblique):**

```csv
X,Y,Z,Yaw,Pitch,Roll
-91500.0,16150.0,25000.0,0.0,-90.0,0.0
-91500.0,22150.0,25000.0,45.0,-45.0,0.0
...
```

### 3.4 Custom Boundary Definition

Replace the default boundary with your own area:

```python
# In OrthoPlanning.py or obliquePlanning.py
boundary_points = [
    (-91500, 16150),   # Point 1 (X, Y in cm)
    (-38750, -62100),  # Point 2
    (39250, -62900),   # Point 3
    (78750, 15600),    # Point 4
    (26050, 68050)     # Point 5
]
```

---

## 4. Sequence Generation in UE5

### 4.1 Generate Nadir Sequence

1. Open UE5 Output Log: **Window → Developer Tools → Output Log**

2. Run the script in UE5 Python console:

```python
# Method 1: Direct execution
exec(open('F:/YourProject/Content/Python/utils_ortho_sequencer.py').read())

# Method 2: Import and run
import utils_ortho_sequencer
utils_ortho_sequencer.main()
```

3. Configure script parameters before running:

```python
# In utils_ortho_sequencer.py - main() function
csv_file_path = 'F:/YourProject/Content/Python/drone_nadir.csv'  # Absolute path!
sequence_dir = '/Game/Sequences/NightTime'
sequence_name = 'Nadir_Sequence'
seq_fps = 30
fov = 45.0
camera_pitch = -90.0  # Vertical downward
rotation_mode = 'FIXED'
```

### 4.2 Generate Oblique Sequence

```python
# In utils_oblique_sequencer.py - main() function
csv_file_path = 'F:/YourProject/Content/Python/drone_oblique.csv'
sequence_dir = '/Game/Sequences/NightTime'
sequence_name = 'Oblique_Sequence'
seq_fps = 30
fov = 45.0
```

### 4.3 Verify Generated Sequence

1. Open Content Browser: `/Game/Sequences/NightTime/`
2. Double-click the generated sequence to open Sequencer
3. Verify:
   - Camera track exists with keyframes
   - Camera Cut track is properly configured
   - Total frame count matches CSV row count

---

## 5. Rendering Configuration

### 5.1 Movie Render Queue Setup

1. Open your sequence in Sequencer
2. Click **Render** button → **Movie Render Queue**
3. Click **+ Setting** to add render passes

### 5.2 Recommended Render Settings

**Output Settings:**

| Setting | Value | Notes |
|---------|-------|-------|
| Output Directory | `{project}/Saved/MovieRenders/` | Use tokens |
| File Name Format | `{sequence_name}/{camera_name}/{frame_number}` | Organized structure |
| Output Resolution | 8196 × 5460 | ~45MP |
| Frame Rate | 30 fps | Match sequence |

**Image Output Settings:**

```
Format: PNG (lossless) or EXR (HDR)
Bit Depth: 16-bit (recommended for post-processing)
Compression: None or ZIP
```

### 5.3 High Resolution Settings

1. Add **High Resolution** setting
2. Configure:
   - Tile Count: 2×2 or 4×4 (for ultra-high resolution)
   - Overlap Ratio: 0.1
   - Override Resolution: Enabled

### 5.4 Console Variables for Quality

Add **Console Variables** setting:

```
r.ScreenPercentage=200
r.DepthOfFieldQuality=4
r.MotionBlurQuality=0
r.AmbientOcclusionLevels=3
r.Shadow.MaxResolution=4096
```

### 5.5 Additional Render Passes

For ground truth generation, add these passes:

| Pass Type | Purpose | Output |
|-----------|---------|--------|
| Deferred Rendering | RGB images | PNG/EXR |
| World Normal | Surface normals | EXR |
| Scene Depth | Depth maps | EXR (32-bit) |
| Custom Stencil | Segmentation | PNG |

---

## 6. Batch Rendering

### 6.1 Multi-Illumination Rendering

For each lighting condition:

1. Adjust world lighting (DirectionalLight, SkyLight, etc.)
2. Save the level with lighting preset
3. Queue the render job

**Recommended Illumination Settings:**

| Condition | Sun Altitude | Time of Day | Notes |
|-----------|--------------|-------------|-------|
| Noon | 90° | 12:00 | Direct overhead |
| Afternoon | 45° | 15:00 | Warm lighting |
| Dusk | 10° | 18:30 | Golden hour |
| Early Night | -10° | 20:00 | Twilight |
| Late Night | -45° | 23:00 | Artificial lights only |
| Base Color | N/A | N/A | Unlit mode |

### 6.2 Render Queue Workflow

```
1. Create render jobs for each illumination
2. Configure output paths with illumination suffix
3. Start batch rendering (can run overnight)
4. Verify outputs after completion
```

### 6.3 Output Organization

```
MovieRenders/
├── Noon/
│   ├── Nadir/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── Oblique/
│       └── ...
├── Afternoon/
│   └── ...
├── Dusk/
│   └── ...
├── EarlyNight/
│   └── ...
├── LateNight/
│   └── ...
└── BaseColor/
    └── ...
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CSV file not found | Use absolute paths with forward slashes |
| Sequence not created | Check UE5 Output Log for errors |
| Camera not moving | Verify keyframe interpolation is CONSTANT |
| Black renders | Check lighting and exposure settings |
| Out of memory | Reduce tile count or resolution |

### Performance Tips

1. Close unnecessary editor windows during rendering
2. Disable real-time rendering in viewport
3. Use GPU with sufficient VRAM (16GB+ recommended)
4. Render to SSD for faster I/O

---

## Reference

### Camera Parameters

Default camera model based on typical UAV specifications:

| Parameter | Value | Real-world Equivalent |
|-----------|-------|----------------------|
| Focal Length | 35mm | Standard wide-angle |
| Sensor Size | 36×24mm | Full-frame |
| Resolution | 6000×4000 | 24MP |
| Field of View | 45° | Moderate wide |

### Coordinate Conversion

```python
# UE5 uses centimeters, convert from meters:
x_cm = x_m * 100
y_cm = y_m * 100
z_cm = z_m * 100

# Rotation: UE5 uses (Roll, Pitch, Yaw) order
# Roll  = Rotation around X axis
# Pitch = Rotation around Y axis  
# Yaw   = Rotation around Z axis
```

---

## License

This tutorial is part of the NightUAV-Sim project. See the main repository for license details.
