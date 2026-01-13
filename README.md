# NightUAV-Sim

**A Synthetic Benchmark for Nighttime UAV 3D Reconstruction**

🌐 **Project Page**: [https://yourusername.github.io/NightUAV-Sim/](https://yourusername.github.io/NightUAV-Sim/)

---

## 📖 Overview

NightUAV-Sim is the first synthetic benchmark dataset designed specifically for nighttime UAV 3D reconstruction. The dataset provides:

- **19,836** high-resolution images (8196×5460 pixels, ~45MP)
- **6** illumination conditions spanning the complete day-night cycle
- Pixel-aligned day-night image pairs
- Complete geometric ground truth (depth maps, surface normals, camera parameters)
- **2 km²** photorealistic urban scene coverage

## 📊 Dataset Statistics

| Property | Value |
|----------|-------|
| Total Images | 19,836 |
| Resolution | 8196 × 5460 (~45MP) |
| Lighting Conditions | 6 (Noon → Late Night) |
| Scene Coverage | 2 km² |
| Imaging Modes | Nadir (551) + Oblique (2,755) |
| Overlap | 80% forward / 60% side |

## 🌙 Illumination Conditions

1. **Noon** - Full daylight
2. **Afternoon** - Golden hour lighting
3. **Dusk** - Twilight transition
4. **Early Night** - City lights emerging
5. **Late Night** - Full nighttime
6. **Base Color** - Albedo reference

## 📥 Download

| Data | Size | Link |
|------|------|------|
| RGB Images | ~XX GB | Coming Soon |
| Ground Truth | ~XX GB | Coming Soon |
| Code & Models | - | [GitHub](https://github.com/yourusername/NightUAV-Sim) |

## 🗂️ Dataset Structure

```
NightUAV-Sim/
├── images/
│   ├── noon/
│   │   ├── nadir/
│   │   └── oblique/
│   ├── afternoon/
│   ├── dusk/
│   ├── early_night/
│   ├── late_night/
│   └── base_color/
├── ground_truth/
│   ├── depth/
│   ├── normals/
│   └── camera_params/
└── metadata.json
```

## 🔧 Usage

```python
# Example: Load dataset
from nightuav import NightUAVDataset

dataset = NightUAVDataset(
    root='path/to/NightUAV-Sim',
    lighting='late_night',
    mode='oblique'
)

for image, depth, normal, camera in dataset:
    # Your code here
    pass
```

## 📄 License

This dataset is released under the [MIT License](LICENSE).

## 📧 Contact

For questions or collaborations, please contact:
- Email: your-email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/NightUAV-Sim/issues)

## 🙏 Acknowledgments

This work was developed for the IGARSS 2026 Student Competition.

---

⭐ If you find this dataset useful, please consider giving us a star!
