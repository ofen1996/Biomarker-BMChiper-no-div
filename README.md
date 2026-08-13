# Whole-Slide Image Alignment Toolkit

A desktop toolkit for chip-array and whole-slide image workflows. It supports region detection, geometric correction, large-image stitching, and multi-channel merging for MRXS, SVS, TIFF, OME-TIFF, and Zarr data.

> This repository is a portfolio implementation. Data, models, and acquisition settings are platform-specific and should be validated for each deployment.

## Highlights

- ONNX keypoint detection for array localization
- Whole-slide image reading and region selection
- Large-image stitching and multi-channel merging
- libvips-based processing to reduce memory pressure
- Configurable magnification, scanner, and tile presets
- PyQt5 desktop interface and PyInstaller specifications

## Getting started

Recommended: Windows and Python 3.9.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pillow opencv-python matplotlib PyQt5 onnxruntime tifffile pyvips scikit-learn
python BMChiper.py
```

MRXS support requires OpenSlide. libvips scripts require a system libvips installation available on PATH.

## Workflow

1. Launch `BMChiper.py`.
2. Open a slide or image directory and select a matching `setting/*.ini` preset.
3. Review the detected array region, then run correction or stitching.
4. Use the libvips utilities to stitch or merge multi-channel datasets.

## Key modules

| Path | Purpose |
| --- | --- |
| `BMChiper.py` | PyQt5 desktop application |
| `match_imgs.py` | Array matching, keypoint localization, geometric alignment |
| `stitch_pic.py` | Image-stitching workflow |
| `merge_*_by_libvips.py` | Memory-efficient large-image merge utilities |
| `need/` | Image IO, UI, correction, and support modules |
| `model/*.onnx` | Keypoint-detection models |
| `setting/` | Scanner and chip presets |

## Notes

Large-image processing can consume significant RAM, disk space, and processing time. Validate settings on representative cropped data before batch runs. Some paths and configurations are Windows-oriented and may require adjustment on other platforms.

## License

No open-source license is currently declared. Contact the repository owner before reusing the code, models, or configuration assets.