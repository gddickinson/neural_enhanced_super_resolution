# Standalone Scripts

These scripts can be run independently of the main NESR package for debugging,
testing, or one-off tasks.

## Scripts

### direct_esrgan.py
A standalone ESRGAN upscaler that bypasses the NESR framework. Use this to verify
that the Real-ESRGAN model works correctly on its own, independent of the full
pipeline. Helpful for diagnosing whether an issue is in ESRGAN or the NESR wrapper.

Usage: `python direct_esrgan.py --input image.png --output result.png`

### download-x3-model.py
Downloads a 3-channel ESRGAN model (v0.3.0) to avoid channel mismatch issues that
occur with the default 12-channel model on certain image types. Run this if you
encounter "expected 3 channels but got 12" errors.

Usage: `python download-x3-model.py`

### superres_project.py
An earlier standalone version of the full super-resolution pipeline. Contains a
self-contained `SuperResolutionPipeline` class that predates the modular `nesr/`
package. Kept for reference; prefer using the `nesr` package directly.
