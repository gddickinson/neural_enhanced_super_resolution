# Neural Enhanced Super-Resolution — Roadmap

## Current State
A well-structured Python package (`nesr/`) with proper `__init__.py`, a GUI app (`nesr/gui/app.py`), utility modules (`image_utils.py`, `downloader.py`), `setup.py`, and `requirements.txt`. The pipeline combines Real-ESRGAN, Stable Diffusion upscaler, and segmentation-based enhancement. Has a `standalone/` directory and `docs/` folder. Good foundation with proper packaging, but depends on heavy GPU libraries with no graceful CPU fallback documentation.

## Short-term Improvements
- [x] Add a `torchvision_patch.py` usage comment or integrate it properly — its purpose is unclear from the name alone
- [ ] Add progress callbacks to `SuperResolutionPipeline.enhance_image()` for GUI integration
- [x] Improve error messages when model weights are missing (currently likely fails with cryptic torch errors)
- [x] Add input image validation (format, minimum size, color mode) before pipeline starts
- [ ] Add `--cpu` flag documentation and ensure all models gracefully fall back to CPU
- [x] Pin dependency versions in `requirements.txt` to avoid breaking changes from diffusers/transformers updates

## Feature Enhancements
- [ ] Add batch processing: accept a directory of images and process them sequentially
- [ ] Add a comparison view in the GUI showing before/after side-by-side
- [ ] Support face-specific enhancement using GFPGAN or CodeFormer alongside Real-ESRGAN
- [ ] Add SSIM/PSNR quality metrics output when a reference image is available
- [ ] Support tiled processing for very large images that exceed GPU memory
- [ ] Add model selection in CLI (choose between Real-ESRGAN variants: x2, x4, anime)
- [ ] Add an option to preserve original EXIF metadata in output images

## Long-term Vision
- [ ] Add a REST API (FastAPI) for remote super-resolution processing
- [ ] Support video super-resolution with temporal consistency (frame-by-frame + optical flow)
- [ ] Implement model quantization (INT8/FP16) for faster inference on consumer hardware
- [ ] Add a model benchmark tool comparing quality/speed across different SR models
- [ ] Publish to PyPI as an installable package

## Technical Debt
- [x] Document the `standalone/` scripts — what do they do and when should they be used?
- [x] Add unit tests for `image_utils.py` (resize, color conversion, normalization)
- [ ] Add integration tests with a small test image to verify the full pipeline
- [x] Review `downloader.py` for proper error handling on network failures
- [ ] Add type hints throughout `nesr.py` pipeline class
- [ ] Set up CI with a lightweight test that runs on CPU only
- [ ] Split `nesr/nesr.py` (1084 lines) into smaller modules (model loading, processing, post-processing)
- [ ] Split `nesr/gui/app.py` (1840 lines) into smaller GUI modules
