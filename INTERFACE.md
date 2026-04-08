# Neural Enhanced Super-Resolution - Interface Map

## Entry Point
- **main.py**: CLI entry point with argparse for running the super-resolution pipeline.

## Core Package: nesr/
- **nesr/__init__.py**: Package init. Applies torchvision compatibility patches, exports `SuperResolutionPipeline`.
- **nesr/nesr.py**: `SuperResolutionPipeline` class - main orchestration for iterative super-resolution.
  Key methods: `enhance_image()`, `_load_models()`, `_apply_esrgan()`, `_apply_diffusion()`,
  `_ensemble_results()`, `_preprocess_image()`, `_postprocess_image()`, `_segment_and_enhance()`,
  `_process_with_tiling()`.

## GUI: nesr/gui/
- **nesr/gui/__init__.py**: GUI subpackage init.
- **nesr/gui/app.py**: PyQt5/PySide GUI application for interactive super-resolution.

## Utilities: nesr/utils/
- **nesr/utils/__init__.py**: Utils subpackage init.
- **nesr/utils/image_utils.py**: Image manipulation functions: `add_noise()`, `blur_image()`,
  `downsample_image()`, `apply_jpeg_compression()`, `create_comparison_image()`, `add_text_to_image()`.
- **nesr/utils/downloader.py**: Model download management. `download_models()`, `download_file()`,
  `check_models_exist()`, `get_model_info()`. Handles Real-ESRGAN weights and HuggingFace models.
- **nesr/utils/torchvision_patch.py**: Compatibility patch for newer torchvision versions.
  Must call `apply_patches()` before importing realesrgan/basicsr.

## Standalone Scripts: standalone/
- **standalone/direct_esrgan.py**: Standalone ESRGAN upscaler for debugging (bypasses NESR framework).
- **standalone/download-x3-model.py**: Downloads 3-channel ESRGAN model variant.
- **standalone/superres_project.py**: Earlier standalone pipeline (predates nesr/ package). Reference only.

## Tests: tests/
- **tests/test_image_utils.py**: 19 unit tests covering all image_utils functions.

## Config
- **requirements.txt**: Pinned dependency versions with upper bounds.
- **setup.py**: Package installation configuration.
