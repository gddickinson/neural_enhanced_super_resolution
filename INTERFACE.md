# Neural Enhanced Super-Resolution - Interface Map

## Entry Point
- **main.py**: CLI entry point with argparse for running the super-resolution pipeline.

## Core Package: nesr/
- **nesr/__init__.py**: Package init. Applies torchvision compatibility patches, exports `SuperResolutionPipeline`.
- **nesr/nesr.py**: `SuperResolutionPipeline` class - main orchestration for iterative super-resolution.
  Key methods (load failures in `model_errors`): `enhance_image()`, `_load_models()`, `_apply_esrgan()`, `_apply_diffusion()`,
  `_ensemble_results()`, `_preprocess_image()`, `_postprocess_image()`, `_segment_and_enhance()`,
  `_process_with_tiling()`.

- **nesr/metrics.py**: `psnr()`, `ssim()` (Gaussian 11x11, sigma 1.5; matches skimage), `ssim_map()`, `gaussian_filter()`.
  Numpy only. Ported to the browser as nesr/web/static/metrics.js.
- **nesr/test_patterns.py**: synthetic test patterns (`PATTERNS`, `make_pattern(name, size)`): zone plate, Siemens star,
  resolution bars, colour shapes. Used for the web examples and tests.

## Web interface: nesr/web/ (FastAPI + plain ES modules; `run_web.py` starts it on 127.0.0.1:8168)
- **nesr/web/app.py**: routes `/api/info`, `/api/examples`, `/api/docs/{name}`, POST `/api/upscale`; `STATIC_ENDPOINTS`.
- **nesr/web/sr_methods.py**: `METHODS` (nearest, bicubic, lanczos, nesr_basic, realesrgan, nesr_esrgan, nesr_full),
  `availability()`, `describe()`, `run(method, image, scale)`: cv2.resize or `SuperResolutionPipeline` (cached per method).
- **nesr/web/uploads.py**: `process(img, method, scale, mode)` for upscale / benchmark requests, `decode()`, `png_data_url()`.
- **nesr/web/examples.py**: precomputed examples (patterns x factors 2, 4 x `EXAMPLE_METHODS`) in `static/examples/`
  (PNGs + examples.json with PSNR/SSIM); `load_manifest()`, `make_lr()`, `hr_image()`.
- **nesr/web/static/**: index.html, style.css (shared kit theme), style_app.css, core.js (router, api), main.js,
  state.js, images.js (pixel helpers), slider.js (before/after), pages home/compare/upscale/metrics_page/docs.js,
  md.js (markdown), static_api.js (static-site data layer), resample.js (port of cv2.resize nearest/bicubic/Lanczos-4)
  and metrics.js, with node tests resample.test.mjs and metrics.test.mjs.

## GUI: nesr/gui/
- **nesr/gui/__init__.py**: GUI subpackage init.
- **nesr/gui/app.py**: PyQt5/PySide GUI application for interactive super-resolution.

## Utilities: nesr/utils/
- **nesr/utils/__init__.py**: Utils subpackage init.
- **nesr/utils/image_utils.py**: Image manipulation functions: `add_noise()`, `blur_image()`,
  `downsample_image()`, `apply_jpeg_compression()`, `create_comparison_image()`, `add_text_to_image()`.
- **nesr/utils/downloader.py**: Model download management (upstream URLs, sizes, SHA-256). `download_models()`, `download_file()`, `calculate_sha256()`,
  `check_models_exist()`, `get_model_info()`. Handles Real-ESRGAN weights and HuggingFace models.
- **nesr/utils/torchvision_patch.py**: Compatibility patch for newer torchvision versions.
  Must call `apply_patches()` before importing realesrgan/basicsr.

## Standalone Scripts: standalone/
- **standalone/direct_esrgan.py**: Standalone ESRGAN upscaler for debugging (bypasses NESR framework).
- **standalone/download-x3-model.py**: Downloads 3-channel ESRGAN model variant.
- **standalone/superres_project.py**: Earlier standalone pipeline (predates nesr/ package). Reference only.

## Scripts: scripts/
- **scripts/make_examples.py**: regenerate nesr/web/static/examples/ (Real-ESRGAN outputs need torch/basicsr/realesrgan).
- **scripts/make_js_fixtures.py**: write tests/fixtures/js_cases.json (OpenCV resize + nesr.metrics outputs for the JS tests).
- **scripts/build_static.py**: export the static GitHub Pages site (site/); **scripts/screenshots.py**: smoke test + docs/img.

## Tests: tests/
- **tests/test_image_utils.py**: 19 unit tests covering all image_utils functions.
- **tests/test_import_without_torch.py**, **tests/test_pipeline_config.py**, **tests/test_esrgan_wiring.py**: regression
  tests for the 2026-11 fixes (the last has a real-model test that needs torch, basicsr, realesrgan).
- **tests/test_metrics.py**: metrics vs scikit-image, test patterns.
- **tests/test_downloader.py**: upstream URLs, sizes and SHA-256 of the Real-ESRGAN weights; hash verification (mocked requests).
- **tests/test_segmentation_loading.py**: SegFormer loaded as semantic segmentation; load failures reported (mocked transformers).
- **tests/test_web.py** (kit: endpoints, static build, node tests), **tests/test_web_api.py** (methods, examples, upscale,
  fixture is current).

## Config
- **requirements.txt**: Pinned dependency versions with upper bounds.
- **requirements-web.txt**: pinned web/test dependencies (no torch); **run_web.py**: starts the web app.
- **.github/workflows/pages.yml**: tests and publishes the static site on push to main.
- **setup.py**: Package installation configuration.
