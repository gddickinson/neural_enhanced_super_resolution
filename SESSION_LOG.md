# Session log

## 2026-11-12: web interface and static site (branch web-interface)

Added a FastAPI web app (`nesr/web/`, `python run_web.py`, port 8168) and a static GitHub Pages export.

- Pages: Home, Compare (before/after slider), Upscale (your image, upscale or benchmark mode), Metrics (PSNR/SSIM table,
  error and SSIM maps, browser-vs-Python check), Pipeline, README / Usage guide / Project structure, About.
- Examples use only synthetic test patterns (new `nesr/test_patterns.py`); `images/test.jpeg` was not used (a photo of a
  mural, third-party artwork with no licence statement).
- New `nesr/metrics.py` (PSNR, SSIM; equal to skimage to 1e-9); ports `resample.js` (cv2.resize nearest / bicubic /
  Lanczos-4, identical 8-bit output on OpenCV 4.10 and 5.0) and `metrics.js`, node-tested against Python fixtures.
- Precomputed outputs (`scripts/make_examples.py`, CPU, torch 2.4.1, basicsr 1.4.2, realesrgan 0.3.0, OpenCV 4.10):
  48 PNGs + examples.json in `nesr/web/static/examples/` (outputs only; no new weights).
- On these patterns the NESR pipeline (CLAHE + sharpening) scores below plain bicubic on PSNR in most cases;
  Real-ESRGAN alone is best on 3 of 8 rows (see the Metrics page). Shown as measured.

Fixes (separate commits, each with a regression test):
1. `import nesr` required torch: `nesr/utils/downloader.py` imported it without using it.
2. `SuperResolutionPipeline(device=...)` with no config raised TypeError (`key not in config` with config None).
3. Real-ESRGAN x2plus was built as RRDBNet(num_in_ch=12, scale=4) and fed copied / brightness-shifted channels, giving
   4x-sized washed-out output (PSNR 16-17 dB vs 25.6 dB upstream on a test pattern). Now RRDBNet(num_in_ch=3, scale=2)
   + RealESRGANer.enhance(outscale=upscale_factor), matching upstream to 1 grey level.
4. README CLI usage (`python -m nesr`, `--output_dir`, `--no_diffusion`, `--intermediate_saves` do not exist) now matches main.py.

Open points for George:
- `images/test.jpeg` carries EXIF GPS coordinates (iPhone XS, 2023-05-31) in a public repo; strip it (and history?) if unwanted.
- `downloader.py` MODELS md5 for RealESRGAN_x2plus (5db904...) differs from the committed weights (f2a817...; sha256
  49fafd45..., size 67,061,725 bytes, which looks like the upstream file). If the constant is wrong, downloads report
  "MD5 verification failed". Not changed (upstream checksum not verified here).
- The GUI's "Force 3-channel mode" checkbox no longer changes anything after fix 3.
- Segmentation loads SegFormer with AutoModelForImageSegmentation (SegFormer is registered for semantic segmentation);
  likely always fails and is silently disabled. Not verified (needs a Hugging Face download).
- `main.py --config file.json` always overrides the file's iterations / upscale_factor with the CLI defaults.
- `nesr/nesr.py` (1013 lines) and `nesr/gui/app.py` (1840 lines) are over the 500-line budget (pre-existing; ROADMAP item).
