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

Follow-up fixes (same day, coordinator's request; each a "Fix ..." commit with tests):
5. `images/test.jpeg` untracked and ignored (EXIF GPS + someone else's mural, no licence); README shows a synthetic
   pattern instead. It is still in git history (removing it needs a history rewrite).
6. Downloader: x2plus URL pointed at release v0.2.5.0 (404; the file is in v0.2.1) in downloader.py, nesr.py and
   standalone/direct_esrgan.py; MD5 constants matched neither upstream file. Verified by downloading the upstream
   assets: x2plus 67,061,725 B sha256 49fafd45...6abb (identical to the committed file), x4plus 67,040,989 B sha256
   4fa0d389...82f1. Now SHA-256 verification; a full-size file that fails verification is downloaded again.
7. GUI "Force 3-channel mode" removed (no effect after fix 3), with its config key and `_apply_esrgan_12channel`.
   Regenerated examples are byte-identical PNGs.
8. Segmentation: SegFormer now loads with AutoModelForSemanticSegmentation + AutoImageProcessor (the old class raised
   "Unrecognized configuration class SegformerConfig", verified with transformers 4.45.2; the new one loads and
   `_segment_and_enhance` runs). Load failures go to `pipeline.model_errors`, warnings and the progress callback.
9. README "Results" now quotes the measured PSNR/SSIM table; unverified quality claims removed.

Open points for George:
- `images/test.jpeg` remains in git history (GPS); a history rewrite would remove it.
- The `preserve_details` setting (GUI checkbox and presets) is not used by the pipeline.
- `main.py --config file.json` always overrides the file's iterations / upscale_factor with the CLI defaults.
- `nesr/nesr.py` (977 lines) and `nesr/gui/app.py` (1828 lines) are over the 500-line budget (pre-existing; ROADMAP item).

## 2026-11-12: git history rewrite (approved by George)
- Purged from all history with git-filter-repo and force-pushed: images/test.jpeg (photo with EXIF GPS).
  The current tree is unchanged (identical HEAD tree before/after); commit hashes changed, so old clones must be
  re-cloned or reset to origin. A pre-rewrite mirror backup was kept outside the repo.
