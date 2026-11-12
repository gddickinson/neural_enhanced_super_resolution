# Neural Enhanced Super-Resolution (NESR)

A Python project for iterative image super-resolution using multiple AI models.

## Overview

NESR combines state-of-the-art super-resolution techniques to enhance image resolution in an iterative manner. The pipeline leverages multiple models:

- **Real-ESRGAN**: Powerful GAN-based upscaler for realistic texture generation
- **Stable Diffusion Upscaler**: Text-guided diffusion model for high-quality upscaling
- **Segmentation-based enhancement**: Targeted improvements for specific image regions
- **Adaptive post-processing**: Detail-aware sharpening and contrast enhancement

Each enhancement iteration can increase resolution while preserving and enhancing details using a multi-model ensemble approach.

## Features

- Multi-stage iterative enhancement
- Model ensemble (equal-weight average of the Real-ESRGAN and diffusion outputs)
- Content-aware processing through segmentation
- Guided upscaling using text prompts
- Configurable pipeline with sensible defaults

## Requirements

```
torch>=1.12.0
diffusers>=0.14.0
transformers>=4.25.0
opencv-python>=4.6.0
Pillow>=9.3.0
numpy>=1.23.0
tqdm>=4.64.0
realesrgan>=0.3.0
basicsr>=1.4.2
```

You'll also need to download model weights:
- Real-ESRGAN weights: [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases)
- Stable Diffusion upscaler is downloaded automatically via Hugging Face

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `weights` directory and download the model weights

## Usage

### Basic usage

```bash
python main.py --cli --input path/to/your/image.jpg
```

Without `--cli` and `--input`, `main.py` opens the desktop GUI (PyQt5).

No sample photo ships with the repo. To try it on a synthetic test pattern (formula-generated, no licence questions):

```bash
python -c "import cv2; from nesr.test_patterns import make_pattern; cv2.imwrite('siemens_star.png', make_pattern('siemens_star', 256))"
python main.py --cli --input siemens_star.png --iterations 1 --device cpu
```

Patterns: `zone_plate`, `siemens_star`, `bar_chart`, `color_shapes` (see `nesr/test_patterns.py`).

### Advanced options

```bash
python main.py --cli --input input.jpg --output results/enhanced.png --iterations 3 --upscale_factor 2.0 --device cpu --prompt "a highly detailed photograph"
```

Parameters (from `main.py`):
- `--cli`: Run in command line mode (`--gui` opens the GUI)
- `--input`, `-i`: Path to input image
- `--output`, `-o`: Output image path (default: `outputs/<name>_enhanced_x<scale><ext>`)
- `--config`, `-c`: JSON file with pipeline settings (any key of the `SuperResolutionPipeline` config, e.g.
  `{"use_diffusion": false, "intermediate_saves": true}`)
- `--iterations`, `-n`: Number of enhancement iterations (default: 3)
- `--upscale_factor`, `-u`: Base upscale factor per iteration (default: 2.0)
- `--device`, `-d`: Device to run on ("cuda" or "cpu", default: "cuda"; falls back to MPS or CPU when CUDA is missing)
- `--prompt`, `-p`: Text prompt for diffusion guidance
- `--download_models`: Download the model weights and exit

### Python API

```python
from nesr import SuperResolutionPipeline

# Configure the pipeline
config = {
    "iterations": 3,
    "upscale_factor": 2.0,
    "use_diffusion": True,
    "output_dir": "results",
}

# Initialize the pipeline
pipeline = SuperResolutionPipeline(device="cuda", config=config)

# Enhance an image
enhanced_image_path = pipeline.enhance_image(
    "input.jpg", 
    prompt="a high resolution detailed photograph"
)

print(f"Enhanced image saved to: {enhanced_image_path}")
```

## How It Works

1. **Preprocessing**: Reduce noise and enhance contrast
2. **Segmentation**: Identify image regions for targeted enhancement
3. **Multi-model upscaling**: Apply different SR models in parallel
4. **Ensemble**: Combine results from different models
5. **Post-processing**: Apply adaptive sharpening to enhance details
6. **Iteration**: Repeat the process for progressive enhancement

## Results

Measured on four synthetic 256 x 256 test patterns (`nesr/test_patterns.py`), each shrunk by 2 or 4 with
`downsample_image(..., 'bicubic')` and restored; scores against the original pattern (PSNR in dB / SSIM, higher is
better, best per row in bold). Produced by `scripts/make_examples.py` on CPU (torch 2.4.1, basicsr 1.4.2,
realesrgan 0.3.0, OpenCV 4.10); the full table is in `nesr/web/static/examples/examples.json`.

| Pattern | Factor | Nearest | Bicubic | Lanczos | NESR basic | Real-ESRGAN | NESR + ESRGAN |
|---|---|---:|---:|---:|---:|---:|---:|
| Zone plate | x2 | **12.47** / **0.413** | 12.43 / 0.318 | 12.11 / 0.305 | 10.26 / 0.314 | 11.46 / 0.189 | 9.70 / 0.183 |
| Zone plate | x4 | 9.58 / 0.053 | **10.08** / **0.087** | 9.88 / 0.083 | 8.08 / 0.075 | 9.86 / 0.049 | 8.42 / 0.042 |
| Siemens star | x2 | 19.81 / 0.898 | 22.89 / **0.932** | 22.93 / 0.886 | 22.04 / 0.841 | **25.89** / 0.895 | 22.98 / 0.864 |
| Siemens star | x4 | 14.51 / 0.697 | 17.11 / 0.748 | 17.02 / 0.659 | 14.59 / 0.624 | **17.19** / **0.752** | 14.01 / 0.621 |
| Resolution bars | x2 | **27.90** / **0.969** | 24.14 / 0.918 | 23.30 / 0.846 | 24.72 / 0.937 | 26.84 / 0.929 | 24.84 / 0.953 |
| Resolution bars | x4 | **21.37** / **0.906** | 17.86 / 0.711 | 17.70 / 0.678 | 17.46 / 0.801 | 17.70 / 0.817 | 13.79 / 0.719 |
| Colour shapes | x2 | 26.94 / 0.959 | 29.57 / **0.970** | 29.54 / 0.965 | 21.58 / 0.898 | **30.32** / 0.958 | 21.64 / 0.911 |
| Colour shapes | x4 | 21.66 / 0.884 | 23.04 / **0.893** | **23.11** / 0.874 | 14.59 / 0.694 | 20.94 / 0.873 | 15.01 / 0.751 |
- *NESR basic*: `enhance_image` with no neural models (denoising + CLAHE, bicubic x2 per iteration, adaptive sharpening).
- *Real-ESRGAN*: the pipeline's RealESRGAN_x2plus step alone, once per x2.
- *NESR + ESRGAN*: `enhance_image` with Real-ESRGAN, without diffusion or segmentation.

On these patterns the pipeline's pre- and post-processing lowers PSNR and SSIM compared with bicubic in most rows, and
Real-ESRGAN alone has the best PSNR in 3 of 8 rows. The default configuration with the Stable Diffusion upscaler and
segmentation has not been measured. Synthetic patterns are an unusual test (Real-ESRGAN was trained on photographs), so
these numbers say nothing about photographs; no photographic benchmark has been run.

## Web interface

A browser front end to the same code, plus a static showcase site (GitHub Pages:
https://gddickinson.github.io/neural_enhanced_super_resolution/).

![Home](docs/img/home.png)

- **Compare**: before/after slider on synthetic test patterns (zone plate, Siemens star, resolution bars, colour shapes;
  `nesr/test_patterns.py`) shrunk by 2 or 4 and restored by nearest, bicubic, Lanczos-4, the NESR pipeline without neural
  models, Real-ESRGAN x2plus alone, and the NESR pipeline with Real-ESRGAN.
- **Upscale**: your own image or a pattern, in "upscale" mode or "benchmark" mode (shrink, restore, score).
- **Metrics**: PSNR and SSIM (`nesr/metrics.py`) of every precomputed output, error and SSIM maps.
- **Pipeline**: what each stage of `SuperResolutionPipeline` does.

![Compare](docs/img/compare.png)
![Metrics](docs/img/metrics.png)

### Run the web app

```bash
pip install -r requirements-web.txt   # classical methods, pipeline without neural models, metrics
pip install -r requirements.txt       # optional: Real-ESRGAN and the diffusion upscaler
python run_web.py                     # http://127.0.0.1:8168 (local only)
```

In the local app every method runs in Python on your computer (models only if installed). On the static site the model
outputs are precomputed (`python scripts/make_examples.py`, CPU), while nearest, bicubic and Lanczos (`resample.js`, an exact
port of `cv2.resize`) and PSNR/SSIM (`metrics.js`) run in the browser; node tests check both against Python
(`tests/fixtures/js_cases.json`, from `scripts/make_js_fixtures.py`). Build the site with `python scripts/build_static.py`.

Tests: `python -m pytest -q tests` (the real Real-ESRGAN test runs only when torch, basicsr and realesrgan are installed).

## Limitations

- High GPU memory requirements when using all models
- Processing time increases with iterations and image size
- May produce artifacts with very low-quality input images

## License

MIT


---
*Built with AI assistance from [Claude (Anthropic)](https://claude.com/).*
