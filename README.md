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
- Model ensemble for better results than any single approach
- Content-aware processing through segmentation
- Detail preservation techniques
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
python -m nesr --input path/to/your/image.jpg
```

### Advanced options

```bash
python -m nesr --input input.jpg --output_dir results --iterations 3 --upscale_factor 2.0 --device cuda --prompt "a highly detailed photograph" --intermediate_saves
```

Parameters:
- `--input`, `-i`: Path to input image (required)
- `--output_dir`, `-o`: Output directory (default: "outputs")
- `--iterations`, `-n`: Number of enhancement iterations (default: 3)
- `--upscale_factor`, `-u`: Base upscale factor per iteration (default: 2.0)
- `--device`, `-d`: Device to run on ("cuda" or "cpu", default: "cuda")
- `--prompt`, `-p`: Text prompt for diffusion guidance
- `--no_diffusion`: Disable diffusion-based upscaling
- `--intermediate_saves`: Save intermediate results after each iteration

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

NESR typically achieves higher quality results than single-model approaches, with particularly noticeable improvements in:
- Texture detail
- Edge sharpness
- Natural-looking results
- Artifact reduction

## Limitations

- High GPU memory requirements when using all models
- Processing time increases with iterations and image size
- May produce artifacts with very low-quality input images

## License

MIT
