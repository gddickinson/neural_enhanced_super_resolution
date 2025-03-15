# NESR Usage Guide

This guide explains how to install and use the Neural Enhanced Super-Resolution (NESR) application.

## Installation

### Prerequisites

- Python 3.7 or newer
- CUDA-compatible GPU (recommended, but CPU mode is supported)
- 8GB+ RAM (16GB+ recommended for larger images)
- CUDA Toolkit 11.0+ and appropriate drivers (for GPU acceleration)

### Method 1: Install from Source

1. Clone the repository or extract the source code
2. Navigate to the project directory
3. Install the package and dependencies:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# For GUI support, install additional dependencies
pip install -e ".[gui]"
```

### Method 2: Direct Installation

You can also install dependencies directly:

```bash
pip install -r requirements.txt
```

## Running the Application

### GUI Mode

The easiest way to use NESR is through the graphical interface:

```bash
python main.py
```

or

```bash
python main.py --gui
```

### Command Line Mode

For batch processing or automation, use the command line mode:

```bash
python main.py --cli --input image.jpg --iterations 3 --upscale_factor 2.0
```

Common CLI options:

- `--input`: Path to input image (required for CLI mode)
- `--output`: Custom output path (optional)
- `--iterations`: Number of enhancement iterations (default: 3)
- `--upscale_factor`: Base upscale factor per iteration (default: 2.0)
- `--device`: Computing device ("cuda" or "cpu")
- `--prompt`: Text prompt for diffusion-based guidance
- `--download_models`: Download required models and exit

## Using the GUI

### Main Interface

1. **Open an image**: Click "Open Image" or use the toolbar button
2. **View the image**: Use zoom controls to inspect details
3. **Configure settings**: Adjust parameters in the settings tabs
4. **Enhance**: Click "Enhance" to start the process
5. **Save result**: Click "Save Result" to save the enhanced image

### Image Degradation (Testing)

To test enhancement on artificially degraded images:

1. Open an image
2. Go to the "Degradation" tab
3. Configure degradation parameters:
   - Downscaling: Reduce resolution
   - Noise: Add various types of noise
   - Blur: Apply different blur effects
   - JPEG compression: Add compression artifacts
4. Click "Preview" to see the effect
5. Click "Apply" to use the degraded image as input for enhancement

### Enhancement Settings

The "Enhancement" tab provides options to configure the super-resolution process:

- **Basic Settings**:
  - Iterations: Number of enhancement passes (more = higher quality but slower)
  - Upscale factor: Resolution multiplier per iteration
  - Save intermediate: Save results from each iteration

- **Models**:
  - Select which models to use (ESRGAN, Diffusion, Segmentation)
  - Configure text prompt for diffusion guidance

- **Advanced Settings**:
  - Denoise level: Strength of noise reduction
  - Preserve details: Enhance important image features
  - Adaptive sharpening: Intelligently sharpen based on content
  - Device selection: GPU or CPU processing

### Presets

Save and load configurations for different use cases:

- **Default**: Balanced settings for general use
- **High Quality**: Maximum quality, slower processing
- **Fast**: Quick enhancement with fewer models

## Tips for Best Results

1. **Start with small images**: Processing time increases with image size
2. **Adjust prompt**: For diffusion-based enhancement, tailor the prompt to the image content (e.g., "a detailed landscape photograph" for landscapes)
3. **Use multiple iterations**: For severely degraded images, more iterations often yield better results
4. **Balance models**: For speed, disable diffusion; for quality, enable all models
5. **Try different presets**: Compare results with different configuration presets

## Troubleshooting

- **Out of memory errors**: Reduce image size or number of iterations, or switch to CPU mode
- **Missing models**: Use the "Download Models" button or run with `--download_models`
- **Slow processing**: Enable fewer models, reduce iterations, or process smaller images
- **Artifacts in results**: Adjust denoise level or try different model combinations
