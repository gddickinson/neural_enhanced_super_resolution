"""
NESR - Neural Enhanced Super-Resolution
Main module containing the SuperResolutionPipeline class.
Updated with Apple Silicon (M1/M2/M3) GPU support via MPS.
"""

import os
import cv2
import numpy as np
import logging
import time
from PIL import Image
import math

# Set up logging
logger = logging.getLogger('nesr')

class SuperResolutionPipeline:
    """Main orchestration class for the iterative super-resolution process."""

    def __init__(self, device='auto', config=None):
        """Initialize the super resolution pipeline.

        Args:
            device: Device to run models on ('cuda', 'mps', 'cpu', or 'auto')
            config: Configuration dictionary for customizing the pipeline
        """
        # Check for available devices
        self.device = self._determine_device(device)
        logger.info(f"Using device: {self.device}")

        # Default configuration
        self.config = {
            'iterations': 3,
            'use_diffusion': True,
            'use_esrgan': True,
            'use_swinir': False,  # Not implemented in this version
            'preserve_details': True,
            'adaptive_sharpening': True,
            'segment_enhancement': True,
            'denoise_level': 0.5,
            'upscale_factor': 2,
            'intermediate_saves': False,
            'output_dir': 'outputs',
            'progress_callback': None,
            'image_callback': None,
            # Memory management options
            'force_3channel': False,     # Force using 3-channel mode even on smaller images
            'max_tile_size': 512,        # Maximum tile size for tiled processing
            'enable_tiling': True,       # Enable tiled processing for large images
            'memory_efficient': False,   # Use more memory-efficient but potentially slower processing
            'device_specific': {         # Device-specific configurations
                'mps': {                 # Apple Silicon specific settings
                    'force_3channel': True,  # MPS works better with 3-channel
                    'max_megapixels': 4,     # Max megapixels before using tiling on MPS
                    'fallback_to_cpu': True, # Fallback to CPU for unsupported operations
                },
                'cuda': {
                    'half_precision': True,   # Use half precision on CUDA for memory efficiency
                },
                'cpu': {
                    'max_megapixels': 2,      # Lower threshold for CPU mode
                }
            }
        }

        # Update with user config if provided
        if config:
            self.config.update(config)

        # Apply device-specific configs if available
        if self.device in self.config['device_specific']:
            device_config = self.config['device_specific'][self.device]
            for key, value in device_config.items():
                # Only set if not explicitly overridden in user config
                if key not in config:
                    self.config[key] = value

        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Initialize models
        self.models = {}
        self._init_models()

    def _determine_device(self, requested_device='auto'):
        """Determine the best available device to use.

        Args:
            requested_device: Requested device ('cuda', 'mps', 'cpu', or 'auto')

        Returns:
            String representing the device to use
        """
        try:
            import torch

            # Auto-detect the best available device
            if requested_device == 'auto':
                if torch.cuda.is_available():
                    return 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    return 'cpu'

            # Specific device requested - check if available
            if requested_device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif requested_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            elif requested_device == 'cpu':
                return 'cpu'

            # Requested device not available, fall back
            if requested_device == 'cuda':
                logger.warning("CUDA requested but not available, checking for MPS")
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    logger.warning("CUDA and MPS not available, falling back to CPU")
                    return 'cpu'
            elif requested_device == 'mps':
                logger.warning("MPS requested but not available, falling back to CPU")
                return 'cpu'

            # Default fallback
            return 'cpu'

        except ImportError:
            logger.warning("PyTorch not found, using CPU mode only")
            return 'cpu'

    def _init_models(self):
        """Initialize the super-resolution models based on config."""
        # Only initialize models when they're first needed, not at startup
        pass


    def _load_models(self):
        """Load models on demand with improved error handling."""
        # Only load models that aren't already loaded
        if self.config['use_esrgan'] and 'esrgan' not in self.models:
            try:
                logger.info("Loading Real-ESRGAN model...")
                # Import with explicit error handling
                try:
                    # First check if the modules are available
                    import sys
                    import importlib.util

                    # Check if basicsr is installed
                    if importlib.util.find_spec("basicsr") is None:
                        raise ImportError("basicsr module not found")

                    # Check if realesrgan is installed
                    if importlib.util.find_spec("realesrgan") is None:
                        raise ImportError("realesrgan module not found")

                    # Try importing the required modules
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer

                    # Get model path - use a direct approach
                    from pathlib import Path
                    import os

                    # Determine model directory structure
                    home = os.path.expanduser("~")
                    if sys.platform == 'darwin':
                        # macOS
                        base_dir = os.path.join(home, 'Library', 'Application Support', 'NESR')
                    elif sys.platform == 'win32':
                        # Windows
                        base_dir = os.path.join(os.environ['APPDATA'], 'NESR')
                    else:
                        # Linux/Unix
                        base_dir = os.path.join(home, '.nesr')

                    # Look for the model in several possible locations
                    possible_paths = [
                        os.path.join(base_dir, 'models', 'weights', 'RealESRGAN_x2plus.pth'),
                        os.path.join('models', 'weights', 'RealESRGAN_x2plus.pth'),
                        os.path.join('weights', 'RealESRGAN_x2plus.pth'),
                        os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', 'RealESRGAN_x2plus.pth'),
                        os.path.join(os.path.dirname(__file__), '..', 'weights', 'RealESRGAN_x2plus.pth'),
                        os.path.join(os.getcwd(), 'models', 'weights', 'RealESRGAN_x2plus.pth'),
                    ]

                    # Find the first path that exists
                    model_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            model_path = path
                            logger.info(f"Found ESRGAN model at: {path}")
                            break

                    if model_path is None:
                        # Download the model if it doesn't exist
                        logger.info("ESRGAN model not found, attempting to download...")
                        os.makedirs(os.path.join(base_dir, 'models', 'weights'), exist_ok=True)
                        model_path = os.path.join(base_dir, 'models', 'weights', 'RealESRGAN_x2plus.pth')

                        import requests
                        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth"
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            with open(model_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logger.info(f"ESRGAN model downloaded to: {model_path}")
                        else:
                            raise ValueError(f"Failed to download model: HTTP {response.status_code}")

                    # Create the model architecture
                    model = RRDBNet(num_in_ch=12, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)


                    # Initialize the ESRGAN model - disable tiling to avoid tensor size issues
                    self.models['esrgan'] = RealESRGANer(
                        scale=int(self.config['upscale_factor']),  # Force integer scale
                        model_path=model_path,
                        model=model,
                        tile=0,  # Disable tiling to avoid the size error
                        tile_pad=0,
                        pre_pad=0,
                        half=False,  # Use full precision for better compatibility
                        device=self.device
                    )
                    logger.info("Real-ESRGAN model loaded successfully")

                except ImportError as e:
                    logger.warning(f"Real-ESRGAN import error: {e}")
                    logger.warning("Install with: pip install realesrgan basicsr")
                    raise
                except Exception as e:
                    logger.error(f"Failed to initialize Real-ESRGAN: {e}")
                    raise

            except Exception as e:
                logger.error(f"Error loading Real-ESRGAN model: {e}")

        # Load diffusion model if enabled and requested
        if self.config['use_diffusion'] and 'diffusion' not in self.models:
            try:
                logger.info("Loading Stable Diffusion upscaler...")
                try:
                    import torch
                    from diffusers import StableDiffusionUpscalePipeline

                    # Use appropriate device and dtype
                    if self.device == 'cuda':
                        torch_dtype = torch.float16
                    elif self.device == 'mps':
                        # MPS doesn't fully support float16 for all operations
                        # Using float32 is safer but uses more memory
                        torch_dtype = torch.float32
                    else:
                        torch_dtype = torch.float32

                    self.models['diffusion'] = StableDiffusionUpscalePipeline.from_pretrained(
                        "stabilityai/stable-diffusion-x4-upscaler",
                        torch_dtype=torch_dtype
                    )

                    # Handle MPS device specifically
                    if self.device == 'mps':
                        # When using MPS, it's better to use float32 for attention calculations
                        if hasattr(self.models['diffusion'], "unet"):
                            self.models['diffusion'].unet.to(memory_format=torch.contiguous_format)

                        # Safety mechanism for MPS - fallback to CPU for operations that might not be supported
                        self.models['diffusion'].safety_checker = None

                    self.models['diffusion'] = self.models['diffusion'].to(self.device)
                    logger.info("Stable Diffusion upscaler loaded successfully")

                except ImportError:
                    logger.warning("Diffusers import failed. Install with: pip install diffusers transformers")
                    self.config['use_diffusion'] = False
            except Exception as e:
                logger.error(f"Error loading Stable Diffusion upscaler: {e}")
                self.config['use_diffusion'] = False

        # Load segmentation model if enabled
        if self.config['segment_enhancement'] and 'segmentation' not in self.models:
            try:
                logger.info("Loading segmentation model for targeted enhancement...")
                try:
                    import torch
                    from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation

                    # Get appropriate device
                    device = self.device

                    self.models['segmentation'] = AutoModelForImageSegmentation.from_pretrained(
                        "nvidia/segformer-b0-finetuned-ade-512-512"
                    ).to(device)
                    self.models['segmentation_extractor'] = AutoFeatureExtractor.from_pretrained(
                        "nvidia/segformer-b0-finetuned-ade-512-512"
                    )
                    logger.info("Segmentation model loaded successfully")
                except ImportError:
                    logger.warning("Transformers import failed. Install with: pip install transformers")
                    self.config['segment_enhancement'] = False
            except Exception as e:
                logger.error(f"Error loading segmentation model: {e}")
                self.config['segment_enhancement'] = False


    def _process_with_tiling(self, processor_func, image, tile_size=512, padding=10):
        """
        Process a large image by splitting it into tiles and then combining the results.

        Args:
            processor_func: Function that processes a single tile
            image: Input image
            tile_size: Size of tiles (square)
            padding: Padding to avoid boundary artifacts

        Returns:
            Processed image
        """
        # Import required modules
        import math

        # Get image dimensions
        h, w, c = image.shape

        # No need for tiling if image is smaller than tile_size
        if h <= tile_size and w <= tile_size:
            return processor_func(image)

        # Calculate number of tiles needed
        num_tiles_h = math.ceil(h / tile_size)
        num_tiles_w = math.ceil(w / tile_size)

        # Determine output size based on upscale factor
        upscale_factor = self.config['upscale_factor']
        out_h = int(h * upscale_factor)
        out_w = int(w * upscale_factor)

        # Initialize the output image
        output = np.zeros((out_h, out_w, c), dtype=np.uint8)

        logger.info(f"Processing image in {num_tiles_h}x{num_tiles_w} tiles")

        # Try direct processing first on a small tile to see if it works
        try:
            # Get a small corner of the image
            test_size = min(256, tile_size)
            test_tile = image[:test_size, :test_size]
            test_result = processor_func(test_tile)

            # If we get here, the processor worked
            logger.info("Tile processor test successful")
            processor_works = True
        except Exception as e:
            logger.warning(f"Tile processor test failed: {e}")
            processor_works = False

            # If processor doesn't work, we'll fall back to bicubic for all tiles
            if not processor_works:
                logger.warning("Falling back to bicubic scaling for all tiles")

        # Process each tile
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Calculate tile coordinates with padding
                y_start = max(0, i * tile_size - padding)
                y_end = min(h, (i + 1) * tile_size + padding)
                x_start = max(0, j * tile_size - padding)
                x_end = min(w, (j + 1) * tile_size + padding)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]

                # Process tile
                try:
                    if processor_works:
                        processed_tile = processor_func(tile)
                    else:
                        # Use bicubic upscaling as fallback
                        processed_tile = cv2.resize(
                            tile,
                            (int(tile.shape[1] * upscale_factor), int(tile.shape[0] * upscale_factor)),
                            interpolation=cv2.INTER_CUBIC
                        )

                    # Calculate output coordinates
                    out_y_start = int(y_start * upscale_factor)
                    out_y_end = int(y_end * upscale_factor)
                    out_x_start = int(x_start * upscale_factor)
                    out_x_end = int(x_end * upscale_factor)

                    # Account for padding in the output
                    if padding > 0:
                        pad_upscaled = int(padding * upscale_factor)
                        if y_start > 0:
                            out_y_start += pad_upscaled
                        if y_end < h:
                            out_y_end -= pad_upscaled
                        if x_start > 0:
                            out_x_start += pad_upscaled
                        if x_end < w:
                            out_x_end -= pad_upscaled

                    # Get the corresponding region in the processed tile
                    tile_out_h, tile_out_w = processed_tile.shape[:2]

                    # Calculate the scaling exactly
                    scale_y = tile_out_h / tile.shape[0]
                    scale_x = tile_out_w / tile.shape[1]

                    # Calculate the region in the processed tile
                    tile_y_start = 0 if y_start == 0 else int(padding * scale_y)
                    tile_y_end = tile_out_h if y_end == h else int(tile_out_h - padding * scale_y)
                    tile_x_start = 0 if x_start == 0 else int(padding * scale_x)
                    tile_x_end = tile_out_w if x_end == w else int(tile_out_w - padding * scale_x)

                    # Ensure valid regions (sometimes numerical precision issues)
                    tile_y_start = max(0, min(tile_y_start, tile_out_h-1))
                    tile_y_end = max(tile_y_start+1, min(tile_y_end, tile_out_h))
                    tile_x_start = max(0, min(tile_x_start, tile_out_w-1))
                    tile_x_end = max(tile_x_start+1, min(tile_x_end, tile_out_w))

                    # Assign to output with boundary checks
                    out_height = out_y_end - out_y_start
                    out_width = out_x_end - out_x_start
                    if out_height <= 0 or out_width <= 0:
                        logger.warning(f"Invalid output region for tile ({i},{j}): height={out_height}, width={out_width}")
                        continue

                    # Extract the relevant part from the processed tile
                    processed_region = processed_tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

                    # Resize if shape doesn't match
                    if processed_region.shape[0] != out_height or processed_region.shape[1] != out_width:
                        processed_region = cv2.resize(
                            processed_region,
                            (out_width, out_height),
                            interpolation=cv2.INTER_LANCZOS4
                        )

                    # Assign to output
                    output[out_y_start:out_y_end, out_x_start:out_x_end] = processed_region

                except Exception as e:
                    logger.warning(f"Error processing tile ({i},{j}): {e}")
                    # Use bicubic upscaling as a fallback for this tile
                    bicubic_tile = cv2.resize(
                        tile,
                        (int(tile.shape[1] * upscale_factor), int(tile.shape[0] * upscale_factor)),
                        interpolation=cv2.INTER_CUBIC
                    )

                    # Calculate output coordinates without padding
                    out_y_start = int(i * tile_size * upscale_factor)
                    out_y_end = min(int(h * upscale_factor), int((i + 1) * tile_size * upscale_factor))
                    out_x_start = int(j * tile_size * upscale_factor)
                    out_x_end = min(int(w * upscale_factor), int((j + 1) * tile_size * upscale_factor))

                    # Ensure valid output region
                    if out_y_end > out_y_start and out_x_end > out_x_start:
                        # Resize bicubic tile to match the output region if needed
                        if bicubic_tile.shape[0] != out_y_end - out_y_start or bicubic_tile.shape[1] != out_x_end - out_x_start:
                            bicubic_tile = cv2.resize(
                                bicubic_tile,
                                (out_x_end - out_x_start, out_y_end - out_y_start),
                                interpolation=cv2.INTER_CUBIC
                            )

                        output[out_y_start:out_y_end, out_x_start:out_x_end] = bicubic_tile

        return output

    def enhance_image(self, image_path, prompt=None):
        """
        Enhance an image through multiple iterations of super-resolution.

        Args:
            image_path: Path to the input image
            prompt: Optional text prompt to guide diffusion-based upscaling

        Returns:
            Path to the final enhanced image
        """
        # Load models on demand
        self._load_models()

        # Check if any models were loaded
        if not self.models:
            logger.warning("No models were loaded. Using basic upscaling only.")

        # Load the initial image
        image = self._load_image(image_path)
        original_h, original_w = image.shape[:2]

        # Track the current image throughout iterations
        current_image = image

        # Generate a default prompt if none provided
        if prompt is None and self.config['use_diffusion']:
            prompt = "a high resolution, detailed photograph"

        # Report progress callback if available
        if self.config.get('progress_callback'):
            self.config['progress_callback'](
                "Starting enhancement",
                0,
                self.config['iterations'],
                f"Image size: {original_w}x{original_h}"
            )

        # Perform iterative enhancement
        for iteration in range(self.config['iterations']):
            iteration_start = time.time()
            logger.info(f"Starting iteration {iteration+1}/{self.config['iterations']}")

            if self.config.get('progress_callback'):
                self.config['progress_callback'](
                    "Enhancement",
                    iteration,
                    self.config['iterations'],
                    f"Starting iteration {iteration+1}/{self.config['iterations']}"
                )

            # 1. Apply pre-processing (denoise, adjust contrast)
            if self.config.get('progress_callback'):
                self.config['progress_callback'](
                    "Preprocessing",
                    iteration,
                    self.config['iterations'],
                    "Applying denoising and contrast enhancement"
                )

            current_image = self._preprocess_image(current_image)

            # 2. Perform segmentation-based enhancement if enabled and available
            if self.config['segment_enhancement'] and 'segmentation' in self.models:
                if self.config.get('progress_callback'):
                    self.config['progress_callback'](
                        "Segmentation",
                        iteration,
                        self.config['iterations'],
                        "Performing region-based analysis and enhancement"
                    )

                current_image = self._segment_and_enhance(current_image)

            # 3. Apply super-resolution models
            upscaled_images = []

            # 3.1 ESRGAN upscaling
            if self.config['use_esrgan'] and 'esrgan' in self.models:
                logger.info("Applying Real-ESRGAN upscaling...")

                if self.config.get('progress_callback'):
                    self.config['progress_callback'](
                        "ESRGAN",
                        iteration,
                        self.config['iterations'],
                        "Applying Real-ESRGAN upscaling"
                    )

                esrgan_result = self._apply_esrgan(current_image)
                if esrgan_result is not None:
                    upscaled_images.append(esrgan_result)

            # 3.2 Diffusion-based upscaling
            if self.config['use_diffusion'] and 'diffusion' in self.models:
                logger.info("Applying diffusion-based upscaling...")

                if self.config.get('progress_callback'):
                    self.config['progress_callback'](
                        "Diffusion",
                        iteration,
                        self.config['iterations'],
                        f"Applying diffusion-based upscaling with prompt: {prompt}"
                    )

                diffusion_result = self._apply_diffusion(current_image, prompt)
                if diffusion_result is not None:
                    upscaled_images.append(diffusion_result)

            # 4. Combine results from different models
            if self.config.get('progress_callback'):
                self.config['progress_callback'](
                    "Ensemble",
                    iteration,
                    self.config['iterations'],
                    "Combining results from multiple models"
                )

            if len(upscaled_images) > 0:
                current_image = self._ensemble_results(upscaled_images)
            else:
                # Fallback to bicubic upscaling if all models failed
                logger.warning("All models failed, falling back to bicubic upscaling")
                h, w = current_image.shape[:2]
                current_image = cv2.resize(
                    current_image,
                    (int(w * self.config['upscale_factor']), int(h * self.config['upscale_factor'])),
                    interpolation=cv2.INTER_CUBIC
                )

            # 5. Apply post-processing
            if self.config.get('progress_callback'):
                self.config['progress_callback'](
                    "Postprocessing",
                    iteration,
                    self.config['iterations'],
                    "Applying final enhancements"
                )

            current_image = self._postprocess_image(current_image)

            # 6. Save intermediate result if configured
            if self.config['intermediate_saves']:
                intermediate_path = os.path.join(
                    self.config['output_dir'],
                    f"intermediate_iter{iteration+1}.png"
                )
                cv2.imwrite(intermediate_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved intermediate result: {intermediate_path}")

            # 7. Report current image to callback
            if self.config.get('image_callback'):
                self.config['image_callback'](current_image)

            # Log iteration time
            iteration_time = time.time() - iteration_start
            logger.info(f"Completed iteration {iteration+1} in {iteration_time:.1f}s")

        # Save and return final enhanced image
        final_h, final_w = current_image.shape[:2]
        scale_achieved = round(final_h / original_h, 1)

        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        final_path = os.path.join(
            self.config['output_dir'],
            f"{base_name}_enhanced_x{scale_achieved}{ext}"
        )

        cv2.imwrite(final_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Enhancement complete. Final image saved to: {final_path}")
        logger.info(f"Original size: {original_w}x{original_h}, Enhanced size: {final_w}x{final_h} (x{scale_achieved})")

        # Final progress update
        if self.config.get('progress_callback'):
            self.config['progress_callback'](
                "Complete",
                self.config['iterations'],
                self.config['iterations'],
                f"Enhancement complete: {original_w}x{original_h} → {final_w}x{final_h} (x{scale_achieved})"
            )

        return final_path

    def _load_image(self, image_path):
        """Load an image from path and convert to RGB."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _preprocess_image(self, image):
        """Apply preprocessing steps like denoising and contrast enhancement."""
        # Apply denoising if configured
        if self.config['denoise_level'] > 0:
            strength = self.config['denoise_level'] * 10  # Scale to reasonable range for h parameter
            try:
                image = cv2.fastNlMeansDenoisingColored(image, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21)
            except Exception as e:
                logger.warning(f"Denoising failed: {e}, skipping")

        # Enhance contrast using CLAHE
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logger.warning(f"CLAHE contrast enhancement failed: {e}, skipping")

        return image

    def _segment_and_enhance(self, image):
        """Use segmentation to identify and enhance specific regions."""
        try:
            # Skip if missing dependencies
            if 'segmentation' not in self.models or 'segmentation_extractor' not in self.models:
                return image

            import torch

            # Convert to PIL for transformers compatibility
            pil_image = Image.fromarray(image)

            # Resize if image is too large
            orig_size = pil_image.size
            max_size = 1024  # Maximum dimension for segmentation
            if max(orig_size) > max_size:
                scale = max_size / max(orig_size)
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)

            # Get segmentation map
            inputs = self.models['segmentation_extractor'](images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.models['segmentation'](inputs.pixel_values)

            # Convert segmentation logits to class probabilities
            seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy()

            # Resize back to original size if needed
            if max(orig_size) > max_size:
                seg_map = cv2.resize(
                    seg_map,
                    (orig_size[0], orig_size[1]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Create an enhanced version of the image
            enhanced = image.copy()

            # Apply different enhancements based on segment classes
            # For demonstration, let's enhance edges in "object" regions (non-background)
            object_mask = (seg_map > 0).astype(np.uint8)
            object_mask = cv2.resize(object_mask, (image.shape[1], image.shape[0]))

            # Dilate mask slightly for better coverage
            kernel = np.ones((3, 3), np.uint8)
            object_mask = cv2.dilate(object_mask, kernel, iterations=1)

            # Apply unsharp mask enhancement to object areas
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

            # Blend based on mask
            enhanced = np.where(
                np.expand_dims(object_mask, axis=2) == 1,
                sharpened,
                enhanced
            )

            return enhanced
        except Exception as e:
            logger.warning(f"Segmentation enhancement failed: {e}")
            return image

    def _apply_esrgan(self, image):
        """Apply Real-ESRGAN super-resolution with memory management."""
        # Check if ESRGAN is enabled and model is available
        if not self.config['use_esrgan'] or 'esrgan' not in self.models:
            return None

        # Get image dimensions and check if tiling should be used
        h, w, c = image.shape
        image_megapixels = (h * w) / (1024 * 1024)

        # Determine if we need tiling based on size and config
        use_tiling = False
        if self.config['enable_tiling']:
            # Determine threshold based on device
            if self.device == 'cpu':
                megapixel_threshold = self.config.get('cpu_megapixel_threshold', 2)
            elif self.device == 'mps':
                megapixel_threshold = self.config.get('mps_megapixel_threshold', 4)
            else:  # cuda
                megapixel_threshold = self.config.get('cuda_megapixel_threshold', 8)

            # Use tiling if image is larger than threshold
            use_tiling = image_megapixels > megapixel_threshold

        # Determine if we should use 3-channel or 12-channel mode
        use_3channel = self.config['force_3channel']

        # Automatically enable 3-channel mode for MPS on larger images
        if self.device == 'mps' and image_megapixels > 1:
            use_3channel = True
            logger.info(f"Using 3-channel mode for {image_megapixels:.1f} MP image on MPS")

        # For very large images, enforce tiling regardless of settings
        if image_megapixels > 16:
            use_tiling = True
            use_3channel = True
            logger.info(f"Forcing tiling and 3-channel mode for {image_megapixels:.1f} MP image")

        # Get tile size from config
        tile_size = self.config['max_tile_size']

        try:
            # If tiling is needed, use the tiled processing approach
            if use_tiling:
                logger.info(f"Using tiled processing with {tile_size}x{tile_size} tiles")

                # Create a processor function that will be applied to each tile
                if use_3channel:
                    processor = lambda tile: self._apply_esrgan_3channel(tile)
                else:
                    processor = lambda tile: self._apply_esrgan_12channel(tile)

                # Apply the tiled processing
                return self._process_with_tiling(processor, image, tile_size=tile_size, padding=16)

            # If no tiling needed, directly apply the selected method
            if use_3channel:
                return self._apply_esrgan_3channel(image)
            else:
                return self._apply_esrgan_12channel(image)

        except Exception as e:
            logger.warning(f"ESRGAN processing failed: {e}")

            # Fall back to simpler method if the selected method fails
            try:
                # If 12-channel failed, try 3-channel
                if not use_3channel:
                    logger.info("Falling back to 3-channel mode")
                    return self._apply_esrgan_3channel(image)

                # If non-tiled 3-channel failed, try tiled 3-channel
                if not use_tiling:
                    logger.info("Falling back to tiled processing")
                    return self._process_with_tiling(
                        lambda tile: self._apply_esrgan_3channel(tile),
                        image,
                        tile_size=256,  # Smaller tiles for fallback
                        padding=16
                    )
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}")

            # If all else fails, use bicubic upscaling
            logger.warning("All ESRGAN methods failed, using bicubic upscaling")
            return cv2.resize(
                image,
                (int(w * self.config['upscale_factor']), int(h * self.config['upscale_factor'])),
                interpolation=cv2.INTER_CUBIC
            )

    def _apply_esrgan_12channel(self, image):
        """Apply Real-ESRGAN using the 12-channel input approach."""
        import torch
        import numpy as np

        # Real-ESRGAN expects BGR format for the image
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create a PyTorch tensor from the image
        img_tensor = torch.from_numpy(np.transpose(bgr_image, (2, 0, 1))).float()

        # First, normalize to [0, 1]
        img_tensor = img_tensor / 255.0

        # Create 12-channel input with variations
        channels = []

        # Original RGB
        channels.append(img_tensor)

        # Brightness variations
        channels.append(torch.clamp(img_tensor * 1.1, 0, 1))  # Brighter
        channels.append(torch.clamp(img_tensor * 0.9, 0, 1))  # Darker

        # Add a fourth variation (e.g., slightly blurred version)
        blurred = torch.tensor(
            np.transpose(
                cv2.GaussianBlur(bgr_image, (3, 3), 0),
                (2, 0, 1)
            )
        ).float() / 255.0
        channels.append(blurred)

        # Combine all channels into a 12-channel tensor
        img_12ch = torch.cat(channels, dim=0)

        # Add batch dimension and move to device
        img_12ch = img_12ch.unsqueeze(0).to(self.device)

        # Use direct inference with the model
        with torch.no_grad():
            # Extract the model from RealESRGANer
            model = self.models['esrgan'].model
            model.eval()

            # Forward pass through the model
            output = model(img_12ch)

            # Move back to CPU and convert to numpy
            output = output.squeeze().cpu().numpy()

            # Convert from CHW to HWC format and scale back to 0-255
            output = np.transpose(output, (1, 2, 0)) * 255.0
            output = np.clip(output, 0, 255).astype(np.uint8)

            # Convert back to RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            return output_rgb

    def _apply_esrgan_3channel(self, image):
        """Apply Real-ESRGAN using 3-channel input with channel adaptation.

        This method adapts the 3-channel input to the 12-channel model by duplicating
        the channels to match the expected input format.
        """
        import torch
        import numpy as np

        # Convert to BGR (expected by Real-ESRGAN)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # Create 3-channel tensor (normalized to [0,1])
            img_tensor = torch.from_numpy(np.transpose(bgr_image, (2, 0, 1))).float() / 255.0

            # Expand from 3 to 12 channels by repeating
            # This is the key fix - we need to duplicate the 3 channels to create 12 channels
            # to match the model's expected input format
            img_12ch = torch.cat([img_tensor, img_tensor, img_tensor, img_tensor], dim=0)

            # Add batch dimension and move to right device
            img_12ch = img_12ch.unsqueeze(0).to(self.device)

            # Extract model and do inference
            model = self.models['esrgan'].model
            model.eval()

            with torch.no_grad():
                # Process with model
                output = model(img_12ch)
                output = output.squeeze().cpu().numpy()

                # Convert from CHW to HWC and scale back to 0-255
                output = np.transpose(output, (1, 2, 0)) * 255.0
                output = np.clip(output, 0, 255).astype(np.uint8)

                # Convert back to RGB
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                return output_rgb

        except Exception as e:
            logger.warning(f"Error in 3-channel ESRGAN processing: {e}")

            # Try CPU fallback if configured and we're on MPS
            if self.device == 'mps' and self.config.get('fallback_to_cpu', True):
                try:
                    logger.info("Trying CPU fallback for 3-channel processing")

                    # Create tensor on CPU instead
                    img_tensor = torch.from_numpy(np.transpose(bgr_image, (2, 0, 1))).float() / 255.0
                    img_12ch = torch.cat([img_tensor, img_tensor, img_tensor, img_tensor], dim=0)
                    img_12ch = img_12ch.unsqueeze(0)  # Keep on CPU

                    # Move model temporarily to CPU
                    model = self.models['esrgan'].model
                    original_device = next(model.parameters()).device
                    model = model.to('cpu')

                    with torch.no_grad():
                        # Process on CPU
                        output = model(img_12ch)
                        output = output.squeeze().clamp_(0, 1).numpy()
                        output = output.transpose(1, 2, 0) * 255.0
                        output = np.clip(output, 0, 255).astype(np.uint8)

                    # Move model back to original device
                    model = model.to(original_device)

                    # Convert back to RGB
                    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                    return output_rgb

                except Exception as cpu_err:
                    logger.error(f"CPU fallback also failed: {cpu_err}")
                    # Move model back in case of error
                    model = model.to(original_device)

            # If we get here, both attempts failed
            raise

    def _apply_diffusion(self, image, prompt):
        """Apply stable diffusion upscaling with text guidance."""
        try:
            # Skip if model not loaded or missing dependencies
            if 'diffusion' not in self.models:
                return None

            # Convert to PIL for diffusers
            pil_image = Image.fromarray(image)

            # Run upscaling pipeline - adjust settings based on device
            if self.device == 'cpu':
                # Use minimal inference steps on CPU
                result = self.models['diffusion'](
                    prompt=prompt,
                    image=pil_image,
                    noise_level=20,
                    num_inference_steps=10,  # Reduced for CPU
                    guidance_scale=7.5,
                ).images[0]
            elif self.device == 'mps':
                # Moderate settings for MPS
                result = self.models['diffusion'](
                    prompt=prompt,
                    image=pil_image,
                    noise_level=20,
                    num_inference_steps=15,  # Balanced for MPS
                    guidance_scale=7.5,
                ).images[0]
            else:
                # Full settings for CUDA
                result = self.models['diffusion'](
                    prompt=prompt,
                    image=pil_image,
                    noise_level=20,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                ).images[0]

            # Convert back to numpy
            return np.array(result)
        except Exception as e:
            logger.warning(f"Diffusion-based upscaling failed: {e}")
            return None

    def _ensemble_results(self, upscaled_images):
        """Combine results from multiple super-resolution models."""
        if len(upscaled_images) == 1:
            return upscaled_images[0]

        # Ensure all images have the same dimensions
        target_h, target_w = max([(img.shape[0], img.shape[1]) for img in upscaled_images])
        aligned_images = []

        for img in upscaled_images:
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            aligned_images.append(img)

        # Weighted average ensemble (could be improved with more sophisticated methods)
        weights = np.ones(len(aligned_images)) / len(aligned_images)
        ensemble = np.zeros_like(aligned_images[0], dtype=np.float32)

        for i, img in enumerate(aligned_images):
            ensemble += img.astype(np.float32) * weights[i]

        return ensemble.astype(np.uint8)

    def _postprocess_image(self, image):
        """Apply post-processing steps like sharpening."""
        # Apply adaptive sharpening if configured
        if self.config['adaptive_sharpening']:
            try:
                # Calculate local variance as a measure of detail
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                variance = cv2.GaussianBlur(gray, (0, 0), 2)
                variance = cv2.subtract(gray, variance)
                variance = cv2.convertScaleAbs(variance)

                # Create a sharpened version of the image
                blurred = cv2.GaussianBlur(image, (0, 0), 3)
                sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

                # Normalize variance to create an alpha mask for blending
                _, variance_binary = cv2.threshold(variance, 10, 255, cv2.THRESH_BINARY)
                alpha = variance_binary.astype(np.float32) / 255.0

                # Apply sharpening selectively to detailed areas
                result = np.zeros_like(image)
                for c in range(3):
                    result[:, :, c] = image[:, :, c] * (1 - alpha) + sharpened[:, :, c] * alpha

                return result.astype(np.uint8)
            except Exception as e:
                logger.warning(f"Adaptive sharpening failed: {e}")

        return image
