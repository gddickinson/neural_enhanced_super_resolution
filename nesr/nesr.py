"""
NESR - Neural Enhanced Super-Resolution
Main module containing the SuperResolutionPipeline class.
This is a modified version of the original implementation that includes
callbacks for the GUI application.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import logging
from tqdm import tqdm
import time

# Set up logging
logger = logging.getLogger('nesr')

class SuperResolutionPipeline:
    """Main orchestration class for the iterative super-resolution process."""
    
    def __init__(self, device='cuda', config=None):
        """Initialize the super resolution pipeline.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            config: Configuration dictionary for customizing the pipeline
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
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
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize models
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize the super-resolution models based on config."""
        # Only initialize models when they're first needed, not at startup
        pass
    
    def _load_models(self):
        """Load models on demand."""
        # Only load models that aren't already loaded
        if self.config['use_esrgan'] and 'esrgan' not in self.models:
            logger.info("Loading Real-ESRGAN model...")
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
                
                # Get model path
                from .utils.downloader import get_model_path
                model_path = get_model_path('esrgan_x2')
                
                self.models['esrgan'] = RealESRGANer(
                    scale=self.config['upscale_factor'],
                    model_path=model_path,
                    model=model,
                    device=self.device
                )
                logger.info("Real-ESRGAN model loaded successfully")
            except ImportError:
                logger.warning("Real-ESRGAN import failed. Install with: pip install realesrgan basicsr")
            except Exception as e:
                logger.error(f"Error loading Real-ESRGAN model: {e}")
        
        if self.config['use_diffusion'] and 'diffusion' not in self.models:
            logger.info("Loading Stable Diffusion upscaler...")
            try:
                from diffusers import StableDiffusionUpscalePipeline
                
                self.models['diffusion'] = StableDiffusionUpscalePipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler", 
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
                self.models['diffusion'] = self.models['diffusion'].to(self.device)
                logger.info("Stable Diffusion upscaler loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Stable Diffusion upscaler: {e}")
        
        if self.config['segment_enhancement'] and 'segmentation' not in self.models:
            logger.info("Loading segmentation model for targeted enhancement...")
            try:
                from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation
                
                self.models['segmentation'] = AutoModelForImageSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                ).to(self.device)
                self.models['segmentation_extractor'] = AutoFeatureExtractor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                logger.info("Segmentation model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading segmentation model: {e}")
    
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
            
            # 2. Perform segmentation-based enhancement if enabled
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
            image = cv2.fastNlMeansDenoisingColored(image, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def _segment_and_enhance(self, image):
        """Use segmentation to identify and enhance specific regions."""
        try:
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
        """Apply Real-ESRGAN super-resolution."""
        try:
            # Real-ESRGAN expects BGR format
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output, _ = self.models['esrgan'].enhance(bgr_image)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Real-ESRGAN upscaling failed: {e}")
            return None
    
    def _apply_diffusion(self, image, prompt):
        """Apply stable diffusion upscaling with text guidance."""
        try:
            # Convert to PIL for diffusers
            pil_image = Image.fromarray(image)
            
            # Run upscaling pipeline
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
        
        return image
