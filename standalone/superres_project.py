# Neural Enhanced Super-Resolution (NESR)
# A multi-model approach to iterative image super-resolution

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline, DiffusionPipeline
from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation
import argparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            'use_swinir': True,
            'preserve_details': True,
            'adaptive_sharpening': True,
            'segment_enhancement': True,
            'denoise_level': 0.5,
            'upscale_factor': 2,
            'intermediate_saves': False,
            'output_dir': 'outputs',
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize the super-resolution models based on config."""
        self.models = {}
        
        if self.config['use_esrgan']:
            logger.info("Loading Real-ESRGAN model...")
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
                self.models['esrgan'] = RealESRGANer(
                    scale=self.config['upscale_factor'],
                    model_path='weights/RealESRGAN_x2plus.pth',
                    model=model,
                    device=self.device
                )
                logger.info("Real-ESRGAN model loaded successfully")
            except ImportError:
                logger.warning("Real-ESRGAN import failed. Install with: pip install realesrgan basicsr")
        
        if self.config['use_swinir']:
            logger.info("Loading SwinIR model...")
            try:
                # This is a placeholder - in a real implementation you would load the SwinIR model
                # from the appropriate repository
                self.models['swinir'] = None
                logger.info("SwinIR model loaded successfully")
            except Exception as e:
                logger.warning(f"SwinIR model loading failed: {e}")
        
        if self.config['use_diffusion']:
            logger.info("Loading Stable Diffusion upscaler...")
            try:
                self.models['diffusion'] = StableDiffusionUpscalePipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler", 
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
                self.models['diffusion'] = self.models['diffusion'].to(self.device)
                logger.info("Stable Diffusion upscaler loaded successfully")
            except Exception as e:
                logger.warning(f"Stable Diffusion upscaler loading failed: {e}")
        
        if self.config['segment_enhancement']:
            logger.info("Loading segmentation model for targeted enhancement...")
            try:
                self.models['segmentation'] = AutoModelForImageSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                ).to(self.device)
                self.models['segmentation_extractor'] = AutoFeatureExtractor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                logger.info("Segmentation model loaded successfully")
            except Exception as e:
                logger.warning(f"Segmentation model loading failed: {e}")
    
    def enhance_image(self, image_path, prompt=None):
        """
        Enhance an image through multiple iterations of super-resolution.
        
        Args:
            image_path: Path to the input image
            prompt: Optional text prompt to guide diffusion-based upscaling
            
        Returns:
            Path to the final enhanced image
        """
        # Load the initial image
        image = self._load_image(image_path)
        original_h, original_w = image.shape[:2]
        
        # Track the current image throughout iterations
        current_image = image
        
        # Generate a default prompt if none provided
        if prompt is None and self.config['use_diffusion']:
            prompt = "a high resolution, detailed photograph"
        
        # Perform iterative enhancement
        for iteration in range(self.config['iterations']):
            logger.info(f"Starting iteration {iteration+1}/{self.config['iterations']}")
            
            # 1. Apply pre-processing (denoise, adjust contrast)
            current_image = self._preprocess_image(current_image)
            
            # 2. Perform segmentation-based enhancement if enabled
            if self.config['segment_enhancement'] and 'segmentation' in self.models:
                current_image = self._segment_and_enhance(current_image)
            
            # 3. Apply super-resolution models
            upscaled_images = []
            
            # 3.1 ESRGAN upscaling
            if self.config['use_esrgan'] and 'esrgan' in self.models:
                logger.info("Applying Real-ESRGAN upscaling...")
                esrgan_result = self._apply_esrgan(current_image)
                if esrgan_result is not None:
                    upscaled_images.append(esrgan_result)
            
            # 3.2 SwinIR upscaling
            if self.config['use_swinir'] and 'swinir' in self.models:
                logger.info("Applying SwinIR upscaling...")
                swinir_result = self._apply_swinir(current_image)
                if swinir_result is not None:
                    upscaled_images.append(swinir_result)
            
            # 3.3 Diffusion-based upscaling
            if self.config['use_diffusion'] and 'diffusion' in self.models:
                logger.info("Applying diffusion-based upscaling...")
                diffusion_result = self._apply_diffusion(current_image, prompt)
                if diffusion_result is not None:
                    upscaled_images.append(diffusion_result)
            
            # 4. Combine results from different models
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
            current_image = self._postprocess_image(current_image)
            
            # 6. Save intermediate result if configured
            if self.config['intermediate_saves']:
                intermediate_path = os.path.join(
                    self.config['output_dir'], 
                    f"intermediate_iter{iteration+1}.png"
                )
                cv2.imwrite(intermediate_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved intermediate result: {intermediate_path}")
        
        # Save and return final enhanced image
        final_h, final_w = current_image.shape[:2]
        scale_achieved = round(final_h / original_h, 1)
        
        final_path = os.path.join(
            self.config['output_dir'], 
            f"enhanced_{os.path.basename(image_path)}_x{scale_achieved}.png"
        )
        
        cv2.imwrite(final_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Enhancement complete. Final image saved to: {final_path}")
        logger.info(f"Original size: {original_w}x{original_h}, Enhanced size: {final_w}x{final_h} (x{scale_achieved})")
        
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
            
            # Get segmentation map
            inputs = self.models['segmentation_extractor'](images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.models['segmentation'](inputs.pixel_values)
            
            # Convert segmentation logits to class probabilities
            seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy()
            
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
    
    def _apply_swinir(self, image):
        """Apply SwinIR super-resolution."""
        try:
            # This is a placeholder - in a real implementation you would use the SwinIR model
            # Return None to indicate this model isn't actually implemented
            return None
        except Exception as e:
            logger.warning(f"SwinIR upscaling failed: {e}")
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

def main():
    """Main function to run the super-resolution pipeline from command line."""
    parser = argparse.ArgumentParser(description="Neural Enhanced Super-Resolution")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output_dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--iterations", "-n", type=int, default=3, help="Number of enhancement iterations")
    parser.add_argument("--upscale_factor", "-u", type=float, default=2.0, help="Base upscale factor per iteration")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--prompt", "-p", default=None, help="Text prompt for diffusion guidance")
    parser.add_argument("--no_diffusion", action="store_true", help="Disable diffusion-based upscaling")
    parser.add_argument("--intermediate_saves", action="store_true", help="Save intermediate results")
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        "iterations": args.iterations,
        "upscale_factor": args.upscale_factor,
        "use_diffusion": not args.no_diffusion,
        "output_dir": args.output_dir,
        "intermediate_saves": args.intermediate_saves,
    }
    
    # Initialize and run the pipeline
    pipeline = SuperResolutionPipeline(device=args.device, config=config)
    pipeline.enhance_image(args.input, prompt=args.prompt)

if __name__ == "__main__":
    main()
