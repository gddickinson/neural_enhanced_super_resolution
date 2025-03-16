#!/usr/bin/env python3
"""
Direct ESRGAN upscaler - A standalone version of the ESRGAN upscaler without NESR dependencies.
This can help verify that ESRGAN itself is working correctly independent of the NESR framework.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('esrgan')

# Apply torchvision compatibility patch
def patch_torchvision():
    """Apply compatibility patches for torchvision."""
    try:
        import importlib.util
        
        # Check if the module already exists (no need to patch)
        if importlib.util.find_spec('torchvision.transforms.functional_tensor') is not None:
            logger.info("torchvision.transforms.functional_tensor already exists, no patch needed")
            return
        
        # Create a compatibility module
        import types
        import torchvision.transforms.functional as F
        
        functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
        
        # Define required functions
        if hasattr(F, 'rgb_to_grayscale'):
            functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
        else:
            # Create a fallback implementation
            import torch
            def rgb_to_grayscale(img):
                """Convert RGB image to grayscale."""
                if img.shape[0] != 3:
                    return img
                return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).unsqueeze(0)
            functional_tensor.rgb_to_grayscale = rgb_to_grayscale
        
        # Add normalize function
        if hasattr(F, 'normalize'):
            functional_tensor.normalize = F.normalize
        else:
            def normalize(tensor, mean, std):
                """Normalize a tensor image with mean and standard deviation."""
                for t, m, s in zip(tensor, mean, std):
                    t.sub_(m).div_(s)
                return tensor
            functional_tensor.normalize = normalize
        
        # Insert the module into sys.modules
        sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
        logger.info("Successfully patched torchvision.transforms.functional_tensor")
    
    except Exception as e:
        logger.error(f"Failed to apply patch: {e}")
        return False
    
    return True

def enhance_image(input_path, output_path=None, scale=2, device='mps'):
    """
    Enhance an image using Real-ESRGAN.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image (default: auto-generate)
        scale: Upscaling factor (default: 2)
        device: Device to use (default: mps for Apple Silicon)
    
    Returns:
        Path to the enhanced image
    """
    # Apply compatibility patch
    patch_torchvision()
    
    try:
        # Import Real-ESRGAN
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # Get model path
        model_path = find_model_path()
        if not model_path:
            logger.error("Could not find ESRGAN model")
            return None
        
        logger.info(f"Using model: {model_path}")
        
        # Create model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        
        # Print device information
        if device == 'mps':
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Using MPS (Apple Silicon GPU)")
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                device = 'cpu'
        
        logger.info(f"Using device: {device}")
        
        # Initialize enhancer
        upscaler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=512,  # Process image in tiles to avoid memory issues
            tile_pad=10,
            pre_pad=0,
            half=False,  # Using full precision for MPS compatibility
            device=device
        )
        
        # Load input image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Could not load image: {input_path}")
            return None
        
        # Print input image info
        height, width, channels = img.shape
        logger.info(f"Input image: {width}x{height}, {channels} channels")
        
        # Check if image is too large
        max_pixels = 1920 * 1080  # HD resolution
        if width * height > max_pixels:
            logger.warning(f"Image is large ({width}x{height}), processing may take a while")
        
        # Time the processing
        start_time = time.time()
        
        # Process the image
        output, _ = upscaler.enhance(img)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        logger.info(f"Processing took {elapsed:.2f} seconds")
        
        # Get output dimensions
        out_height, out_width = output.shape[:2]
        logger.info(f"Output image: {out_width}x{out_height}")
        
        # Generate output path if not provided
        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_filename = os.path.basename(input_path)
            input_name, input_ext = os.path.splitext(input_filename)
            output_path = os.path.join(input_dir, f"{input_name}_enhanced_x{scale}{input_ext}")
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the output image
        cv2.imwrite(output_path, output)
        logger.info(f"Enhanced image saved to: {output_path}")
        
        return output_path
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure realesrgan and basicsr are installed: pip install realesrgan basicsr")
        return None
    
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return None

def find_model_path():
    """Find the ESRGAN model path."""
    # Common paths to check
    home = os.path.expanduser("~")
    
    possible_paths = [
        # Check current directory first
        os.path.join(os.getcwd(), "models", "weights", "RealESRGAN_x2plus.pth"),
        os.path.join(os.getcwd(), "weights", "RealESRGAN_x2plus.pth"),
        os.path.join(os.getcwd(), "RealESRGAN_x2plus.pth"),
        
        # Check NESR directories
        os.path.join(home, "Library", "Application Support", "NESR", "models", "weights", "RealESRGAN_x2plus.pth"),
        os.path.join(home, ".nesr", "models", "weights", "RealESRGAN_x2plus.pth"),
        os.path.join(home, "AppData", "Roaming", "NESR", "models", "weights", "RealESRGAN_x2plus.pth"),
        
        # Check script directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights", "RealESRGAN_x2plus.pth"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "RealESRGAN_x2plus.pth"),
    ]
    
    # Check all paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, try to download
    logger.warning("Model not found, attempting to download...")
    try:
        # Create a directory to store the model
        download_dir = os.path.join(os.getcwd(), "weights")
        os.makedirs(download_dir, exist_ok=True)
        
        model_path = os.path.join(download_dir, "RealESRGAN_x2plus.pth")
        
        # Download the model
        import requests
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth"
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Model downloaded to: {model_path}")
            return model_path
        else:
            logger.error(f"Failed to download model: HTTP {response.status_code}")
            return None
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Real-ESRGAN Upscaler")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path (optional)")
    parser.add_argument("--scale", "-s", type=int, default=2, help="Upscaling factor (default: 2)")
    parser.add_argument("--device", "-d", default="mps", choices=["cuda", "mps", "cpu"], 
                       help="Device to use (default: mps for Apple Silicon)")
    
    args = parser.parse_args()
    
    # Enhance the image
    output_path = enhance_image(
        args.input,
        args.output,
        args.scale,
        args.device
    )
    
    if output_path:
        print(f"\nSuccess! Enhanced image saved to: {output_path}")
    else:
        print("\nFailed to enhance image. Check the logs for details.")
        sys.exit(1)
