"""
NESR - Neural Enhanced Super-Resolution
Image utility functions for processing and manipulating images.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
import io

def add_noise(image, noise_type='gaussian', amount=0.1):
    """
    Add noise to an image.
    
    Args:
        image: Input image as numpy array (RGB)
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle', 'poisson')
        amount: Noise intensity (0.0 to 1.0)
    
    Returns:
        Noisy image as numpy array
    """
    if amount <= 0:
        return image.copy()
    
    result = image.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        # Gaussian noise
        mean = 0
        std = amount * 255
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        result += noise
    
    elif noise_type == 'salt & pepper' or noise_type == 'salt_pepper':
        # Salt and pepper noise
        s_vs_p = 0.5
        salt = np.ceil(amount * image.size * s_vs_p)
        pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        
        # Add salt (white) noise
        coords = [np.random.randint(0, i - 1, int(salt)) for i in image.shape]
        result[coords[0], coords[1], :] = 255
        
        # Add pepper (black) noise
        coords = [np.random.randint(0, i - 1, int(pepper)) for i in image.shape]
        result[coords[0], coords[1], :] = 0
    
    elif noise_type == 'speckle':
        # Speckle noise (multiplicative)
        noise = np.random.normal(0, amount, image.shape).astype(np.float32)
        result += result * noise
    
    elif noise_type == 'poisson':
        # Poisson noise
        scaling = amount * 10  # Scale factor to control noise intensity
        noise = np.random.poisson(image / 255.0 * scaling) / scaling * 255
        result = noise.astype(np.float32)
    
    # Clip values to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def blur_image(image, blur_type='gaussian', radius=3):
    """
    Apply blur to an image.
    
    Args:
        image: Input image as numpy array (RGB)
        blur_type: Type of blur ('gaussian', 'box', 'motion')
        radius: Blur radius in pixels
    
    Returns:
        Blurred image as numpy array
    """
    if radius <= 0:
        return image.copy()
    
    if blur_type == 'gaussian':
        # Gaussian blur
        return cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)
    
    elif blur_type == 'box':
        # Box blur
        return cv2.boxFilter(image, -1, (radius*2+1, radius*2+1))
    
    elif blur_type == 'motion':
        # Motion blur
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
    
    # Default to Gaussian if unknown type
    return cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)

def downsample_image(image, scale_factor=0.5, interpolation='bicubic'):
    """
    Downsample an image by a scale factor.
    
    Args:
        image: Input image as numpy array (RGB)
        scale_factor: Scale factor (0.1 to 1.0)
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic', 'lanczos')
    
    Returns:
        Downsampled image as numpy array
    """
    if scale_factor >= 1.0:
        return image.copy()
    
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Map string interpolation names to OpenCV constants
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp_method = interp_map.get(interpolation.lower(), cv2.INTER_CUBIC)
    
    return cv2.resize(image, (new_width, new_height), interpolation=interp_method)

def apply_jpeg_compression(image, quality=75):
    """
    Apply JPEG compression artifacts to an image.
    
    Args:
        image: Input image as numpy array (RGB)
        quality: JPEG quality (1-100, lower values mean more artifacts)
    
    Returns:
        Compressed image as numpy array
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to a BytesIO buffer with JPEG compression
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    
    # Load back from buffer
    buffer.seek(0)
    compressed = np.array(Image.open(buffer))
    
    return compressed

def create_comparison_image(original, enhanced, orientation='horizontal'):
    """
    Create a side-by-side comparison image.
    
    Args:
        original: Original image as numpy array (RGB)
        enhanced: Enhanced image as numpy array (RGB)
        orientation: 'horizontal' or 'vertical' layout
    
    Returns:
        Combined comparison image as numpy array
    """
    if orientation.lower() == 'horizontal':
        # Resize if heights don't match
        orig_h, orig_w = original.shape[:2]
        enh_h, enh_w = enhanced.shape[:2]
        
        if orig_h != enh_h:
            # Scale to match heights
            scale = orig_h / enh_h
            new_width = int(enh_w * scale)
            enhanced = cv2.resize(enhanced, (new_width, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create horizontal concatenation
        comparison = np.hstack((original, enhanced))
        
        # Add a dividing line
        comparison[:, orig_w:orig_w+1, :] = [255, 0, 0]  # Red line
    
    else:  # vertical
        # Resize if widths don't match
        orig_h, orig_w = original.shape[:2]
        enh_h, enh_w = enhanced.shape[:2]
        
        if orig_w != enh_w:
            # Scale to match widths
            scale = orig_w / enh_w
            new_height = int(enh_h * scale)
            enhanced = cv2.resize(enhanced, (orig_w, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create vertical concatenation
        comparison = np.vstack((original, enhanced))
        
        # Add a dividing line
        comparison[orig_h:orig_h+1, :, :] = [255, 0, 0]  # Red line
    
    return comparison

def add_text_to_image(image, text, position='top', font_scale=1.0, color=(255, 255, 255)):
    """
    Add text overlay to an image.
    
    Args:
        image: Input image as numpy array (RGB)
        text: Text to add
        position: 'top', 'bottom', or (x, y) coordinates
        font_scale: Font scale factor
        color: Text color as (R, G, B) tuple
    
    Returns:
        Image with text overlay
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position
    if position == 'top':
        x = int((image.shape[1] - text_width) / 2)
        y = text_height + 10
    elif position == 'bottom':
        x = int((image.shape[1] - text_width) / 2)
        y = image.shape[0] - 10
    else:
        x, y = position
    
    # Add a dark background rectangle for better visibility
    cv2.rectangle(
        result, 
        (x - 5, y - text_height - 5), 
        (x + text_width + 5, y + 5), 
        (0, 0, 0), 
        -1
    )
    
    # Add text
    cv2.putText(
        result, 
        text, 
        (x, y), 
        font, 
        font_scale, 
        color, 
        thickness
    )
    
    return result
