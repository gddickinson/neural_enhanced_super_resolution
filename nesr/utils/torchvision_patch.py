"""
Compatibility patch for torchvision to support Real-ESRGAN with newer torchvision versions.
"""

import sys
import logging
import importlib.util

logger = logging.getLogger('nesr.patch')

def apply_patches():
    """Apply compatibility patches for various modules."""
    # Fix for 'torchvision.transforms.functional_tensor' missing in newer versions
    patch_torchvision_functional_tensor()

def patch_torchvision_functional_tensor():
    """
    Patch for missing torchvision.transforms.functional_tensor in newer torchvision versions.
    """
    try:
        # Check if the module already exists (no need to patch)
        if importlib.util.find_spec('torchvision.transforms.functional_tensor') is not None:
            logger.info("torchvision.transforms.functional_tensor already exists, no patch needed")
            return

        # First, check if we can import from the new location
        try:
            import torchvision.transforms.functional as F
            logger.info("Found torchvision.transforms.functional, creating compatibility module")

            # Create a module to serve as functional_tensor
            import types
            functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')

            # Copy necessary functions from functional to functional_tensor
            if hasattr(F, 'rgb_to_grayscale'):
                functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
            elif hasattr(F, 'to_grayscale'):
                functional_tensor.rgb_to_grayscale = F.to_grayscale
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
                # Create a fallback implementation
                def normalize(tensor, mean, std):
                    """Normalize a tensor image with mean and standard deviation."""
                    for t, m, s in zip(tensor, mean, std):
                        t.sub_(m).div_(s)
                    return tensor
                functional_tensor.normalize = normalize

            # Insert the module into sys.modules so it can be imported
            sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
            logger.info("Successfully patched torchvision.transforms.functional_tensor")

        except ImportError:
            logger.error("Could not import torchvision.transforms.functional for patching")
            raise

    except Exception as e:
        logger.warning(f"Failed to apply torchvision patch: {e}")
