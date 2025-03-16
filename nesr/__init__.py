"""
Neural Enhanced Super-Resolution (NESR)
A multi-model approach to iterative image super-resolution.
"""

# Apply compatibility patches before importing any other modules
from .utils.torchvision_patch import apply_patches
apply_patches()

# Now import the rest of the modules
from .nesr import SuperResolutionPipeline

__version__ = '0.1.0'
__author__ = 'NESR Team'
