#!/usr/bin/env python3
"""
Download a 3-channel ESRGAN model to avoid the channel mismatch issue.
This script downloads the new v0.3.0 model that has 3 input channels.
"""

import os
import sys
import requests
import logging
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nesr.model_downloader')

def get_models_dir():
    """Get the directory for model storage."""
    # Default locations
    if sys.platform == 'win32':
        # Windows: User's AppData folder
        base_dir = os.path.join(os.environ['APPDATA'], 'NESR')
    elif sys.platform == 'darwin':
        # macOS: User's Library folder
        base_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'NESR')
    else:
        # Linux/Unix: User's home directory
        base_dir = os.path.join(os.path.expanduser('~'), '.nesr')
    
    # Create directory structure
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create weights directory
    weights_dir = os.path.join(models_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    return models_dir, weights_dir

def download_file(url, destination, desc=None):
    """Download a file with progress tracking."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB
        
        desc = desc or os.path.basename(destination)
        with open(destination, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    bar.update(size)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def main():
    """Main function to download the 3-channel model."""
    # Get the models directory
    models_dir, weights_dir = get_models_dir()
    
    # Model URLs - these are the new 3-channel models from Real-ESRGAN v0.3.0
    models = [
        {
            "name": "RealESRGAN v0.3.0 x2 (3-channel)",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/realesr-general-x2v3.pth",
            "path": os.path.join(weights_dir, "realesr-general-x2v3.pth")
        },
        {
            "name": "RealESRGAN v0.3.0 x4 (3-channel)",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/realesr-general-wdn-x4v3.pth",
            "path": os.path.join(weights_dir, "realesr-general-x4v3.pth")
        }
    ]
    
    # Download each model
    for model in models:
        if os.path.exists(model["path"]):
            logger.info(f"{model['name']} already exists at {model['path']}")
        else:
            logger.info(f"Downloading {model['name']}...")
            if download_file(model["url"], model["path"]):
                logger.info(f"Downloaded {model['name']} to {model['path']}")
            else:
                logger.error(f"Failed to download {model['name']}")
    
    # Create symbolic link/copy for standard name
    try:
        # Find the x2 model
        x2_model = next((m for m in models if "x2" in m["name"]), None)
        if x2_model and os.path.exists(x2_model["path"]):
            standard_name = os.path.join(weights_dir, "RealESRGAN_x2plus.pth")
            # Make a backup of existing model
            if os.path.exists(standard_name):
                backup_name = standard_name + ".backup"
                logger.info(f"Backing up existing model to {backup_name}")
                shutil.move(standard_name, backup_name)
            
            # Create a copy
            logger.info(f"Creating copy as {standard_name}")
            shutil.copy2(x2_model["path"], standard_name)
            logger.info(f"Successfully copied model to standard location")
    except Exception as e:
        logger.error(f"Failed to create standard model link: {e}")
    
    print("\nDownloaded models to:")
    print(f"  {weights_dir}\n")
    print("Available models:")
    for model in models:
        status = "✓" if os.path.exists(model["path"]) else "✗"
        print(f"  - {model['name']}: {status}")
    
    # Check if standard name exists
    standard_name = os.path.join(weights_dir, "RealESRGAN_x2plus.pth")
    status = "✓" if os.path.exists(standard_name) else "✗"
    print(f"  - Standard name (RealESRGAN_x2plus.pth): {status}")
    
    print("\nNext steps:")
    print("1. Restart the NESR application")
    print("2. The 3-channel model should now be used automatically")

if __name__ == "__main__":
    main()
