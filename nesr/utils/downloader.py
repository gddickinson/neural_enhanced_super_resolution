"""
NESR - Neural Enhanced Super-Resolution
Model downloader utility to handle downloading and checking required models.
"""

import os
import sys
import json
import hashlib
import logging
import requests
from pathlib import Path
from tqdm import tqdm
import torch

# Set up logging
logger = logging.getLogger('nesr.downloader')

# Model configurations with download info
MODELS = {
    "esrgan_x2": {
        "name": "Real-ESRGAN x2 Model",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth",
        "path": "weights/RealESRGAN_x2plus.pth",
        "size": 67010191,  # Approximate size in bytes
        "md5": "5db904e3e9f0dbf5c64b7ae665527e62",
        "required": True
    },
    "esrgan_x4": {
        "name": "Real-ESRGAN x4 Model",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "path": "weights/RealESRGAN_x4plus.pth",
        "size": 67010191,  # Approximate size in bytes
        "md5": "94df4e7c584b55e2e9a5d2b8f161860e",
        "required": False
    },
    "sd_upscaler": {
        "name": "Stable Diffusion Upscaler",
        "huggingface_id": "stabilityai/stable-diffusion-x4-upscaler",
        "path": None,  # Will be downloaded via huggingface_hub
        "size": 1789525015,  # Approximate size in bytes
        "required": True
    },
    "segmentation": {
        "name": "Segmentation Model",
        "huggingface_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "path": None,  # Will be downloaded via huggingface_hub
        "size": 31256892,  # Approximate size in bytes
        "required": True
    }
}

def get_models_dir():
    """Get the directory for model storage."""
    # First, check for user-defined model directory in environment
    if 'NESR_MODELS_DIR' in os.environ:
        models_dir = os.environ['NESR_MODELS_DIR']
        os.makedirs(models_dir, exist_ok=True)
        return models_dir

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

    return models_dir

def get_model_path(model_key):
    """Get the full path for a specific model."""
    models_dir = get_models_dir()
    model_info = MODELS[model_key]

    if model_info.get('path'):
        return os.path.join(models_dir, model_info['path'])

    # For huggingface models, return the cache directory
    huggingface_dir = os.path.join(models_dir, 'huggingface')
    return os.path.join(huggingface_dir, model_key.split('/')[-1])

def check_models_exist():
    """Check if required models exist.

    Returns:
        Dict mapping model keys to boolean existence status
    """
    result = {}
    models_dir = get_models_dir()

    for model_key, model_info in MODELS.items():
        if model_info.get('huggingface_id'):
            # Hugging Face models are handled differently
            try:
                import huggingface_hub
                # Check if model is in cache
                result[model_key] = huggingface_hub.model_info(
                    model_info['huggingface_id'],
                    token=None,  # Use anonymous access
                    local_files_only=True
                ) is not None
            except Exception:
                result[model_key] = False
        elif model_info.get('path'):
            # Direct file models
            full_path = os.path.join(models_dir, model_info['path'])
            result[model_key] = os.path.exists(full_path)
        else:
            result[model_key] = False

    return result

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, destination, expected_size=None, expected_md5=None, callback=None):
    """Download a file with progress tracking.

    Args:
        url: URL to download
        destination: Destination file path
        expected_size: Expected file size in bytes (for progress calculation)
        expected_md5: Expected MD5 hash for verification
        callback: Progress callback function(progress_percent, message)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Setup for resuming download if file exists
        headers = {}
        mode = 'wb'
        initial_size = 0

        if os.path.exists(destination):
            initial_size = os.path.getsize(destination)
            if expected_size and initial_size >= expected_size:
                # File already exists and is complete, verify hash if available
                if expected_md5:
                    actual_md5 = calculate_md5(destination)
                    if actual_md5 == expected_md5:
                        if callback:
                            callback(100, f"File already exists and verified: {os.path.basename(destination)}")
                        return True
                else:
                    # No MD5 to verify, but file exists with correct size
                    if callback:
                        callback(100, f"File already exists: {os.path.basename(destination)}")
                    return True

            # Resume download
            headers['Range'] = f'bytes={initial_size}-'
            mode = 'ab'

        # Start download
        response = requests.get(url, headers=headers, stream=True, timeout=10)

        # Get total size
        total_size = int(response.headers.get('content-length', 0)) + initial_size
        if expected_size:
            total_size = expected_size

        # Create progress bar
        desc = os.path.basename(destination)
        with open(destination, mode) as f:
            with tqdm(
                initial=initial_size,
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=desc,
                disable=True  # Disable tqdm's own progress display
            ) as pbar:

                # Download chunks
                last_percent = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress = pbar.n + len(chunk)
                        pbar.update(len(chunk))

                        # Call progress callback
                        if callback:
                            percent = int(100 * progress / total_size) if total_size else 0
                            if percent > last_percent:
                                last_percent = percent
                                callback(percent, f"Downloading {desc}: {percent}%")

        # Verify download if MD5 is provided
        if expected_md5:
            actual_md5 = calculate_md5(destination)
            if actual_md5 != expected_md5:
                logger.error(f"MD5 verification failed for {destination}")
                if callback:
                    callback(0, f"Error: MD5 verification failed for {os.path.basename(destination)}")
                return False

        if callback:
            callback(100, f"Download complete: {os.path.basename(destination)}")

        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        if callback:
            callback(0, f"Error downloading {os.path.basename(destination)}: {str(e)}")
        return False

def download_huggingface_model(model_id, callback=None):
    """Download a model from Hugging Face using the hub API.

    Args:
        model_id: Hugging Face model ID
        callback: Progress callback function(progress_percent, message)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Notify starting download
        if callback:
            callback(0, f"Preparing to download {model_id}")

        # Import huggingface_hub
        from huggingface_hub import snapshot_download

        # Get the local directory path
        local_dir = os.path.join(get_models_dir(), 'huggingface', model_id.split('/')[-1])
        os.makedirs(local_dir, exist_ok=True)

        # Use the simplest form without progress_callback
        if callback:
            callback(10, f"Downloading {model_id} (progress not available)")

        # Basic download without progress tracking
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        if callback:
            callback(100, f"Successfully downloaded {model_id}")

        return True

    except Exception as e:
        logger.error(f"Error downloading model from Hugging Face {model_id}: {str(e)}")
        if callback:
            callback(0, f"Error downloading {model_id}: {str(e)}")
        return False

def download_models(model_keys=None, progress_callback=None):
    """Download required models.

    Args:
        model_keys: List of specific model keys to download, or None for all required models
        progress_callback: Progress callback function(progress_percent, message)

    Returns:
        True if all downloads succeeded, False otherwise
    """
    # Determine which models to download
    if model_keys is None:
        model_keys = [key for key, info in MODELS.items() if info.get('required', False)]

    # Check which models already exist
    existing_models = check_models_exist()
    models_to_download = [key for key in model_keys if not existing_models.get(key, False)]

    if not models_to_download:
        if progress_callback:
            progress_callback(100, "All required models are already downloaded")
        return True

    # Calculate total download size for progress tracking
    total_size = sum(MODELS[key].get('size', 0) for key in models_to_download)
    downloaded_size = 0

    # Download each model
    success = True
    for model_key in models_to_download:
        model_info = MODELS[model_key]
        model_size = model_info.get('size', 0)

        # Create a progress callback for this specific model
        def model_progress_callback(progress, message):
            if progress_callback:
                # Calculate overall progress
                if total_size > 0:
                    # Weighted progress based on model size
                    overall_progress = int(
                        (downloaded_size + (model_size * progress / 100)) / total_size * 100
                    )
                else:
                    # Equal weighting if sizes are unknown
                    completed_models = sum(1 for k in model_keys if existing_models.get(k, False))
                    current_model_progress = progress / 100
                    overall_progress = int(
                        (completed_models + current_model_progress) / len(model_keys) * 100
                    )

                progress_callback(overall_progress, message)

        # Log download start
        logger.info(f"Downloading model: {model_info['name']}")
        if progress_callback:
            progress_callback(
                int(downloaded_size / total_size * 100) if total_size > 0 else 0,
                f"Downloading {model_info['name']}..."
            )

        # Download based on model type
        if 'huggingface_id' in model_info:
            model_success = download_huggingface_model(
                model_info['huggingface_id'],
                callback=model_progress_callback
            )
        else:
            destination = os.path.join(get_models_dir(), model_info['path'])
            model_success = download_file(
                model_info['url'],
                destination,
                expected_size=model_info.get('size'),
                expected_md5=model_info.get('md5'),
                callback=model_progress_callback
            )

        # Update progress
        if model_success:
            logger.info(f"Successfully downloaded model: {model_info['name']}")
            downloaded_size += model_size
        else:
            logger.error(f"Failed to download model: {model_info['name']}")
            success = False

    # Final log
    if success:
        logger.info("All models downloaded successfully")
        if progress_callback:
            progress_callback(100, "All models downloaded successfully")
    else:
        logger.error("Some models failed to download")
        if progress_callback:
            progress_callback(100, "Some models failed to download")

    return success

def get_model_info():
    """Get information about all models."""
    models_status = check_models_exist()

    info = []
    for model_key, model_info in MODELS.items():
        info.append({
            "key": model_key,
            "name": model_info["name"],
            "installed": models_status.get(model_key, False),
            "required": model_info.get("required", False),
            "size": model_info.get("size", 0),
            "size_str": f"{model_info.get('size', 0) / (1024 * 1024):.1f} MB"
        })

    return info

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="NESR Model Downloader")
    parser.add_argument("--list", action="store_true", help="List all models and their status")
    parser.add_argument("--download", action="store_true", help="Download all required models")
    parser.add_argument("--model", type=str, help="Specific model to download")
    args = parser.parse_args()

    # Setup console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    if args.list:
        print("Available models:")
        for info in get_model_info():
            status = "✓ Installed" if info["installed"] else "✗ Not installed"
            required = "[Required]" if info["required"] else "[Optional]"
            print(f"- {info['name']} ({info['key']}): {status} {required} ({info['size_str']})")

    elif args.download:
        if args.model:
            if args.model in MODELS:
                print(f"Downloading model: {MODELS[args.model]['name']}")
                download_models([args.model])
            else:
                print(f"Error: Unknown model '{args.model}'")
                print("Available models: " + ", ".join(MODELS.keys()))
        else:
            print("Downloading all required models")
            download_models()

    else:
        parser.print_help()
