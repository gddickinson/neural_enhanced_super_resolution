#!/usr/bin/env python3
# NESR - Neural Enhanced Super-Resolution
# Main entry point for both CLI and GUI application

import os
import sys
import argparse
import logging
import importlib.util

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nesr')

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    # Core dependencies
    core_deps = ["torch", "numpy", "PIL", "cv2"]
    for dep in core_deps:
        if importlib.util.find_spec(dep) is None:
            missing_deps.append(dep)
    
    # GUI dependencies (only check if not in CLI mode)
    if not "--cli" in sys.argv:
        gui_deps = ["PyQt5.QtWidgets"]
        for dep in gui_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep.split('.')[0])
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            if dep == "torch":
                print("  - torch: pip install torch torchvision")
            elif dep == "cv2":
                print("  - opencv: pip install opencv-python")
            elif dep == "PIL":
                print("  - Pillow: pip install Pillow")
            elif dep == "PyQt5":
                print("  - PyQt5: pip install PyQt5")
            else:
                print(f"  - {dep}: pip install {dep}")
        print("\nInstall all dependencies with:")
        print("pip install torch torchvision opencv-python Pillow PyQt5 numpy diffusers transformers")
        return False
    
    return True

def setup_environment():
    """Configure environment variables and check dependencies."""
    # Set environment variables for better GPU performance
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
    
    # Path setup for the application
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.append(app_dir)
    
    # Check if torch is available and configured correctly
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            logger.info(f"CUDA is available: {device_count} device(s), using {device_name}")
        else:
            logger.info("CUDA is not available, using CPU mode")
    except ImportError:
        logger.warning("PyTorch not found, some functionality may not work")

def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="NESR - Neural Enhanced Super-Resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main mode selection
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface")
    parser.add_argument("--cli", action="store_true", help="Run in command line mode")
    
    # CLI mode arguments
    parser.add_argument("--input", "-i", help="Input image path (CLI mode)")
    parser.add_argument("--output", "-o", help="Output path (CLI mode)")
    parser.add_argument("--config", "-c", help="Path to configuration JSON file")
    parser.add_argument("--iterations", "-n", type=int, default=3, help="Number of enhancement iterations")
    parser.add_argument("--upscale_factor", "-u", type=float, default=2.0, help="Base upscale factor per iteration")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--prompt", "-p", default=None, help="Text prompt for diffusion guidance")
    parser.add_argument("--download_models", action="store_true", help="Download required models")
    
    return parser.parse_args()

def run_cli_mode(args):
    """Run the application in command line mode."""
    try:
        from nesr import SuperResolutionPipeline
        import json
        
        print("Starting NESR in CLI mode")
        
        # Handle model downloads if requested
        if args.download_models:
            from nesr.utils.downloader import download_models
            download_models()
            print("Model download complete.")
            return
        
        # Check if input is provided
        if not args.input:
            print("Error: Input image path is required in CLI mode.")
            print("Use --input or -i to specify the input image path.")
            return
        
        # Load configuration if provided
        config = None
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                print(f"Loaded configuration from {args.config}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
                return
        
        # Create base configuration from arguments
        base_config = {
            "iterations": args.iterations,
            "upscale_factor": args.upscale_factor,
            "output_dir": os.path.dirname(args.output) if args.output else "outputs",
        }
        
        # Merge with loaded config if available
        if config:
            config.update(base_config)
        else:
            config = base_config
        
        # Initialize and run the pipeline
        try:
            pipeline = SuperResolutionPipeline(device=args.device, config=config)
            output_path = pipeline.enhance_image(args.input, prompt=args.prompt)
            
            # Move/rename output if specific path requested
            if args.output and output_path != args.output:
                import shutil
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                shutil.copy2(output_path, args.output)
                print(f"Enhanced image saved to: {args.output}")
        except Exception as e:
            print(f"Error during image enhancement: {e}")
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure all dependencies are installed.")

def launch_gui():
    """Launch the graphical user interface."""
    # First make sure torch is available before attempting to import GUI
    try:
        import torch
        import numpy as np
        import cv2
        from PIL import Image
    except ImportError as e:
        print(f"Error importing core dependencies: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision opencv-python Pillow numpy")
        return
    
    # Now try to import and create PyQt application
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt6.QtWidgets import QApplication
            print("Warning: Using PyQt6 instead of PyQt5. Some features may not work correctly.")
        except ImportError:
            print("Error: Neither PyQt5 nor PyQt6 could be imported.")
            print("Please install PyQt5 with: pip install PyQt5")
            return
    
    # Create the application instance first
    app = QApplication(sys.argv)
    
    # Now import and create the main window
    try:
        from nesr.gui.app import NESRApplication
        main_window = NESRApplication()
        
        # Check if qtmodern is available for styling
        try:
            import qtmodern.styles
            import qtmodern.windows
            qtmodern.styles.dark(app)
            modern_window = qtmodern.windows.ModernWindow(main_window)
            modern_window.show()
        except ImportError:
            # Fall back to regular styling
            main_window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install PyQt5 qtmodern pyqtgraph")

def main():
    """Main entry point for the application."""
    # Configure environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine mode
    if args.gui or (not args.cli and not args.input):
        # Default to GUI if no mode specified and no input provided
        launch_gui()
    else:
        # Run in CLI mode
        run_cli_mode(args)

if __name__ == "__main__":
    main()
