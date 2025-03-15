# NESR Project Structure

```
nesr/
├── main.py                        # Main entry point
├── README.md                      # Project documentation
├── requirements.txt               # Package dependencies
├── setup.py                       # Installation script
├── nesr/                          # Main package
│   ├── __init__.py                # Package initialization
│   ├── nesr.py                    # Core super-resolution implementation
│   ├── gui/                       # GUI module
│   │   ├── __init__.py            # GUI module initialization
│   │   ├── app.py                 # Main application window
│   │   └── dialogs/               # Additional dialogs (not implemented yet)
│   └── utils/                     # Utility modules
│       ├── __init__.py            # Utilities initialization
│       ├── downloader.py          # Model downloading utilities
│       └── image_utils.py         # Image processing utilities
└── models/                        # Directory for downloaded models
    └── weights/                   # Model weights directory
```

## Key Components

### Core Files

- **main.py**: Entry point for both CLI and GUI modes
- **nesr.py**: Main super-resolution pipeline implementation
- **setup.py**: Package installation script

### GUI Components

- **app.py**: Main window and application logic
- **ImageViewer**: Widget for displaying and comparing images
- **EnhancementSettings**: Widget for configuring enhancement parameters
- **DegradationSettings**: Widget for adding degradation to test images

### Utilities

- **downloader.py**: Model management (downloading, checking)
- **image_utils.py**: Image processing functions for degradation, comparison, etc.

## Data Flow

1. User loads an image through the GUI
2. Optional: User applies degradation to simulate a low-quality image
3. User configures enhancement settings and starts the process
4. Pipeline processes the image through multiple iterations:
   - Pre-processing (denoising, contrast enhancement)
   - Segmentation-based enhancement
   - Multi-model super-resolution (ESRGAN, Diffusion)
   - Model ensemble and post-processing
5. Progress and intermediate results are displayed in real-time
6. Final enhanced image is displayed side-by-side with the original

## Model Management

The application automatically checks for required models on startup and offers to download them if missing. Models are downloaded to a central location and loaded on demand to minimize memory usage.
