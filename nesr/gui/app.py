"""
NESR - Neural Enhanced Super-Resolution GUI Application
Main GUI application module that provides a user-friendly interface for image enhancement.
"""

import os
import sys
import time
import logging
import threading
import queue
import json
from datetime import datetime
from io import BytesIO  # Added missing import

# Import core dependencies explicitly
import torch
import numpy as np
import cv2
from PIL import Image, ImageQt

# Import PyQt modules
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QTabWidget,
    QSlider, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QProgressBar, QLineEdit, QMessageBox,
    QDockWidget, QAction, QToolBar, QStatusBar, QMenu, QScrollArea,
    QDialog, QListWidget, QSizePolicy, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QTextCursor, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QSettings

# Try to import qtmodern for styling
try:
    import qtmodern.styles
    import qtmodern.windows
    HAS_QTMODERN = True
except ImportError:
    HAS_QTMODERN = False

# Import the NESR core - with proper error handling
try:
    from nesr import SuperResolutionPipeline
    from nesr.utils.downloader import download_models, check_models_exist
    from nesr.utils.image_utils import add_noise, blur_image, downsample_image, apply_jpeg_compression
except ImportError as e:
    logging.error(f"Failed to import NESR modules: {e}")
    # We'll handle this in the constructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('nesr.gui')

class LogHandler(logging.Handler):
    """Custom logging handler that emits signals for GUI updates."""

    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(record.levelno, msg)

class EnhancementWorker(QThread):
    """Worker thread for running image enhancement."""

    progress_signal = pyqtSignal(int, str)  # Progress percentage, message
    image_signal = pyqtSignal(object)  # Intermediate image result
    finished_signal = pyqtSignal(str)  # Output path
    error_signal = pyqtSignal(str)  # Error message

    def __init__(self, pipeline, input_path, config, prompt=None):
        super().__init__()
        self.pipeline = pipeline
        self.input_path = input_path
        self.config = config
        self.prompt = prompt
        self.running = True

    def run(self):
        """Run the enhancement process."""
        try:
            # Create a custom progress callback
            last_progress = [0]
            last_update = [time.time()]
            update_interval = 0.5  # Update GUI every 0.5 seconds

            def progress_callback(stage, iteration, total_iterations, message=None):
                if not self.running:
                    return False  # Cancel processing

                # Calculate overall progress percentage
                overall_progress = int((100 * iteration) / total_iterations)

                # Throttle updates to avoid GUI overload
                now = time.time()
                if overall_progress != last_progress[0] and (now - last_update[0]) > update_interval:
                    status_msg = f"{stage}: {message}" if message else f"{stage}"
                    self.progress_signal.emit(overall_progress, status_msg)
                    last_progress[0] = overall_progress
                    last_update[0] = now

                return True  # Continue processing

            # Set up intermediate image callback
            def image_callback(current_image):
                if not self.running:
                    return False

                # Emit the current image for display
                self.image_signal.emit(current_image)
                return True

            # Add callbacks to config
            self.config["progress_callback"] = progress_callback
            self.config["image_callback"] = image_callback

            # Run the enhancement
            output_path = self.pipeline.enhance_image(self.input_path, prompt=self.prompt)

            if self.running:
                self.finished_signal.emit(output_path)

        except Exception as e:
            logger.exception("Enhancement failed")
            self.error_signal.emit(str(e))

    def stop(self):
        """Stop the enhancement process."""
        self.running = False

class ModelDownloadWorker(QThread):
    """Worker thread for downloading models."""

    progress_signal = pyqtSignal(int, str)  # Progress percentage, message
    finished_signal = pyqtSignal(bool)  # Success status
    error_signal = pyqtSignal(str)  # Error message

    def __init__(self, models=None):
        super().__init__()
        self.models = models  # List of specific models to download, or None for all

    def run(self):
        """Run the download process."""
        try:
            def progress_callback(progress, message):
                self.progress_signal.emit(progress, message)
                return True  # Continue downloading

            download_models(self.models, progress_callback)
            self.finished_signal.emit(True)

        except Exception as e:
            logger.exception("Model download failed")
            self.error_signal.emit(str(e))

class ImageViewer(QWidget):
    """Widget for displaying and comparing images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image = None
        self.enhanced_image = None
        self.zoom_level = 1.0

        # Set a minimum size to ensure adequate display space
        self.setMinimumHeight(400)

        # Default view mode
        self.current_view_mode = "Side by Side"

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins to maximize space

        # Zoom controls in a toolbar-like layout
        zoom_layout = QHBoxLayout()
        zoom_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.set_zoom)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(self.zoom_in)

        self.zoom_reset_btn = QPushButton("100%")
        self.zoom_reset_btn.setFixedWidth(50)
        self.zoom_reset_btn.clicked.connect(self.reset_zoom)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(40)

        # Add fit to view button
        self.fit_view_btn = QPushButton("Fit")
        self.fit_view_btn.setFixedWidth(40)
        self.fit_view_btn.clicked.connect(self.fit_to_view)

        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_reset_btn)
        zoom_layout.addWidget(self.fit_view_btn)
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addStretch()

        # View mode controls
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Side by Side", "Split View", "Before/After"])
        self.view_mode_combo.setCurrentText("Side by Side")
        self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)

        zoom_layout.addWidget(QLabel("View:"))
        zoom_layout.addWidget(self.view_mode_combo)

        # Image display
        self.splitter = QSplitter(Qt.Horizontal)

        # Make sure the splitter can expand to fill available space
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Original image
        self.original_container = QScrollArea()
        self.original_container.setWidgetResizable(True)
        self.original_label = QLabel("No image loaded")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_container.setWidget(self.original_label)

        # Enhanced image
        self.enhanced_container = QScrollArea()
        self.enhanced_container.setWidgetResizable(True)
        self.enhanced_label = QLabel("No enhanced image")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_container.setWidget(self.enhanced_label)

        self.splitter.addWidget(self.original_container)
        self.splitter.addWidget(self.enhanced_container)
        self.splitter.setSizes([int(self.width()/2), int(self.width()/2)])

        # Image info labels
        info_layout = QHBoxLayout()
        self.original_info = QLabel("Original: N/A")
        self.enhanced_info = QLabel("Enhanced: N/A")
        info_layout.addWidget(self.original_info)
        info_layout.addSpacing(20)
        info_layout.addWidget(self.enhanced_info)
        info_layout.addStretch()

        # Add to main layout
        layout.addLayout(zoom_layout)
        layout.addWidget(self.splitter, 1)
        layout.addLayout(info_layout)

    def set_original_image(self, image_path=None, image=None):
        """Set the original image from path or numpy array."""
        if image_path and os.path.exists(image_path):
            self.original_image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            height, width = self.original_image.shape[:2]
            self.original_info.setText(f"Original: {width}x{height}")
        elif image is not None:
            self.original_image = image.copy()
            height, width = self.original_image.shape[:2]
            self.original_info.setText(f"Original: {width}x{height}")
        else:
            self.original_image = None
            self.original_info.setText("Original: N/A")
            self.original_label.setText("No image loaded")
            return

        self.update_display()

        # Auto-fit to view when loading a new image
        self.fit_to_view()

    def set_enhanced_image(self, image_path=None, image=None):
        """Set the enhanced image from path or numpy array."""
        if image_path and os.path.exists(image_path):
            self.enhanced_image = cv2.imread(image_path)
            self.enhanced_image = cv2.cvtColor(self.enhanced_image, cv2.COLOR_BGR2RGB)
            height, width = self.enhanced_image.shape[:2]
            self.enhanced_info.setText(f"Enhanced: {width}x{height}")
        elif image is not None:
            self.enhanced_image = image.copy()
            height, width = self.enhanced_image.shape[:2]
            self.enhanced_info.setText(f"Enhanced: {width}x{height}")
        else:
            self.enhanced_image = None
            self.enhanced_info.setText("Enhanced: N/A")
            self.enhanced_label.setText("No enhanced image")
            return

        self.update_display()

        # Auto-fit to view when loading a new enhanced image
        self.fit_to_view()

    def update_display(self):
        """Update the image displays with current zoom level and view mode."""
        # Handle different view modes
        if self.current_view_mode == "Side by Side":
            self._update_side_by_side_view()
        elif self.current_view_mode == "Split View":
            self._update_split_view()
        elif self.current_view_mode == "Before/After":
            self._update_before_after_view()
        else:
            # Default to side by side
            self._update_side_by_side_view()

    def _update_side_by_side_view(self):
        """Update for standard side-by-side view."""
        # Update original image display
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
            new_width = int(width * self.zoom_level)
            new_height = int(height * self.zoom_level)

            scaled_image = cv2.resize(
                self.original_image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA if self.zoom_level < 1 else cv2.INTER_LANCZOS4
            )

            q_image = QImage(
                scaled_image.data,
                new_width,
                new_height,
                scaled_image.strides[0],
                QImage.Format_RGB888
            )

            self.original_label.setPixmap(QPixmap.fromImage(q_image))
            self.original_label.setMinimumSize(new_width, new_height)

        # Update enhanced image display
        if self.enhanced_image is not None:
            height, width = self.enhanced_image.shape[:2]
            new_width = int(width * self.zoom_level)
            new_height = int(height * self.zoom_level)

            scaled_image = cv2.resize(
                self.enhanced_image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA if self.zoom_level < 1 else cv2.INTER_LANCZOS4
            )

            q_image = QImage(
                scaled_image.data,
                new_width,
                new_height,
                scaled_image.strides[0],
                QImage.Format_RGB888
            )

            self.enhanced_label.setPixmap(QPixmap.fromImage(q_image))
            self.enhanced_label.setMinimumSize(new_width, new_height)

    def _update_split_view(self):
        """Update for split view (half original, half enhanced)."""
        if self.original_image is None or self.enhanced_image is None:
            self._update_side_by_side_view()
            return

        # Create a composite image with left half original and right half enhanced
        try:
            # First, resize to the same dimensions if needed
            org_h, org_w = self.original_image.shape[:2]
            enh_h, enh_w = self.enhanced_image.shape[:2]

            # Use the larger dimensions
            target_w = max(org_w, enh_w)
            target_h = max(org_h, enh_h)

            # Resize both images to target size
            org_resized = cv2.resize(self.original_image, (target_w, target_h))
            enh_resized = cv2.resize(self.enhanced_image, (target_w, target_h))

            # Create composite image
            composite = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            mid_point = target_w // 2
            composite[:, :mid_point] = org_resized[:, :mid_point]
            composite[:, mid_point:] = enh_resized[:, mid_point:]

            # Add a dividing line
            composite[:, mid_point-1:mid_point+1] = [255, 0, 0]  # Red line

            # Apply zoom
            new_width = int(target_w * self.zoom_level)
            new_height = int(target_h * self.zoom_level)

            scaled_composite = cv2.resize(
                composite,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA if self.zoom_level < 1 else cv2.INTER_LANCZOS4
            )

            # Convert to QImage and display
            q_image = QImage(
                scaled_composite.data,
                new_width,
                new_height,
                scaled_composite.strides[0],
                QImage.Format_RGB888
            )

            # Hide the second image panel
            self.enhanced_container.hide()

            # Show in original panel
            self.original_label.setPixmap(QPixmap.fromImage(q_image))
            self.original_label.setMinimumSize(new_width, new_height)

        except Exception as e:
            print(f"Error creating split view: {e}")
            # Fall back to side-by-side if there's an error
            self._update_side_by_side_view()

    def _update_before_after_view(self):
        """Update for before/after toggle view."""
        # Simply use the same side-by-side view but add toggle button if needed
        self._update_side_by_side_view()

        # In a more complete implementation, we would add a toggle button
        # to switch between before and after views

    def change_view_mode(self, mode):
        """Change the image view mode."""
        self.current_view_mode = mode

        # Reset UI elements based on view mode
        if mode == "Side by Side":
            self.enhanced_container.show()
            self.splitter.setSizes([int(self.width()/2), int(self.width()/2)])
        elif mode == "Split View":
            self.enhanced_container.hide()
        elif mode == "Before/After":
            self.enhanced_container.show()
            self.splitter.setSizes([int(self.width()/2), int(self.width()/2)])

        # Update display with new mode
        self.update_display()

    def zoom_in(self):
        """Increase zoom level."""
        self.zoom_level = min(4.0, self.zoom_level + 0.1)
        self.zoom_slider.setValue(int(self.zoom_level * 100))
        self.update_zoom_label()
        self.update_display()

    def zoom_out(self):
        """Decrease zoom level."""
        self.zoom_level = max(0.1, self.zoom_level - 0.1)
        self.zoom_slider.setValue(int(self.zoom_level * 100))
        self.update_zoom_label()
        self.update_display()

    def set_zoom(self, value):
        """Set zoom level from slider value."""
        self.zoom_level = value / 100
        self.update_zoom_label()
        self.update_display()

    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_level = 1.0
        self.zoom_slider.setValue(100)
        self.update_zoom_label()
        self.update_display()

    def fit_to_view(self):
        """Fit images to view."""
        if self.original_image is None and self.enhanced_image is None:
            return

        # Get available space
        available_width = self.splitter.width() // 2  # Half for each image in side-by-side
        available_height = self.splitter.height()

        # Get max image dimensions
        max_width = 0
        max_height = 0

        if self.original_image is not None:
            org_h, org_w = self.original_image.shape[:2]
            max_width = max(max_width, org_w)
            max_height = max(max_height, org_h)

        if self.enhanced_image is not None:
            enh_h, enh_w = self.enhanced_image.shape[:2]
            max_width = max(max_width, enh_w)
            max_height = max(max_height, enh_h)

        if max_width == 0 or max_height == 0:
            return

        # Calculate zoom factor to fit
        width_factor = available_width / max_width if max_width > 0 else 1
        height_factor = available_height / max_height if max_height > 0 else 1
        fit_factor = min(width_factor, height_factor) * 0.9  # 90% of space for margin

        # Set zoom level
        self.zoom_level = fit_factor
        self.zoom_slider.setValue(int(self.zoom_level * 100))
        self.update_zoom_label()
        self.update_display()

    def update_zoom_label(self):
        """Update the zoom level display."""
        self.zoom_label.setText(f"{int(self.zoom_level * 100)}%")

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        # Dynamically update the fit on resize
        QTimer.singleShot(100, self.fit_to_view)

class LogConsole(QWidget):
    """Widget for displaying log messages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font = QFont("Courier")
        self.log_text.setFont(font)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear Console")
        self.clear_btn.clicked.connect(self.clear_log)

        self.save_btn = QPushButton("Save Log")
        self.save_btn.clicked.connect(self.save_log)

        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()

        # Add to layout
        layout.addWidget(self.log_text)
        layout.addLayout(btn_layout)

    def add_log(self, level, message):
        """Add a log message to the console."""
        # Set color based on log level
        if level >= logging.ERROR:
            color = "red"
        elif level >= logging.WARNING:
            color = "orange"
        elif level >= logging.INFO:
            color = "black"
        else:
            color = "gray"

        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add formatted message
        self.log_text.append(f'<span style="color:{color};">[{timestamp}] {message}</span>')

        # Scroll to bottom
        self.log_text.moveCursor(QTextCursor.End)

    def clear_log(self):
        """Clear the log console."""
        self.log_text.clear()

    def save_log(self):
        """Save the log to a file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.log_text.toPlainText())
                logger.info(f"Log saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save log: {e}")

class DegradationSettings(QWidget):
    """Widget for configuring image degradation settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Downscale settings
        downscale_group = QGroupBox("Downscale")
        downscale_layout = QVBoxLayout()

        self.enable_downscale = QCheckBox("Enable downscaling")
        self.enable_downscale.setChecked(True)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale factor:"))
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.1, 0.9)
        self.scale_factor.setSingleStep(0.1)
        self.scale_factor.setValue(0.5)
        scale_layout.addWidget(self.scale_factor)

        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Interpolation:"))
        self.interpolation = QComboBox()
        self.interpolation.addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"])
        self.interpolation.setCurrentIndex(2)  # Bicubic default
        interp_layout.addWidget(self.interpolation)

        downscale_layout.addWidget(self.enable_downscale)
        downscale_layout.addLayout(scale_layout)
        downscale_layout.addLayout(interp_layout)
        downscale_group.setLayout(downscale_layout)

        # Noise settings
        noise_group = QGroupBox("Noise")
        noise_layout = QVBoxLayout()

        self.enable_noise = QCheckBox("Add noise")
        self.enable_noise.setChecked(False)

        noise_type_layout = QHBoxLayout()
        noise_type_layout.addWidget(QLabel("Noise type:"))
        self.noise_type = QComboBox()
        self.noise_type.addItems(["Gaussian", "Salt & Pepper", "Speckle", "Poisson"])
        noise_type_layout.addWidget(self.noise_type)

        noise_amount_layout = QHBoxLayout()
        noise_amount_layout.addWidget(QLabel("Amount:"))
        self.noise_amount = QSlider(Qt.Horizontal)
        self.noise_amount.setMinimum(1)
        self.noise_amount.setMaximum(50)
        self.noise_amount.setValue(10)
        self.noise_amount_label = QLabel("10%")
        self.noise_amount.valueChanged.connect(
            lambda v: self.noise_amount_label.setText(f"{v}%")
        )
        noise_amount_layout.addWidget(self.noise_amount)
        noise_amount_layout.addWidget(self.noise_amount_label)

        noise_layout.addWidget(self.enable_noise)
        noise_layout.addLayout(noise_type_layout)
        noise_layout.addLayout(noise_amount_layout)
        noise_group.setLayout(noise_layout)

        # Blur settings
        blur_group = QGroupBox("Blur")
        blur_layout = QVBoxLayout()

        self.enable_blur = QCheckBox("Add blur")
        self.enable_blur.setChecked(False)

        blur_type_layout = QHBoxLayout()
        blur_type_layout.addWidget(QLabel("Blur type:"))
        self.blur_type = QComboBox()
        self.blur_type.addItems(["Gaussian", "Box", "Motion"])
        blur_type_layout.addWidget(self.blur_type)

        blur_amount_layout = QHBoxLayout()
        blur_amount_layout.addWidget(QLabel("Radius:"))
        self.blur_amount = QSlider(Qt.Horizontal)
        self.blur_amount.setMinimum(1)
        self.blur_amount.setMaximum(10)
        self.blur_amount.setValue(3)
        self.blur_amount_label = QLabel("3px")
        self.blur_amount.valueChanged.connect(
            lambda v: self.blur_amount_label.setText(f"{v}px")
        )
        blur_amount_layout.addWidget(self.blur_amount)
        blur_amount_layout.addWidget(self.blur_amount_label)

        blur_layout.addWidget(self.enable_blur)
        blur_layout.addLayout(blur_type_layout)
        blur_layout.addLayout(blur_amount_layout)
        blur_group.setLayout(blur_layout)

        # JPEG compression settings
        jpeg_group = QGroupBox("JPEG Compression")
        jpeg_layout = QVBoxLayout()

        self.enable_jpeg = QCheckBox("Apply JPEG compression")
        self.enable_jpeg.setChecked(False)

        jpeg_quality_layout = QHBoxLayout()
        jpeg_quality_layout.addWidget(QLabel("Quality:"))
        self.jpeg_quality = QSlider(Qt.Horizontal)
        self.jpeg_quality.setMinimum(1)
        self.jpeg_quality.setMaximum(95)
        self.jpeg_quality.setValue(75)
        self.jpeg_quality_label = QLabel("75%")
        self.jpeg_quality.valueChanged.connect(
            lambda v: self.jpeg_quality_label.setText(f"{v}%")
        )
        jpeg_quality_layout.addWidget(self.jpeg_quality)
        jpeg_quality_layout.addWidget(self.jpeg_quality_label)

        jpeg_layout.addWidget(self.enable_jpeg)
        jpeg_layout.addLayout(jpeg_quality_layout)
        jpeg_group.setLayout(jpeg_layout)

        # Buttons
        btn_layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview")

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        self.reset_btn = QPushButton("Reset")

        btn_layout.addWidget(self.preview_btn)
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.reset_btn)

        # Add to main layout
        layout.addWidget(downscale_group)
        layout.addWidget(noise_group)
        layout.addWidget(blur_group)
        layout.addWidget(jpeg_group)
        layout.addLayout(btn_layout)
        layout.addStretch()

    def get_settings(self):
        """Get the current degradation settings."""
        settings = {
            "downscale": {
                "enabled": self.enable_downscale.isChecked(),
                "factor": self.scale_factor.value(),
                "interpolation": self.interpolation.currentText().lower(),
            },
            "noise": {
                "enabled": self.enable_noise.isChecked(),
                "type": self.noise_type.currentText().lower(),
                "amount": self.noise_amount.value() / 100,
            },
            "blur": {
                "enabled": self.enable_blur.isChecked(),
                "type": self.blur_type.currentText().lower(),
                "radius": self.blur_amount.value(),
            },
            "jpeg": {
                "enabled": self.enable_jpeg.isChecked(),
                "quality": self.jpeg_quality.value(),
            }
        }

        return settings

class EnhancementSettings(QWidget):
    """Widget for configuring basic enhancement settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Basic settings
        basic_group = QGroupBox("Processing Parameters")
        basic_layout = QVBoxLayout()

        # Iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 5)
        self.iterations.setValue(3)
        iter_layout.addWidget(self.iterations)

        # Upscale factor
        factor_layout = QHBoxLayout()
        factor_layout.addWidget(QLabel("Upscale factor per iteration:"))
        self.upscale_factor = QDoubleSpinBox()
        self.upscale_factor.setRange(1.1, 4.0)
        self.upscale_factor.setSingleStep(0.1)
        self.upscale_factor.setValue(2.0)
        factor_layout.addWidget(self.upscale_factor)

        # Denoise level
        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(QLabel("Denoise level:"))
        self.denoise_level = QSlider(Qt.Horizontal)
        self.denoise_level.setMinimum(0)
        self.denoise_level.setMaximum(10)
        self.denoise_level.setValue(5)
        self.denoise_level_label = QLabel("0.5")
        self.denoise_level.valueChanged.connect(
            lambda v: self.denoise_level_label.setText(f"{v/10:.1f}")
        )
        denoise_layout.addWidget(self.denoise_level)
        denoise_layout.addWidget(self.denoise_level_label)

        # Add to basic layout
        basic_layout.addLayout(iter_layout)
        basic_layout.addLayout(factor_layout)
        basic_layout.addLayout(denoise_layout)
        basic_group.setLayout(basic_layout)

        # Model selection
        model_group = QGroupBox("Models")
        model_layout = QVBoxLayout()

        self.use_esrgan = QCheckBox("Use Real-ESRGAN")
        self.use_esrgan.setChecked(True)

        self.use_diffusion = QCheckBox("Use Stable Diffusion upscaler")
        self.use_diffusion.setChecked(True)

        self.use_segment = QCheckBox("Use segmentation-based enhancement")
        self.use_segment.setChecked(True)

        # Text prompt for diffusion
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt = QLineEdit("a high resolution detailed photograph")
        prompt_layout.addWidget(self.prompt)

        # Add to model layout
        model_layout.addWidget(self.use_esrgan)
        model_layout.addWidget(self.use_diffusion)
        model_layout.addWidget(self.use_segment)
        model_layout.addLayout(prompt_layout)
        model_group.setLayout(model_layout)

        # Image quality options
        quality_group = QGroupBox("Image Quality")
        quality_layout = QVBoxLayout()

        # Detail preservation and sharpening
        self.preserve_details = QCheckBox("Preserve details")
        self.preserve_details.setChecked(True)

        self.adaptive_sharpen = QCheckBox("Adaptive sharpening")
        self.adaptive_sharpen.setChecked(True)

        # Add to quality layout
        quality_layout.addWidget(self.preserve_details)
        quality_layout.addWidget(self.adaptive_sharpen)
        quality_group.setLayout(quality_layout)

        # Preset management
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Default")
        self.preset_combo.addItem("High Quality")
        self.preset_combo.addItem("Fast")
        self.preset_combo.addItem("Apple Silicon Optimized")  # Add the new preset

        # Auto-select "Apple Silicon Optimized" if running on MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.preset_combo.setCurrentText("Apple Silicon Optimized")
        except:
            pass  # Fallback to default if there's an error

        self.load_preset_btn = QPushButton("Load")
        self.load_preset_btn.clicked.connect(self.load_preset)

        self.save_preset_btn = QPushButton("Save As...")
        self.save_preset_btn.clicked.connect(self.save_preset)

        # Add to preset layout
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.load_preset_btn)
        preset_layout.addWidget(self.save_preset_btn)
        preset_group.setLayout(preset_layout)

        # Add all groups to main layout
        layout.addWidget(basic_group)
        layout.addWidget(model_group)
        layout.addWidget(quality_group)
        layout.addWidget(preset_group)
        layout.addStretch()

    def get_settings(self):
        """Get the current enhancement settings as a configuration dictionary."""
        config = {
            "iterations": self.iterations.value(),
            "upscale_factor": self.upscale_factor.value(),
            "use_esrgan": self.use_esrgan.isChecked(),
            "use_diffusion": self.use_diffusion.isChecked(),
            "segment_enhancement": self.use_segment.isChecked(),
            "preserve_details": self.preserve_details.isChecked(),
            "adaptive_sharpening": self.adaptive_sharpen.isChecked(),
            "denoise_level": self.denoise_level.value() / 10,
        }

        return config

    def get_prompt(self):
        """Get the text prompt for diffusion guidance."""
        if self.use_diffusion.isChecked():
            return self.prompt.text()
        return None

    def load_preset(self):
        """Load settings from a preset."""
        preset_name = self.preset_combo.currentText()

        if preset_name == "Default":
            self.iterations.setValue(3)
            self.upscale_factor.setValue(2.0)
            self.use_esrgan.setChecked(True)
            self.use_diffusion.setChecked(True)
            self.use_segment.setChecked(True)
            self.prompt.setText("a high resolution detailed photograph")
            self.denoise_level.setValue(5)
            self.preserve_details.setChecked(True)
            self.adaptive_sharpen.setChecked(True)

        elif preset_name == "High Quality":
            self.iterations.setValue(3)
            self.upscale_factor.setValue(2.0)
            self.use_esrgan.setChecked(True)
            self.use_diffusion.setChecked(True)
            self.use_segment.setChecked(True)
            self.prompt.setText("a highly detailed professional photograph")
            self.denoise_level.setValue(3)
            self.preserve_details.setChecked(True)
            self.adaptive_sharpen.setChecked(True)

        elif preset_name == "Fast":
            self.iterations.setValue(2)
            self.upscale_factor.setValue(2.0)
            self.use_esrgan.setChecked(True)
            self.use_diffusion.setChecked(False)
            self.use_segment.setChecked(False)
            self.prompt.setText("")
            self.denoise_level.setValue(5)
            self.preserve_details.setChecked(True)
            self.adaptive_sharpen.setChecked(True)

        # Add a new preset specifically for Apple Silicon
        elif preset_name == "Apple Silicon Optimized":
            self.iterations.setValue(2)
            self.upscale_factor.setValue(2.0)
            self.use_esrgan.setChecked(True)
            self.use_diffusion.setChecked(True)
            self.use_segment.setChecked(False)
            self.prompt.setText("a detailed photograph")
            self.denoise_level.setValue(5)
            self.preserve_details.setChecked(True)
            self.adaptive_sharpen.setChecked(True)

            # Signal to parent to update advanced settings too
            if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'update_advanced_for_preset'):
                self.parent().parent().update_advanced_for_preset(preset_name)

    def save_preset(self):
        """Save current settings as a preset."""
        # In a real implementation, this would show a dialog for naming and saving
        preset_name, ok = QInputDialog.getText(
            self, "Save Preset", "Enter preset name:"
        )

        if ok and preset_name:
            # Add to combo box
            self.preset_combo.addItem(preset_name)
            self.preset_combo.setCurrentText(preset_name)

            # In a real implementation, this would save to settings or file

class AdvancedSettings(QWidget):
    """Widget for configuring advanced enhancement settings including memory management."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)

        # Use a scroll area to ensure all content is accessible on smaller screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Memory Management settings
        memory_group = QGroupBox("Memory Management")
        memory_layout = QVBoxLayout()

        # Force 3-channel mode
        self.force_3channel = QCheckBox("Force 3-channel mode (lower quality, less memory)")
        memory_tips = QLabel("Recommended for larger images and Apple Silicon")
        memory_tips.setStyleSheet("color: gray; font-style: italic;")

        # Enable tiling
        tiling_layout = QHBoxLayout()
        self.enable_tiling = QCheckBox("Enable tiled processing")
        self.enable_tiling.setChecked(True)
        self.tile_size = QComboBox()
        self.tile_size.addItems(["128", "256", "512", "1024"])
        self.tile_size.setCurrentText("512")
        tiling_layout.addWidget(self.enable_tiling)
        tiling_layout.addWidget(QLabel("Tile size:"))
        tiling_layout.addWidget(self.tile_size)

        # Memory efficiency mode
        self.memory_efficient = QCheckBox("Memory-efficient mode (slower)")
        self.memory_efficient.setChecked(False)

        # Add to memory layout
        memory_layout.addWidget(self.force_3channel)
        memory_layout.addWidget(memory_tips)
        memory_layout.addLayout(tiling_layout)
        memory_layout.addWidget(self.memory_efficient)
        memory_group.setLayout(memory_layout)

        # Device specific settings
        device_group = QGroupBox("Device-Specific Settings")
        device_layout = QVBoxLayout()

        # Device selection
        device_select_layout = QHBoxLayout()
        device_select_layout.addWidget(QLabel("Device:"))
        self.device = QComboBox()

        # Check for available devices
        try:
            import torch

            devices = ["CPU"]

            # Check for CUDA (NVIDIA GPUs)
            if torch.cuda.is_available():
                cuda_device = "CUDA (GPU)"
                if torch.cuda.device_count() > 0:
                    cuda_device += f" - {torch.cuda.get_device_name(0)}"
                devices.insert(0, cuda_device)

            # Check for MPS (Apple Silicon GPUs)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_device = "MPS (Apple GPU)"
                devices.insert(0, mps_device)

            self.device.addItems(devices)

            if len(devices) > 1:
                # Default to first available GPU
                self.device.setCurrentIndex(0)
            else:
                # Only CPU available
                self.device.setCurrentIndex(0)

        except ImportError:
            self.device.addItems(["CPU"])
            self.device.setEnabled(False)

        device_select_layout.addWidget(self.device)

        # CPU fallback for MPS
        self.cpu_fallback = QCheckBox("Fall back to CPU for unsupported operations")
        self.cpu_fallback.setChecked(True)

        # Only enable CPU fallback if we're on MPS
        mps_detected = False
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_detected = True
        except ImportError:
            pass

        self.cpu_fallback.setEnabled(mps_detected)

        # Add to device layout
        device_layout.addLayout(device_select_layout)
        device_layout.addWidget(self.cpu_fallback)
        device_group.setLayout(device_layout)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output directory:"))
        self.output_dir = QLineEdit("outputs")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output_dir)
        dir_layout.addWidget(self.output_dir)
        dir_layout.addWidget(self.browse_btn)

        # Intermediate saves option
        self.save_intermediate = QCheckBox("Save intermediate results")

        # Add to output layout
        output_layout.addLayout(dir_layout)
        output_layout.addWidget(self.save_intermediate)
        output_group.setLayout(output_layout)

        # Add all groups to the scroll layout
        scroll_layout.addWidget(memory_group)
        scroll_layout.addWidget(device_group)
        scroll_layout.addWidget(output_group)
        scroll_layout.addStretch()

        # Set the scroll content and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    def browse_output_dir(self):
        """Select output directory."""
        dir_dialog = QFileDialog()
        dir_path = dir_dialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir.text()
        )

        if dir_path:
            self.output_dir.setText(dir_path)

    def get_settings(self):
        """Get the current advanced settings as a configuration dictionary."""
        config = {
            "force_3channel": self.force_3channel.isChecked(),
            "enable_tiling": self.enable_tiling.isChecked(),
            "max_tile_size": int(self.tile_size.currentText()),
            "memory_efficient": self.memory_efficient.isChecked(),
            "output_dir": self.output_dir.text(),
            "intermediate_saves": self.save_intermediate.isChecked(),
        }

        # Device-specific settings
        device_text = self.device.currentText().lower()
        if "mps" in device_text or "apple" in device_text:
            config["device_specific"] = {
                "mps": {
                    "fallback_to_cpu": self.cpu_fallback.isChecked()
                }
            }

        return config

class NESRApplication(QMainWindow):
    """Main application window for NESR GUI."""

    log_signal = pyqtSignal(int, str)  # Log level, message

    def __init__(self):
        super().__init__()

        # Initialize state
        self.input_path = None
        self.output_path = None
        self.pipeline = None
        self.worker = None
        self.current_image = None
        self.degraded_image = None

        # Check if required modules are available
        try:
            # Just to validate imports
            import torch
            import numpy as np
            import cv2
            from PIL import Image
            from nesr import SuperResolutionPipeline
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Missing Dependencies",
                f"Required module not found: {e}\n\nPlease install required packages with:\npip install torch torchvision opencv-python Pillow diffusers transformers"
            )
            logger.error(f"Missing dependencies: {e}")
            # Continue anyway to show interface, but functionality will be limited

        # Set up logging
        self.setup_logging()

        # Initialize UI
        self.init_ui()

        # Check models
        try:
            self.check_models()
        except Exception as e:
            logger.error(f"Failed to check models: {e}")

    def setup_logging(self):
        """Set up logging for the application."""
        # Create and add the custom handler
        handler = LogHandler(self.log_signal)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger('nesr').addHandler(handler)

        # Connect signal to log console
        self.log_signal.connect(self.log_message)

    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("NESR - Neural Enhanced Super-Resolution")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create toolbar
        self.create_toolbar()

        # Create main splitter - with fixed initial sizes to prioritize image display
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setStretchFactor(0, 3)  # Give more space to image viewer
        main_splitter.setStretchFactor(1, 1)  # Less space to settings area

        # Create image viewer with larger initial size
        self.image_viewer = ImageViewer()
        main_splitter.addWidget(self.image_viewer)

        # Create tabbed settings area
        settings_tabs = QTabWidget()

        # Enhancement settings tab - split into sub-tabs for better organization
        enhancement_container = QWidget()
        enhancement_layout = QVBoxLayout(enhancement_container)

        # Create sub-tabs for enhancement settings
        enhancement_sub_tabs = QTabWidget()

        # Basic tab - contains essential settings
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        self.enhancement_settings = EnhancementSettings(basic_tab)
        basic_layout.addWidget(self.enhancement_settings)
        enhancement_sub_tabs.addTab(basic_tab, "Basic")

        # Advanced tab - contains memory management and other advanced options
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        self.advanced_settings = AdvancedSettings(advanced_tab)  # New class to be created
        advanced_layout.addWidget(self.advanced_settings)
        enhancement_sub_tabs.addTab(advanced_tab, "Advanced")

        # Add the sub-tabs to the enhancement container
        enhancement_layout.addWidget(enhancement_sub_tabs)

        # Add the enhancement container to the main tabs
        settings_tabs.addTab(enhancement_container, "Enhancement")

        # Degradation settings tab
        self.degradation_settings = DegradationSettings()
        self.degradation_settings.preview_btn.clicked.connect(self.preview_degradation)
        self.degradation_settings.apply_btn.clicked.connect(self.apply_degradation)
        self.degradation_settings.reset_btn.clicked.connect(self.reset_degradation)
        settings_tabs.addTab(self.degradation_settings, "Degradation")

        # Log console
        self.log_console = LogConsole()
        settings_tabs.addTab(self.log_console, "Console")

        # Add settings area to main splitter
        main_splitter.addWidget(settings_tabs)

        # Set initial sizes to prioritize image display
        main_splitter.setSizes([600, 200])

        # Add to main layout
        main_layout.addWidget(main_splitter)

        # Add status bar with progress
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(200)
        self.statusBar.addPermanentWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)

        # Apply modern style if available
        if HAS_QTMODERN:
            try:
                import qtmodern.styles
                qtmodern.styles.dark(QApplication.instance())
            except Exception as e:
                logger.warning(f"Could not apply modern styling: {e}")

        # Log application start
        logger.info("NESR application started")

    def create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open image action
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        # Save image action
        save_action = QAction("Save Result", self)
        save_action.triggered.connect(self.save_result)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Enhance action
        self.enhance_action = QAction("Enhance", self)
        self.enhance_action.triggered.connect(self.start_enhancement)
        self.enhance_action.setEnabled(False)
        toolbar.addAction(self.enhance_action)

        # Stop action
        self.stop_action = QAction("Stop", self)
        self.stop_action.triggered.connect(self.stop_enhancement)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        # Download models action
        download_action = QAction("Download Models", self)
        download_action.triggered.connect(self.download_models)
        toolbar.addAction(download_action)

    def check_models(self):
        """Check if required models are available."""
        try:
            from nesr.utils.downloader import check_models_exist

            models_status = check_models_exist()
            missing_models = [model for model, exists in models_status.items() if not exists]

            if missing_models:
                logger.warning(f"Missing models: {', '.join(missing_models)}")

                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Missing Models")
                msg_box.setText("Some required models are not available.")
                msg_box.setInformativeText(f"Missing: {', '.join(missing_models)}\n\nWould you like to download them now?")
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

                if msg_box.exec_() == QMessageBox.Yes:
                    self.download_models(missing_models)
            else:
                logger.info("All required models are available")

        except Exception as e:
            logger.error(f"Error checking models: {e}")

    def download_models(self, models=None):
        """Download required models."""
        # Create progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Downloading Models")
        progress_dialog.setModal(True)
        dialog_layout = QVBoxLayout(progress_dialog)

        # Add progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        dialog_layout.addWidget(progress_bar)

        # Add status label
        status_label = QLabel("Preparing download...")
        dialog_layout.addWidget(status_label)

        # Add cancel button
        cancel_button = QPushButton("Cancel")
        dialog_layout.addWidget(cancel_button)

        # Create worker
        self.download_worker = ModelDownloadWorker(models)

        # Connect signals
        self.download_worker.progress_signal.connect(
            lambda p, m: (progress_bar.setValue(p), status_label.setText(m))
        )

        self.download_worker.finished_signal.connect(
            lambda success: (
                progress_dialog.accept(),
                QMessageBox.information(
                    self, "Download Complete", "All models downloaded successfully."
                ) if success else None
            )
        )

        self.download_worker.error_signal.connect(
            lambda error: (
                progress_dialog.reject(),
                QMessageBox.critical(
                    self, "Download Error", f"Error downloading models: {error}"
                )
            )
        )

        cancel_button.clicked.connect(progress_dialog.reject)

        # Start worker and show dialog
        self.download_worker.start()
        progress_dialog.exec_()

        # If dialog is rejected, stop the worker
        if progress_dialog.result() == QDialog.Rejected:
            self.download_worker.terminate()

    def open_image(self):
        """Open an image file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )

        if file_path:
            try:
                self.input_path = file_path
                self.current_image = None
                self.degraded_image = None

                # Load and display image
                self.image_viewer.set_original_image(image_path=file_path)
                self.image_viewer.set_enhanced_image(image=None)

                # Enable enhancement
                self.enhance_action.setEnabled(True)

                logger.info(f"Opened image: {file_path}")
                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"Error opening image: {e}")
                QMessageBox.critical(
                    self, "Error", f"Failed to open image: {e}"
                )

    def preview_degradation(self):
        """Preview image degradation with current settings."""
        if self.input_path is None and self.current_image is None:
            QMessageBox.warning(
                self, "Warning", "Please open an image first."
            )
            return

        try:
            # Get settings
            settings = self.degradation_settings.get_settings()

            # Get original image
            if self.current_image is None:
                image = cv2.imread(self.input_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = self.current_image.copy()

            # Apply degradation
            degraded = self.degrade_image(image, settings)

            # Show preview
            self.image_viewer.set_enhanced_image(image=degraded)

            logger.info("Degradation preview applied")

        except Exception as e:
            logger.error(f"Error previewing degradation: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to preview degradation: {e}"
            )

    def apply_degradation(self):
        """Apply degradation to the image and set as the current image."""
        if self.input_path is None and self.current_image is None:
            QMessageBox.warning(
                self, "Warning", "Please open an image first."
            )
            return

        try:
            # Get settings
            settings = self.degradation_settings.get_settings()

            # Get original image
            if self.current_image is None:
                image = cv2.imread(self.input_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image.copy()
            else:
                image = self.current_image.copy()

            # Apply degradation
            self.degraded_image = self.degrade_image(image, settings)

            # Update display
            self.image_viewer.set_original_image(image=self.degraded_image)
            self.image_viewer.set_enhanced_image(image=None)

            logger.info("Degradation applied as new input image")

            # Enable enhancement
            self.enhance_action.setEnabled(True)

        except Exception as e:
            logger.error(f"Error applying degradation: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to apply degradation: {e}"
            )

    def reset_degradation(self):
        """Reset to the original image."""
        if self.input_path is None and self.current_image is None:
            return

        try:
            if self.input_path:
                self.image_viewer.set_original_image(image_path=self.input_path)
                self.degraded_image = None
            elif self.current_image is not None:
                self.image_viewer.set_original_image(image=self.current_image)
                self.degraded_image = None

            self.image_viewer.set_enhanced_image(image=None)

            logger.info("Reset to original image")

        except Exception as e:
            logger.error(f"Error resetting image: {e}")

    def degrade_image(self, image, settings):
        """Apply degradation to an image based on settings."""
        from nesr.utils.image_utils import add_noise, blur_image, downsample_image, apply_jpeg_compression

        result = image.copy()

        # Apply downscaling if enabled
        if settings["downscale"]["enabled"]:
            factor = settings["downscale"]["factor"]
            interp_method = settings["downscale"]["interpolation"]

            result = downsample_image(
                result,
                scale_factor=factor,
                interpolation=interp_method
            )

        # Apply noise if enabled
        if settings["noise"]["enabled"]:
            noise_type = settings["noise"]["type"]
            amount = settings["noise"]["amount"]

            result = add_noise(
                result,
                noise_type=noise_type,
                amount=amount
            )

        # Apply blur if enabled
        if settings["blur"]["enabled"]:
            blur_type = settings["blur"]["type"]
            radius = settings["blur"]["radius"]

            result = blur_image(
                result,
                blur_type=blur_type,
                radius=radius
            )

        # Apply JPEG compression if enabled
        if settings["jpeg"]["enabled"]:
            quality = settings["jpeg"]["quality"]

            # Convert to PIL, apply compression, and back to numpy
            pil_image = Image.fromarray(result)
            jpeg_buffer = BytesIO()
            pil_image.save(jpeg_buffer, format="JPEG", quality=quality)
            jpeg_buffer.seek(0)
            result = np.array(Image.open(jpeg_buffer))

        return result


    def update_advanced_for_preset(self, preset_name):
        """Update the advanced settings based on a preset name."""
        if preset_name == "Default":
            self.advanced_settings.force_3channel.setChecked(False)
            self.advanced_settings.enable_tiling.setChecked(True)
            self.advanced_settings.tile_size.setCurrentText("512")
            self.advanced_settings.memory_efficient.setChecked(False)
            self.advanced_settings.cpu_fallback.setChecked(True)
            self.advanced_settings.save_intermediate.setChecked(False)

        elif preset_name == "High Quality":
            self.advanced_settings.force_3channel.setChecked(False)
            self.advanced_settings.enable_tiling.setChecked(True)
            self.advanced_settings.tile_size.setCurrentText("1024")
            self.advanced_settings.memory_efficient.setChecked(False)
            self.advanced_settings.cpu_fallback.setChecked(True)
            self.advanced_settings.save_intermediate.setChecked(True)

        elif preset_name == "Fast":
            self.advanced_settings.force_3channel.setChecked(True)
            self.advanced_settings.enable_tiling.setChecked(True)
            self.advanced_settings.tile_size.setCurrentText("256")
            self.advanced_settings.memory_efficient.setChecked(False)
            self.advanced_settings.cpu_fallback.setChecked(True)
            self.advanced_settings.save_intermediate.setChecked(False)

        elif preset_name == "Apple Silicon Optimized":
            self.advanced_settings.force_3channel.setChecked(True)
            self.advanced_settings.enable_tiling.setChecked(True)
            self.advanced_settings.tile_size.setCurrentText("512")
            self.advanced_settings.memory_efficient.setChecked(True)
            self.advanced_settings.cpu_fallback.setChecked(True)
            self.advanced_settings.save_intermediate.setChecked(False)


    def start_enhancement(self):
        """Start the enhancement process."""
        if (self.input_path is None and self.degraded_image is None) or self.worker is not None:
            return

        try:
            # Import core module
            from nesr import SuperResolutionPipeline

            # Get basic enhancement settings
            basic_config = self.enhancement_settings.get_settings()
            prompt = self.enhancement_settings.get_prompt()

            # Get advanced settings
            advanced_config = self.advanced_settings.get_settings()

            # Combine configurations
            config = {**basic_config, **advanced_config}

            # Check models status and configure for fallback if needed
            try:
                from nesr.utils.downloader import check_models_exist
                models_status = check_models_exist()

                # Disable models that aren't available
                if not models_status.get('sd_upscaler', False):
                    if config['use_diffusion']:
                        logger.info("Stable Diffusion model not available, disabling diffusion upscaling")
                        config['use_diffusion'] = False

                if not models_status.get('segmentation', False):
                    if config['segment_enhancement']:
                        logger.info("Segmentation model not available, disabling segmentation-based enhancement")
                        config['segment_enhancement'] = False
            except Exception as e:
                logger.warning(f"Couldn't check model status: {e}, continuing with requested config")

            # Determine device
            device_text = self.advanced_settings.device.currentText().lower()
            if "cuda" in device_text:
                device = "cuda"
            elif "mps" in device_text or "apple" in device_text:
                device = "mps"
            else:
                device = "cpu"

            # Add to the _determine_device method in nesr.py
            if device == 'mps':
                logger.info(f"Using Apple Silicon GPU with MPS backend")
                # Test tensor operation on MPS
                try:
                    import torch
                    test_tensor = torch.ones(1, device='mps')
                    logger.info(f"MPS test successful: {test_tensor.device}")
                except Exception as e:
                    logger.warning(f"MPS test failed: {e}, falling back to CPU")
                    return 'cpu'

            logger.info(f"Using device: {device}")

            # Create temporary file for degraded image if needed
            input_image_path = self.input_path
            if self.degraded_image is not None:
                tmp_dir = os.path.join(config["output_dir"], "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(tmp_dir, "degraded_input.png")
                cv2.imwrite(tmp_path, cv2.cvtColor(self.degraded_image, cv2.COLOR_BGR2BGR))
                input_image_path = tmp_path

            # Initialize pipeline
            if self.pipeline is None:
                self.pipeline = SuperResolutionPipeline(device=device, config=config)
            else:
                # Update pipeline config
                self.pipeline.config.update(config)

            # Create and start worker
            self.worker = EnhancementWorker(
                self.pipeline, input_image_path, config, prompt
            )

            # Connect signals
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.image_signal.connect(self.update_intermediate_image)
            self.worker.finished_signal.connect(self.enhancement_finished)
            self.worker.error_signal.connect(self.enhancement_error)

            # Update UI
            self.enhance_action.setEnabled(False)
            self.stop_action.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Enhancing...")

            # Start worker
            self.worker.start()

            logger.info("Started enhancement process")

        except Exception as e:
            logger.error(f"Error starting enhancement: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to start enhancement: {e}"
            )
            self.worker = None
            self.enhance_action.setEnabled(True)
            self.stop_action.setEnabled(False)

    def stop_enhancement(self):
        """Stop the enhancement process."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping...")
            logger.info("Stopping enhancement process")

    def update_progress(self, progress, message):
        """Update progress bar and status."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def update_intermediate_image(self, image):
        """Update the display with an intermediate image."""
        self.image_viewer.set_enhanced_image(image=image)

    def enhancement_finished(self, output_path):
        """Handle enhancement completion."""
        self.output_path = output_path

        # Update UI
        self.worker = None
        self.enhance_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Enhancement complete!")

        # Display the final result
        self.image_viewer.set_enhanced_image(image_path=output_path)

        logger.info(f"Enhancement completed: {output_path}")

    def enhancement_error(self, error_message):
        """Handle enhancement error."""
        # Update UI
        self.worker = None
        self.enhance_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Enhancement failed!")

        # Show error message
        QMessageBox.critical(
            self, "Enhancement Error", f"Failed to enhance image: {error_message}"
        )

        logger.error(f"Enhancement failed: {error_message}")

    def save_result(self):
        """Save the enhanced image to a custom location."""
        if self.output_path is None:
            QMessageBox.warning(
                self, "Warning", "No enhanced image available to save."
            )
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Enhanced Image", "", "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )

        if file_path:
            try:
                # Read from output path and save to new location
                image = cv2.imread(self.output_path)
                cv2.imwrite(file_path, image)

                logger.info(f"Saved enhanced image to: {file_path}")
                self.status_label.setText(f"Saved: {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"Error saving image: {e}")
                QMessageBox.critical(
                    self, "Error", f"Failed to save image: {e}"
                )

    def log_message(self, level, message):
        """Add a message to the log console."""
        self.log_console.add_log(level, message)
