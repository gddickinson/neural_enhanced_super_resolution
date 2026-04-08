"""
Unit tests for nesr.utils.image_utils module.

Tests cover noise addition, blurring, downsampling, JPEG compression,
comparison image creation, and text overlay.
"""

import unittest
import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nesr.utils.image_utils import (
    add_noise,
    blur_image,
    downsample_image,
    apply_jpeg_compression,
    create_comparison_image,
    add_text_to_image,
)


def _make_test_image(height=100, width=100, channels=3):
    """Create a simple test image."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


class TestAddNoise(unittest.TestCase):
    """Tests for add_noise function."""

    def test_gaussian_noise_changes_image(self):
        img = _make_test_image()
        noisy = add_noise(img, noise_type='gaussian', amount=0.3)
        self.assertEqual(img.shape, noisy.shape)
        self.assertFalse(np.array_equal(img, noisy))

    def test_zero_amount_returns_copy(self):
        img = _make_test_image()
        result = add_noise(img, amount=0.0)
        np.testing.assert_array_equal(img, result)

    def test_salt_pepper_noise(self):
        img = _make_test_image()
        noisy = add_noise(img, noise_type='salt_pepper', amount=0.05)
        self.assertEqual(img.shape, noisy.shape)

    def test_speckle_noise(self):
        img = _make_test_image()
        noisy = add_noise(img, noise_type='speckle', amount=0.2)
        self.assertEqual(img.shape, noisy.shape)

    def test_output_range(self):
        """Output should be clipped to [0, 255]."""
        img = _make_test_image()
        noisy = add_noise(img, noise_type='gaussian', amount=1.0)
        self.assertTrue(noisy.min() >= 0)
        self.assertTrue(noisy.max() <= 255)
        self.assertEqual(noisy.dtype, np.uint8)


class TestBlurImage(unittest.TestCase):
    """Tests for blur_image function."""

    def test_gaussian_blur(self):
        img = _make_test_image()
        blurred = blur_image(img, blur_type='gaussian', radius=3)
        self.assertEqual(img.shape, blurred.shape)

    def test_box_blur(self):
        img = _make_test_image()
        blurred = blur_image(img, blur_type='box', radius=2)
        self.assertEqual(img.shape, blurred.shape)

    def test_motion_blur(self):
        img = _make_test_image()
        blurred = blur_image(img, blur_type='motion', radius=3)
        self.assertEqual(img.shape, blurred.shape)

    def test_zero_radius_returns_copy(self):
        img = _make_test_image()
        result = blur_image(img, radius=0)
        np.testing.assert_array_equal(img, result)


class TestDownsampleImage(unittest.TestCase):
    """Tests for downsample_image function."""

    def test_half_size(self):
        img = _make_test_image(100, 100)
        result = downsample_image(img, scale_factor=0.5)
        self.assertEqual(result.shape[0], 50)
        self.assertEqual(result.shape[1], 50)

    def test_scale_one_returns_copy(self):
        img = _make_test_image()
        result = downsample_image(img, scale_factor=1.0)
        np.testing.assert_array_equal(img, result)

    def test_various_interpolations(self):
        img = _make_test_image(100, 100)
        for interp in ['nearest', 'bilinear', 'bicubic', 'lanczos']:
            result = downsample_image(img, scale_factor=0.5, interpolation=interp)
            self.assertEqual(result.shape[0], 50, f"Failed for {interp}")


class TestJpegCompression(unittest.TestCase):
    """Tests for apply_jpeg_compression function."""

    def test_compression_returns_same_shape(self):
        img = _make_test_image()
        compressed = apply_jpeg_compression(img, quality=50)
        self.assertEqual(img.shape, compressed.shape)

    def test_low_quality_changes_image(self):
        img = _make_test_image()
        compressed = apply_jpeg_compression(img, quality=1)
        self.assertFalse(np.array_equal(img, compressed))


class TestCreateComparisonImage(unittest.TestCase):
    """Tests for create_comparison_image function."""

    def test_horizontal_comparison(self):
        img1 = _make_test_image(100, 80)
        img2 = _make_test_image(100, 80)
        result = create_comparison_image(img1, img2, orientation='horizontal')
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 160)

    def test_vertical_comparison(self):
        img1 = _make_test_image(80, 100)
        img2 = _make_test_image(80, 100)
        result = create_comparison_image(img1, img2, orientation='vertical')
        self.assertEqual(result.shape[0], 160)
        self.assertEqual(result.shape[1], 100)


class TestAddTextToImage(unittest.TestCase):
    """Tests for add_text_to_image function."""

    def test_add_text_top(self):
        img = _make_test_image(200, 200)
        result = add_text_to_image(img, "Hello", position='top')
        self.assertEqual(img.shape, result.shape)

    def test_add_text_bottom(self):
        img = _make_test_image(200, 200)
        result = add_text_to_image(img, "Hello", position='bottom')
        self.assertEqual(img.shape, result.shape)

    def test_original_not_modified(self):
        img = _make_test_image(200, 200)
        original = img.copy()
        add_text_to_image(img, "Test")
        np.testing.assert_array_equal(img, original)


if __name__ == "__main__":
    unittest.main()
