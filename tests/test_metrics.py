"""nesr.metrics against scikit-image, and the test patterns."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nesr.metrics import psnr, ssim  # noqa: E402
from nesr.test_patterns import PATTERNS, make_pattern  # noqa: E402


def _pair(shape, seed=0, noise=20):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, shape).astype(np.uint8)
    return a, np.clip(a + rng.normal(0, noise, shape), 0, 255).astype(np.uint8)


def test_psnr_simple_values():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    assert psnr(a, b) == float("inf")
    b[:] = 255
    assert psnr(a, b) == pytest.approx(0.0)
    b[:] = 1
    assert psnr(a, b) == pytest.approx(10 * np.log10(255 ** 2))


def test_ssim_identical_is_one():
    a, _ = _pair((20, 20, 3))
    assert ssim(a, a) == pytest.approx(1.0)


@pytest.mark.parametrize("shape", [(40, 37, 3), (64, 64), (11, 15, 3)])
def test_against_skimage(shape):
    skm = pytest.importorskip("skimage.metrics")
    a, b = _pair(shape)
    kw = dict(data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    if len(shape) == 3:
        kw["channel_axis"] = 2
    assert ssim(a, b) == pytest.approx(skm.structural_similarity(a, b, **kw), rel=1e-9)
    assert psnr(a, b) == pytest.approx(skm.peak_signal_noise_ratio(a, b, data_range=255), rel=1e-12)


def test_errors():
    a, b = _pair((10, 10, 3))
    with pytest.raises(ValueError):
        ssim(a, b)  # smaller than the 11 x 11 window
    with pytest.raises(ValueError):
        psnr(a, b[:5])


@pytest.mark.parametrize("name", list(PATTERNS))
def test_patterns(name):
    img = make_pattern(name, 128)
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert img.min() < 60 and img.max() > 180  # real contrast
    np.testing.assert_array_equal(img, make_pattern(name, 128))
    with pytest.raises(ValueError):
        make_pattern("nope")
