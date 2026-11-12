"""
NESR - image quality metrics against a reference image (numpy only).

    psnr(reference, test, data_range=255)   peak signal-to-noise ratio in dB (inf for identical images)
    ssim(reference, test, data_range=255)   structural similarity (Wang et al. 2004), mean over channels

SSIM uses the standard settings: an 11x11 Gaussian window with sigma 1.5, K1 = 0.01, K2 = 0.03, population
(co)variances, mirror ('reflect') borders, and the mean taken over the image minus a 5-pixel border. These are the
settings of skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False),
which tests/test_metrics.py checks against. The browser port is nesr/web/static/metrics.js.
"""

import numpy as np

SIGMA = 1.5
RADIUS = 5  # int(3.5 * SIGMA + 0.5), as scipy.ndimage.gaussian_filter with truncate=3.5
K1, K2 = 0.01, 0.03


def _pair(reference, test):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"images differ in shape: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        a, b = a[:, :, None], b[:, :, None]
    if a.ndim != 3:
        raise ValueError("expected a 2-D (grey) or 3-D (H, W, C) image")
    return a, b


def psnr(reference, test, data_range=255.0):
    """Peak signal-to-noise ratio in dB over all pixels and channels; inf when the images are identical."""
    a, b = _pair(reference, test)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def gaussian_kernel(sigma=SIGMA, radius=RADIUS):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _reflect_index(n, radius):
    """Source indices for a 'reflect' (d c b a | a b c d) padded axis of length n."""
    idx = np.arange(-radius, n + radius)
    period = 2 * n
    idx = np.mod(idx, period)
    return np.where(idx >= n, period - 1 - idx, idx)


def gaussian_filter(img, sigma=SIGMA, radius=RADIUS):
    """Separable Gaussian blur of a 2-D float array with mirrored borders (same output size)."""
    k = gaussian_kernel(sigma, radius)
    h, w = img.shape
    rows = img[:, _reflect_index(w, radius)]
    tmp = sum(k[i] * rows[:, i:i + w] for i in range(2 * radius + 1))
    cols = tmp[_reflect_index(h, radius), :]
    return sum(k[i] * cols[i:i + h, :] for i in range(2 * radius + 1))


def ssim_map(a, b, data_range=255.0):
    """Per-pixel SSIM of two 2-D float arrays (same size)."""
    c1, c2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    ux, uy = gaussian_filter(a), gaussian_filter(b)
    vx = gaussian_filter(a * a) - ux * ux
    vy = gaussian_filter(b * b) - uy * uy
    vxy = gaussian_filter(a * b) - ux * uy
    return ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))


def ssim(reference, test, data_range=255.0):
    """Mean structural similarity over the image (minus a RADIUS-pixel border) and over channels."""
    a, b = _pair(reference, test)
    if min(a.shape[0], a.shape[1]) < 2 * RADIUS + 1:
        raise ValueError(f"SSIM needs images of at least {2 * RADIUS + 1} x {2 * RADIUS + 1} pixels")
    r = RADIUS
    vals = [ssim_map(a[:, :, c], b[:, :, c], data_range)[r:-r, r:-r].mean() for c in range(a.shape[2])]
    return float(np.mean(vals))
