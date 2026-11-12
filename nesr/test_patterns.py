"""
NESR - synthetic test patterns for super-resolution demos and benchmarks.

Every pattern is generated from a formula (no external images, no licence questions), rendered with 4x
supersampling and box-averaged down, so edges are anti-aliased like a real photograph of a sharp target.
Returns RGB uint8 arrays of shape (size, size, 3).

    PATTERNS: name -> (title, description, function(size) -> ndarray)
    make_pattern(name, size=256)
"""

import numpy as np

SUPERSAMPLE = 4


def _grid(size):
    """Pixel-centre coordinates of a supersampled grid, in units of output pixels, centred on the image."""
    n = size * SUPERSAMPLE
    c = (np.arange(n) + 0.5) / SUPERSAMPLE - size / 2.0
    return np.meshgrid(c, c)


def _box_down(img):
    """Average SUPERSAMPLE x SUPERSAMPLE blocks of a float image (H*s, W*s[, C]) to (H, W[, C])."""
    s = SUPERSAMPLE
    h, w = img.shape[0] // s, img.shape[1] // s
    return img.reshape(h, s, w, s, *img.shape[2:]).mean(axis=(1, 3))


def _to_rgb8(gray_or_rgb):
    a = np.asarray(gray_or_rgb, dtype=np.float64)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def zone_plate(size=256):
    """Circular zone plate: cos(pi r^2 / size). Spatial frequency rises linearly from the centre to
    the Nyquist limit at the edge, so every frequency is present and aliasing shows up as moire rings."""
    x, y = _grid(size)
    r2 = x * x + y * y
    return _to_rgb8(_box_down(0.5 + 0.5 * np.cos(np.pi * r2 / size)))


def siemens_star(size=256, spokes=36):
    """Siemens star: black and white wedges whose period shrinks towards the centre."""
    x, y = _grid(size)
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    star = (np.sin(spokes / 2.0 * theta) > 0).astype(np.float64)
    star[r > size * 0.46] = 0.85
    return _to_rgb8(_box_down(star))


def bar_chart(size=256):
    """Resolution bars: groups of three dark bars with periods 2 to 12 px, horizontal in the top half
    and vertical in the bottom half, on a light background."""
    n = size * SUPERSAMPLE
    img = np.full((n, n), 0.92)
    periods = [12, 10, 8, 6, 5, 4, 3, 2]
    half = n // 2
    for row, vertical in ((0, False), (half, True)):
        x0 = 4 * SUPERSAMPLE
        for p in periods:
            pp = p * SUPERSAMPLE
            group_w = 6 * pp
            if x0 + group_w > n:
                break
            for b in range(3):
                a0 = x0 + 2 * b * pp
                if vertical:
                    img[row + 8 * SUPERSAMPLE: n - 8 * SUPERSAMPLE, a0:a0 + pp] = 0.08
                else:
                    img[8 * SUPERSAMPLE: half - 8 * SUPERSAMPLE, a0:a0 + pp] = 0.08
            x0 += group_w + 2 * pp
    return _to_rgb8(_box_down(img))


def color_shapes(size=256):
    """Coloured discs, a rotated square and thin lines on a smooth two-colour gradient: smooth areas,
    sharp colour edges and fine lines in one image."""
    x, y = _grid(size)
    u = (x / size) + 0.5
    v = (y / size) + 0.5
    img = np.stack([0.15 + 0.5 * u, 0.2 + 0.3 * v, 0.55 - 0.35 * u], axis=-1)
    discs = [(-0.22, -0.2, 0.17, (0.95, 0.25, 0.3)), (0.2, -0.18, 0.12, (0.2, 0.85, 0.4)),
             (0.05, 0.22, 0.2, (0.98, 0.85, 0.2))]
    for cx, cy, rad, col in discs:
        m = (x / size - cx) ** 2 + (y / size - cy) ** 2 < rad ** 2
        img[m] = col
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    xr = (x / size + 0.25) * c + (y / size - 0.25) * s
    yr = -(x / size + 0.25) * s + (y / size - 0.25) * c
    img[(np.abs(xr) < 0.1) & (np.abs(yr) < 0.1)] = (0.3, 0.45, 1.0)
    for k, w in enumerate((0.003, 0.005, 0.008)):
        m = np.abs((x / size) - (y / size) * 0.3 - 0.3 - 0.04 * k) < w
        img[m] = (1.0, 1.0, 1.0)
    return _to_rgb8(_box_down(img))


PATTERNS = {
    "zone_plate": ("Zone plate", "Circular zone plate: every spatial frequency up to Nyquist; aliasing shows as moire rings.",
                   zone_plate),
    "siemens_star": ("Siemens star", "36 black and white wedges that get finer towards the centre.", siemens_star),
    "bar_chart": ("Resolution bars", "Groups of three bars with periods from 12 px down to 2 px, both orientations.",
                  bar_chart),
    "color_shapes": ("Colour shapes", "Discs, a rotated square and thin lines on a smooth gradient.", color_shapes),
}


def make_pattern(name, size=256):
    """Render the named pattern (see PATTERNS) as an RGB uint8 array of shape (size, size, 3)."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern {name!r}; choose from {sorted(PATTERNS)}")
    if size < 16:
        raise ValueError("size must be at least 16")
    return PATTERNS[name][2](size)
