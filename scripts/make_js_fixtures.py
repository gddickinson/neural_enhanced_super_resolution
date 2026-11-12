"""Write tests/fixtures/js_cases.json: inputs and Python outputs for the browser ports.

- resize: cv2.resize (OpenCV, as used by the pipeline) on uint8 and float32 images (nearest, bicubic, Lanczos-4),
  up- and downscaling, checked by nesr/web/static/resample.test.mjs
- metrics: nesr.metrics.psnr / ssim on image pairs, checked by nesr/web/static/metrics.test.mjs

tests/test_web.py checks the fixture still matches what Python computes now. Run: python scripts/make_js_fixtures.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nesr.metrics import psnr, ssim  # noqa: E402
from nesr.test_patterns import make_pattern  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "js_cases.json"
INTERP = {"nearest": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
SIZES = [(2.0, 2.0), (3.0, 3.0), (1.5, 1.5), (0.5, 0.5), (2.3, 1.7)]


def inputs():
    """Named uint8 test images: random noise (worst case for rounding), a grey ramp and pattern crops."""
    rng = np.random.default_rng(42)
    ramp = np.add.outer(np.arange(7) * 30, np.arange(10) * 20).clip(0, 255).astype(np.uint8)
    return {
        "noise_13x9": rng.integers(0, 256, (9, 13, 3), dtype=np.uint8),
        "ramp_grey_10x7": ramp[:, :, None],
        "zone_24": make_pattern("zone_plate", 24),
        "star_20": np.ascontiguousarray(make_pattern("siemens_star", 64)[22:42, 22:42]),
    }


def _as3(a):
    return a if a.ndim == 3 else a[:, :, None]


def resize_cases(ins):
    cases = []
    for name, img in ins.items():
        h, w = img.shape[:2]
        for sx, sy in SIZES:
            ow, oh = max(1, round(w * sx)), max(1, round(h * sy))
            for method, flag in INTERP.items():
                out = _as3(cv2.resize(img, (ow, oh), interpolation=flag))
                cases.append({"input": name, "method": method, "out_w": ow, "out_h": oh,
                              "out": out.reshape(-1).tolist()})
    return cases


def float_cases():
    rng = np.random.default_rng(3)
    img = rng.random((8, 11, 3)).astype(np.float32)
    cases = []
    for ow, oh in ((22, 16), (25, 19), (6, 4)):
        for method in ("bicubic", "lanczos"):
            out = cv2.resize(img, (ow, oh), interpolation=INTERP[method])
            cases.append({"method": method, "out_w": ow, "out_h": oh,
                          "out": out.reshape(-1).astype(float).tolist()})
    return {"w": 11, "h": 8, "cn": 3, "data": img.reshape(-1).astype(float).tolist(), "cases": cases}


def metric_pairs():
    """Reference/test pairs: a pattern against its bicubic x2 round trip, noise against noisy noise, a grey pair."""
    rng = np.random.default_rng(5)
    hr = make_pattern("color_shapes", 48)
    lr = cv2.resize(hr, (24, 24), interpolation=cv2.INTER_CUBIC)
    a = rng.integers(0, 256, (21, 17, 3), dtype=np.uint8)
    g = np.ascontiguousarray(make_pattern("zone_plate", 32)[:, :, :1])
    pairs = {
        "shapes_bicubic": (hr, cv2.resize(lr, (48, 48), interpolation=cv2.INTER_CUBIC)),
        "noise_noisy": (a, np.clip(a + rng.normal(0, 25, a.shape), 0, 255).astype(np.uint8)),
        "zone_grey_blur": (g, _as3(cv2.GaussianBlur(g, (5, 5), 0))),
        "identical": (a, a.copy()),
    }
    out = []
    for name, (ref, test) in pairs.items():
        p = psnr(ref, test)
        out.append({"name": name, "w": ref.shape[1], "h": ref.shape[0], "cn": ref.shape[2],
                    "ref": ref.reshape(-1).tolist(), "test": test.reshape(-1).tolist(),
                    "psnr": None if np.isinf(p) else p, "ssim": ssim(ref, test)})
    return out


def build():
    ins = inputs()
    return {
        "opencv": cv2.__version__,
        "inputs": {k: {"w": v.shape[1], "h": v.shape[0], "cn": v.shape[2], "data": v.reshape(-1).tolist()}
                   for k, v in ins.items()},
        "resize": resize_cases(ins),
        "resize_float": float_cases(),
        "metrics": metric_pairs(),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
