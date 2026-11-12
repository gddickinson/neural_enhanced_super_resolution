"""Upscale / benchmark requests for the local web app: decode the image, run a method, score it.

Modes
  upscale    the image is the low-resolution input; the result is `scale` times larger (no reference, no metrics)
  benchmark  the image is the reference: it is cropped to a multiple of `scale`, reduced with the GUI's degradation
             tool (downsample_image, bicubic), restored with the method, and PSNR / SSIM are computed against it
"""
import base64

import cv2
import numpy as np

from nesr.metrics import psnr, ssim
from nesr.test_patterns import PATTERNS, make_pattern
from nesr.utils.image_utils import downsample_image

from .sr_methods import METHODS, MethodError, run

MAX_UPLOAD = 25 * 1024 * 1024
MAX_SIDE = {"classical": 2048, "model": 768, "pipeline": 768}  # longest input side (CPU-friendly for models)


def decode(data: bytes) -> np.ndarray:
    """Image file bytes -> RGB uint8 (alpha dropped, grey expanded)."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise MethodError("could not read the image (use PNG, JPEG, BMP, TIFF or WebP)")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def png_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise MethodError("could not encode the result")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def source_image(data: bytes | None, pattern: str, size: int) -> tuple[np.ndarray, str]:
    if data:
        return decode(data), "upload"
    if pattern in PATTERNS:
        return make_pattern(pattern, size), PATTERNS[pattern][0]
    raise MethodError("upload an image or choose a test pattern")


def process(img: np.ndarray, method: str, scale: int, mode: str) -> dict:
    if method not in METHODS:
        raise MethodError(f"unknown method {method!r}")
    if mode not in ("upscale", "benchmark"):
        raise MethodError("mode must be 'upscale' or 'benchmark'")
    ref = None
    if mode == "benchmark":
        h, w = img.shape[:2]
        ref = img[: h - h % scale, : w - w % scale]
        lr = downsample_image(ref, scale_factor=1.0 / scale, interpolation="bicubic")
        limit = MAX_SIDE[METHODS[method]["kind"]] * scale
    else:
        lr = img
        limit = MAX_SIDE[METHODS[method]["kind"]]
    if max(img.shape[:2]) > limit:
        raise MethodError(f"image too large for {METHODS[method]['label']} in the web app (longest side {limit} px "
                          f"in {mode} mode); crop it or use the desktop app / command line")
    out, info = run(method, np.ascontiguousarray(lr), scale)
    result = {"mode": mode, "info": info, "input_size": [lr.shape[1], lr.shape[0]],
              "output_size": [out.shape[1], out.shape[0]], "input": png_data_url(lr), "output": png_data_url(out)}
    if ref is not None:
        result["reference"] = png_data_url(ref)
        if out.shape == ref.shape:
            result["metrics"] = {"psnr": psnr(ref, out), "ssim": ssim(ref, out)}
        else:
            result["metrics_note"] = f"output {out.shape[1]}x{out.shape[0]} differs from the reference size"
    return result
