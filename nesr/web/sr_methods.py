"""Super-resolution methods offered by the web app, all run through the repo's own code.

- nearest / bicubic / lanczos: cv2.resize, the call the pipeline itself uses for its bicubic fallback and ensemble
  alignment (also ported to the browser: static/resample.js).
- nesr_basic: SuperResolutionPipeline.enhance_image with every neural model switched off, i.e. the pipeline's
  pre-processing (non-local-means denoising + CLAHE), its bicubic fallback and its adaptive sharpening.
- realesrgan: the pipeline's Real-ESRGAN x2plus step (_apply_esrgan) on its own, applied log2(scale) times.
- nesr_esrgan: enhance_image with Real-ESRGAN (no diffusion, no segmentation).
- nesr_full: enhance_image with the repo defaults: Real-ESRGAN + Stable Diffusion x4 upscaler + segmentation
  (downloads several GB from Hugging Face on first use).

run(method, image, scale) returns (rgb_uint8, info). Scales are powers of two because the pipeline works in x2 steps.
"""
import importlib.util
import os
import tempfile
import time

import cv2
import numpy as np

from nesr.nesr import SuperResolutionPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "models", "weights", "RealESRGAN_x2plus.pth")
SCALES = (2, 4)
INTERP = {"nearest": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}

METHODS = {
    "nearest": dict(short="Nearest", label="Nearest neighbour", kind="classical", browser=True,
                    description="Each low-resolution pixel becomes a block: shows exactly what the input contains."),
    "bicubic": dict(short="Bicubic", label="Bicubic", kind="classical", browser=True,
                    description="cv2.resize INTER_CUBIC (a = -0.75), the pipeline's own fallback when no model loads."),
    "lanczos": dict(short="Lanczos", label="Lanczos-4", kind="classical", browser=True,
                    description="cv2.resize INTER_LANCZOS4 (8x8 windowed sinc), used by the pipeline to align ensemble members."),
    "nesr_basic": dict(short="NESR basic", label="NESR pipeline, no neural models", kind="pipeline", browser=False,
                       description="enhance_image with all models off: denoise + CLAHE, bicubic x2 per iteration, "
                                   "adaptive sharpening.", config=dict(use_esrgan=False, use_diffusion=False,
                                                                       segment_enhancement=False)),
    "realesrgan": dict(short="Real-ESRGAN", label="Real-ESRGAN x2plus", kind="model", browser=False,
                       description="The pipeline's Real-ESRGAN step alone (RealESRGAN_x2plus.pth, committed in "
                                   "models/weights), applied once per x2.",
                       config=dict(use_esrgan=True, use_diffusion=False, segment_enhancement=False)),
    "nesr_esrgan": dict(short="NESR + ESRGAN", label="NESR pipeline with Real-ESRGAN", kind="pipeline", browser=False,
                        description="enhance_image with Real-ESRGAN: denoise + CLAHE, Real-ESRGAN x2, adaptive "
                                    "sharpening, per iteration.",
                        config=dict(use_esrgan=True, use_diffusion=False, segment_enhancement=False)),
    "nesr_full": dict(short="NESR full", label="NESR pipeline, all models", kind="pipeline", browser=False,
                      description="The repo defaults: Real-ESRGAN + Stable Diffusion x4 upscaler ensemble and "
                                  "segmentation-guided sharpening. Downloads several GB on first use; slow without a GPU.",
                      config=dict(use_esrgan=True, use_diffusion=True, segment_enhancement=True)),
}


class MethodError(ValueError):
    """Bad request or a method that cannot run here (message is shown to the user)."""


def _have(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def availability():
    """{method: (available, reason)} on this machine."""
    esrgan = _have("torch", "basicsr", "realesrgan")
    weights = os.path.exists(WEIGHTS)
    out = {}
    for m, spec in METHODS.items():
        if m in ("realesrgan", "nesr_esrgan"):
            ok = esrgan and weights
            why = "" if ok else ("needs torch, basicsr and realesrgan (pip install -r requirements.txt)" if not esrgan
                                 else "needs models/weights/RealESRGAN_x2plus.pth")
        elif m == "nesr_full":
            ok = esrgan and weights and _have("diffusers", "transformers")
            why = "" if ok else "needs torch, basicsr, realesrgan, diffusers and transformers (requirements.txt)"
        else:
            ok, why = True, ""
        out[m] = (ok, why)
    return out


def describe():
    """Method list for the API: id, label, kind, description, browser, available, reason."""
    av = availability()
    return [dict(id=m, label=s["label"], short=s["short"], kind=s["kind"], description=s["description"], browser=s["browser"],
                 available=av[m][0], reason=av[m][1]) for m, s in METHODS.items()]


_PIPELINES = {}


def pipeline(method, device="auto"):
    """A cached SuperResolutionPipeline for a method (models load once per process)."""
    key = (method, device)
    if key not in _PIPELINES:
        cfg = dict(METHODS[method]["config"], upscale_factor=2, output_dir=tempfile.gettempdir())
        _PIPELINES[key] = SuperResolutionPipeline(device=device, config=cfg)
    return _PIPELINES[key]


def _check(image, scale):
    if scale not in SCALES:
        raise MethodError(f"scale must be one of {SCALES}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise MethodError("expected an RGB uint8 image")
    if min(image.shape[:2]) < 16:
        raise MethodError("image too small: the pipeline needs at least 16 x 16 pixels")


def run(method, image, scale, device="auto"):
    """Upscale an RGB uint8 image by `scale` with `method`. Returns (image, info)."""
    if method not in METHODS:
        raise MethodError(f"unknown method {method!r}")
    _check(image, scale)
    ok, why = availability()[method]
    if not ok:
        raise MethodError(f"{METHODS[method]['label']} is not available here: {why}")
    t0 = time.time()
    h, w = image.shape[:2]
    info = {"method": method, "label": METHODS[method]["label"], "scale": scale}
    if method in INTERP:
        out = cv2.resize(image, (w * scale, h * scale), interpolation=INTERP[method])
    else:
        iterations = int(round(np.log2(scale)))
        p = pipeline(method, device)
        p._load_models()
        info["device"] = p.device
        if method == "realesrgan":
            if "esrgan" not in p.models:
                raise MethodError("Real-ESRGAN failed to load (see the server log)")
            out = image
            for _ in range(iterations):
                out = p._apply_esrgan(out)
        else:
            out = _enhance(p, image, iterations)
        info["models_loaded"] = sorted(k for k in p.models if not k.endswith("_extractor"))
        if p.model_errors:
            info["model_errors"] = dict(p.model_errors)
    info["seconds"] = round(time.time() - t0, 2)
    return out, info


def _enhance(p, image, iterations):
    """SuperResolutionPipeline.enhance_image on an array (it reads and writes files), in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.png")
        cv2.imwrite(src, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        p.config["output_dir"] = tmp
        p.config["iterations"] = iterations
        path = p.enhance_image(src)
        out = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return out
