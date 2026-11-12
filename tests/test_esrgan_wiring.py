"""Regression tests: Real-ESRGAN x2plus is built and called the way the weights expect.

RealESRGAN_x2plus.pth is an RRDBNet with scale=2: the network pixel-unshuffles a 3-channel image (that is why its first
convolution has 12 input channels) and upsamples x4, so the output is 2x the input. The pipeline used to build
RRDBNet(num_in_ch=12) with the default scale=4 and feed it four copies (or brightness variants) of the RGB image,
which gave a 4x-sized, washed-out result instead of a 2x Real-ESRGAN upscale.

The first test needs no torch (fake basicsr/realesrgan modules record how they are called); the second runs the real
model when torch, basicsr and realesrgan are installed.
"""
import importlib.machinery
import os
import sys
import types

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nesr.nesr import SuperResolutionPipeline  # noqa: E402

WEIGHTS = os.path.join(ROOT, "models", "weights", "RealESRGAN_x2plus.pth")


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__dict__.update(attrs)
    return m


def _image(h=24, w=30):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_esrgan_built_and_called_as_x2(tmp_path, monkeypatch):
    calls = {}

    class FakeRRDBNet:
        def __init__(self, **kw):
            calls["rrdb"] = kw

    class FakeUpsampler:
        def __init__(self, **kw):
            calls["upsampler"] = kw
            self.model, self.device = kw["model"], kw["device"]

        def enhance(self, img, outscale=None):
            calls.setdefault("enhance", []).append((img.copy(), outscale))
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * outscale), int(h * outscale)), interpolation=cv2.INTER_NEAREST), "RGB"

    for name, attrs in {"basicsr": {}, "basicsr.archs": {}, "basicsr.archs.rrdbnet_arch": {"RRDBNet": FakeRRDBNet},
                        "realesrgan": {"RealESRGANer": FakeUpsampler}}.items():
        monkeypatch.setitem(sys.modules, name, _module(name, **attrs))
    monkeypatch.chdir(ROOT)
    p = SuperResolutionPipeline(device="cpu", config={"use_diffusion": False, "segment_enhancement": False,
                                                      "output_dir": str(tmp_path)})
    p._load_models()
    assert calls["rrdb"]["num_in_ch"] == 3 and calls["rrdb"]["scale"] == 2
    assert calls["upsampler"]["scale"] == 2
    img = _image()
    for _ in range(2):
        out = p._apply_esrgan(img)
        assert out.shape == (48, 60, 3)
        sent, outscale = calls["enhance"][-1]
        assert outscale == 2
        np.testing.assert_array_equal(sent, img[:, :, ::-1])  # RealESRGANer expects BGR
        np.testing.assert_array_equal(out, cv2.resize(img, (60, 48), interpolation=cv2.INTER_NEAREST))


def _have_real_model():
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("torch", "basicsr", "realesrgan")) and os.path.exists(WEIGHTS)


@pytest.mark.skipif(not _have_real_model(), reason="needs torch, basicsr, realesrgan and the committed weights")
def test_real_esrgan_matches_upstream_usage(tmp_path, monkeypatch):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    monkeypatch.chdir(ROOT)
    p = SuperResolutionPipeline(device="cpu", config={"use_diffusion": False, "segment_enhancement": False,
                                                      "output_dir": str(tmp_path)})
    p._load_models()
    img = _image(32, 40)
    out = p._apply_esrgan(img)
    ref_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    ref = RealESRGANer(scale=2, model_path=WEIGHTS, model=ref_net, tile=0, pre_pad=0, half=False, device="cpu")
    expected, _ = ref.enhance(img[:, :, ::-1], outscale=2)
    assert out.shape == (64, 80, 3)
    assert np.abs(out.astype(int) - expected[:, :, ::-1].astype(int)).max() <= 1


def test_no_dead_three_channel_option():
    """After the x2plus fix both ESRGAN 'modes' ran the same code, so the GUI's "Force 3-channel mode (lower quality,
    less memory)" checkbox did nothing; it was removed with the config key and the 12-channel method."""
    gui = open(os.path.join(ROOT, "nesr", "gui", "app.py"), encoding="utf-8").read()
    assert "force_3channel" not in gui and "3-channel" not in gui
    assert not hasattr(SuperResolutionPipeline, "_apply_esrgan_12channel")


def test_old_configs_with_force_3channel_still_load(tmp_path):
    p = SuperResolutionPipeline(device="cpu", config={"force_3channel": True, "output_dir": str(tmp_path)})
    assert p.config["force_3channel"] is True  # accepted and ignored
