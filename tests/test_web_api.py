"""Project-specific web API tests: methods, examples, the upscale endpoint and the JS fixture being current."""
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nesr.metrics import psnr, ssim  # noqa: E402
from nesr.web import examples as ex  # noqa: E402
from nesr.web.app import app  # noqa: E402

client = TestClient(app)


def _decode(url):
    raw = base64.b64decode(url.split(",", 1)[1])
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def test_info_lists_methods():
    info = client.get("/api/info").json()
    ids = {m["id"]: m for m in info["methods"]}
    assert {"nearest", "bicubic", "lanczos", "nesr_basic", "realesrgan", "nesr_esrgan", "nesr_full"} <= set(ids)
    assert ids["bicubic"]["available"] and ids["bicubic"]["browser"]
    assert info["n_examples"] == len(ex.PATTERNS) * len(ex.FACTORS)


def test_examples_complete_and_scores_reproduce():
    m = client.get("/api/examples").json()
    assert len(m["examples"]) == len(ex.PATTERNS) * len(ex.FACTORS)
    assert m["methods"] == ex.method_list()
    for e in m["examples"]:
        hr = ex.read_png(e["hr"])
        np.testing.assert_allclose(hr, ex.hr_image(e["pattern"]), atol=1)  # patterns regenerate
        lr = ex.read_png(e["lr"])
        assert np.abs(lr.astype(int) - ex.make_lr(hr, e["factor"]).astype(int)).max() <= 1
        assert set(e["outputs"]) == set(ex.EXAMPLE_METHODS)
        for method, o in e["outputs"].items():
            out = ex.read_png(o["file"])
            assert out.shape == hr.shape
            assert psnr(hr, out) == pytest.approx(o["psnr"], abs=1e-5)
            assert ssim(hr, out) == pytest.approx(o["ssim"], abs=1e-5)
            if method in ("nearest", "bicubic", "lanczos"):  # classical outputs are reproducible with OpenCV
                flag = {"nearest": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[method]
                again = cv2.resize(lr, hr.shape[1::-1], interpolation=flag)
                assert np.abs(again.astype(int) - out.astype(int)).max() <= 1


def test_nesr_basic_example_reproduces():
    """The pipeline without neural models (CPU, OpenCV only) gives the committed output again."""
    from nesr.web.sr_methods import run
    hr = ex.hr_image("siemens_star")
    out, info = run("nesr_basic", ex.make_lr(hr, 2), 2, device="cpu")
    assert info["models_loaded"] == []
    assert psnr(ex.read_png("siemens_star_x2_nesr_basic.png"), out) > 40


@pytest.mark.parametrize("method", ["nearest", "bicubic", "lanczos", "nesr_basic"])
def test_upscale_pattern_benchmark(method):
    r = client.post("/api/upscale", data={"pattern": "color_shapes", "size": 64, "method": method, "scale": 2,
                                          "mode": "benchmark"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input_size"] == [32, 32] and j["output_size"] == [64, 64]
    out, ref = _decode(j["output"]), _decode(j["reference"])
    assert j["metrics"]["psnr"] == pytest.approx(psnr(ref, out))
    assert 0 < j["metrics"]["ssim"] <= 1


def test_upscale_uploaded_file():
    img = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/api/upscale", files={"file": ("x.png", buf.tobytes(), "image/png")},
                    data={"method": "lanczos", "scale": 4, "mode": "upscale"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["output_size"] == [120, 80] and "metrics" not in j and j["source"] == "x.png"
    expect = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (120, 80), interpolation=cv2.INTER_LANCZOS4)
    np.testing.assert_array_equal(_decode(j["output"]), expect)


@pytest.mark.parametrize("data,msg", [
    ({"pattern": "zone_plate", "method": "magic"}, "unknown method"),
    ({"pattern": "zone_plate", "method": "bicubic", "scale": 3}, "scale"),
    ({"method": "bicubic"}, "upload an image"),
    ({"pattern": "zone_plate", "method": "bicubic", "mode": "x"}, "mode"),
])
def test_upscale_errors(data, msg):
    r = client.post("/api/upscale", data=data)
    assert r.status_code == 400 and msg in r.text


def test_upscale_bad_file():
    r = client.post("/api/upscale", files={"file": ("x.png", b"not an image", "image/png")}, data={"method": "bicubic"})
    assert r.status_code == 400 and "could not read" in r.text


def test_js_fixture_is_current():
    import make_js_fixtures
    old = json.loads((ROOT / "tests" / "fixtures" / "js_cases.json").read_text())
    new = make_js_fixtures.build()
    assert old["inputs"] == new["inputs"]
    for a, b in zip(old["resize"], new["resize"], strict=True):  # OpenCV builds may differ by 1 grey level
        assert a["out_w"] == b["out_w"] and np.abs(np.array(a["out"]) - np.array(b["out"])).max() <= 1
    for a, b in zip(old["resize_float"]["cases"], new["resize_float"]["cases"], strict=True):
        np.testing.assert_allclose(a["out"], b["out"], atol=1e-5)
    for a, b in zip(old["metrics"], new["metrics"], strict=True):
        assert a["ref"] == b["ref"]
        if np.abs(np.array(a["test"]) - np.array(b["test"])).max() == 0:
            assert a["ssim"] == pytest.approx(b["ssim"], rel=1e-9)
