"""Generate the precomputed examples for the web app (nesr/web/static/examples/: PNGs + examples.json).

Every pattern x factor x method in nesr/web/examples.py is run through nesr/web/sr_methods.py (the repo's own code)
on the CPU. Real-ESRGAN methods need torch, basicsr and realesrgan; without them those outputs are kept from the
previous run if present (use --require-esrgan to fail instead).

Run: python scripts/make_examples.py [--require-esrgan]
"""
import argparse
import json
import logging
import platform
from importlib import metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nesr.web import examples as ex  # noqa: E402
from nesr.web.sr_methods import availability, run  # noqa: E402


def versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
         "machine": f"{platform.system()} {platform.machine()}", "device": "cpu"}
    for pkg in ("torch", "basicsr", "realesrgan"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-esrgan", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    av = availability()
    if a.require_esrgan and not av["realesrgan"][0]:
        sys.exit("Real-ESRGAN not available: " + av["realesrgan"][1])
    old = {e["id"]: e for e in ex.load_manifest()["examples"]}
    entries = []
    for pattern in ex.PATTERNS:
        hr = ex.hr_image(pattern)
        ex.write_png(ex.file_name(pattern), hr)
        for factor in ex.FACTORS:
            lr = ex.make_lr(hr, factor)
            ex.write_png(ex.file_name(pattern, factor, "lr"), lr)
            results = {}
            for method in ex.EXAMPLE_METHODS:
                if not av[method][0]:
                    prev = old.get(ex.example_id(pattern, factor), {}).get("outputs", {}).get(method)
                    if prev:
                        results[method] = {k: v for k, v in prev.items() if k != "file"}
                    print(f"  skip {pattern} x{factor} {method}: {av[method][1]}")
                    continue
                out, info = run(method, lr, factor, device="cpu")
                out = ex.check_shapes(hr, lr, out, factor)
                ex.write_png(ex.file_name(pattern, factor, method), out)
                res = ex.scores(hr, out)
                res["seconds"] = info["seconds"]
                if "models_loaded" in info:
                    res["models_loaded"] = info["models_loaded"]
                results[method] = res
                print(f"{pattern:13s} x{factor} {method:12s} PSNR {res['psnr']:6.2f} dB  SSIM {res['ssim']:.4f}")
            entries.append(ex.entry(pattern, factor, results))
    manifest = {"size": ex.SIZE, "factors": list(ex.FACTORS), "patterns": ex.pattern_list(),
                "methods": ex.method_list(), "versions": versions(),
                "degradation": "downsample_image(hr, 1/factor, 'bicubic') from nesr/utils/image_utils.py",
                "examples": entries}
    Path(ex.MANIFEST).write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in Path(ex.EXAMPLES_DIR).glob("*")) / 1e6
    print(f"wrote {len(entries)} examples, {size:.1f} MB in {ex.EXAMPLES_DIR}")


if __name__ == "__main__":
    main()
