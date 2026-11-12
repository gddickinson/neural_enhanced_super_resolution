"""FastAPI web app for Neural Enhanced Super-Resolution: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8168, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

Upscaling runs the repo's own code (sr_methods.py: cv2.resize, SuperResolutionPipeline, Real-ESRGAN); the examples
shown on the static site were precomputed by scripts/make_examples.py (examples.py).
"""
import math
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import examples as ex
from .sr_methods import SCALES, MethodError, describe
from .uploads import MAX_UPLOAD, MAX_SIDE, process, source_image

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[2]
TITLE = "Neural Enhanced Super-Resolution"
DOCS = {"readme": ("README", "README.md"), "usage": ("Usage guide", "docs/usage_guide.md"),
        "structure": ("Project structure", "docs/project_structure.md")}

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary: methods (with availability on this machine), patterns, scales and example counts."""
    m = ex.load_manifest()
    return {
        "title": TITLE, "version": "1", "scales": list(SCALES), "methods": describe(),
        "patterns": ex.pattern_list(), "n_examples": len(m["examples"]), "example_size": m["size"],
        "example_methods": [x["id"] for x in m["methods"]], "example_versions": m.get("versions", {}),
        "max_side": MAX_SIDE, "docs": [{"id": k, "title": v[0], "file": v[1]} for k, v in DOCS.items()],
    }


@app.get("/api/examples")
def examples():
    """The precomputed examples: files (under static/examples/) and Python PSNR / SSIM against the ground truth."""
    return ex.load_manifest()


@app.get("/api/docs/{name}")
def docs(name: str):
    if name not in DOCS:
        raise HTTPException(404, "unknown document")
    title, file = DOCS[name]
    return {"id": name, "title": title, "file": file, "markdown": (ROOT / file).read_text(encoding="utf-8")}


@app.post("/api/upscale")
async def upscale(file: UploadFile | None = File(None), pattern: str = Form(""), size: int = Form(256),
                  method: str = Form("bicubic"), scale: int = Form(2), mode: str = Form("upscale")):
    """Upscale an uploaded image (or a test pattern) with one method; in benchmark mode also degrade it first and
    report PSNR / SSIM against the original."""
    data = None
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(400, "file too large (limit 25 MB)")
    try:
        if not 16 <= size <= 1024:
            raise MethodError("pattern size must be 16 to 1024 px")
        img, source = source_image(data, pattern, size)
        result = await run_in_threadpool(process, img, method, scale, mode)
    except MethodError as e:
        raise HTTPException(400, str(e))
    result["source"] = file.filename if data else source
    for k in ("psnr", "ssim"):
        v = result.get("metrics", {}).get(k)
        if v is not None and math.isinf(v):
            result["metrics"][k] = None  # identical images: JSON has no infinity
    return result


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/examples"] + [f"/docs/{d}" for d in DOCS]
