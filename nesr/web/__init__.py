"""NESR web interface (FastAPI app + static front end, also exported as a GitHub Pages site).

Contents:
    app.py         FastAPI routes under /api and STATIC_ENDPOINTS for the static export
    sr_methods.py  the upscaling methods offered (cv2.resize, the NESR pipeline, Real-ESRGAN) and their availability
    examples.py    precomputed examples on synthetic test patterns (static/examples/, made by scripts/make_examples.py)
    uploads.py     decoding uploads, PNG data URLs and the upscale / benchmark request
    static/        the browser front end (plain ES modules; resample.js and metrics.js are tested JS ports)
"""
