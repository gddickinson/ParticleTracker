"""FastAPI web app for ParticleTracker: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8211, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

Everything is computed by the repo's own code (src.core.pipeline.ParticleTracker, src.detection, src.tracking,
src.analysis); the demo movie is simulated by src.core.simulation (see demo.py).
"""
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import demo as D
from .upload import RunError, run_upload

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 1024 * 1024 * 1024

app = FastAPI(title="ParticleTracker")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Detectors, pipeline defaults and the demo's description."""
    return D.info()


@app.get("/api/demo/meta")
def demo_meta():
    """The demo movie (shape, simulation parameters, display range) and its ground-truth trajectories."""
    return D.meta()


@app.get("/api/demo/detections/{detector}")
def demo_detections(detector: str):
    """One detector with its default parameters on the demo movie, scored against the ground truth."""
    if detector not in D.DETECTORS:
        raise HTTPException(404, "unknown detector")
    return D.detections(detector)


@app.get("/api/demo/pipeline")
def demo_pipeline():
    """The default pipeline (wavelet detection, LAP linking, MSD analysis and classification) on the demo."""
    return D.pipeline()


def _numbers(d, name):
    if not isinstance(d, dict) or not all(isinstance(v, (int, float)) for v in d.values()):
        raise HTTPException(422, f"{name} must be an object of numbers (or true/false)")
    return d


INT_PARAMS = ("min_distance", "gap_closing", "min_track_length", "num_sigma", "fit_window")


@app.post("/api/demo/run")
def demo_run(body: dict = Body(...)):
    """Detect, link and analyse the demo movie with any detector and parameters."""
    det = body.get("detector", "wavelet")
    if det not in D.DETECTORS:
        raise HTTPException(422, "unknown detector")
    dp = dict(_numbers(body.get("det_params", {}), "det_params"))
    lp = dict(_numbers(body.get("link_params", {}), "link_params"))
    for d in (dp, lp):
        for k in INT_PARAMS:
            if k in d:
                d[k] = int(d[k])
    if "subpixel" in dp:
        dp["subpixel"] = bool(dp["subpixel"])
    return D.run(det, dp, lp)


@app.post("/api/run")
async def run(file: UploadFile = File(...), parameters: str = Form("{}")):
    """The pipeline on an uploaded TIFF stack."""
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file larger than 1 GB")
    try:
        return run_upload(file.filename, data, parameters)
    except RunError as e:
        raise HTTPException(422, str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = (["/info", "/demo/meta", "/demo/pipeline"]
                    + [f"/demo/detections/{d}" for d in D.DETECTORS])
