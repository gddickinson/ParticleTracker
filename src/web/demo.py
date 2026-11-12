"""The demo movie and the analyses the web app shows for it.

The movie is simulated by the repo's own simulator, src/core/simulation.py
(create_synthetic_data, seeded): 12 particles, 4 each of 'brownian', 'directed'
and 'confined' motion, with known trajectories. static/demo/movie.u16 holds the
pixels (rounded to uint16, raw little-endian, frames x height x width), written
by scripts/make_demo.py; the browser and every Python analysis here read that
same file, and a test checks it still matches the simulator (within 1 count).

Seed 220 was picked from the first 300 seeds because its particles stay at
least 6 px apart and away from the frame edge (the simulator clamps at the edge).

Units: the simulator works in pixels and frames; the demo uses the
ImageStack defaults 0.1 um per pixel and 0.1 s per frame, so its D values are
in um^2/s on that assumed scale.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.simulation import create_synthetic_data
from . import analysis as A
from .scoring import detection_scores, tracking_scores

STATIC = Path(__file__).resolve().parent / "static"
MOVIE_FILE = STATIC / "demo" / "movie.u16"
TYPES = ["brownian"] * 4 + ["directed"] * 4 + ["confined"] * 4
DEMO = {
    "seed": 220, "n_frames": 50, "frame_shape": [112, 112], "n_particles": 12, "motion_types": TYPES,
    "noise_level": 10.0, "diffusion_coeff": 0.3, "velocity": 0.4, "confinement_radius": 2.0, "margin": 14,
    "background": 100.0, "psf_sigma": 1.5, "intensity": 1000.0, "pixel_size": 0.1, "time_interval": 0.1,
}
DETECTORS = ["wavelet", "log", "dog", "gaussian"]


def simulate():
    """(float32 movie, ground truth) straight from the simulator."""
    p = dict(DEMO)
    n_frames, n_particles, shape = p.pop("n_frames"), p.pop("n_particles"), tuple(p.pop("frame_shape"))
    noise = p.pop("noise_level")
    stack, truth = create_synthetic_data(n_frames, n_particles, shape, noise, return_ground_truth=True, **p)
    return stack.data, truth


def to_uint16(movie):
    return np.clip(np.rint(movie), 0, 65535).astype("<u2")


@lru_cache(maxsize=1)
def truth():
    return simulate()[1]


@lru_cache(maxsize=1)
def movie():
    """The committed demo movie, uint16 [T, H, W]."""
    T, (H, W) = DEMO["n_frames"], DEMO["frame_shape"]
    return np.fromfile(MOVIE_FILE, dtype="<u2").reshape(T, H, W)


def meta():
    m = movie()
    lo, hi = np.percentile(m[0], [1, 99.95])
    return A.finish({
        **DEMO, "shape": list(m.shape), "dtype": "uint16", "file": "static/demo/movie.u16",
        "display": [float(lo), float(hi)],
        "generator": "src.core.simulation.create_synthetic_data (seeded)",
        "ground_truth": [{"particle": g["track_id"], "motion_type": g["motion_type"], "x": g["x"], "y": g["y"]}
                         for g in truth()],
    }, 4)


def info():
    d = A.defaults()
    return {
        "title": "ParticleTracker", "version": "1", "detectors": DETECTORS,
        "defaults": {"detection": d["detection"], "tracking": d["tracking"], "analysis": d["analysis"]},
        "demo": {k: DEMO[k] for k in ("seed", "n_frames", "frame_shape", "n_particles", "motion_types",
                                       "pixel_size", "time_interval")},
    }


def _tracker():
    return A.make_tracker(movie(), DEMO["pixel_size"], DEMO["time_interval"])


def _scored(pt, out):
    gt = truth()
    out["detection_scores"] = detection_scores(out["particles"], gt)
    out["tracking_scores"] = tracking_scores(out["tracks"], gt)
    return out


@lru_cache(maxsize=None)
def detections(method):
    """Default-parameter detections of one detector, scored against the truth."""
    pt = _tracker()
    params = A.defaults()["detection"] if method == "wavelet" else {}
    params = {k: v for k, v in params.items() if k != "method"}
    dets = A.detect(pt, method, params)
    parts = A.per_frame(dets, pt.image_stack.n_frames)
    return A.finish({"detector": method, "det_params": params, "particles": parts,
                     "detection_scores": detection_scores(parts, truth())}, 5)


def run(detector="wavelet", det_params=None, link_params=None):
    """Detection + LAP linking + analysis on the demo movie, scored against the truth."""
    pt = _tracker()
    base = A.defaults()["detection"] if detector == "wavelet" else {}
    dp = {**{k: v for k, v in base.items() if k != "method"}, **(det_params or {})}
    A.detect(pt, detector, dp)
    out = A.analyse(pt, link_params)
    out.update({"detector": detector, "det_params": dp, "computed_in": "python"})
    return A.finish(_scored(pt, out), 6)


@lru_cache(maxsize=1)
def pipeline():
    """The pipeline with all defaults (wavelet + LAP + MSD analysis), as ParticleTracker runs it."""
    return run()
