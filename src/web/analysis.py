"""Run the repo's own pipeline steps and turn the results into compact JSON for the browser.

Everything here calls ParticleTracker / src.analysis; nothing is re-implemented. static/analysis_engine.js mirrors
per_track() and ensemble() for the static site (node-tested against this module's output).
"""
import logging
import warnings

import numpy as np

from ..analysis.classification import classify_track_motion
from ..analysis.motion import (calculate_directionality, calculate_msd, detect_confinement,
                               fit_diffusion_model)
from ..core.data_structures import ImageStack
from ..core.pipeline import ParticleTracker
from .jsonify import jsonable

logging.getLogger("src.core.pipeline").setLevel(logging.WARNING)

MODELS = ("brownian", "anomalous", "directed", "confined")


def defaults():
    """The pipeline's own defaults (ParticleTracker._default_config)."""
    return ParticleTracker._default_config()


def make_tracker(movie, pixel_size, time_interval):
    pt = ParticleTracker(config=defaults())
    pt.image_stack = ImageStack(np.asarray(movie, dtype=np.float32), pixel_size=pixel_size,
                                time_interval=time_interval)
    return pt


def detect(pt, method="wavelet", params=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # scipy OptimizeWarning from the Gaussian refinement
        return pt.detect(method=method, **(params or {}))


def per_frame(detections, n_frames):
    """Detections as [[x, y, intensity, snr], ...] per frame."""
    out = [[] for _ in range(n_frames)]
    for d in detections:
        out[d.t].append([d.x, d.y, d.intensity, d.snr])
    return out


def compact_tracks(result):
    return [{"track_id": int(t.track_id), "frames": [int(v) for v in t.times],
             "x": [float(d.x) for d in t.detections], "y": [float(d.y) for d in t.detections],
             "gaps": [int(g) for g in t.gap_frames]} for t in result.tracks]


def per_track(track, pixel_size, time_interval):
    """The numbers the MSD classifier looks at, for one track (None where the track is too short)."""
    row = {"track_id": int(track.track_id), "length": track.length,
           "motion_class": classify_track_motion(track, method="msd", time_interval=time_interval)}
    if track.length >= 5:
        max_lag = max(3, min(track.length // 2, 20))
        msd_px = calculate_msd([track], max_lag=max_lag, time_interval=time_interval)
        an = fit_diffusion_model(msd_px, "anomalous")
        msd_um = calculate_msd([track], max_lag=max_lag, time_interval=time_interval, pixel_size=pixel_size)
        br = fit_diffusion_model(msd_um, "brownian")
        row["alpha"] = an.get("alpha") if an.get("success") else None
        row["D"] = br.get("diffusion_coefficient") if br.get("success") else None
    row["directionality"] = calculate_directionality(track)["directionality_ratio"]
    conf = detect_confinement(track)
    row["confinement_level"] = conf["confinement_level"]
    row["confined"] = conf["is_confined"]
    row["confined_legacy"] = detect_confinement(track, method="rg_legacy")["is_confined"]
    return row


def ensemble(tracks, pixel_size, time_interval, max_lag, models=MODELS):
    msd = calculate_msd(tracks, max_lag=max_lag, time_interval=time_interval, pixel_size=pixel_size)
    fits = {}
    for m in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = fit_diffusion_model(msd, m)
        fits[m] = {k: v for k, v in f.items() if k not in ("param_errors",)}
    return {"msd": {k: msd[k] for k in ("lags", "time_lags", "mean_msd", "std_msd", "n_tracks_per_lag", "n_tracks")},
            "fits": fits, "units": msd["units"]}


def analyse(pt, tracker_params=None):
    """Link pt.detections with the LAP tracker (pipeline defaults + params) and analyse the tracks."""
    st = pt.image_stack
    params = {**defaults()["tracking"], **(tracker_params or {})}
    method = params.pop("method", "lap")
    result = pt.track(method=method, **params)
    rows = [per_track(t, st.pixel_size, st.time_interval) for t in result.tracks]
    classes = {}
    for r in rows:
        classes[r["motion_class"]] = classes.get(r["motion_class"], 0) + 1
    return {
        "link_params": {**params, "method": method},
        "units": {"length": "um", "time": "s", "pixel_size_um": st.pixel_size, "time_interval_s": st.time_interval},
        "particles": per_frame(pt.detections, st.n_frames),
        "tracks": compact_tracks(result),
        "per_track": rows,
        "class_counts": classes,
        "ensemble": ensemble(result.tracks, st.pixel_size, st.time_interval, defaults()["analysis"]["msd_lag_max"]),
    }


def finish(d, digits=6):
    return jsonable(d, digits)
