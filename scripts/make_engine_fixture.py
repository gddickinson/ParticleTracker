"""Write tests/fixtures/engine_cases.json: Python results the JS engines must reproduce (engines.test.mjs).

- detect: the wavelet detector with subpixel=False on three demo frames, three parameter sets
- link/analyse: the default pipeline's detections (subpixel Gaussian refinement) linked by the LAP tracker with three
  parameter sets, with the per-track analysis, the ensemble MSD and fits and the ground-truth scores
- classify: the MSD classifier's features and classes (three option sets) on 72 simulated tracks (demo motion
  parameters, three seeds, lengths 12-60, some with missing frames)

Run: python scripts/make_engine_fixture.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.web import analysis as A  # noqa: E402
from src.web import demo as D  # noqa: E402
from src.web.jsonify import jsonable  # noqa: E402
from src.web.scoring import detection_scores, tracking_scores  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
FRAMES = [0, 17, 33]
DETECT = [{"sigma": 1.5, "threshold": 3.0, "min_distance": 3},
          {"sigma": 1.2, "threshold": 4.0, "min_distance": 2},
          {"sigma": 2.0, "threshold": 2.5, "min_distance": 4}]
LINK = [{}, {"max_displacement": 2.0, "gap_closing": 0}, {"max_displacement": 8.0, "gap_closing": 5, "min_track_length": 10}]


CLASSIFY_OPTS = [{}, {"alpha_classes": "anomalous", "drift_p": 0.05},
                 {"confinement_fit_lags": 4, "confinement_threshold": 3.16, "fit_fraction": 0.5, "max_fit_lag": 20,
                  "alpha_sub": 0.85, "alpha_super": 1.15, "min_length": 15}]


def classify_cases():
    import numpy as np
    from src.analysis.msd_classifier import classify_features, msd_track_features
    from src.core.data_structures import Detection, Track
    from src.core.simulation import simulate_trajectories
    tracks = []
    for seed in (1, 2, 3):
        rng = np.random.RandomState(seed)
        types = ["brownian"] * 8 + ["directed"] * 8 + ["confined"] * 8
        for i, g in enumerate(simulate_trajectories(60, 24, (2000, 2000), rng, 0.3, types, 0.4, 2.0, margin=500)):
            n = [12, 20, 35, 60][i % 4]
            frames = np.arange(n)
            if i % 3 == 0:
                frames = frames[rng.random_sample(n) > 0.15]       # missing frames
            t = Track(len(tracks))
            for f in frames:
                t.add_detection(Detection(x=g["x"][f] + rng.standard_normal() * 0.05,
                                          y=g["y"][f] + rng.standard_normal() * 0.05, t=int(f)))
            tracks.append(t)
    return [{"frames": [int(v) for v in t.times], "x": [d.x for d in t.detections], "y": [d.y for d in t.detections],
             "features": [msd_track_features(t, **o) for o in CLASSIFY_OPTS],
             "classes": [classify_features(msd_track_features(t, **o), **o) for o in CLASSIFY_OPTS]} for t in tracks]


def build():
    truth = D.truth()
    ps, dt = D.DEMO["pixel_size"], D.DEMO["time_interval"]
    detect_cases = []
    for p in DETECT:
        pt = D._tracker()
        dets = pt.detect(method="wavelet", frames=FRAMES, subpixel=False, **p)
        per = A.per_frame(dets, pt.image_stack.n_frames)
        detect_cases.append({"params": p, "frames": FRAMES, "expected": [per[f] for f in FRAMES]})
    pt = D._tracker()
    A.detect(pt, "wavelet", {})
    particles = A.per_frame(pt.detections, pt.image_stack.n_frames)
    link_cases = []
    for p in LINK:
        out = A.analyse(pt, p)
        ens = out["ensemble"]
        ens["fits"].pop("confined")
        link_cases.append({"params": p, "tracks": out["tracks"], "per_track": out["per_track"], "ensemble": ens,
                           "tracking_scores": tracking_scores(out["tracks"], truth)})
    return jsonable({
        "movie": {"file": "src/web/static/demo/movie.u16", "shape": list(D.movie().shape)},
        "pixel_size": ps, "time_interval": dt,
        "truth": [{"x": g["x"], "y": g["y"]} for g in truth],
        "detect_cases": detect_cases,
        "particles": particles, "detection_scores": detection_scores(particles, truth),
        "link_cases": link_cases,
        "classify_opts": CLASSIFY_OPTS, "classify_cases": classify_cases(),
    })


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
