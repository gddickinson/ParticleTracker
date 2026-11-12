"""Write tests/fixtures/engine_cases.json: Python results the JS engines must reproduce (engines.test.mjs).

- detect: the wavelet detector with subpixel=False on three demo frames, three parameter sets
- link/analyse: the default pipeline's detections (subpixel Gaussian refinement) linked by the LAP tracker with three
  parameter sets, with the per-track analysis, the ensemble MSD and fits and the ground-truth scores

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
    })


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
