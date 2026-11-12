"""Confusion matrices of the MSD classifier against the simulator's ground truth -> docs/msd_classifier.md.

1. Trajectories: simulate_trajectories (src/core/simulation.py) on a grid of track lengths, diffusion coefficients,
   localisation errors, corral radii and drift speeds, 12 particles per run (4 brownian, 4 directed, 4 confined),
   seeds 0-5 per condition; Gaussian localisation error added to the true positions.
2. End to end: movies with the web demo's parameters (src/web/demo.py) on seed 220 (the demo) and 10 other seeds,
   through wavelet detection + LAP linking with the pipeline defaults; each track is matched to the simulated
   particle most of its points sit on (web/scoring.tracking_scores) and takes that particle's motion type.

Both 'msd_legacy' (the rule before 2026-11-12) and 'msd' (analysis/msd_classifier.py, defaults) are scored.
Also reports, per test, the fraction of free-diffusion tracks it flags (false-positive rate) by track length.

    python scripts/msd_classifier_study.py            # writes docs/msd_classifier.md (about 2 minutes)
"""
import itertools
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.classification import classify_track_motion  # noqa: E402
from src.analysis.msd_classifier import MSD_CLASSIFIER_DEFAULTS as DEF, msd_track_features  # noqa: E402
from src.core.data_structures import Detection, Track  # noqa: E402
from src.core.simulation import create_synthetic_data, simulate_trajectories  # noqa: E402

TRUE = ("brownian", "directed", "confined")
PRED = ("brownian", "directed", "confined", "subdiffusive", "superdiffusive", "insufficient_data")
GRID = {"n": (20, 50, 100), "D": (0.1, 0.3, 1.0), "sigma": (0.05, 0.2), "R": (2.0, 4.0), "v": (0.2, 0.4)}
SEEDS = range(6)
MOVIE_SEEDS = [220] + list(range(1, 11))
METHODS = ("msd_legacy", "msd")


def make_track(frames, x, y, tid=0):
    t = Track(tid)
    for f, a, b in zip(frames, x, y):
        t.add_detection(Detection(x=float(a), y=float(b), t=int(f)))
    return t


def trajectory_runs():
    """Yield (condition, truth class, Track) for every simulated trajectory of the grid."""
    for ci, (n, D, sig, R, v) in enumerate(itertools.product(*GRID.values())):
        for s in SEEDS:
            rng = np.random.RandomState(1000 * ci + s)
            types = ["brownian"] * 4 + ["directed"] * 4 + ["confined"] * 4
            for g in simulate_trajectories(n, 12, (4000, 4000), rng, D, types, v, R, margin=1500):
                x = np.array(g["x"]) + rng.standard_normal(n) * sig
                y = np.array(g["y"]) + rng.standard_normal(n) * sig
                yield (n, D, sig, R, v), g["motion_type"], make_track(range(n), x, y, g["track_id"])


def movie_runs():
    """Yield (seed, truth class, Track) for every linked track of the demo-like movies."""
    from src.web import analysis as A
    from src.web.demo import DEMO
    from src.web.scoring import tracking_scores
    for seed in MOVIE_SEEDS:
        p = {**DEMO, "seed": seed}
        n_frames, n_part, shape, noise = p.pop("n_frames"), p.pop("n_particles"), tuple(p.pop("frame_shape")), \
            p.pop("noise_level")
        stack, truth = create_synthetic_data(n_frames, n_part, shape, noise, return_ground_truth=True, **p)
        pt = A.make_tracker(stack.data, DEMO["pixel_size"], DEMO["time_interval"])
        det = {k: v for k, v in A.defaults()["detection"].items() if k != "method"}
        A.detect(pt, "wavelet", det)
        lp = dict(A.defaults()["tracking"])
        result = pt.track(method=lp.pop("method"), **lp)
        tracks = A.compact_tracks(result)
        sc = tracking_scores(tracks, truth)
        for tr, m in zip(result.tracks, sc["per_track"]):
            if m["particle"] is not None:
                yield seed, truth[m["particle"]]["motion_type"], tr


def confusion(rows, method):
    c = {t: Counter() for t in TRUE}
    for truth, track in rows:
        c[truth][classify_track_motion(track, method=method)] += 1
    return c


def table(c):
    cols = [p for p in PRED if any(c[t][p] for t in TRUE)]
    out = ["| true \\ classified | " + " | ".join(cols) + " | recall |", "|---" * (len(cols) + 2) + "|"]
    for t in TRUE:
        tot = sum(c[t].values())
        rec = c[t][t] / tot if tot else float("nan")
        out.append(f"| {t} | " + " | ".join(str(c[t][p]) for p in cols) + f" | {rec:.0%} |")
    return "\n".join(out)


def fp_rates(rows):
    """Fraction of free-diffusion tracks flagged by each test (default thresholds), by track length."""
    out = ["| track length | confinement level > %g | drift p < %g | alpha < %g | alpha > %g | any (classified non-Brownian) |"
           % (DEF["confinement_threshold"], DEF["drift_p"], DEF["alpha_sub"], DEF["alpha_super"]), "|---" * 6 + "|"]
    for n in GRID["n"]:
        f = [msd_track_features(t) for (cond, truth, t) in rows if truth == "brownian" and cond[0] == n]
        L = np.mean([x["confinement_level"] > DEF["confinement_threshold"] for x in f])
        p = np.mean([x["drift_p"] < DEF["drift_p"] for x in f])
        lo = np.mean([x["alpha"] < DEF["alpha_sub"] for x in f])
        hi = np.mean([x["alpha"] > DEF["alpha_super"] for x in f])
        anyf = np.mean([classify_track_motion(t) != "brownian" for (cond, truth, t) in rows
                        if truth == "brownian" and cond[0] == n])
        out.append(f"| {n} | {L:.1%} | {p:.1%} | {lo:.1%} | {hi:.1%} | {anyf:.1%} |")
    return "\n".join(out)


def section(title, rows, note=""):
    s = [f"### {title}", note] if note else [f"### {title}"]
    for m in METHODS:
        label = "Before (`msd_legacy`)" if m == "msd_legacy" else "After (`msd`, defaults)"
        s += [f"**{label}**", table(confusion(rows, m))]
    return "\n\n".join(s)


def main():
    warnings.simplefilter("ignore")
    traj = list(trajectory_runs())
    parts = []
    for n in GRID["n"]:
        parts.append(section(f"Simulated trajectories, {n} points", [(t, tr) for c, t, tr in traj if c[0] == n],
                             f"{sum(1 for c, *_ in traj if c[0] == n)} tracks: every combination of D, localisation "
                             "error, corral radius and drift speed below."))
    demo_like = [(t, tr) for c, t, tr in traj if c == (50, 0.3, 0.05, 2.0, 0.4)]
    parts.append(section("Simulated trajectories, the demo's motion parameters (50 points, D 0.3, R 2, v 0.4)",
                         demo_like))
    movies = list(movie_runs())
    parts.append(section(f"End to end: {len(MOVIE_SEEDS)} demo-like movies (seed 220 = the web demo, and seeds 1-10)",
                         [(t, tr) for s, t, tr in movies],
                         f"{len(movies)} linked tracks matched to a simulated particle (wavelet + LAP defaults)."))
    parts.append(section("End to end: the web demo movie only (seed 220)",
                         [(t, tr) for s, t, tr in movies if s == 220]))
    grid = ", ".join(f"{k} {list(v)}" for k, v in GRID.items())
    doc = TEMPLATE.format(grid=grid, seeds=len(SEEDS), fp=fp_rates(traj), body="\n\n".join(parts),
                          opts="\n".join(f"| `{k}` | {v!r} |" for k, v in DEF.items()))
    out = ROOT / "docs" / "msd_classifier.md"
    out.write_text(doc)
    print("wrote", out)


TEMPLATE = """# MSD classifier: confusion matrices against the simulator

Generated by `scripts/msd_classifier_study.py` (do not edit by hand). Ground truth from the repo's own simulator
(`src/core/simulation.py`): 'brownian' (free diffusion, D px^2/frame), 'directed' (the same plus a drift of
v px/frame) and 'confined' (free diffusion inside a disc of radius R px). Units are pixels and frames; every test is
dimensionless.

Grid: {grid}; localisation error sigma (px) added to every position; {seeds} seeds per condition, 4 tracks of each
class per seed. Many grid corners cannot be told apart even in principle (a slow particle in a 4 px corral over
20 frames never feels the wall; a 0.2 px/frame drift next to D = 1 px^2/frame is buried), so the per-length tables
are a lower bound; the demo-parameter and end-to-end tables are the realistic case.

## What was wrong (method `msd_legacy`)
- alpha was fitted over lags up to n/2 (max 20), where the time-averaged MSD is very noisy: about 20 % of
  free-diffusion tracks of 50 points fall outside the 0.85-1.15 band and were called sub- or superdiffusive.
- alpha > 1.15 returned 'superdiffusive' before directed motion was tested, so diffusion with drift was never
  'directed'; the directionality ratio (> 0.7) it would have used is far lower for drift plus diffusion anyway.
- The confinement level used the MSD slope over lags 1-4; in a small corral the MSD is already levelling off
  there, so D and the level came out too low and corralled tracks fell through to 'subdiffusive'.

## The classifier now (method `msd`, `src/analysis/msd_classifier.py`)
1. confined if the Saxton/Simson confinement level L > `confinement_threshold`, with D from lags 1-2;
2. directed if the drift test (net displacement against the step spread, chi-square with 2 d.o.f.) gives
   p < `drift_p`;
3. alpha from lags 1..K, K = clip(int(n * `fit_fraction`), 4, `max_fit_lag`): alpha < `alpha_sub` is confined,
   alpha > `alpha_super` directed (or 'subdiffusive' / 'superdiffusive' with `alpha_classes='anomalous'`);
4. otherwise brownian. Tracks shorter than `min_length` are 'insufficient_data'.

Options (pass to `classify_track_motion(track, 'msd', **options)`, `ParticleTracker.classify_motion(**options)`
or `config['classification']`):

| option | default |
|---|---|
{opts}

Why these defaults: each test flags about 1-2 % of free-diffusion tracks of 50 points (table below), so a Brownian
track is called Brownian about 95 % of the time; L > 2 means a free walker would stay that close to its start with
probability below 1 % (Saxton 1993); drift p < 0.01 is a 1 % test; lags up to a quarter of the track (max 10)
keep alpha's spread small (the time-averaged MSD at long lags uses few independent displacements). Lower
`confinement_threshold`/`drift_p` or narrow the alpha band for more sensitivity at the cost of more false calls;
`confinement_threshold=3.16` with `confinement_fit_lags=4` is the original Simson et al. setting.

Limitations: localisation error adds a constant to the MSD and lowers alpha for slow particles; per-track alpha
cannot separate a corral from power-law subdiffusion at these lengths and the simulator has only corrals (hence
`alpha_classes`); the tests are less reliable for tracks under about 30 points.

## Free-diffusion tracks flagged by each test (default thresholds)

{fp}

## Confusion matrices

{body}
"""

if __name__ == "__main__":
    main()
