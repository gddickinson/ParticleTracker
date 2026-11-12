"""Regression tests for the MSD classifier (classify_track_motion(method='msd')).

Before the fix, on seeded simulated tracks with the web demo's motion parameters, directed motion came out
'superdiffusive' (alpha > 1.15 was tested before directed motion), corralled motion mostly 'subdiffusive'
(confinement level from lags 1-4 too low) and about 40 % of free diffusion sub- or superdiffusive (alpha fitted over
noisy lags up to n/2). docs/msd_classifier.md has the full before/after confusion matrices.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.analysis.classification import classify_track_motion  # noqa: E402
from src.analysis.msd_classifier import (MSD_CLASSIFIER_DEFAULTS, classify_features, drift_test,  # noqa: E402
                                         msd_track_features)
from src.core.data_structures import Detection, Track  # noqa: E402
from src.core.pipeline import ParticleTracker  # noqa: E402
from src.core.simulation import simulate_trajectories  # noqa: E402

TYPES = ["brownian"] * 4 + ["directed"] * 4 + ["confined"] * 4


def _track(x, y, frames=None, tid=0):
    t = Track(tid)
    for f, a, b in zip(frames if frames is not None else range(len(x)), x, y):
        t.add_detection(Detection(x=float(a), y=float(b), t=int(f)))
    return t


def _demo_like(seeds, n=50, sigma=0.05):
    """(truth, Track) for the web demo's motion: D 0.3 px^2/frame, drift 0.4 px/frame, corral radius 2 px."""
    out = []
    for s in seeds:
        rng = np.random.RandomState(s)
        for g in simulate_trajectories(n, 12, (2000, 2000), rng, 0.3, TYPES, 0.4, 2.0, margin=500):
            x = np.array(g["x"]) + rng.standard_normal(n) * sigma
            y = np.array(g["y"]) + rng.standard_normal(n) * sigma
            out.append((g["motion_type"], _track(x, y, tid=g["track_id"])))
    return out


def _recall(rows, method="msd", **opts):
    got = {t: [] for t in ("brownian", "directed", "confined")}
    for truth, tr in rows:
        got[truth].append(classify_track_motion(tr, method=method, **opts) == truth)
    return {t: float(np.mean(v)) for t, v in got.items()}


def test_recovers_simulated_classes():
    rec = _recall(_demo_like(range(20)))
    assert rec["brownian"] >= 0.85
    assert rec["directed"] >= 0.7
    assert rec["confined"] >= 0.8


def test_legacy_rule_kept_for_comparison():
    rows = _demo_like(range(5))
    assert _recall(rows, "msd_legacy")["directed"] == 0.0     # all 'superdiffusive'
    labels = {classify_track_motion(tr, "msd_legacy") for _, tr in rows}
    assert "superdiffusive" in labels and "subdiffusive" in labels


def test_free_diffusion_false_positives_low():
    rng = np.random.RandomState(7)
    calls = [classify_track_motion(_track(*(100 + np.cumsum(rng.standard_normal((2, 100)) * 0.8, axis=1))))
             for _ in range(300)]
    assert np.mean([c != "brownian" for c in calls]) < 0.06


def test_drift_test_calibrated_and_handles_gaps():
    rng = np.random.RandomState(3)
    ps = [drift_test(range(60), np.cumsum(rng.standard_normal((60, 2)), 0))[1] for _ in range(2000)]
    assert 0.005 < np.mean(np.array(ps) < 0.01) < 0.02
    frames = np.r_[0:20, 23:40]                       # a 3-frame gap
    pos = np.c_[frames * 1.0, np.zeros(frames.size)] + 50 + rng.standard_normal((frames.size, 2)) * 0.3
    z2, p = drift_test(frames, pos)
    assert p < 1e-6


def test_anomalous_labels_option():
    rows = _demo_like(range(5))
    labels = {classify_track_motion(tr, alpha_classes="anomalous") for _, tr in rows}
    assert labels <= {"brownian", "directed", "confined", "subdiffusive", "superdiffusive"}
    feats = {"length": 50, "alpha": 0.3, "drift_p": 0.5, "confinement_level": 0.0}
    assert classify_features(feats) == "confined"
    assert classify_features(feats, alpha_classes="anomalous") == "subdiffusive"
    assert classify_features({**feats, "alpha": 1.8}, alpha_classes="anomalous") == "superdiffusive"
    assert classify_features({**feats, "length": 9}) == "insufficient_data"


def test_options_validated_and_reach_the_classifier():
    with pytest.raises(ValueError):
        classify_track_motion(_track(range(20), range(20)), bogus=1)
    rows = _demo_like(range(3))
    strict = _recall(rows, confinement_threshold=1e9, alpha_sub=-1e9)
    assert strict["confined"] == 0.0
    pt = ParticleTracker()
    assert pt.config["classification"] == MSD_CLASSIFIER_DEFAULTS
    tracks = [tr for _, tr in rows]
    assert set(pt.classify_motion(tracks, drift_p=0.0, alpha_super=1e9).values()) <= {"brownian", "confined"}


def test_features_dimensionless():
    rows = _demo_like(range(1))
    for _, tr in rows:
        f1 = msd_track_features(tr)
        scaled = _track([d.x * 7.0 for d in tr.detections], [d.y * 7.0 for d in tr.detections])
        f2 = msd_track_features(scaled)
        for k in ("alpha", "drift_z2", "confinement_level"):
            assert f1[k] == pytest.approx(f2[k], rel=1e-9, abs=1e-12)


def test_classify_motion_type_uses_the_same_classifier():
    from src.analysis.motion import classify_motion_type
    for _, tr in _demo_like(range(3)):
        assert classify_motion_type(tr) == classify_track_motion(tr)
