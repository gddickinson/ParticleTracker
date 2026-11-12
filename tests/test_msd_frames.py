"""Regression tests for calculate_msd.

Before the fix: (1) pairs were looked for only between rows exactly `lag` apart, so in a track with missing frames
the valid pairs that are `lag` frames apart but fewer rows apart were never used (a track observed every other frame
contributed nothing at all); (2) a track too short for a lag contributed MSD = 0 at that lag to the ensemble mean,
pulling the mean MSD down at long lags.
"""
import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import Detection, Track  # noqa: E402
from src.analysis.motion import calculate_msd  # noqa: E402


def _line(frames, step=1.0, track_id=0):
    t = Track(track_id)
    for f in frames:
        t.add_detection(Detection(x=5 + step * f, y=5, t=f))
    return t


def test_track_sampled_every_other_frame():
    r = calculate_msd([_line(range(0, 40, 2))], max_lag=6)
    assert r["n_tracks"] == 1
    m = r["mean_msd"]
    assert np.isnan(m[0]) and np.isclose(m[1], 4.0) and np.isnan(m[2]) and np.isclose(m[3], 16.0)


def test_pairs_across_gap_used():
    frames = [f for f in range(20) if f != 7]
    r = calculate_msd([_line(frames)], max_lag=3)
    assert np.allclose(r["mean_msd"], [1, 4, 9])
    assert list(r["n_pairs"]) == [17, 16, 15]   # pairs exactly 1, 2, 3 frames apart


def test_short_track_does_not_pull_down_long_lags():
    long_, short = _line(range(30), track_id=0), _line(range(6), track_id=1)
    r = calculate_msd([long_, short], max_lag=10)
    assert np.allclose(r["mean_msd"], np.arange(1, 11) ** 2)
    assert list(r["n_tracks_per_lag"]) == [2] * 5 + [1] * 5


def test_gap_free_unchanged():
    rng = np.random.RandomState(3)
    t = Track(0)
    xy = np.cumsum(rng.randn(50, 2), axis=0) + 100
    for f, (x, y) in enumerate(xy):
        t.add_detection(Detection(x=x, y=y, t=f))
    r = calculate_msd([t], max_lag=5)
    for k in range(1, 6):
        d = xy[k:] - xy[:-k]
        assert np.isclose(r["mean_msd"][k - 1], np.mean(np.sum(d ** 2, axis=1)))
