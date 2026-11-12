"""Regression tests for NearestNeighborTracker.

Before the fix its documented `gap_closing` setting was stored and never used: a particle missed in one frame always
ended its track and started a new one. Two more problems in the same loop: tracks were linked in track order, so a
track whose nearest detection had already been taken by an earlier track ended even when another detection was in
range; and a frame with no detections at all was skipped, so tracks were linked across it whatever gap_closing said.
Gap convention as in LAPTracker: a link may span a frame difference of up to `gap_closing` (so up to gap_closing - 1
missing frames), within max_gap_displacement * sqrt(frame difference).
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import Detection  # noqa: E402
from src.tracking.nearest_tracker import NearestNeighborTracker  # noqa: E402


def _dets(points):
    return [Detection(x=x, y=y, t=t) for t, x, y in points]


def _blink(missing):
    """One particle moving 1 px/frame along x, frames 0-9, absent in `missing`; a static one far away."""
    pts = [(t, 10.0 + t, 10.0) for t in range(10) if t not in missing]
    pts += [(t, 80.0, 80.0) for t in range(10)]
    return _dets(pts)


def test_gap_closing_bridges_missing_frames():
    res = NearestNeighborTracker({"max_displacement": 3.0, "gap_closing": 3}).track(_blink({4, 5}))
    moving = [t for t in res.tracks if t.detections[0].x < 50]
    assert len(moving) == 1
    assert moving[0].times.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    assert moving[0].gap_frames == [4, 5]
    assert res.metadata["gap_closing"] == 3


def test_gap_longer_than_setting_splits_track():
    res = NearestNeighborTracker({"max_displacement": 3.0, "gap_closing": 2}).track(_blink({4, 5}))
    assert len([t for t in res.tracks if t.detections[0].x < 50]) == 2
    res0 = NearestNeighborTracker({"max_displacement": 3.0}).track(_blink({4}))     # default 0: no gaps
    assert len([t for t in res0.tracks if t.detections[0].x < 50]) == 2


def test_empty_frame_is_a_gap():
    pts = [(t, 10.0 + t, 10.0) for t in range(6) if t != 3]      # nothing at all in frame 3
    assert len(NearestNeighborTracker({"max_displacement": 3.0}).track(_dets(pts)).tracks) == 2
    assert len(NearestNeighborTracker({"max_displacement": 3.0, "gap_closing": 2}).track(_dets(pts)).tracks) == 1


def test_links_closest_pairs_first():
    # frame 1: detection A at x=11 is closest to track 1 (x=11.5, 0.5 px) and next to track 0 (x=10, 1 px);
    # track 0 also has B at x=8 within range (2 px). Track order must not matter.
    pts = [(0, 10.0, 10.0), (0, 11.5, 10.0), (1, 11.0, 10.0), (1, 8.0, 10.0)]
    res = NearestNeighborTracker({"max_displacement": 3.0}).track(_dets(pts))
    assert sorted(t.length for t in res.tracks) == [2, 2]
