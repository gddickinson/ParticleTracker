"""Regression tests for LAPTracker linking.

Before the fix the birth/death blocks of the augmented cost matrix were np.eye(n) * cost: their off-diagonal entries
were 0 instead of forbidden, so with two or more particles 'end every track and start a new one' cost nothing and no
detection was ever linked (every track had length 1, and the pipeline's min_track_length filter removed them all).
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import Detection  # noqa: E402
from src.tracking.lap_tracker import LAPTracker  # noqa: E402


def _lines(n_frames=10, skip=()):
    dets = []
    for t in range(n_frames):
        dets.append(Detection(x=10 + 0.5 * t, y=10, t=t))
        if t not in skip:
            dets.append(Detection(x=50, y=50 + 0.5 * t, t=t))
    return dets


def test_two_particles_linked_frame_to_frame():
    r = LAPTracker({"gap_closing": 0, "merge_splits": False}).track(_lines())
    assert sorted(t.length for t in r.tracks) == [10, 10]
    for t in r.tracks:
        assert len({round(d.x) if d.y == 10 else round(d.y) for d in t.detections}) > 1
        assert len({d.y == 10 for d in t.detections}) == 1   # never jumps between the two particles


def test_gap_closed_into_one_track():
    # particle 2 is missing in frames 4 and 5: gap closing (default 3 frames) joins the two pieces
    r = LAPTracker({}).track(_lines(skip=(4, 5)))
    assert sorted(t.length for t in r.tracks) == [8, 10]
    long2 = [t for t in r.tracks if t.length == 8][0]
    assert long2.gap_frames == [4, 5]


def test_gap_closing_chain_has_no_duplicates():
    # particle 2 missing in frames 3 and 6: three pieces chained into one track, each detection used once
    r = LAPTracker({}).track(_lines(n_frames=10, skip=(3, 6)))
    total = sum(t.length for t in r.tracks)
    assert total == 18
    assert sorted(t.length for t in r.tracks) == [8, 10]
    assert sorted([t for t in r.tracks if t.length == 8][0].gap_frames) == [3, 6]


def test_far_detections_not_linked():
    dets = [Detection(x=10, y=10, t=0), Detection(x=60, y=60, t=0),
            Detection(x=40, y=10, t=1), Detection(x=60, y=90, t=1)]
    r = LAPTracker({"gap_closing": 0, "merge_splits": False}).track(dets)
    assert sorted(t.length for t in r.tracks) == [1, 1, 1, 1]
