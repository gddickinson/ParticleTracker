"""Tests for LAP tracker."""

import numpy as np
import pytest
import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.core.data_structures import Detection, Track, TrackingResult
from src.tracking.lap_tracker import LAPTracker


def _make_simple_detections(n_frames=10, n_particles=3, displacement=1.0):
    """Create simple test detections with known trajectories."""
    rng = np.random.RandomState(42)
    detections = []

    # Create particles moving in straight lines
    for p in range(n_particles):
        x0 = 20.0 + p * 30.0
        y0 = 20.0 + p * 30.0

        for t in range(n_frames):
            det = Detection(
                x=x0 + t * displacement + rng.randn() * 0.1,
                y=y0 + t * displacement * 0.5 + rng.randn() * 0.1,
                t=t,
                intensity=1000.0
            )
            detections.append(det)

    return detections


class TestLAPTracker:
    """Tests for LAPTracker."""

    def test_instantiation(self):
        config = {'max_displacement': 5.0, 'gap_closing': 3}
        tracker = LAPTracker(config)
        assert tracker.max_displacement == 5.0
        assert tracker.gap_closing == 3

    def test_empty_detections(self):
        config = {'max_displacement': 5.0}
        tracker = LAPTracker(config)
        result = tracker.track([])
        assert isinstance(result, TrackingResult)
        assert result.n_tracks == 0

    def test_single_frame(self):
        config = {'max_displacement': 5.0}
        tracker = LAPTracker(config)
        detections = [
            Detection(x=10.0, y=10.0, t=0),
            Detection(x=50.0, y=50.0, t=0),
        ]
        result = tracker.track(detections)
        assert result.n_tracks == 2
        for track in result.tracks:
            assert track.length == 1

    def test_tracks_simple_motion(self):
        config = {'max_displacement': 10.0, 'gap_closing': 0}
        tracker = LAPTracker(config)
        detections = _make_simple_detections(n_frames=5, n_particles=2, displacement=0.5)
        result = tracker.track(detections)
        # Should produce tracks from the detections
        assert result.n_tracks >= 1
        # Total detections should be preserved
        total_dets = sum(t.length for t in result.tracks)
        assert total_dets == len(detections)

    def test_organize_by_frame(self):
        config = {'max_displacement': 5.0}
        tracker = LAPTracker(config)
        detections = [
            Detection(x=1.0, y=1.0, t=0),
            Detection(x=2.0, y=2.0, t=0),
            Detection(x=3.0, y=3.0, t=1),
        ]
        by_frame = tracker._organize_by_frame(detections)
        assert len(by_frame[0]) == 2
        assert len(by_frame[1]) == 1

    def test_metadata_populated(self):
        config = {'max_displacement': 5.0, 'gap_closing': 2}
        tracker = LAPTracker(config)
        detections = _make_simple_detections(n_frames=3, n_particles=1)
        result = tracker.track(detections)
        assert result.metadata['method'] == 'lap'
        assert result.metadata['max_displacement'] == 5.0
