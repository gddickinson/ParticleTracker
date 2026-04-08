"""Tests for core data structures."""

import numpy as np
import pytest
import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.core.data_structures import (
    ImageStack, Detection, Track, TrackingResult, create_synthetic_data
)


class TestImageStack:
    """Tests for ImageStack dataclass."""

    def test_creation_3d(self):
        data = np.random.rand(10, 64, 64).astype(np.float32)
        stack = ImageStack(data=data)
        assert stack.n_frames == 10
        assert stack.shape == (10, 64, 64)
        assert not stack.is_3d

    def test_creation_2d_promotes_to_3d(self):
        data = np.random.rand(64, 64).astype(np.float32)
        stack = ImageStack(data=data)
        assert stack.n_frames == 1
        assert stack.shape == (1, 64, 64)

    def test_get_frame(self):
        data = np.random.rand(5, 32, 32).astype(np.float32)
        stack = ImageStack(data=data)
        frame = stack.get_frame(2)
        np.testing.assert_array_equal(frame, data[2])

    def test_get_substack(self):
        data = np.random.rand(10, 32, 32).astype(np.float32)
        stack = ImageStack(data=data)
        sub = stack.get_substack(2, 5)
        assert sub.n_frames == 3

    def test_pixel_size_default(self):
        data = np.random.rand(5, 32, 32)
        stack = ImageStack(data=data)
        assert stack.pixel_size == 0.1
        assert stack.time_interval == 0.1


class TestDetection:
    """Tests for Detection dataclass."""

    def test_creation(self):
        det = Detection(x=10.5, y=20.3, t=0)
        assert det.x == 10.5
        assert det.y == 20.3
        assert det.t == 0

    def test_position_2d(self):
        det = Detection(x=10.0, y=20.0)
        pos = det.position
        np.testing.assert_array_equal(pos, [10.0, 20.0])

    def test_position_3d(self):
        det = Detection(x=10.0, y=20.0, z=5.0)
        pos = det.position
        np.testing.assert_array_equal(pos, [10.0, 20.0, 5.0])
        assert det.is_3d

    def test_distance_2d(self):
        d1 = Detection(x=0.0, y=0.0)
        d2 = Detection(x=3.0, y=4.0)
        assert d1.distance_to(d2) == pytest.approx(5.0)

    def test_negative_coords_raises(self):
        with pytest.raises(ValueError):
            Detection(x=-1.0, y=0.0)


class TestTrack:
    """Tests for Track class."""

    def test_empty_track(self):
        track = Track(track_id=0)
        assert track.length == 0
        assert track.duration == 0
        assert track.start_frame == -1

    def test_add_detections(self):
        track = Track(track_id=1)
        track.add_detection(Detection(x=10.0, y=10.0, t=0))
        track.add_detection(Detection(x=11.0, y=11.0, t=1))
        track.add_detection(Detection(x=12.0, y=12.0, t=2))
        assert track.length == 3
        assert track.start_frame == 0
        assert track.end_frame == 2
        assert track.duration == 3

    def test_positions_array(self):
        track = Track(track_id=0)
        track.add_detection(Detection(x=1.0, y=2.0, t=0))
        track.add_detection(Detection(x=3.0, y=4.0, t=1))
        pos = track.positions
        assert pos.shape == (2, 2)

    def test_to_dataframe(self):
        track = Track(track_id=5)
        track.add_detection(Detection(x=1.0, y=2.0, t=0, intensity=100.0))
        track.add_detection(Detection(x=3.0, y=4.0, t=1, intensity=200.0))
        df = track.to_dataframe()
        assert len(df) == 2
        assert 'x' in df.columns
        assert 'track_id' in df.columns
        assert df['track_id'].iloc[0] == 5

    def test_gap_frames(self):
        track = Track(track_id=0)
        track.add_gap(5)
        track.add_gap(7)
        assert track.gap_frames == [5, 7]

    def test_properties(self):
        track = Track(track_id=0)
        track.set_property('motion_class', 'brownian')
        assert track.get_property('motion_class') == 'brownian'
        assert track.get_property('missing', 'default') == 'default'


class TestTrackingResult:
    """Tests for TrackingResult class."""

    def test_empty_result(self):
        result = TrackingResult()
        assert result.n_tracks == 0
        assert result.n_detections == 0

    def test_add_tracks(self):
        result = TrackingResult()
        t1 = Track(track_id=0)
        t1.add_detection(Detection(x=1.0, y=1.0, t=0))
        t2 = Track(track_id=1)
        t2.add_detection(Detection(x=5.0, y=5.0, t=0))
        result.add_track(t1)
        result.add_track(t2)
        assert result.n_tracks == 2

    def test_filter_tracks(self):
        result = TrackingResult()
        short = Track(track_id=0)
        short.add_detection(Detection(x=1.0, y=1.0, t=0))

        long = Track(track_id=1)
        for i in range(10):
            long.add_detection(Detection(x=float(i), y=float(i), t=i))

        result.add_track(short)
        result.add_track(long)

        filtered = result.filter_tracks(min_length=5)
        assert filtered.n_tracks == 1
        assert filtered.tracks[0].track_id == 1

    def test_to_dataframe(self):
        result = TrackingResult()
        track = Track(track_id=0)
        track.add_detection(Detection(x=1.0, y=2.0, t=0, intensity=100.0))
        result.add_track(track)
        df = result.to_dataframe()
        assert len(df) == 1


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_create_synthetic(self):
        stack = create_synthetic_data(
            n_frames=10, n_particles=5, frame_shape=(64, 64)
        )
        assert isinstance(stack, ImageStack)
        assert stack.n_frames == 10
        assert stack.shape == (10, 64, 64)
        assert stack.metadata.get('synthetic') is True

    def test_synthetic_has_signal(self):
        stack = create_synthetic_data(
            n_frames=5, n_particles=3, frame_shape=(64, 64),
            intensity=5000.0, noise_level=1.0
        )
        # Signal should be present (max > noise floor)
        assert np.max(stack.data) > 50.0
