"""Tests for motion analysis module."""

import numpy as np
import pytest
import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.core.data_structures import Detection, Track
from src.analysis.motion import (
    calculate_msd,
    fit_diffusion_model,
    calculate_velocity,
    calculate_directionality,
    detect_confinement,
    classify_motion_type,
)


def _make_brownian_track(n_frames=50, diffusion_coeff=1.0, track_id=0):
    """Create a track with Brownian motion."""
    rng = np.random.RandomState(42)
    track = Track(track_id=track_id)
    x, y = 50.0, 50.0

    for t in range(n_frames):
        x += rng.randn() * np.sqrt(2 * diffusion_coeff)
        y += rng.randn() * np.sqrt(2 * diffusion_coeff)
        track.add_detection(Detection(x=max(0, x), y=max(0, y), t=t, intensity=100.0))

    return track


def _make_directed_track(n_frames=50, velocity=2.0, track_id=0):
    """Create a track with directed motion."""
    track = Track(track_id=track_id)
    x, y = 50.0, 50.0

    rng = np.random.RandomState(42)
    for t in range(n_frames):
        x += velocity + rng.randn() * 0.1
        y += velocity * 0.5 + rng.randn() * 0.1
        track.add_detection(Detection(x=max(0, x), y=max(0, y), t=t, intensity=100.0))

    return track


class TestMSD:
    """Tests for MSD calculation."""

    def test_basic_msd(self):
        track = _make_brownian_track(n_frames=50)
        result = calculate_msd([track], max_lag=10)
        assert 'mean_msd' in result
        assert 'time_lags' in result
        assert len(result['mean_msd']) == 10
        assert result['n_tracks'] == 1

    def test_msd_increases(self):
        track = _make_brownian_track(n_frames=100, diffusion_coeff=2.0)
        result = calculate_msd([track], max_lag=20)
        msd = result['mean_msd']
        # MSD should generally increase with lag (for Brownian)
        assert msd[-1] > msd[0]

    def test_empty_tracks(self):
        result = calculate_msd([], max_lag=10)
        assert result['n_tracks'] == 0
        assert np.all(result['mean_msd'] == 0)

    def test_short_track_skipped(self):
        track = Track(track_id=0)
        for t in range(3):
            track.add_detection(Detection(x=float(t), y=float(t), t=t))
        result = calculate_msd([track], max_lag=10)
        assert result['n_tracks'] == 0  # Too short (< 5)


class TestDiffusionFitting:
    """Tests for diffusion model fitting."""

    def test_brownian_fit(self):
        track = _make_brownian_track(n_frames=100, diffusion_coeff=1.0)
        msd = calculate_msd([track], max_lag=20, time_interval=1.0)
        fit = fit_diffusion_model(msd, model='brownian')
        assert fit['success'] is True
        assert fit['model'] == 'brownian'
        assert 'diffusion_coefficient' in fit

    def test_anomalous_fit(self):
        track = _make_brownian_track(n_frames=100)
        msd = calculate_msd([track], max_lag=20, time_interval=1.0)
        fit = fit_diffusion_model(msd, model='anomalous')
        assert fit['success'] is True
        assert 'alpha' in fit

    def test_insufficient_data(self):
        msd = {
            'time_lags': np.array([1.0]),
            'mean_msd': np.array([0.5]),
        }
        fit = fit_diffusion_model(msd, model='brownian')
        assert fit['success'] is False


class TestVelocity:
    """Tests for velocity calculation."""

    def test_basic_velocity(self):
        track = _make_directed_track(n_frames=20, velocity=2.0)
        result = calculate_velocity(track, time_interval=1.0)
        assert result['mean_velocity'] > 0
        assert 'std_velocity' in result

    def test_single_detection_velocity(self):
        track = Track(track_id=0)
        track.add_detection(Detection(x=1.0, y=1.0, t=0))
        result = calculate_velocity(track)
        assert result['mean_velocity'] == 0.0


class TestDirectionality:
    """Tests for directionality calculation."""

    def test_directed_track_high_ratio(self):
        track = _make_directed_track(n_frames=30, velocity=5.0)
        result = calculate_directionality(track)
        assert result['directionality_ratio'] > 0.5
        assert result['net_displacement'] > 0

    def test_short_track(self):
        track = Track(track_id=0)
        track.add_detection(Detection(x=1.0, y=1.0, t=0))
        result = calculate_directionality(track)
        assert result['directionality_ratio'] == 0.0


class TestConfinement:
    """Tests for confinement detection."""

    def test_confined_motion(self):
        track = Track(track_id=0)
        rng = np.random.RandomState(42)
        for t in range(30):
            x = 50.0 + rng.randn() * 0.5  # Very confined
            y = 50.0 + rng.randn() * 0.5
            track.add_detection(Detection(x=max(0, x), y=max(0, y), t=t))

        result = detect_confinement(track, window_size=10, method='rg_legacy')
        assert result['confinement_probability'] > 0.5

    def test_short_track_not_confined(self):
        track = Track(track_id=0)
        for t in range(3):
            track.add_detection(Detection(x=float(t), y=float(t), t=t))
        result = detect_confinement(track, window_size=10)
        assert result['is_confined'] is False
