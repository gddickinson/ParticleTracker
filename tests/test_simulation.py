"""Tests for the synthetic movie generator (core/simulation.py).

Regression: the original generator rendered each particle at int(position) (truncated, so up to 1 px from its
true position, -0.5 px on average) and returned no ground truth, so localisation accuracy could not be measured.
"""
import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import create_synthetic_data  # noqa: E402
from src.core.simulation import render_frame  # noqa: E402


def test_spot_centred_on_subpixel_position():
    img = render_frame([(20.3, 11.7)], (32, 40), 1000.0, 1.5)
    yy, xx = np.mgrid[:32, :40]
    assert np.isclose((img * xx).sum() / img.sum(), 20.3, atol=1e-6)
    assert np.isclose((img * yy).sum() / img.sum(), 11.7, atol=1e-6)


def test_seeded_and_ground_truth():
    a, ta = create_synthetic_data(10, 4, (48, 48), seed=3, return_ground_truth=True,
                                  motion_types=['brownian', 'directed', 'confined', 'brownian'])
    b, tb = create_synthetic_data(10, 4, (48, 48), seed=3, return_ground_truth=True,
                                  motion_types=['brownian', 'directed', 'confined', 'brownian'])
    assert np.array_equal(a.data, b.data) and ta == tb
    assert [t['motion_type'] for t in ta] == ['brownian', 'directed', 'confined', 'brownian']
    assert all(len(t['x']) == 10 for t in ta)


def test_confined_stays_in_disc():
    _, truth = create_synthetic_data(200, 1, (64, 64), seed=1, return_ground_truth=True, margin=10,
                                     motion_types=['confined'], confinement_radius=2.0)
    x, y = np.array(truth[0]['x']), np.array(truth[0]['y'])
    # the start position is before frame 0; every frame lies within 2 px of it, so within 4 px of each other
    assert np.max(np.hypot(x - x[0], y - y[0])) <= 4.0 + 1e-9
