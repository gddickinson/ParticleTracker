"""Regression tests for detect_confinement.

Before the fix a track was 'confined' when the radius of gyration of more than 70 % of its 10-point windows was
below a fixed 2 pixels. That threshold ignores how fast the particle moves: free diffusion slower than about
0.3 px^2/frame (common: 0.05 um^2/s at 0.1 um/px and 50 ms frames is 0.25 px^2/frame) was called confined, and
fast motion in a corral wider than a few pixels never was. The default is now the Saxton (1993) / Simson et al.
(1995) confinement level, which is dimensionless and does not depend on the speed, pixel size or track length for
free diffusion.
"""
import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import Detection, Track  # noqa: E402
from src.analysis.motion import detect_confinement  # noqa: E402
from src.analysis.confinement import confinement_level  # noqa: E402


def _walk(rng, n, D, radius=None, start=(100.0, 100.0)):
    p = np.array(start, float)
    out = [p.copy()]
    for _ in range(n - 1):
        q = p + rng.randn(2) * np.sqrt(2 * D)
        if radius is not None:
            while np.hypot(*(q - start)) > radius:   # reject steps leaving the corral
                q = p + rng.randn(2) * np.sqrt(2 * D)
        p = q
        out.append(p.copy())
    return np.array(out)


def _track(xy, tid=0):
    t = Track(tid)
    for f, (x, y) in enumerate(xy):
        t.add_detection(Detection(x=x, y=y, t=f))
    return t


def _fraction_confined(D, n, radius=None, k=150, seed=1):
    rng = np.random.RandomState(seed)
    return np.mean([detect_confinement(_track(_walk(rng, n, D, radius)))['is_confined'] for _ in range(k)])


def test_slow_free_diffusion_not_confined():
    assert _fraction_confined(D=0.25, n=100) < 0.05
    assert _fraction_confined(D=0.25, n=40) < 0.05


def test_fast_motion_in_wide_corral_confined():
    assert _fraction_confined(D=2.0, n=100, radius=5.0) > 0.9


def test_level_does_not_depend_on_scale():
    rng = np.random.RandomState(5)
    xy = _walk(rng, 60, 1.0)
    f = np.arange(60)
    assert np.isclose(confinement_level(f, xy), confinement_level(f, xy * 0.107))


def test_legacy_rule_still_available():
    rng = np.random.RandomState(1)
    r = detect_confinement(_track(_walk(rng, 100, 0.25)), method='rg_legacy')
    assert 'confinement_probability' in r and r['method'] == 'rg_legacy'
