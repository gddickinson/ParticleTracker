"""Regression test: _extract_track_features (used by classify_track_motion(method='ml')) called np.cross on 2D
vectors, which NumPy 2 rejects (ValueError), so the 'ml' method crashed on every track of 3+ points."""
import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.data_structures import Detection, Track  # noqa: E402
from src.analysis.classification import _extract_track_features, classify_track_motion  # noqa: E402


def test_turning_angles_2d():
    t = Track(0)
    for f, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)] * 3):
        t.add_detection(Detection(x=x + 5, y=y + 5, t=f))
    feats = _extract_track_features(t, 1.0)
    # every turn is 90 degrees except the 'return' steps (0,1)->(0,0)->(1,0) also 90 degrees
    assert np.isclose(feats['mean_turning_angle'], np.pi / 2)
    assert isinstance(classify_track_motion(t, method='ml'), str)
