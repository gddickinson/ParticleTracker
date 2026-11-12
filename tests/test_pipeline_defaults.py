"""Regression test: ParticleTracker.detect(method=...) must not apply another detector's defaults.

Before the fix detect() merged the configured defaults (the wavelet detector's: threshold 3.0 in noise standard
deviations, sigma 1.5, min_distance 3) into every method's parameters. LoG and DoG read 'threshold' on a 0-1
normalised image, so tracker.detect(method='log') or ('dog') found nothing on any image.
"""
import logging
import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.core.pipeline import ParticleTracker  # noqa: E402
from src.core.simulation import create_synthetic_data  # noqa: E402

logging.disable(logging.INFO)


def _tracker():
    pt = ParticleTracker()
    pt.image_stack = create_synthetic_data(3, 6, (64, 64), seed=2, margin=8, background=100)
    return pt


def test_log_and_dog_use_their_own_defaults():
    for method in ("log", "dog"):
        pt = _tracker()
        assert len(pt.detect(method=method)) >= 3 * 4, method


def test_wavelet_still_uses_configured_defaults():
    pt = _tracker()
    pt.config['detection']['threshold'] = 1e9
    assert pt.detect(method='wavelet') == []
