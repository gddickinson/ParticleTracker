"""Regression tests: every detector the factory documents can be created.

Before the fix, wavelet_detector imported PyWavelets (never used), so the default detector (and the whole test module)
failed to import without it, and create_detector('dog' / 'gaussian') imported modules that do not exist.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.detection.detector_factory import create_detector  # noqa: E402


def _spot_image():
    from scipy.ndimage import gaussian_filter
    img = np.zeros((48, 48), np.float32)
    img[20, 30] = 1000.0
    img = gaussian_filter(img, 1.5)
    return img + np.random.RandomState(0).randn(48, 48).astype(np.float32)


def test_wavelet_importable_without_pywavelets(monkeypatch):
    monkeypatch.setitem(sys.modules, "pywt", None)   # import pywt now raises ImportError
    sys.modules.pop("src.detection.wavelet_detector", None)
    det = create_detector({"method": "wavelet"})
    assert isinstance(det.detect(_spot_image()), list)


@pytest.mark.parametrize("method", ["log", "dog", "gaussian"])
def test_documented_methods_create(method):
    det = create_detector({"method": method})
    assert hasattr(det, "detect")


def test_deep_method_says_not_implemented():
    with pytest.raises(NotImplementedError):
        create_detector({"method": "deep"})
