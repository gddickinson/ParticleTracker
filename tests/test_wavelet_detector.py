"""Tests for wavelet detector."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Import via the src package to keep relative imports working
_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.core.data_structures import Detection
from src.detection.wavelet_detector import WaveletDetector


def _make_test_image_with_spots(n_spots=3, size=128, sigma=1.5, intensity=1000.0):
    """Create a test image with Gaussian spots at known locations."""
    from scipy.ndimage import gaussian_filter

    image = np.zeros((size, size), dtype=np.float32)
    positions = []

    rng = np.random.RandomState(42)
    for _ in range(n_spots):
        y = rng.randint(10, size - 10)
        x = rng.randint(10, size - 10)
        image[y, x] = intensity
        positions.append((x, y))

    image = gaussian_filter(image, sigma=sigma)
    image += rng.randn(size, size).astype(np.float32) * 10.0

    return image, positions


class TestWaveletDetector:
    """Tests for WaveletDetector."""

    def test_instantiation(self):
        config = {'sigma': 1.5, 'threshold': 3.0}
        detector = WaveletDetector(config)
        assert detector.sigma == 1.5
        assert detector.threshold == 3.0

    def test_detect_returns_list(self):
        config = {'sigma': 1.5, 'threshold': 2.0, 'subpixel': False}
        detector = WaveletDetector(config)
        image, _ = _make_test_image_with_spots(n_spots=3, intensity=5000.0)
        detections = detector.detect(image, frame_index=0)
        assert isinstance(detections, list)
        for d in detections:
            assert isinstance(d, Detection)

    def test_detects_bright_spots(self):
        config = {'sigma': 1.5, 'threshold': 2.0, 'subpixel': False}
        detector = WaveletDetector(config)
        image, positions = _make_test_image_with_spots(
            n_spots=3, intensity=10000.0, sigma=1.5
        )
        detections = detector.detect(image, frame_index=5)
        # Should find at least some detections
        assert len(detections) > 0
        # Frame index should be set
        for d in detections:
            assert d.t == 5

    def test_empty_image(self):
        config = {'sigma': 1.5, 'threshold': 5.0, 'subpixel': False}
        detector = WaveletDetector(config)
        image = np.zeros((64, 64), dtype=np.float32)
        detections = detector.detect(image, frame_index=0)
        assert len(detections) == 0

    def test_noise_estimation(self):
        config = {'sigma': 1.5}
        detector = WaveletDetector(config)
        noise = np.random.randn(100, 100).astype(np.float32) * 10.0
        estimated = detector._estimate_noise(noise)
        # Should be roughly 10.0
        assert 5.0 < estimated < 20.0


def test_mexican_hat_finds_bright_spot_and_kernel_is_zero_sum():
    """Regression: the kernel (1 - r^2/s^2) exp(-r^2/2s^2) summed to a negative number, so bright spots gave a
    negative response and the detector reported noise/edges instead (the spot at (30, 20) was missed)."""
    from scipy.ndimage import gaussian_filter
    img = np.zeros((48, 48), np.float32)
    img[20, 30] = 1000.0
    img = gaussian_filter(img, 1.5) + np.random.RandomState(0).randn(48, 48).astype(np.float32)
    det = WaveletDetector({})
    w = det._apply_wavelet_transform(img)
    assert np.unravel_index(np.argmax(w), w.shape) == (20, 30)
    found = det.detect(img)
    assert any(abs(d.x - 30) < 0.2 and abs(d.y - 20) < 0.2 for d in found)


def test_mexican_hat_kernel_symmetric():
    """The kernel must be symmetric about its centre so the response peak is not shifted."""
    det = WaveletDetector({})
    k = det._mexican_hat_kernel(1.5)
    assert k.shape[0] % 2 == 1 and np.allclose(k, k[::-1, ::-1]) and abs(k.sum()) < 1e-6
