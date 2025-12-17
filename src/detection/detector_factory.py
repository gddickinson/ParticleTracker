"""
Factory for creating particle detectors.

This module provides a factory pattern for instantiating different
particle detection algorithms.
"""

from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod

try:
    from ..core.data_structures import Detection
except:
    from core.data_structures import Detection


class BaseDetector(ABC):
    """
    Abstract base class for particle detectors.

    All detector implementations should inherit from this class
    and implement the detect() method.
    """

    def __init__(self, config: Dict):
        """
        Initialize detector with configuration.

        Args:
            config: Dictionary with detector parameters
        """
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Detect particles in a single image frame.

        Args:
            image: 2D or 3D numpy array
            frame_index: Time index of this frame

        Returns:
            List of Detection objects
        """
        pass

    def _apply_threshold(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """Apply threshold to detection map."""
        return image > threshold

    def _find_peaks(self, image: np.ndarray,
                   threshold: float,
                   min_distance: int = 3) -> np.ndarray:
        """
        Find local maxima in image.

        Args:
            image: 2D numpy array
            threshold: Minimum peak value
            min_distance: Minimum distance between peaks

        Returns:
            Array of peak coordinates (N x 2)
        """
        from skimage.feature import peak_local_max

        coordinates = peak_local_max(
            image,
            min_distance=min_distance,
            threshold_abs=threshold
        )

        return coordinates


def create_detector(config: Dict) -> BaseDetector:
    """
    Factory function to create detector instance.

    Args:
        config: Configuration dictionary with 'method' key

    Returns:
        Detector instance
    """
    method = config.get('method', 'wavelet').lower()

    if method == 'wavelet':
        from .wavelet_detector import WaveletDetector
        return WaveletDetector(config)
    elif method == 'log':
        from .log_detector import LoGDetector
        return LoGDetector(config)
    elif method == 'dog':
        from .dog_detector import DoGDetector
        return DoGDetector(config)
    elif method == 'deep':
        from .deep_detector import DeepDetector
        return DeepDetector(config)
    elif method == 'gaussian':
        from .gaussian_detector import GaussianFitDetector
        return GaussianFitDetector(config)
    else:
        raise ValueError(f"Unknown detection method: {method}")
