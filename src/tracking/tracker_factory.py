"""
Factory for creating particle trackers.

This module provides a factory pattern for instantiating different
particle tracking algorithms.
"""

from typing import Dict, List
from abc import ABC, abstractmethod

try:
    from ..core.data_structures import Detection, Track, TrackingResult
except:
    from core.data_structures import Detection, Track, TrackingResult


class BaseTracker(ABC):
    """
    Abstract base class for particle trackers.

    All tracker implementations should inherit from this class
    and implement the track() method.
    """

    def __init__(self, config: Dict):
        """
        Initialize tracker with configuration.

        Args:
            config: Dictionary with tracker parameters
        """
        self.config = config

    @abstractmethod
    def track(self, detections: List[Detection]) -> TrackingResult:
        """
        Link detections into tracks.

        Args:
            detections: List of Detection objects from all frames

        Returns:
            TrackingResult object containing tracks
        """
        pass

    def _organize_by_frame(self, detections: List[Detection]) -> Dict[int, List[Detection]]:
        """
        Organize detections by frame number.

        Args:
            detections: List of all detections

        Returns:
            Dictionary mapping frame_index -> list of detections
        """
        by_frame = {}
        for detection in detections:
            if detection.t not in by_frame:
                by_frame[detection.t] = []
            by_frame[detection.t].append(detection)
        return by_frame


def create_tracker(config: Dict) -> BaseTracker:
    """
    Factory function to create tracker instance.

    Args:
        config: Configuration dictionary with 'method' key

    Returns:
        Tracker instance
    """
    method = config.get('method', 'lap').lower()

    if method == 'lap':
        from .lap_tracker import LAPTracker
        return LAPTracker(config)
    elif method == 'nearest':
        from .nearest_tracker import NearestNeighborTracker
        return NearestNeighborTracker(config)
    elif method == 'mht':
        from .mht_tracker import MHTTracker
        return MHTTracker(config)
    elif method == 'deep':
        from .deep_tracker import DeepLearningTracker
        return DeepLearningTracker(config)
    else:
        raise ValueError(f"Unknown tracking method: {method}")
