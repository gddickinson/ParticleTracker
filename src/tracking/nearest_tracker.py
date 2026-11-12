"""
Nearest neighbor tracker - simple and fast baseline method.

Links particles to their nearest neighbor in the next frame within
a search radius.
"""

import numpy as np
from typing import List, Dict
from scipy.spatial.distance import cdist

try:
    from .tracker_factory import BaseTracker
    from ..core.data_structures import Detection, Track, TrackingResult
except ImportError:
    from tracking.tracker_factory import BaseTracker
    from core.data_structures import Detection, Track, TrackingResult


class NearestNeighborTracker(BaseTracker):
    """
    Nearest neighbor linking tracker.

    Simple frame-to-frame linking based on nearest neighbor within
    a search radius. Fast but less robust than LAP for dense fields.

    Config parameters:
        max_displacement: Maximum linking distance (default: 5.0)
        gap_closing: Frames to interpolate gaps (default: 0)
    """

    def __init__(self, config: Dict):
        """Initialize nearest neighbor tracker."""
        super().__init__(config)

        self.max_displacement = config.get('max_displacement', 5.0)
        self.gap_closing = config.get('gap_closing', 0)

    def track(self, detections: List[Detection]) -> TrackingResult:
        """
        Track using nearest neighbor linking.

        Args:
            detections: List of all detections

        Returns:
            TrackingResult
        """
        if not detections:
            return TrackingResult()

        # Organize by frame
        by_frame = self._organize_by_frame(detections)
        frames = sorted(by_frame.keys())

        # Initialize tracks
        tracks = []
        track_id = 0

        # Start tracks from first frame
        active_tracks = {}  # Maps track_id to Track object

        for detection in by_frame[frames[0]]:
            track = Track(track_id=track_id)
            track.add_detection(detection)
            active_tracks[track_id] = track
            tracks.append(track)
            track_id += 1

        # Process subsequent frames
        for i in range(1, len(frames)):
            frame = frames[i]
            current_detections = by_frame[frame]

            if not current_detections:
                continue

            # Get last positions of active tracks
            track_ids = list(active_tracks.keys())
            if not track_ids:
                # No active tracks, start new ones
                for detection in current_detections:
                    track = Track(track_id=track_id)
                    track.add_detection(detection)
                    active_tracks[track_id] = track
                    tracks.append(track)
                    track_id += 1
                continue

            last_positions = np.array([
                active_tracks[tid].detections[-1].position
                for tid in track_ids
            ])

            current_positions = np.array([
                det.position for det in current_detections
            ])

            # Calculate distance matrix
            distances = cdist(last_positions, current_positions)

            # Find nearest neighbors
            assigned_tracks = set()
            assigned_dets = set()

            for track_idx in range(len(track_ids)):
                # Find nearest detection
                det_idx = np.argmin(distances[track_idx])
                min_dist = distances[track_idx, det_idx]

                # Check if within search radius and not already assigned
                if min_dist < self.max_displacement and det_idx not in assigned_dets:
                    # Link to track
                    tid = track_ids[track_idx]
                    active_tracks[tid].add_detection(current_detections[det_idx])
                    assigned_tracks.add(track_idx)
                    assigned_dets.add(det_idx)

            # Remove tracks that weren't linked (end them)
            for track_idx, tid in enumerate(track_ids):
                if track_idx not in assigned_tracks:
                    del active_tracks[tid]

            # Start new tracks for unassigned detections
            for det_idx, detection in enumerate(current_detections):
                if det_idx not in assigned_dets:
                    track = Track(track_id=track_id)
                    track.add_detection(detection)
                    active_tracks[track_id] = track
                    tracks.append(track)
                    track_id += 1

        # Create result
        result = TrackingResult()
        result.tracks = tracks
        result.detections = detections
        result.metadata = {
            'method': 'nearest_neighbor',
            'max_displacement': self.max_displacement
        }

        return result
