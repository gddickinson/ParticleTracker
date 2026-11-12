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

    Frame by frame, every (track, detection) pair within reach is linked in
    order of increasing cost, each track and detection at most once (greedy
    nearest-neighbour matching; LAPTracker solves the same assignment
    optimally). Fast but less robust than LAP for dense fields.

    Gap closing, same convention as LAPTracker: a track whose last detection
    is k frames back (k = frame difference, 1 for consecutive frames) can be
    continued when k <= max(1, gap_closing) and the distance is below
    max_gap_displacement * sqrt(k). The cost is distance + 0.5 * (k - 1), so
    a consecutive-frame link wins over a gap link at the same distance.
    Missing frames are recorded in Track.gap_frames.

    Config parameters:
        max_displacement: Maximum linking distance between consecutive frames
            (default: 5.0)
        gap_closing: Largest frame difference a link may span (default: 0,
            consecutive frames only; 3 bridges up to 2 missing frames)
        max_gap_displacement: Distance scale for gap links (default:
            max_displacement)
    """

    def __init__(self, config: Dict):
        """Initialize nearest neighbor tracker."""
        super().__init__(config)

        self.max_displacement = config.get('max_displacement', 5.0)
        self.gap_closing = int(config.get('gap_closing', 0) or 0)
        self.max_gap_displacement = config.get('max_gap_displacement', self.max_displacement)

    def _reach(self, frame_diff: int) -> float:
        if frame_diff == 1:
            return self.max_displacement
        return self.max_gap_displacement * np.sqrt(frame_diff)

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

        by_frame = self._organize_by_frame(detections)
        max_diff = max(1, self.gap_closing)
        tracks: List[Track] = []
        open_tracks: List[Track] = []     # tracks that may still be continued

        for frame in sorted(by_frame.keys()):
            current = by_frame[frame]
            open_tracks = [t for t in open_tracks if frame - t.detections[-1].t <= max_diff]
            assigned_dets = set()
            if open_tracks and current:
                last = np.array([t.detections[-1].position for t in open_tracks])
                diffs = np.array([frame - t.detections[-1].t for t in open_tracks])
                dist = cdist(last, np.array([d.position for d in current]))
                reach = np.array([self._reach(int(k)) for k in diffs])
                cost = dist + 0.5 * (diffs - 1)[:, None]
                rows, cols = np.nonzero(dist < reach[:, None])
                order = np.lexsort((cols, rows, cost[rows, cols]))
                linked = set()
                for k in order:
                    r, c = int(rows[k]), int(cols[k])
                    if r in linked or c in assigned_dets:
                        continue
                    trk = open_tracks[r]
                    for gap in range(trk.detections[-1].t + 1, frame):
                        trk.add_gap(gap)
                    trk.add_detection(current[c])
                    linked.add(r)
                    assigned_dets.add(c)
            for c, det in enumerate(current):
                if c not in assigned_dets:
                    trk = Track(track_id=len(tracks))
                    trk.add_detection(det)
                    tracks.append(trk)
                    open_tracks.append(trk)

        result = TrackingResult()
        result.tracks = tracks
        result.detections = detections
        result.metadata = {
            'method': 'nearest_neighbor',
            'max_displacement': self.max_displacement,
            'gap_closing': self.gap_closing,
            'max_gap_displacement': self.max_gap_displacement
        }

        return result
