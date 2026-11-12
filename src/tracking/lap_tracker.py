"""
Linear Assignment Problem (LAP) based tracker.

This implements the global optimization approach from U-track (Jaqaman et al.),
formulating tracking as a series of LAP solutions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    from .tracker_factory import BaseTracker
    from ..core.data_structures import Detection, Track, TrackingResult
except ImportError:
    from tracker_factory import BaseTracker
    from core.data_structures import Detection, Track, TrackingResult


class LAPTracker(BaseTracker):
    """
    Linear Assignment Problem tracker.

    Implements two-step LAP approach:
    1. Frame-to-frame linking
    2. Gap closing and track segment merging

    Config parameters:
        max_displacement: Maximum distance for linking (pixels) (default: 5.0)
        gap_closing: Maximum frames to close gaps (default: 3)
        merge_splits: Whether to handle merging/splitting (default: True)
        max_gap_displacement: Max displacement for gap closing (default: 10.0)
        motion_model: Motion model ('brownian', 'directed') (default: 'brownian')
        brownian_search_radius_multiplier: Multiplier for search radius (default: 2.0)
    """

    def __init__(self, config: Dict):
        """Initialize LAP tracker."""
        super().__init__(config)

        self.max_displacement = config.get('max_displacement', 5.0)
        self.gap_closing = config.get('gap_closing', 3)
        self.merge_splits = config.get('merge_splits', True)
        self.max_gap_displacement = config.get('max_gap_displacement', 10.0)
        self.motion_model = config.get('motion_model', 'brownian')
        self.search_multiplier = config.get('brownian_search_radius_multiplier', 2.0)

    def track(self, detections: List[Detection]) -> TrackingResult:
        """
        Track particles using two-step LAP.

        Args:
            detections: List of all detections

        Returns:
            TrackingResult with tracks
        """
        if not detections:
            return TrackingResult()

        # Organize detections by frame
        by_frame = self._organize_by_frame(detections)

        # Step 1: Frame-to-frame linking
        track_segments = self._frame_to_frame_linking(by_frame)

        # Step 2: Gap closing and merging
        if self.gap_closing > 0 or self.merge_splits:
            final_tracks = self._gap_closing_and_merging(track_segments, by_frame)
        else:
            final_tracks = track_segments

        # Create result
        result = TrackingResult()
        result.tracks = final_tracks
        result.detections = detections
        result.metadata = {
            'method': 'lap',
            'n_frames': len(by_frame),
            'max_displacement': self.max_displacement,
            'gap_closing': self.gap_closing
        }

        return result

    def _frame_to_frame_linking(self, by_frame: Dict[int, List[Detection]]) -> List[Track]:
        """
        Perform frame-to-frame linking.

        Args:
            by_frame: Detections organized by frame

        Returns:
            List of track segments
        """
        frames = sorted(by_frame.keys())

        # Initialize tracks from first frame
        tracks = []
        track_id = 0

        for detection in by_frame[frames[0]]:
            track = Track(track_id=track_id)
            track.add_detection(detection)
            tracks.append(track)
            track_id += 1

        # Link subsequent frames
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]

            # Get active tracks (those ending in current frame)
            active_tracks = [t for t in tracks if t.end_frame == current_frame]

            if not active_tracks:
                # Start new tracks
                for detection in by_frame[next_frame]:
                    track = Track(track_id=track_id)
                    track.add_detection(detection)
                    tracks.append(track)
                    track_id += 1
                continue

            # Get detections in next frame
            next_detections = by_frame[next_frame]

            if not next_detections:
                continue

            # Solve assignment problem
            assignments = self._solve_lap(active_tracks, next_detections)

            # Apply assignments
            for track_idx, det_idx in assignments:
                if track_idx < len(active_tracks) and det_idx < len(next_detections):
                    tracks[active_tracks[track_idx].track_id].add_detection(
                        next_detections[det_idx]
                    )

            # Create new tracks for unassigned detections
            assigned_dets = {det_idx for _, det_idx in assignments}
            for det_idx, detection in enumerate(next_detections):
                if det_idx not in assigned_dets:
                    track = Track(track_id=track_id)
                    track.add_detection(detection)
                    tracks.append(track)
                    track_id += 1

        return tracks

    def _solve_lap(self, tracks: List[Track],
                   detections: List[Detection]) -> List[Tuple[int, int]]:
        """
        Solve Linear Assignment Problem for one frame.

        Args:
            tracks: List of track segments
            detections: List of detections in next frame

        Returns:
            List of (track_index, detection_index) pairs
        """
        n_tracks = len(tracks)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return []

        # Build cost matrix
        cost_matrix = np.full((n_tracks, n_dets), np.inf)

        for i, track in enumerate(tracks):
            last_detection = track.detections[-1]

            for j, detection in enumerate(detections):
                distance = last_detection.distance_to(detection)

                # Adaptive search radius based on motion model
                if self.motion_model == 'brownian':
                    # For Brownian motion, search radius increases with sqrt(dt)
                    dt = detection.t - last_detection.t
                    search_radius = self.max_displacement * self.search_multiplier * np.sqrt(dt)
                else:
                    search_radius = self.max_displacement

                if distance < search_radius:
                    cost_matrix[i, j] = distance

        # Augmented matrix (Jaqaman et al. 2008): tracks may end (death) and
        # detections may start new tracks (birth), each at cost max_cost / 2.
        max_cost = self.max_displacement * 2
        assignments = solve_augmented_lap(cost_matrix, max_cost * 0.5, max_cost * 0.5)
        assignments = [(r, c) for r, c in assignments if cost_matrix[r, c] < max_cost]

        return assignments

    def _gap_closing_and_merging(self, track_segments: List[Track],
                                 by_frame: Dict[int, List[Detection]]) -> List[Track]:
        """
        Close gaps and merge track segments.

        This is the second LAP step that handles:
        - Gap closing (particle disappears temporarily)
        - Track merging (two tracks are same particle)
        - Track splitting (detection artifacts)

        Args:
            track_segments: Initial track segments
            by_frame: Detections by frame

        Returns:
            Final tracks with gaps closed
        """
        if not track_segments:
            return []

        # Build cost matrix for segment linking
        n_segments = len(track_segments)

        # Create segment start/end lists
        segment_ends = []  # (track_idx, end_frame, end_position)
        segment_starts = []  # (track_idx, start_frame, start_position)

        for i, track in enumerate(track_segments):
            if track.detections:
                end_det = track.detections[-1]
                start_det = track.detections[0]

                segment_ends.append((i, end_det.t, end_det.position))
                segment_starts.append((i, start_det.t, start_det.position))

        # Build cost matrix for gap closing
        n_ends = len(segment_ends)
        n_starts = len(segment_starts)

        if n_ends == 0 or n_starts == 0:
            return track_segments

        gap_cost_matrix = np.full((n_ends, n_starts), np.inf)

        for i, (track_i, end_frame, end_pos) in enumerate(segment_ends):
            for j, (track_j, start_frame, start_pos) in enumerate(segment_starts):
                # Can't link segment to itself
                if track_i == track_j:
                    continue

                # Check temporal consistency (start must be after end)
                gap_frames = start_frame - end_frame

                if gap_frames > 0 and gap_frames <= self.gap_closing:
                    # Calculate predicted position assuming Brownian motion
                    distance = np.linalg.norm(end_pos - start_pos)

                    # Expected displacement for Brownian motion
                    expected_disp = self.max_displacement * np.sqrt(gap_frames)

                    # Cost increases with gap length and distance
                    if distance < self.max_gap_displacement * np.sqrt(gap_frames):
                        gap_cost_matrix[i, j] = distance + gap_frames * 0.5

        # Solve gap closing LAP
        if not np.any(np.isfinite(gap_cost_matrix)):
            return track_segments

        max_gap_cost = self.max_gap_displacement * 2
        links = solve_augmented_lap(gap_cost_matrix, max_gap_cost * 0.5, max_gap_cost * 0.5)

        # Segment index -> index of the segment that continues it
        next_of = {}
        for r, c in links:
            if gap_cost_matrix[r, c] < max_gap_cost:
                next_of[segment_ends[r][0]] = segment_starts[c][0]
        continued = set(next_of.values())

        # Walk each chain from its first segment, so chains of any length and
        # any processing order give one track with every detection used once.
        final_tracks = []
        for i, track in enumerate(track_segments):
            if i in continued:
                continue
            j = i
            while j in next_of:
                nxt = track_segments[next_of[j]]
                for gap_frame in range(track_segments[j].end_frame + 1, nxt.start_frame):
                    track.add_gap(gap_frame)
                for gap_frame in nxt.gap_frames:
                    track.add_gap(gap_frame)
                for det in nxt.detections:
                    track.add_detection(det)
                j = next_of[j]
            final_tracks.append(track)

        return final_tracks


def solve_augmented_lap(cost: np.ndarray, death_cost: float,
                        birth_cost: float) -> List[Tuple[int, int]]:
    """
    Solve a linking LAP with the U-track augmented cost matrix.

    cost[i, j] is the cost of linking row object i to column object j
    (np.inf = forbidden). Row i may instead be left unlinked at death_cost and
    column j at birth_cost. The off-diagonal entries of the birth and death
    blocks are forbidden (a dummy may only stand in for its own object), and
    the lower-right block is 0, so the total cost is the sum of the chosen
    links plus the birth/death costs of everything left unlinked.

    Returns:
        List of (row, column) pairs that are real links.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    big = 1e12  # stands in for "forbidden"; linear_sum_assignment needs a feasible finite path
    finite = np.where(np.isfinite(cost), cost, big)
    death = np.full((n_rows, n_rows), big)
    np.fill_diagonal(death, death_cost)
    birth = np.full((n_cols, n_cols), big)
    np.fill_diagonal(birth, birth_cost)
    augmented = np.block([[finite, death], [birth, np.zeros((n_cols, n_rows))]])
    rows, cols = linear_sum_assignment(augmented)
    return [(int(r), int(c)) for r, c in zip(rows, cols)
            if r < n_rows and c < n_cols and np.isfinite(cost[r, c])]
