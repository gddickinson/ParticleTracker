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
except:
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

        # Add dummy rows/columns for birth/death
        # Augmented matrix allows tracks to end and detections to start new tracks
        max_cost = self.max_displacement * 2

        augmented = np.full((n_tracks + n_dets, n_tracks + n_dets), max_cost)
        augmented[:n_tracks, :n_dets] = cost_matrix

        # Diagonal blocks for birth/death
        augmented[n_tracks:, :n_dets] = np.eye(n_dets) * max_cost * 0.5
        augmented[:n_tracks, n_dets:] = np.eye(n_tracks) * max_cost * 0.5

        # Solve using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(augmented)

        # Extract real assignments (not dummy)
        assignments = []
        for r, c in zip(row_ind, col_ind):
            if r < n_tracks and c < n_dets:
                if augmented[r, c] < max_cost:
                    assignments.append((r, c))

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
        if np.any(np.isfinite(gap_cost_matrix)):
            # Add dummy variables for segments that don't link
            max_gap_cost = self.max_gap_displacement * 2

            augmented = np.full((n_ends + n_starts, n_ends + n_starts), max_gap_cost)
            augmented[:n_ends, :n_starts] = gap_cost_matrix

            # Diagonal blocks
            augmented[n_ends:, :n_starts] = np.eye(n_starts) * max_gap_cost * 0.5
            augmented[:n_ends, n_starts:] = np.eye(n_ends) * max_gap_cost * 0.5

            row_ind, col_ind = linear_sum_assignment(augmented)

            # Apply gap closing
            merged_tracks = {}  # Maps segment index to final track
            next_track_id = max(t.track_id for t in track_segments) + 1

            for r, c in zip(row_ind, col_ind):
                if r < n_ends and c < n_starts:
                    if augmented[r, c] < max_gap_cost:
                        end_idx = segment_ends[r][0]
                        start_idx = segment_starts[c][0]

                        # Merge these segments
                        if end_idx not in merged_tracks:
                            merged_tracks[end_idx] = track_segments[end_idx]

                        # Add gap frames
                        gap_frames_list = list(range(
                            track_segments[end_idx].end_frame + 1,
                            track_segments[start_idx].start_frame
                        ))
                        for gap_frame in gap_frames_list:
                            merged_tracks[end_idx].add_gap(gap_frame)

                        # Add detections from second segment
                        for det in track_segments[start_idx].detections:
                            merged_tracks[end_idx].add_detection(det)

                        # Mark start segment as merged
                        merged_tracks[start_idx] = merged_tracks[end_idx]

            # Collect final tracks (keeping only unique tracks)
            final_tracks = []
            seen_ids = set()

            for i, track in enumerate(track_segments):
                if i in merged_tracks:
                    final_track = merged_tracks[i]
                    if id(final_track) not in seen_ids:
                        seen_ids.add(id(final_track))
                        final_tracks.append(final_track)
                else:
                    # Unmerged segment
                    final_tracks.append(track)

            return final_tracks
        else:
            return track_segments
