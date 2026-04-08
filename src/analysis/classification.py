"""
Track motion classification module.

Provides methods for classifying particle motion types.
"""

import numpy as np
from typing import Dict, List

try:
    from ..core.data_structures import Track
except ImportError:
    from core.data_structures import Track

try:
    from .motion import (
        calculate_msd,
        fit_diffusion_model,
        calculate_directionality,
        detect_confinement
    )
except ImportError:
    from motion import (
        calculate_msd,
        fit_diffusion_model,
        calculate_directionality,
        detect_confinement
    )


def classify_track_motion(track: Track,
                         method: str = 'msd',
                         time_interval: float = 1.0) -> str:
    """
    Classify motion type of a single track.

    Args:
        track: Track object
        method: Classification method ('msd', 'ml', 'hmm')
        time_interval: Time between frames

    Returns:
        Motion class string
    """
    if method == 'msd':
        return _classify_by_msd(track, time_interval)
    elif method == 'ml':
        return _classify_by_ml(track, time_interval)
    elif method == 'hmm':
        return _classify_by_hmm(track, time_interval)
    else:
        raise ValueError(f"Unknown classification method: {method}")


def _classify_by_msd(track: Track, time_interval: float) -> str:
    """Classify using MSD analysis."""
    if track.length < 10:
        return 'insufficient_data'

    # Calculate MSD
    max_lag = min(track.length // 2, 20)
    msd_result = calculate_msd([track], max_lag=max_lag, time_interval=time_interval)

    # Try different models
    brownian_fit = fit_diffusion_model(msd_result, model='brownian')
    directed_fit = fit_diffusion_model(msd_result, model='directed')
    anomalous_fit = fit_diffusion_model(msd_result, model='anomalous')

    # Get directionality
    directionality = calculate_directionality(track)
    confinement = detect_confinement(track)

    # Decision tree
    if confinement['is_confined']:
        return 'confined'

    if anomalous_fit.get('success', False):
        alpha = anomalous_fit['alpha']
        if alpha < 0.85:
            return 'subdiffusive'
        elif alpha > 1.15:
            return 'superdiffusive'

    if directionality['directionality_ratio'] > 0.7:
        return 'directed'

    # Check if directed fit is significantly better than brownian
    if (directed_fit.get('success', False) and
        brownian_fit.get('success', False)):
        r2_directed = directed_fit.get('r_squared', 0)
        r2_brownian = brownian_fit.get('r_squared', 0)

        if r2_directed > r2_brownian + 0.1:  # Significant improvement
            return 'directed'

    return 'brownian'


def _classify_by_ml(track: Track, time_interval: float) -> str:
    """
    Classify using machine learning features.

    This would use pre-trained classifier on trajectory features.
    Placeholder for now.
    """
    # Extract features
    features = _extract_track_features(track, time_interval)

    # TODO: Apply trained classifier
    # For now, fall back to MSD-based
    return _classify_by_msd(track, time_interval)


def _classify_by_hmm(track: Track, time_interval: float) -> str:
    """
    Classify using Hidden Markov Model.

    Can detect state transitions (e.g., switching between diffusive and directed).
    Placeholder for now.
    """
    # TODO: Implement HMM-based classification
    return _classify_by_msd(track, time_interval)


def _extract_track_features(track: Track, time_interval: float) -> Dict:
    """
    Extract features for machine learning classification.

    Returns:
        Dictionary of features
    """
    from .motion import calculate_velocity, calculate_directionality

    features = {}

    # Basic features
    features['length'] = track.length
    features['duration'] = track.duration * time_interval

    # Velocity features
    velocity_stats = calculate_velocity(track, time_interval)
    features['mean_velocity'] = velocity_stats['mean_velocity']
    features['std_velocity'] = velocity_stats['std_velocity']

    # Directionality
    dir_stats = calculate_directionality(track)
    features['directionality'] = dir_stats['directionality_ratio']
    features['net_displacement'] = dir_stats['net_displacement']
    features['path_length'] = dir_stats['path_length']

    # MSD-based features
    if track.length >= 10:
        max_lag = min(track.length // 2, 10)
        msd_result = calculate_msd([track], max_lag=max_lag, time_interval=time_interval)

        # Fit anomalous diffusion
        anomalous_fit = fit_diffusion_model(msd_result, model='anomalous')
        if anomalous_fit.get('success', False):
            features['alpha'] = anomalous_fit['alpha']
            features['diffusion_coeff'] = anomalous_fit['diffusion_coefficient']
        else:
            features['alpha'] = 1.0
            features['diffusion_coeff'] = 0.0
    else:
        features['alpha'] = 1.0
        features['diffusion_coeff'] = 0.0

    # Turning angle statistics
    if track.length >= 3:
        positions = track.positions
        angles = []

        for i in range(1, len(positions) - 1):
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]

            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            angles.append(abs(angle))

        features['mean_turning_angle'] = np.mean(angles) if angles else 0
        features['std_turning_angle'] = np.std(angles) if angles else 0
    else:
        features['mean_turning_angle'] = 0
        features['std_turning_angle'] = 0

    return features


def batch_classify_tracks(tracks: List[Track],
                          method: str = 'msd',
                          time_interval: float = 1.0) -> Dict[int, str]:
    """
    Classify multiple tracks.

    Args:
        tracks: List of Track objects
        method: Classification method
        time_interval: Time between frames

    Returns:
        Dictionary mapping track_id to motion class
    """
    classifications = {}

    for track in tracks:
        motion_class = classify_track_motion(track, method, time_interval)
        classifications[track.track_id] = motion_class

    return classifications


def get_classification_summary(classifications: Dict[int, str]) -> Dict[str, int]:
    """
    Get summary statistics of classifications.

    Args:
        classifications: Dictionary of track_id -> class

    Returns:
        Dictionary of class -> count
    """
    summary = {}

    for class_name in classifications.values():
        summary[class_name] = summary.get(class_name, 0) + 1

    return summary
