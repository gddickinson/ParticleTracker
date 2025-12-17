"""
Motion analysis tools.

This module provides functions for analyzing particle trajectories,
including mean squared displacement (MSD) calculation and diffusion
coefficient estimation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import curve_fit

try:
    from ..core.data_structures import Track
except:
    from core.data_structures import Track


def calculate_msd(tracks: List[Track],
                  max_lag: int = 50,
                  time_interval: float = 1.0) -> Dict:
    """
    Calculate mean squared displacement for tracks.

    MSD(Δt) = <[r(t+Δt) - r(t)]²>

    Args:
        tracks: List of Track objects
        max_lag: Maximum time lag to calculate
        time_interval: Time between frames (seconds)

    Returns:
        Dictionary with MSD results
    """
    # Collect all MSDs
    all_msds = []
    lags = np.arange(1, max_lag + 1)

    for track in tracks:
        if track.length < 5:  # Skip very short tracks
            continue

        positions = track.positions
        times = track.times

        # Calculate MSD for this track
        track_msd = np.zeros(len(lags))
        counts = np.zeros(len(lags))

        for i, lag in enumerate(lags):
            # All pairs separated by 'lag' frames
            displacements_squared = []

            for j in range(len(positions) - lag):
                # Check if times are actually separated by lag (handle gaps)
                time_diff = times[j + lag] - times[j]
                if time_diff == lag:  # No gaps
                    displacement = positions[j + lag] - positions[j]
                    disp_squared = np.sum(displacement**2)
                    displacements_squared.append(disp_squared)

            if displacements_squared:
                track_msd[i] = np.mean(displacements_squared)
                counts[i] = len(displacements_squared)

        if np.any(counts > 0):
            all_msds.append(track_msd)

    if not all_msds:
        return {
            'lags': lags,
            'time_lags': lags * time_interval,
            'mean_msd': np.zeros_like(lags),
            'std_msd': np.zeros_like(lags),
            'n_tracks': 0
        }

    # Compute ensemble average
    all_msds = np.array(all_msds)
    mean_msd = np.mean(all_msds, axis=0)
    std_msd = np.std(all_msds, axis=0)

    return {
        'lags': lags,
        'time_lags': lags * time_interval,
        'mean_msd': mean_msd,
        'std_msd': std_msd,
        'individual_msds': all_msds,
        'n_tracks': len(all_msds)
    }


def fit_diffusion_model(msd_results: Dict,
                        model: str = 'brownian') -> Dict:
    """
    Fit diffusion model to MSD data.

    Models:
        - 'brownian': MSD = 4Dt (2D) or 6Dt (3D)
        - 'directed': MSD = 4Dt + (vt)²
        - 'confined': MSD = A(1 - exp(-4Dt/A))
        - 'anomalous': MSD = 4Dt^α

    Args:
        msd_results: Output from calculate_msd()
        model: Diffusion model to fit

    Returns:
        Dictionary with fitted parameters
    """
    time_lags = msd_results['time_lags']
    mean_msd = msd_results['mean_msd']

    # Remove zeros and nans
    valid = (mean_msd > 0) & np.isfinite(mean_msd)
    time_lags = time_lags[valid]
    mean_msd = mean_msd[valid]

    if len(time_lags) < 3:
        return {'success': False, 'message': 'Insufficient data for fitting'}

    try:
        if model == 'brownian':
            # MSD = 4Dt for 2D, 6Dt for 3D
            # Fit: MSD = slope * t
            slope, intercept = np.polyfit(time_lags, mean_msd, 1)
            D = slope / 4.0  # Assume 2D

            return {
                'success': True,
                'model': 'brownian',
                'diffusion_coefficient': D,
                'slope': slope,
                'intercept': intercept,
                'r_squared': _calculate_r_squared(mean_msd, slope * time_lags + intercept)
            }

        elif model == 'directed':
            # MSD = 4Dt + (vt)²
            def directed_msd(t, D, v):
                return 4 * D * t + (v * t)**2

            popt, pcov = curve_fit(
                directed_msd, time_lags, mean_msd,
                p0=[1.0, 0.1],
                bounds=([0, 0], [np.inf, np.inf])
            )

            D, v = popt

            return {
                'success': True,
                'model': 'directed',
                'diffusion_coefficient': D,
                'velocity': v,
                'param_errors': np.sqrt(np.diag(pcov)),
                'r_squared': _calculate_r_squared(mean_msd, directed_msd(time_lags, *popt))
            }

        elif model == 'confined':
            # MSD = A(1 - exp(-4Dt/A))
            def confined_msd(t, D, A):
                return A * (1 - np.exp(-4 * D * t / A))

            # Initial guess
            A_init = mean_msd[-1] * 1.5  # Plateau value
            D_init = mean_msd[1] / (4 * time_lags[1])

            popt, pcov = curve_fit(
                confined_msd, time_lags, mean_msd,
                p0=[D_init, A_init],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=5000
            )

            D, A = popt

            return {
                'success': True,
                'model': 'confined',
                'diffusion_coefficient': D,
                'confinement_area': A,
                'param_errors': np.sqrt(np.diag(pcov)),
                'r_squared': _calculate_r_squared(mean_msd, confined_msd(time_lags, *popt))
            }

        elif model == 'anomalous':
            # MSD = 4Dt^α
            # Fit in log space: log(MSD) = log(4D) + α*log(t)
            log_t = np.log(time_lags)
            log_msd = np.log(mean_msd)

            alpha, log_4D = np.polyfit(log_t, log_msd, 1)
            D = np.exp(log_4D) / 4.0

            return {
                'success': True,
                'model': 'anomalous',
                'diffusion_coefficient': D,
                'alpha': alpha,
                'motion_type': _classify_alpha(alpha),
                'r_squared': _calculate_r_squared(log_msd, alpha * log_t + log_4D)
            }

        else:
            return {'success': False, 'message': f'Unknown model: {model}'}

    except Exception as e:
        return {'success': False, 'message': str(e)}


def _calculate_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R² value."""
    ss_res = np.sum((observed - predicted)**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _classify_alpha(alpha: float) -> str:
    """Classify motion based on anomalous diffusion exponent."""
    if alpha < 0.9:
        return 'subdiffusive'
    elif alpha > 1.1:
        return 'superdiffusive'
    else:
        return 'normal_diffusion'


def calculate_velocity(track: Track, time_interval: float = 1.0) -> Dict:
    """
    Calculate instantaneous and mean velocity for a track.

    Args:
        track: Track object
        time_interval: Time between frames

    Returns:
        Dictionary with velocity statistics
    """
    if track.length < 2:
        return {'mean_velocity': 0.0, 'std_velocity': 0.0}

    positions = track.positions
    times = track.times

    # Calculate instantaneous velocities
    velocities = []

    for i in range(len(positions) - 1):
        dt = (times[i+1] - times[i]) * time_interval
        if dt > 0:
            displacement = positions[i+1] - positions[i]
            velocity = np.linalg.norm(displacement) / dt
            velocities.append(velocity)

    if not velocities:
        return {'mean_velocity': 0.0, 'std_velocity': 0.0}

    velocities = np.array(velocities)

    return {
        'mean_velocity': np.mean(velocities),
        'std_velocity': np.std(velocities),
        'median_velocity': np.median(velocities),
        'max_velocity': np.max(velocities),
        'velocities': velocities
    }


def calculate_directionality(track: Track) -> Dict:
    """
    Calculate track directionality metrics.

    Args:
        track: Track object

    Returns:
        Dictionary with directionality metrics
    """
    if track.length < 3:
        return {
            'directionality_ratio': 0.0,
            'straightness': 0.0,
            'net_displacement': 0.0
        }

    positions = track.positions

    # Net displacement
    net_displacement = np.linalg.norm(positions[-1] - positions[0])

    # Total path length
    path_length = 0.0
    for i in range(len(positions) - 1):
        path_length += np.linalg.norm(positions[i+1] - positions[i])

    # Directionality ratio (net / total)
    directionality = net_displacement / path_length if path_length > 0 else 0.0

    # Straightness (similar but different normalization)
    straightness = directionality

    return {
        'directionality_ratio': directionality,
        'straightness': straightness,
        'net_displacement': net_displacement,
        'path_length': path_length
    }


def detect_confinement(track: Track,
                       window_size: int = 10) -> Dict:
    """
    Detect confined motion in trajectory.

    Uses sliding window to detect confined regions.

    Args:
        track: Track object
        window_size: Window size for detection

    Returns:
        Dictionary with confinement metrics
    """
    if track.length < window_size:
        return {'is_confined': False, 'confinement_probability': 0.0}

    positions = track.positions

    # Calculate local mean positions in sliding windows
    is_confined = []

    for i in range(len(positions) - window_size + 1):
        window = positions[i:i+window_size]

        # Calculate radius of gyration
        center = np.mean(window, axis=0)
        distances = np.linalg.norm(window - center, axis=1)
        rg = np.sqrt(np.mean(distances**2))

        # Expected Rg for Brownian motion
        # For confined motion, Rg should plateau

        # Simple criterion: check if radius is small
        is_confined.append(rg < 2.0)  # Threshold in pixels

    confinement_prob = np.mean(is_confined) if is_confined else 0.0

    return {
        'is_confined': confinement_prob > 0.7,
        'confinement_probability': confinement_prob
    }


def classify_motion_type(track: Track,
                        time_interval: float = 1.0) -> str:
    """
    Classify overall motion type of a track.

    Args:
        track: Track object
        time_interval: Time between frames

    Returns:
        Motion type string: 'brownian', 'directed', 'confined', 'subdiffusive'
    """
    if track.length < 10:
        return 'insufficient_data'

    # Calculate MSD for this track
    max_lag = min(track.length // 2, 20)
    msd_result = calculate_msd([track], max_lag=max_lag, time_interval=time_interval)

    # Fit different models
    brownian_fit = fit_diffusion_model(msd_result, model='brownian')
    directed_fit = fit_diffusion_model(msd_result, model='directed')
    anomalous_fit = fit_diffusion_model(msd_result, model='anomalous')

    # Get directionality
    directionality = calculate_directionality(track)

    # Decision logic
    if anomalous_fit['success']:
        alpha = anomalous_fit['alpha']
        if alpha < 0.9:
            return 'subdiffusive'
        elif alpha > 1.1:
            return 'superdiffusive'

    if directionality['directionality_ratio'] > 0.7:
        return 'directed'

    confinement = detect_confinement(track)
    if confinement['is_confined']:
        return 'confined'

    return 'brownian'
