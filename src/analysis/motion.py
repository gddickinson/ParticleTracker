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
    from .msd import frame_msd
    from .confinement import confinement_level, CONFINEMENT_LEVEL_THRESHOLD
except ImportError:
    from core.data_structures import Track
    from analysis.msd import frame_msd
    from analysis.confinement import confinement_level, CONFINEMENT_LEVEL_THRESHOLD


def calculate_msd(tracks: List[Track],
                  max_lag: int = 50,
                  time_interval: Optional[float] = None,
                  pixel_size: Optional[float] = None) -> Dict:
    """
    Calculate mean squared displacement for tracks.

    MSD(Δt) = <[r(t+Δt) - r(t)]²>

    Lags are in frames: pairs of localisations exactly `lag` frames apart are
    used wherever they sit in the track, so tracks with gaps contribute every
    valid pair. The ensemble mean at each lag averages only the tracks that
    have at least one pair at that lag.

    Args:
        tracks: List of Track objects
        max_lag: Maximum time lag to calculate (frames)
        time_interval: Time between frames in seconds (None: lags in frames)
        pixel_size: Pixel size in micrometres; positions (pixels) are scaled
            by it so MSDs are in um^2 (None: MSDs in px^2)

    Returns:
        Dictionary with MSD results: lags (frames), time_lags, mean_msd and
        std_msd across tracks (NaN where no track has that lag),
        individual_msds (tracks x lags, NaN where undefined), n_pairs (pairs
        summed over tracks), n_tracks_per_lag, n_tracks, and units
        ({'length': 'um' or 'px', 'time': 's' or 'frame'}).
    """
    lags = np.arange(1, max_lag + 1)
    units = {'length': 'px' if pixel_size is None else 'um',
             'time': 'frame' if time_interval is None else 's'}
    scale = 1.0 if pixel_size is None else float(pixel_size)
    time_interval = 1.0 if time_interval is None else float(time_interval)
    all_msds, all_pairs = [], []

    for track in tracks:
        if track.length < 5:  # Skip very short tracks
            continue
        track_msd, pairs = frame_msd(track.times, track.positions * scale, max_lag)
        if np.any(pairs > 0):
            all_msds.append(track_msd)
            all_pairs.append(pairs)

    if not all_msds:
        return {
            'lags': lags,
            'time_lags': lags * time_interval,
            'mean_msd': np.zeros_like(lags),
            'std_msd': np.zeros_like(lags),
            'n_pairs': np.zeros_like(lags),
            'n_tracks_per_lag': np.zeros_like(lags),
            'n_tracks': 0,
            'units': units
        }

    all_msds = np.array(all_msds)
    present = np.isfinite(all_msds)
    n_per_lag = present.sum(axis=0)
    filled = np.where(present, all_msds, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_msd = np.where(n_per_lag > 0, filled.sum(axis=0) / n_per_lag, np.nan)
        dev2 = np.where(present, (all_msds - mean_msd) ** 2, 0.0)
        std_msd = np.where(n_per_lag > 0, np.sqrt(dev2.sum(axis=0) / np.maximum(n_per_lag, 1)), np.nan)

    return {
        'lags': lags,
        'time_lags': lags * time_interval,
        'mean_msd': mean_msd,
        'std_msd': std_msd,
        'individual_msds': all_msds,
        'n_pairs': np.sum(all_pairs, axis=0),
        'n_tracks_per_lag': n_per_lag,
        'n_tracks': len(all_msds),
        'units': units
    }


def fit_diffusion_model(msd_results: Dict,
                        model: str = 'brownian') -> Dict:
    """
    Fit a diffusion model to MSD data (see _fit_diffusion_model) and attach
    the units of the fitted parameters, taken from msd_results['units']
    (pixels and frames when absent).
    """
    fit = _fit_diffusion_model(msd_results, model)
    u = msd_results.get('units', {'length': 'px', 'time': 'frame'})
    L, T = u['length'], u['time']
    fit['units'] = {'diffusion_coefficient': f'{L}^2/{T}', 'velocity': f'{L}/{T}',
                    'confinement_area': f'{L}^2', 'msd': f'{L}^2', 'time': T}
    return fit


def _fit_diffusion_model(msd_results: Dict,
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
                       window_size: int = 10,
                       method: str = 'saxton',
                       level_threshold: float = CONFINEMENT_LEVEL_THRESHOLD) -> Dict:
    """
    Detect confined motion in a trajectory.

    method='saxton' (default): Saxton (1993) / Simson et al. (1995)
    confinement level L of the whole track (see analysis/confinement.py);
    confined when L > level_threshold (3.16). Dimensionless: it does not
    depend on the particle's speed, the pixel size or, for free diffusion,
    the track length. Tracks shorter than window_size are not tested.

    method='rg_legacy': the original rule, kept for comparison: confined when
    the radius of gyration of more than 70 % of the window_size-point windows
    is below 2 pixels (calls slow free diffusion confined).

    Args:
        track: Track object
        window_size: Minimum track length (saxton) / window size (rg_legacy)
        method: 'saxton' or 'rg_legacy'
        level_threshold: Confinement level threshold (saxton)

    Returns:
        Dictionary with is_confined, method and confinement_level (saxton) or
        confinement_probability (rg_legacy)
    """
    if method == 'rg_legacy':
        return _detect_confinement_rg_legacy(track, window_size)
    if method != 'saxton':
        raise ValueError(f"Unknown confinement method: {method}")

    if track.length < window_size:
        return {'is_confined': False, 'confinement_level': float('nan'), 'method': 'saxton'}

    level = confinement_level(track.times, track.positions)
    return {
        'is_confined': bool(np.isfinite(level) and level > level_threshold),
        'confinement_level': level,
        'method': 'saxton'
    }


def _detect_confinement_rg_legacy(track: Track, window_size: int = 10) -> Dict:
    """Original rule: Rg < 2 px in more than 70 % of sliding windows."""
    if track.length < window_size:
        return {'is_confined': False, 'confinement_probability': 0.0, 'method': 'rg_legacy'}

    positions = track.positions
    is_confined = []

    for i in range(len(positions) - window_size + 1):
        window = positions[i:i+window_size]
        center = np.mean(window, axis=0)
        distances = np.linalg.norm(window - center, axis=1)
        rg = np.sqrt(np.mean(distances**2))
        is_confined.append(rg < 2.0)  # Threshold in pixels

    confinement_prob = np.mean(is_confined) if is_confined else 0.0

    return {
        'is_confined': bool(confinement_prob > 0.7),
        'confinement_probability': confinement_prob,
        'method': 'rg_legacy'
    }


def classify_motion_type(track: Track,
                        time_interval: float = 1.0,
                        **options) -> str:
    """
    Classify overall motion type of a track: the MSD classifier
    (analysis/msd_classifier.py, same as classify_track_motion(method='msd')).
    It used to run its own copy of the old rule (alpha before directed motion),
    which called diffusion with drift 'superdiffusive'.

    Args:
        track: Track object
        time_interval: Time between frames (unused: the tests are dimensionless)
        **options: MSD classifier options

    Returns:
        'brownian', 'directed', 'confined', 'subdiffusive' / 'superdiffusive'
        (alpha_classes='anomalous') or 'insufficient_data'
    """
    try:
        from .msd_classifier import classify_msd
    except ImportError:
        from analysis.msd_classifier import classify_msd
    return classify_msd(track, **options)
