"""
MSD-based motion classifier (method 'msd' of classify_track_motion).

Three tests, each dimensionless (pixels and frames can be used directly) and
each with a known false-positive rate on free (Brownian) diffusion:

1. Confinement: Saxton (1993) / Simson et al. (1995) confinement level L of the
   whole track (analysis/confinement.py), with D from the MSD slope between
   lags 1 and `confinement_fit_lags`. Confined when L > `confinement_threshold`.
2. Drift: is the mean velocity (net displacement / duration) non-zero? Under
   free diffusion z2 = |net displacement|^2 / (duration * s^2), with s^2 the
   per-axis, per-frame step variance, is chi-square with 2 degrees of freedom,
   so p = exp(-z2 / 2). Directed when p < `drift_p`.
3. Anomalous exponent: alpha from a straight-line fit of log MSD against log
   lag over lags 1..K, K = clip(int(n * fit_fraction), 4, max_fit_lag).
   alpha < alpha_sub -> 'confined' (alpha_classes='motion') or 'subdiffusive'
   (alpha_classes='anomalous'); alpha > alpha_super -> 'directed' or
   'superdiffusive'.

Checked in that order; otherwise 'brownian'. Tracks with fewer than
`min_length` points are 'insufficient_data'.

The defaults were chosen on the repo's simulator (scripts/msd_classifier_study.py,
docs/msd_classifier.md): over free diffusion of 50-point tracks each test calls
about 1-2 % of tracks non-Brownian. Why the old rule failed: alpha was fitted
over lags up to n/2 (max 20), where the time-averaged MSD is very noisy (about
20 % of free-diffusion tracks fell outside 0.85-1.15), and alpha > 1.15 was
reported as 'superdiffusive' before directed motion was considered; the
confinement level used lags 1-4, where a small corral is already saturating, so
D (and L) came out too low.

Limitations: localisation error adds a constant to the MSD and lowers alpha for
slow particles; per-track alpha cannot tell a corral from power-law
subdiffusion at these lengths (the simulator only has corrals), hence the
alpha_classes option; short tracks (< 30 points) are classified much less
reliably (see the study).
"""

import math

import numpy as np

try:
    from .msd import frame_msd
    from .confinement import confinement_level
except ImportError:
    from analysis.msd import frame_msd
    from analysis.confinement import confinement_level

MSD_CLASSIFIER_DEFAULTS = {
    'min_length': 10,             # points; shorter tracks are 'insufficient_data'
    'fit_fraction': 0.25,         # alpha fit uses lags 1..K, K = clip(int(n * fit_fraction), 4, max_fit_lag)
    'max_fit_lag': 10,
    'alpha_sub': 0.6,             # alpha below this: confined / subdiffusive
    'alpha_super': 1.4,           # alpha above this: directed / superdiffusive
    'drift_p': 0.01,              # drift test significance level
    'confinement_fit_lags': 2,    # lags for D in the confinement level
    'confinement_threshold': 2.0,  # confinement level L (-log10 of the free-walk probability)
    'alpha_classes': 'motion',    # 'motion': confined/directed; 'anomalous': subdiffusive/superdiffusive
}


def _options(options):
    unknown = set(options) - set(MSD_CLASSIFIER_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown MSD classifier option(s): {sorted(unknown)}")
    opts = {**MSD_CLASSIFIER_DEFAULTS, **options}
    if opts['alpha_classes'] not in ('motion', 'anomalous'):
        raise ValueError("alpha_classes must be 'motion' or 'anomalous'")
    return opts


def alpha_fit_lags(n, fit_fraction=0.25, max_fit_lag=10):
    """Number of lags K used for the alpha fit of an n-point track."""
    return int(min(max(int(n * fit_fraction), 4), max_fit_lag))


def anomalous_alpha(frames, positions, max_lag):
    """Slope of log MSD against log lag over frame lags 1..max_lag (lags with pairs); NaN if < 3 lags."""
    msd, n = frame_msd(frames, positions, max_lag)
    ok = (n > 0) & (msd > 0)
    if ok.sum() < 3:
        return float('nan')
    x = np.log(np.arange(1, max_lag + 1, dtype=float)[ok])
    y = np.log(msd[ok])
    xm, ym = x.mean(), y.mean()
    return float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))


def drift_test(frames, positions):
    """
    Test for a constant drift: (z2, p). NaN when the track has fewer than 3
    points or no step variance. Steps across gaps are scaled by their length.
    """
    f = np.rint(np.asarray(frames, dtype=float))
    if f.size < 3:
        return float('nan'), float('nan')
    pos = np.asarray(positions, dtype=float).reshape(f.size, -1)
    order = np.argsort(f, kind='stable')
    f, pos = f[order], pos[order]
    dt = np.diff(f)
    keep = dt > 0
    steps, dt = np.diff(pos, axis=0)[keep], dt[keep]
    T = f[-1] - f[0]
    if steps.shape[0] < 2 or T <= 0:
        return float('nan'), float('nan')
    net = pos[-1] - pos[0]
    v = net / T
    resid = steps - dt[:, None] * v
    s2 = np.sum(resid ** 2 / dt[:, None]) / (pos.shape[1] * (steps.shape[0] - 1))
    if s2 <= 0:
        return float('nan'), float('nan')
    z2 = float(np.sum(net ** 2) / (T * s2))
    return z2, float(math.exp(-z2 / 2))


def msd_track_features(track, **options):
    """The numbers the MSD classifier uses for one track (pixels and frames)."""
    opts = _options(options)
    n = track.length
    K = alpha_fit_lags(n, opts['fit_fraction'], opts['max_fit_lag'])
    frames, pos = track.times, track.positions
    z2, p = drift_test(frames, pos) if n >= 3 else (float('nan'), float('nan'))
    level = (confinement_level(frames, pos, opts['confinement_fit_lags'])
             if n >= 2 else float('nan'))
    return {'length': n, 'alpha_fit_lags': K,
            'alpha': anomalous_alpha(frames, pos, K) if n >= 4 else float('nan'),
            'drift_z2': z2, 'drift_p': p, 'confinement_level': level}


def classify_features(features, **options):
    """Apply the decision rules to msd_track_features() output."""
    opts = _options(options)
    if features['length'] < opts['min_length']:
        return 'insufficient_data'
    L, p, a = features['confinement_level'], features['drift_p'], features['alpha']
    if np.isfinite(L) and L > opts['confinement_threshold']:
        return 'confined'
    if np.isfinite(p) and p < opts['drift_p']:
        return 'directed'
    motion = opts['alpha_classes'] == 'motion'
    if np.isfinite(a) and a < opts['alpha_sub']:
        return 'confined' if motion else 'subdiffusive'
    if np.isfinite(a) and a > opts['alpha_super']:
        return 'directed' if motion else 'superdiffusive'
    return 'brownian'


def classify_msd(track, **options):
    """Motion class of one track: 'confined', 'directed', 'brownian', 'subdiffusive',
    'superdiffusive' (alpha_classes='anomalous') or 'insufficient_data'."""
    return classify_features(msd_track_features(track, **options), **options)
