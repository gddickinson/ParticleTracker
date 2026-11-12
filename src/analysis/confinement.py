"""
Saxton/Simson confinement level of a trajectory (length-independent test).

Same method and constants as George's sibling repositories particle_tracking
(mobility_metrics.py) and particle-tracker (ptsuite/confinement.py):

Saxton (1993), Biophys. J. 64:1766-1780, doi:10.1016/S0006-3495(93)81548-0:
probability psi that a free random walker with diffusion coefficient D stays
within radius R of its start for a time t,
    log10(psi) = 0.2048 - 2.5117 * D * t / R**2
Simson, Sheets & Jacobson (1995), Biophys. J. 69:989-993,
doi:10.1016/S0006-3495(95)79972-6, used it with D from the initial slope of
the MSD and R the maximum excursion, and called a track confined when the
level L = -log10(psi) exceeds 3.16 (psi below about 0.07 %).

L is dimensionless (D t / R^2), so pixels and frames can be used directly,
and for free diffusion it does not grow with track length. Whole-track
version (Simson et al. also used sliding windows).
"""

import numpy as np

try:
    from .msd import frame_msd
except ImportError:
    from analysis.msd import frame_msd

SAXTON_A, SAXTON_B = 0.2048, 2.5117
CONFINEMENT_LEVEL_THRESHOLD = 3.16
CONFINEMENT_FIT_LAGS = 4


def initial_slope_D(frames, positions, fit_lags=CONFINEMENT_FIT_LAGS):
    """
    D (length^2 per frame) from a straight-line fit (with intercept, which
    absorbs localisation error) of MSD = 4 D t over frame lags 1..fit_lags.
    NaN if fewer than min(3, fit_lags) of those lags have pairs (fit_lags >= 2).
    """
    if fit_lags < 2:
        raise ValueError("fit_lags must be at least 2 (the fit has an intercept)")
    msd, n = frame_msd(frames, positions, fit_lags)
    ok = n > 0
    if ok.sum() < min(3, fit_lags):
        return np.nan
    t = np.arange(1, fit_lags + 1, dtype=float)[ok]
    m = msd[ok]
    tm, mm = t.mean(), m.mean()
    return float(np.sum((t - tm) * (m - mm)) / np.sum((t - tm) ** 2) / 4.0)


def confinement_level(frames, positions, fit_lags=CONFINEMENT_FIT_LAGS):
    """
    Confinement level L = 2.5117 D t / R^2 - 0.2048 of a whole track.

    D from initial_slope_D, t the duration in frames (last - first frame),
    R the maximum distance from the first position. NaN if R = 0, the track
    has fewer than 2 points or D is undefined.
    """
    f = np.rint(np.asarray(frames, dtype=float))
    if f.size < 2:
        return np.nan
    pos = np.asarray(positions, dtype=float).reshape(f.size, -1)
    order = np.argsort(f, kind="stable")
    f, pos = f[order], pos[order]
    D = initial_slope_D(f, pos, fit_lags)
    t = f[-1] - f[0]
    r2 = np.max(np.sum((pos - pos[0]) ** 2, axis=1))
    if not np.isfinite(D) or r2 == 0:
        return np.nan
    return float(SAXTON_B * D * t / r2 - SAXTON_A)
