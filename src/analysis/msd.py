"""
Frame-lag mean squared displacement of one trajectory (pure numpy).

frame_msd pairs localisations that are exactly `lag` FRAMES apart, whatever
their row positions, so tracks with missing frames (gap closing) use every
valid pair. For a gap-free track it equals the usual row-lag MSD.
"""

import numpy as np


def frame_msd(frames, positions, max_lag):
    """
    Time-averaged MSD of one track at frame lags 1..max_lag.

    Args:
        frames: frame numbers (integers; a repeated frame keeps its first
            localisation in frame order)
        positions: (N, d) positions in any length unit
        max_lag: largest lag in frames

    Returns:
        (msd, n_pairs): arrays of length max_lag; msd[k-1] is the mean squared
        displacement over all pairs exactly k frames apart (NaN when there is
        no such pair) and n_pairs[k-1] the number of those pairs.
    """
    max_lag = int(max_lag)
    msd = np.full(max_lag, np.nan)
    n_pairs = np.zeros(max_lag, dtype=int)
    f = np.rint(np.asarray(frames, dtype=float)).astype(int)
    if f.size == 0 or max_lag < 1:
        return msd, n_pairs
    pos = np.asarray(positions, dtype=float).reshape(f.size, -1)
    order = np.argsort(f, kind="stable")
    f, pos = f[order], pos[order]
    span = int(f[-1] - f[0])
    grid = np.full((span + 1, pos.shape[1]), np.nan)
    idx = f - f[0]
    grid[idx[::-1]] = pos[::-1]  # reversed so the first localisation of a repeated frame wins
    for lag in range(1, min(max_lag, span) + 1):
        d2 = np.sum((grid[lag:] - grid[:-lag]) ** 2, axis=1)
        d2 = d2[np.isfinite(d2)]
        if d2.size:
            msd[lag - 1] = d2.mean()
            n_pairs[lag - 1] = d2.size
    return msd, n_pairs
