"""Per-frame signals derived from the automatic tracking, homography and Smart Tagging.

Everything is computed by code from the provided automatic assets; no frame of this match is
referenced by number except the published half boundaries (games.py / DATA.md).
"""

import numpy as np

from .data import FPS, HOME, AWAY, N_FRAMES, PITCH

L, W = PITCH

# Graded halves, published by the organisers (DATA.md "Which frames are graded").
HALVES = ((11185, 90651), (110808, 190197))


def rolling_mean(x, window):
    """Centred moving average that ignores NaN (NaN where the whole window is NaN)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    k = np.ones(window)
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def rolling_median_present(x, half):
    """Centred median over +/-half frames, computed only where x is present."""
    n = len(x)
    out = np.full(n, np.nan)
    idx = np.flatnonzero(~np.isnan(x))
    if not len(idx):
        return out
    stack = []
    for s in range(-half, half + 1):
        j = np.clip(idx + s, 0, n - 1)
        stack.append(x[j])
    out[idx] = np.nanmedian(np.vstack(stack), axis=0)
    return out


def run_lengths(mask):
    """(start, end_exclusive) of each run of True."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def ball_signals(ball):
    """Smoothed ball position in metres and its speed in m/s (NaN where not tracked)."""
    bx = rolling_median_present(ball[:, 0].astype(np.float64) * L, 2)
    by = rolling_median_present(ball[:, 1].astype(np.float64) * W, 2)
    vx = np.gradient(bx) * FPS
    vy = np.gradient(by) * FPS
    speed = rolling_mean(np.hypot(vx, vy), 5)
    return bx, by, speed


def nearest_by_team(players, teams, bx, by):
    """Distance (m) from the ball to the nearest home and away player; inf if none."""
    px = players[:, :, 0].astype(np.float64) * L
    py = players[:, :, 1].astype(np.float64) * W
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    dh = np.min(np.where(teams[None] == HOME, d, np.inf), axis=1)
    da = np.min(np.where(teams[None] == AWAY, d, np.inf), axis=1)
    return dh, da


def activity(players):
    """Mean player speed (m/s), smoothed over 2 s, relative to the match median in play time."""
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanmean(players[:, :, 2].astype(np.float64), axis=1)
    s = rolling_mean(s, 2 * FPS + 1)
    graded = graded_mask()
    return s / np.nanmedian(s[graded])


def graded_mask():
    g = np.zeros(N_FRAMES, bool)
    for a, b in HALVES:
        g[a:b + 1] = True
    return g


def team_centroids(players, teams):
    """(home_x, away_x) mean x in metres per frame, NaN if no player of that team is tracked."""
    px = players[:, :, 0].astype(np.float64) * L
    with np.errstate(invalid="ignore"):
        hx = np.nanmean(np.where(teams[None] == HOME, px, np.nan), axis=1)
        ax = np.nanmean(np.where(teams[None] == AWAY, px, np.nan), axis=1)
    return hx, ax
