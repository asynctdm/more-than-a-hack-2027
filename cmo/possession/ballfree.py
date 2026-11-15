"""A ball-free possession model, trained on pseudo-labels produced by code.

The ball is untracked for long stretches (median gap ~11 s). During those, the evidence decoder has
nothing to go on except a weak tag prior. But team shape carries possession too: the side with the
ball spreads out and pushes up, the side without it compacts and drops.

So: on frames where the ball IS tracked and one side is unambiguously at it (a code-defined rule,
no human involved), take that side as a pseudo-label; describe each frame with features that do
not use the ball at all; fit a logistic regression; apply it where the ball is missing.

This is self-supervised learning on the provided automatic data, which RULES.md allows ("pseudo-
labels produced by an algorithm, provided no person decided any of them"). It is validated on
held-out pseudo-labelled frames (contiguous blocks, so neighbouring frames cannot leak).
"""

import numpy as np

from .data import FPS, HOME, AWAY, N_FRAMES
from . import features as F


def attack_sign(players, teams):
    """+1 if home attacks towards x=L in each half, -1 otherwise; per frame (0 outside halves).

    Decided per half from where each side's deepest player (the goalkeeper) usually stands.
    """
    sign = np.zeros(N_FRAMES)
    px = players[:, :, 0].astype(np.float64) * F.L
    for h0, h1 in F.HALVES:
        seg = px[h0:h1 + 1]
        hx = np.nanmedian(np.where(teams[None] == HOME, seg, np.nan), axis=1)
        ax = np.nanmedian(np.where(teams[None] == AWAY, seg, np.nan), axis=1)
        # Home's median x lower than away's -> home defends x=0 -> attacks +x.
        s = 1.0 if np.nanmedian(hx - ax) < 0 else -1.0
        sign[h0:h1 + 1] = s
    return sign


def frame_features(players, teams, cam_x, cam_y):
    """Ball-free features per frame, oriented so +x is home's attacking direction."""
    sign = attack_sign(players, teams)
    px = players[:, :, 0].astype(np.float64) * F.L
    py = players[:, :, 1].astype(np.float64) * F.W
    # orient: x' = x if home attacks +x else L - x
    ox = np.where(sign[:, None] >= 0, px, F.L - px)
    feats = []
    names = []
    with np.errstate(invalid="ignore"):
        for team, tag in ((HOME, "h"), (AWAY, "a")):
            m = teams[None] == team
            x = np.where(m, ox, np.nan)
            y = np.where(m, py, np.nan)
            feats += [np.nanmean(x, 1), np.nanstd(x, 1), np.nanstd(y, 1),
                      np.nanmax(x, 1), np.nanmin(x, 1)]
            names += [tag + "_cx", tag + "_sx", tag + "_sy", tag + "_maxx", tag + "_minx"]
        ocx = np.where(sign >= 0, cam_x * F.L, F.L - cam_x * F.L)
        feats += [ocx, np.abs(cam_y * F.W - F.W / 2)]
        names += ["cam_x", "cam_y_off"]
    X = np.vstack(feats).T
    # temporal context: 1 s smoothing plus 1 s velocity of every feature
    Xs = np.vstack([F.rolling_mean(X[:, j], FPS + 1) for j in range(X.shape[1])]).T
    V = np.vstack([np.gradient(F.rolling_mean(X[:, j], 2 * FPS + 1)) * FPS
                   for j in range(X.shape[1])]).T
    names = names + [n + "_v" for n in names]
    return np.hstack([Xs, V]), names


def pseudo_labels(dh, da, inside, bspeed, window=FPS // 2):
    """+1 home / 0 away / -1 unknown, from a sustained, uncontested nearest-player rule."""
    h = (dh < 1.5) & (da > 3.0) & inside & (bspeed < 12)
    a = (da < 1.5) & (dh > 3.0) & inside & (bspeed < 12)
    hs = F.rolling_mean(h.astype(float), 2 * window + 1)
    as_ = F.rolling_mean(a.astype(float), 2 * window + 1)
    y = np.full(N_FRAMES, -1)
    y[(hs > 0.5) & (as_ == 0)] = 1
    y[(as_ > 0.5) & (hs == 0)] = 0
    return y


def fit_logistic(X, y, l2=1.0, iters=300, lr=0.5):
    """Plain L2-regularised logistic regression by gradient descent (numpy only)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def train_and_predict(players, teams, cam_x, cam_y, dh, da, inside, bspeed, report=False):
    """P(home has the ball) per frame from ball-free features; NaN where features are missing."""
    X, names = frame_features(players, teams, cam_x, cam_y)
    y = pseudo_labels(dh, da, inside, bspeed)
    graded = F.graded_mask()
    ok = np.all(np.isfinite(X), axis=1)
    mu = np.nanmean(X[ok & graded], 0)
    sd = np.nanstd(X[ok & graded], 0) + 1e-9
    Z = (X - mu) / sd
    train = ok & graded & (y >= 0)

    if report:
        # Held-out check on contiguous 2-minute blocks (alternate blocks train/test).
        block = (np.arange(N_FRAMES) // (120 * FPS)) % 2
        tr, te = train & (block == 0), train & (block == 1)
        w, b = fit_logistic(Z[tr], y[tr])
        p = 1 / (1 + np.exp(-(Z[te] @ w + b)))
        acc = np.mean((p > 0.5) == (y[te] == 1))
        conf = np.abs(p - 0.5) > 0.3
        print("ball-free model: held-out accuracy {:.3f} on {} frames "
              "(base rate {:.3f}); confident {:.0%} of frames at accuracy {:.3f}".format(
                  acc, te.sum(), max(y[te].mean(), 1 - y[te].mean()), conf.mean(),
                  np.mean(((p > 0.5) == (y[te] == 1))[conf])))

    w, b = fit_logistic(Z[train], y[train])
    prob = np.full(N_FRAMES, np.nan)
    prob[ok] = 1 / (1 + np.exp(-(Z[ok] @ w + b)))
    return prob
