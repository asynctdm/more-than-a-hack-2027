"""Frame evidence -> possession timeline, via a 3-state Viterbi decoder with switching costs.

States: H (home has the ball), A (away has the ball), D (dead ball).

Each frame contributes a score per state (log-likelihood-like "evidence"). The decoder picks the
label sequence that maximises total evidence minus a fixed cost per change of state. That cost is
the possession rule written as arithmetic:

  * possession changes only when the other team shows *sustained* control -- a touch, deflection
    or lost aerial duel adds a little evidence for the other side but not enough to pay the cost;
  * a loose or untracked ball adds no evidence at all, so the decoder keeps the last team;
  * a dead ball needs its own sustained evidence (players stop, ball out, a tagged restart).

All constants are set from football reasoning and from aggregate, label-free properties of the
automatic data (see TECHNICAL.md). None was tuned against any ground truth.
"""

import numpy as np

from .data import FPS, HOME, AWAY, N_FRAMES
from . import features as F

H_, A_, D_ = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = dict(
    control_radius=1.5,       # m; Gaussian kernel width for "player at the ball"
    control_weight=1.0,
    flight_speed=12.0,        # m/s; above this the ball is travelling, not controlled
    flight_factor=0.25,
    out_margin=1.0,           # m beyond touch/goal line (3 m on the far touchline, see below)
    out_margin_far=3.0,
    out_weight=1.0,
    activity_ref=0.75,        # relative mean player speed below which play looks stopped
    activity_scale=0.25,
    activity_weight=0.6,
    missing_weight=0.10,      # untracked ball: slight lean to dead ball
    tag_weight=0.05,          # Smart Tagging phase team prior
    ballfree_weight=0.15,     # ball-free shape model, used where the ball is not usable
    restart_offset=round(3.5 * FPS),   # tag start -> restart (checked on both kick-offs)
    restart_dead=3.0,
    restart_play=2.0,
    restart_team=2.0,
    restart_len=2 * FPS,
    cost_switch=12.0,         # H <-> A
    cost_dead=8.0,            # H/A <-> D
)

# Smart Tagging codes whose Team tag is the side IN possession / the side DEFENDING.
# Established by label-free agreement with the nearest-player signal (TECHNICAL.md).
TAG_IN_POSSESSION = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION"}
TAG_DEFENDING = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                 "DEFENSIVE TRANSITION"}
TEAM_NAMES = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}


def tag_team(event):
    return TEAM_NAMES.get((event["tags"].get("Team") or "").strip().lower())


def restart_team(event, r, players, teams, bx, by, dh, da):
    """Which side takes a restart whose Team tag is N/A (throw-ins, kick-offs)."""
    typ = (event["tags"].get("Type") or "").lower()
    lo, hi = max(r - 2 * FPS, 0), min(r + FPS // 2, N_FRAMES)
    if "throw" in typ:
        # The thrower is the only player standing beyond a touchline.
        py = players[lo:hi, :, 1].astype(np.float64) * F.W
        outside = (py < -0.3) | (py > F.W + 0.3)
        votes = [outside[:, teams == HOME].sum(), outside[:, teams == AWAY].sum()]
        if max(votes) > 0 and votes[0] != votes[1]:
            return int(np.argmax(votes))
    if "kick off" in typ:
        # The kicking side has a player on the centre spot.
        px = players[lo:hi, :, 0].astype(np.float64) * F.L
        py = players[lo:hi, :, 1].astype(np.float64) * F.W
        d = np.hypot(px - F.L / 2, py - F.W / 2)
        d = np.where(np.isnan(d), np.inf, d)
        best = [d[:, teams == HOME].min(axis=1), d[:, teams == AWAY].min(axis=1)]
        close = np.vstack(best) < 1.5
        votes = close.sum(axis=1)
        if votes.max() > 0 and votes[0] != votes[1]:
            return int(np.argmax(votes))
    # Fallback: whoever is at the ball around the restart.
    h = np.sum(dh[lo:hi] < np.minimum(da[lo:hi], 2.0))
    a = np.sum(da[lo:hi] < np.minimum(dh[lo:hi], 2.0))
    if h != a:
        return HOME if h > a else AWAY
    return None


def evidence(players, ball, teams, tags, cam=None, p=PARAMS):
    """E[N, 3]: per-frame score for H, A, D. cam = (x, y) camera centre on the pitch (0-1)."""
    E = np.zeros((N_FRAMES, 3))
    bx, by, bspeed = F.ball_signals(ball)
    dh, da = F.nearest_by_team(players, teams, bx, by)
    seen = ~np.isnan(bx)

    # --- ball location: out of the pitch -> dead
    # The far touchline gets a wider margin: a ball in the air projects beyond it.
    out = seen & ((bx < -p["out_margin"]) | (bx > F.L + p["out_margin"]) |
                  (by < -p["out_margin"]) | (by > F.W + p["out_margin_far"]))
    E[out, D_] += p["out_weight"]
    inside = seen & ~out

    # --- control: a player of one side at the ball, the other side not
    k = lambda d: np.exp(-0.5 * (d / p["control_radius"]) ** 2)
    kh, ka = k(dh), k(da)
    w = np.where(bspeed > p["flight_speed"], p["flight_factor"], 1.0) * p["control_weight"]
    w = np.where(inside, w, 0.0)
    E[:, H_] += w * np.clip(kh - ka, 0, None)
    E[:, A_] += w * np.clip(ka - kh, 0, None)

    # --- ball-free shape model where the ball gives no control information
    if cam is not None and p["ballfree_weight"] > 0:
        from .ballfree import train_and_predict
        prob = train_and_predict(players, teams, cam[0], cam[1], dh, da, inside, bspeed)
        unusable = ~inside | (bspeed > p["flight_speed"])
        s = np.where(unusable & np.isfinite(prob), 2 * prob - 1, 0.0) * p["ballfree_weight"]
        E[:, H_] += np.clip(s, 0, None)
        E[:, A_] += np.clip(-s, 0, None)

    # --- activity: players stop when the ball is dead
    act = F.activity(players)
    act = np.where(np.isnan(act), p["activity_ref"], act)
    E[:, D_] += p["activity_weight"] * np.clip(
        (p["activity_ref"] - act) / p["activity_scale"], -1.5, 2.0)
    E[~seen, D_] += p["missing_weight"]

    # --- Smart Tagging phases: weak team prior
    for ev in tags:
        team = tag_team(ev)
        if team is None:
            continue
        if ev["code"] in TAG_IN_POSSESSION:
            side = team
        elif ev["code"] in TAG_DEFENDING:
            side = 1 - team
        else:
            continue
        E[ev["start"]:ev["end"], side] += p["tag_weight"]

    # --- Tagged restarts: dead before, in play and (if known) one side's ball after
    for ev in tags:
        if ev["code"] != "SET PIECES":
            continue
        r = ev["start"] + p["restart_offset"]
        E[ev["start"]:r - 2, D_] += p["restart_dead"]
        a, b = r, min(r + p["restart_len"], N_FRAMES)
        E[a:b, D_] -= p["restart_play"]
        team = tag_team(ev)
        if team is None:
            team = restart_team(ev, r, players, teams, bx, by, dh, da)
        if team is not None:
            E[a:b, team] += p["restart_team"]
        # Kick-off at the start of a half: everything in the half before it is dead.
        if (ev["tags"].get("Type") or "").lower() == "kick off (start)":
            for h0, h1 in F.HALVES:
                if h0 <= r <= h1:
                    E[h0:r - 2, D_] += 10.0

    diag = dict(bx=bx, by=by, bspeed=bspeed, dh=dh, da=da, act=act, out=out)
    return E, diag


def viterbi(E, cost_switch, cost_dead, start_state=D_):
    """Max-sum decoding of E[n,3] with symmetric switching costs."""
    n = len(E)
    C = np.array([[0, cost_switch, cost_dead],
                  [cost_switch, 0, cost_dead],
                  [cost_dead, cost_dead, 0]], dtype=np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    score = np.full(3, -1e9)
    score[start_state] = 0.0
    score = score + E[0]
    for t in range(1, n):
        cand = score[:, None] - C          # from i to j
        arg = cand.argmax(axis=0)
        back[t] = arg
        score = cand[arg, np.arange(3)] + E[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(E, p=PARAMS):
    """Labels for the whole video: decoded inside each half, D outside the halves."""
    labels = np.full(N_FRAMES, "D", dtype="<U1")
    for h0, h1 in F.HALVES:
        path = viterbi(E[h0:h1 + 1], p["cost_switch"], p["cost_dead"])
        labels[h0:h1 + 1] = LABELS[path]
    return labels
