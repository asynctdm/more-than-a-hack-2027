"""Compare submissions WITHOUT any ground truth. Use this, not the public 5 minutes, to judge changes.

    python cmo/diagnose.py evaluation/results/A.csv evaluation/results/B.csv ...

The rules forbid tuning on the public window (and 17 possessions cannot rank anything anyway), so
every check here is computed from our own output plus the automatic data only:

  poss / stops     possession and stoppage counts in the graded halves
  mean_s           mean possession length in seconds
  short            possessions under 2 s -- mostly flicker; spellF1 punishes each one
  H% A% D%         label shares; real LaLiga matches have the ball dead roughly 40-45% of the time
  ctrl_agree       on frames where one side is clearly on a tracked, slow, in-pitch ball (the
                   code-defined pseudo-label from ballfree.py), how often we say that side.
                   High is necessary but not sufficient -- it only covers easy frames.
  tag_agree        agreement with the Smart Tagging phase team (BUILD UP etc.). Weak: the tags
                   are automatic, phase-level, and also a (small) input to the model.
  changed          frames that differ from the FIRST file given (use your baseline first)

None of these is the score. They catch regressions (flicker, collapsed D, team swaps) and show
what a change actually did. A real improvement should usually move ctrl_agree up or hold it,
keep `short` low, and keep counts plausible.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession.data import load_atd, load_smart_tagging, N_FRAMES, FPS, HOME  # noqa: E402
from possession import features as F  # noqa: E402
from possession.ballfree import pseudo_labels  # noqa: E402
from possession.model import tag_team, TAG_IN_POSSESSION, TAG_DEFENDING  # noqa: E402


def segments(labels):
    out = []
    for h0, h1 in F.HALVES:
        seg = labels[h0:h1 + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], cut))
        ends = np.concatenate((cut, [len(seg)]))
        out += [(seg[a], b - a) for a, b in zip(starts, ends)]
    return out


def main(paths):
    warnings.simplefilter("ignore", RuntimeWarning)
    players, ball, teams = load_atd()
    tags = load_smart_tagging()
    bx, by, bspeed = F.ball_signals(ball)
    dh, da = F.nearest_by_team(players, teams, bx, by)
    inside = ~np.isnan(bx) & (bx > 0) & (bx < F.L) & (by > 0) & (by < F.W)
    pseudo = pseudo_labels(dh, da, inside, bspeed)       # 1 home, 0 away, -1 unknown

    tag_side = np.full(N_FRAMES, -1)
    for ev in tags:
        team = tag_team(ev)
        if team is None:
            continue
        if ev["code"] in TAG_IN_POSSESSION:
            tag_side[ev["start"]:ev["end"]] = team
        elif ev["code"] in TAG_DEFENDING:
            tag_side[ev["start"]:ev["end"]] = 1 - team

    graded = F.graded_mask()
    ref = None
    header = "{:<32} {:>5} {:>5} {:>6} {:>5} {:>5} {:>5} {:>5} {:>10} {:>9} {:>8}".format(
        "file", "poss", "stops", "mean_s", "short", "H%", "A%", "D%",
        "ctrl_agree", "tag_agree", "changed")
    print(header)
    print("-" * len(header))
    for path in paths:
        lab = pd.read_csv(path)["label"].astype(str).values
        if len(lab) != N_FRAMES:
            print("{:<32} wrong length {}".format(os.path.basename(path), len(lab)))
            continue
        segs = segments(lab)
        poss = [n for k, n in segs if k in ("H", "A")]
        stops = sum(1 for k, _ in segs if k == "D")
        g = lab[graded]
        side = np.where(lab == "H", HOME, np.where(lab == "A", 1 - HOME, -1))
        m = graded & (pseudo >= 0)
        ctrl = np.mean(side[m] == np.where(pseudo[m] == 1, HOME, 1 - HOME))
        mt = graded & (tag_side >= 0) & (side >= 0)
        tagag = np.mean(side[mt] == tag_side[mt])
        changed = "" if ref is None else str(int(np.sum(lab != ref)))
        if ref is None:
            ref = lab
        print("{:<32} {:>5} {:>5} {:>6.1f} {:>5} {:>5.1f} {:>5.1f} {:>5.1f} {:>10.3f} {:>9.3f} {:>8}".format(
            os.path.basename(path)[:32], len(poss), stops,
            np.mean(poss) / FPS if poss else 0, sum(1 for n in poss if n < 2 * FPS),
            100 * np.mean(g == "H"), 100 * np.mean(g == "A"), 100 * np.mean(g == "D"),
            ctrl, tagag, changed))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
