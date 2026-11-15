"""One command: regenerate the submission CSV from the provided tactical assets.

    python run.py                          # writes submission/<team>.csv
    python run.py --out path/to/file.csv
    python run.py --out exp.csv --set cost_switch=15 tag_weight=0
    MTAH_DATA_DIR=/path/to/data python run.py

Deterministic: same input files -> byte-identical CSV.
"""

import argparse
import os
import warnings

import numpy as np

from possession.data import load_atd, load_smart_tagging, load_homography, N_FRAMES
from possession.camera import camera_centre
from possession.model import evidence, decode, PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))


def write_csv(path, labels):
    assert len(labels) == N_FRAMES
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def summarise(labels):
    from possession.features import HALVES
    n_poss, n_dead = 0, 0
    for h0, h1 in HALVES:
        seg = labels[h0:h1 + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        kinds = seg[starts]
        n_poss += int(np.sum(kinds != "D"))
        n_dead += int(np.sum(kinds == "D"))
    graded = np.concatenate([labels[a:b + 1] for a, b in HALVES])
    share = {k: float(np.mean(graded == k)) for k in "HAD"}
    print("possessions {}  stoppages {}  share H {:.2f} A {:.2f} D {:.2f}".format(
        n_poss, n_dead, share["H"], share["A"], share["D"]))


def main():
    # Frames with no tracked players give harmless all-NaN reductions; keep the output readable.
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "submission", "submission.csv"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override model constants for an experiment, e.g. --set cost_switch=15")
    args = ap.parse_args()

    params = dict(PARAMS)
    for item in args.set:
        name, value = item.split("=", 1)
        if name not in params:
            raise SystemExit("unknown parameter {!r}; have: {}".format(name, ", ".join(params)))
        params[name] = type(params[name])(float(value)) if isinstance(params[name], int) \
            else float(value)
    if args.set:
        print("overrides:", " ".join(args.set))

    players, ball, teams = load_atd(args.data_dir)
    tags = load_smart_tagging(args.data_dir)
    cam = camera_centre(*load_homography(args.data_dir))
    E, _ = evidence(players, ball, teams, tags, cam=cam, p=params)
    labels = decode(E, params)
    write_csv(args.out, labels)
    summarise(labels)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
