"""Load the provided tactical assets into numpy arrays, indexed by VIDEO frame (0-based).

Nothing here interprets the data; it only parses and aligns it:

    ATD           frame_id 1..N        -> video frame = frame_id - 1
    homography    frame    0..N-1      -> already a video frame
    Smart Tagging seconds from start   -> video frame = round(seconds * 25)

The parsed ATD is cached as a compressed .npz next to the raw file's cache dir, because parsing
196 MB of text takes about a minute and is deterministic. The cache is a pure function of the
input file (keyed on its size and mtime) so it can never change the output.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)          # metres; games.py value measured from corrected tracking
HOME, AWAY = 0, 1
STEM = "BAR-ATH_tactical"


def default_data_dir():
    """MTAH_DATA_DIR if set (the Colab notebook sets it), else <repo root>/data."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../cmo
    return os.environ.get("MTAH_DATA_DIR") or os.path.join(os.path.dirname(pkg_root), "data")


def paths(data_dir=None):
    folder = os.path.join(data_dir or default_data_dir(), "tactical")
    out = {
        "atd_meta": os.path.join(folder, STEM + "_atd.xml"),
        "atd_raw": os.path.join(folder, STEM + "_atd.txt"),
        "homography": os.path.join(folder, STEM + "_homography.csv"),
        "smart_tagging": os.path.join(folder, STEM + "_smart_tagging.xml"),
    }
    missing = [p for p in out.values() if not os.path.exists(p)]
    if missing:
        raise IOError("missing data files:\n  " + "\n  ".join(missing))
    return out


def track_teams(meta_path):
    """Team (HOME/AWAY) of each of the 46 track slots. First <Team> in the file is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    assert len(team_ids) == 2, team_ids
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def _parse_atd(raw_path, n_slots):
    """(players[N, S, 3], ball[N, 3]) float32: x, y (0-1), speed (km/h). NaN where missing."""
    players = np.full((N_FRAMES, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 3), np.nan, dtype=np.float32)
    width = n_slots * 3
    with open(raw_path) as handle:
        for line in handle:
            head, body, tail = line.rstrip("\n").split(":")
            frame = int(head) - 1
            vals = np.array(body.replace(";", ",").split(","), dtype=np.float32)
            if vals.size == width:
                players[frame] = vals.reshape(n_slots, 3)
            ball[frame] = np.array(tail.split(","), dtype=np.float32)
    return players, ball


def load_atd(data_dir=None, cache_dir=None):
    p = paths(data_dir)
    teams = track_teams(p["atd_meta"])
    st = os.stat(p["atd_raw"])
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "atd_{}_{}.npz".format(st.st_size, int(st.st_mtime)))
    if os.path.exists(cache):
        z = np.load(cache)
        return z["players"], z["ball"], teams
    players, ball = _parse_atd(p["atd_raw"], len(teams))
    np.savez_compressed(cache, players=players, ball=ball)
    return players, ball, teams


def load_homography(data_dir=None):
    """(H[N,3,3] pitch(0-1)->pixels, reprojection_error[N], valid[N])."""
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    t = pd.read_csv(paths(data_dir)["homography"])
    assert len(t) == N_FRAMES and (t["frame"].values == np.arange(N_FRAMES)).all()
    return (t[cols].values.astype(np.float64).reshape(-1, 3, 3),
            t["reprojection_error"].values.astype(np.float64),
            t["valid"].values.astype(bool))


def load_smart_tagging(data_dir=None):
    """List of dicts: code, start, end (video frames, end exclusive), tags."""
    out = []
    for inst in ET.parse(paths(data_dir)["smart_tagging"]).iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "tags": {(l.findtext("group") or "").strip(): (l.findtext("text") or "").strip()
                     for l in inst.findall("label")},
        })
    return out
