# More than a Hack 2027 — Phase 1 possession timeline

Labels every frame of the tactical video of Barcelona vs Athletic Club as `H`, `A` or `D`, using
only the provided automatic assets (ATD tracking, homography, Smart Tagging). No hand labels, no
human-produced data about the match. The method is described in [TECHNICAL.md](TECHNICAL.md).

All commands run from the repository root. In Colab, `notebooks/team_colab.ipynb` does all of
this with Runtime → Run all.

## Install

Python 3.9+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r cmo/requirements.txt      # numpy and pandas only
```

## Data

The match data is gated and is **not** in this repository. Unpack the official bundle so that this
folder exists:

```
<data dir>/tactical/BAR-ATH_tactical_atd.xml
<data dir>/tactical/BAR-ATH_tactical_atd.txt
<data dir>/tactical/BAR-ATH_tactical_homography.csv
<data dir>/tactical/BAR-ATH_tactical_smart_tagging.xml
```

By default `<data dir>` is `data/` at the root of this repository (gitignored).
Override it with `MTAH_DATA_DIR=/path/to/data` or `--data-dir`.

## Run — one command

```bash
python cmo/run.py --out evaluation/results/<team_name>.csv
```

About 10 s on a laptop (the first run also caches the parsed tracking in `cache/`). The output is
deterministic: the same input files give a byte-identical CSV.

Validate and score against the public 5 minutes with the competition repo:

```bash
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

## Layout

| file | what it does |
|---|---|
| `run.py` | entry point: load, compute evidence, decode, write CSV |
| `possession/data.py` | parsing and frame alignment of the provided assets |
| `possession/features.py` | ball smoothing, nearest-player distances, player activity |
| `possession/camera.py` | camera centre on the pitch from the homography |
| `possession/ballfree.py` | self-supervised ball-free possession model (team shape) |
| `possession/model.py` | per-frame evidence and the 3-state Viterbi decoder |
| `diagnose.py` | label-free comparison of submissions (use instead of the public score) |

Want to improve it? Read [CONTRIBUTING.md](CONTRIBUTING.md) first.
