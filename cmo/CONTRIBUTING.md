# Working on the pipeline as a team

## Ground rules (these can disqualify us)

1. **No match data in any AI tool.** That includes Colab's Gemini (Tools → Settings → AI
   assistance → off) and "Explain error". Paste code and aggregate numbers only, never rows.
2. **Never tune on the public 5 minutes.** Don't pick a setting because it scores better on
   `run_evaluation.py`. Use `diagnose.py` (below) to compare versions. The public score is only a
   check that nothing is broken.
3. **No hand labels or hand corrections on this match**, in any amount, for any purpose.
   Constants must not be tied to specific frames you saw in the video.
4. **Same input, same file.** No unseeded randomness. The Colab notebook checks this (step 9).
5. **Never commit data**, the `cache/` folder, or CSVs.

## How to try an idea

**Quick experiment, no code changes:** override any constant in `PARAMS`
(`possession/model.py`):

```bash
python cmo/run.py --out evaluation/results/exp_switch15.csv --set cost_switch=15
python cmo/diagnose.py evaluation/results/main.csv evaluation/results/exp_switch15.csv
```

Always generate `main.csv` from the current `main` first, and list it first, so the
`changed` column is relative to it.

**Code change:** work on a branch, not on `main`:

```bash
git checkout -b dead-ball-detector
# edit, run, diagnose
git push -u origin dead-ball-detector
```

Open a pull request on GitHub with the `diagnose.py` table for main vs your branch in the
description. In Colab, set `BRANCH` in the Settings cell to your branch name to run it.

## Reading `diagnose.py`

No column is the score. A good change usually:
- holds or raises `ctrl_agree` (agreement on the easy frames, where one side is clearly on the
  ball);
- keeps `short` (possessions under 2 s) low, because every fake possession costs spellF1;
- keeps counts plausible: in LaLiga the ball is dead about 40–45% of match time. Our `D%` of
  about 30% is probably too low.

**Gotcha:** keep `cost_switch ≤ 2 × cost_dead`. Otherwise the decoder fakes a stoppage at every
turnover (H→D→A becomes cheaper than H→A), and `stops` jumps. Seen at `cost_switch=20`.

## Where improvement is most likely (pick one, say so in the team chat)

1. **Dead-ball detector for fouls.** The ball stays inside the pitch and play stops without a
   set-piece tag. Signals: players walking (`activity`), a stationary ball, players regrouping.
   Probably the biggest gain, because our D share looks low.
2. **Restart side for untagged restarts.** Untagged throw-ins and free kicks get no team evidence
   today. Reuse `restart_team()` on detected stoppage ends.
3. **Better ball-free model** (`ballfree.py`). Try more features (team compactness, pressing
   distance, camera velocity) or a small tree model. Report its held-out accuracy with
   `train_and_predict(..., report=True)`; it is 0.827 today.
4. **Track linking.** Join the 46 track slots into player tracklets so that "the same player
   keeps the ball" becomes a signal.
5. **Ball-out detection.** The ball crossing a line and then vanishing is a strong signal of a
   stoppage starting. Only the "already outside" case is used today.

Update `TECHNICAL.md` when you change the method: the write-up has to describe what we
actually submit.
