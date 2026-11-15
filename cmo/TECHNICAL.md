# Technical document — possession timeline from automatic tracking

## 1. Approach

The task is scored with spellF1, which treats each possession as one object, so a correct per-frame
classifier that flickers scores close to zero. The method is therefore built as a **segmentation**
problem rather than a per-frame classification problem.

1. **Per-frame evidence.** For each frame, compute a score for each of `H`, `A` and `D` from
   several independent signals (section 2).
2. **Decoding with switching costs.** A 3-state Viterbi decoder picks the label sequence that
   maximises total evidence minus a fixed cost for every change of state (12 for H↔A, 8 for
   H/A↔D). The costs are the possession rules expressed as arithmetic:
   - possession changes only when the other team shows *sustained* control, enough evidence to
     pay the switching cost. A touch, a deflection or a lost aerial duel adds a little evidence but
     not enough;
   - a loose or untracked ball adds no team evidence, so the decoder keeps the last team in
     possession;
   - a stoppage needs its own sustained evidence.

Why this rather than majority-vote smoothing (the baseline): a smoothing window trades boundary
precision for stability, and it has no notion that a gap in the evidence should mean "no change".
The decoder keeps exact boundaries where the evidence is clear and holds its state where there
is no evidence, which is what the rules ask for.

## 2. Signals and assets used

All from the provided tactical assets, aligned as documented (ATD `frame_id − 1`, homography
`frame`, Smart Tagging `round(seconds × 25)`).

| signal | source | effect |
|---|---|---|
| Control: Gaussian kernel (σ = 1.5 m) on the distance from the ball to the nearest home/away player, minus the same for the other side | ATD ball + players | H/A evidence; weight ×0.25 when the ball moves > 12 m/s (in flight) |
| Ball beyond touch/goal line (1 m margin; 3 m on the far touchline, where balls in the air project beyond the line) | ATD ball | D evidence |
| Activity: mean player speed over 2 s, relative to the match median | ATD players | D evidence when play has stopped, negative D evidence when play is fast |
| Ball untracked | ATD ball | small D lean |
| Ball-free shape model (section 3) | ATD players + homography | H/A evidence where the ball is missing, out of play or in flight |
| Phase team prior | Smart Tagging | weak H/A evidence |
| Tagged restarts | Smart Tagging `SET PIECES` | D before the restart, in play after it, restart side's ball for 2 s |

**Restart timing.** Every set-piece tag except the kick-offs is a fixed 35 s window. Averaging
player speed over all tags of each type (label-free) gives a minimum about 2–4 s after the tag
start and a recovery after it. The two kick-off tags start 3.5 s before the published kick-off
frames. The restart is therefore placed at tag start + 3.5 s.

**Restart side when Team = N/A.** For throw-ins: the side with a player standing beyond a
touchline. For kick-offs: the side with a player on the centre spot. Otherwise: the side at the
ball around the restart.

**Phase semantics.** We measured how often the tagged team is the nearest-in-control side, from
tracking alone, during each phase code. BUILD UP 0.85, PROGRESSION 0.81, ATTACKING TRANSITION
0.69, so the tagged team has the ball. DEFENDING IN ATTACKING THIRD 0.15, DEFENDING IN MIDDLE
THIRD 0.19, DEFENSIVE TRANSITION 0.31, so the tagged team is defending. The other codes are near
chance and are not used.

**Tried and dropped / not used.**
- The camera centre as a stand-in ball position: its median error against the tracked ball is
  12–16 m. That is too coarse to find the player on the ball, so it is used only as a feature of
  the shape model.
- The ATD speed column for the ball: unreliable, so the ball speed is derived from smoothed
  positions.
- The broadcast feed and the video: not used.

## 3. Training data — self-supervised only

The ball is untracked on 41% of graded frames, mostly in long gaps (median ≈ 11 s). To give the
decoder evidence there, a logistic regression predicts P(home has the ball) from **ball-free**
features: per-team centroid, spread and extremes in the attacking direction, the camera centre,
and the 1 s smoothed value and velocity of each.

- **Labels:** pseudo-labels produced by code. A frame is labelled with a side when, over ±0.5 s,
  that side is within 1.5 m of a tracked, in-pitch, slow ball more than half the time and the other
  side never is (the other side's nearest player is > 3 m away). 28,156 frames.
- **No human labelled anything**, on this match or any other. No external data was used.
- **Validation:** alternating 2-minute blocks for training and testing. Held-out accuracy 0.827,
  against a base rate of 0.565. The 76% of frames where the model is confident reach 0.875.

## 4. Possession rules applied

- **Loose ball / challenges:** no control evidence for either side, so the previous team keeps the
  ball. A contested ball (both sides within a few metres) gives evidence only to the side that is
  clearly closer, and that evidence is small.
- **Deflections, touches without control, aerial duels:** short bursts of evidence below the
  switching cost, so no change of possession.
- **Restarts:** dead from the stoppage, detected from the ball leaving the pitch and players
  stopping, until the restart. After it, the restart side has the ball.
- **Dead ball** is always `D`. Outside the graded halves we write `D`. The second half is dead
  before its detected kick-off, which also handles the 2,934 dead frames before the second-half
  kick-off.
- `X` is never written.

## 5. Constants

All constants (kernel width, flight speed, margins, switching costs, weights) were set from
football reasoning and label-free statistics of the automatic data. **None was tuned against any
ground truth.** The public 5 minutes were used only to check that the pipeline is wired correctly:
it beats the published baseline there, and the predicted possession count (14 vs 17) shows no
flicker. On the full match the output has 245 possessions and 119 stoppages, with `D` on 30% of
graded frames.

## 6. Limitations and what we'd fix with more time

- The share of `D` (30%) is probably low. Ball-in-play time in LaLiga is usually about 55%.
  Stoppages for fouls, where the ball stays inside the pitch and is untracked, are the most likely
  misses. A dedicated stoppage detector (players walking back into shape, the ball stationary at a
  spot) would help.
- Untagged restarts (fouls, untagged throw-ins) get no restart-side evidence.
- The 46 track slots are used without identity. A tracklet-linking step would make "same player
  keeps the ball" an explicit signal.
- The ball-free model is linear and trained on ball-visible frames. Frames where the ball is
  missing are more crowded, so its accuracy there is probably lower than measured.
- The broadcast feed could recover the ball in some tactical gaps, but it needs alignment first.

## 7. Reproduce

```bash
pip install -r cmo/requirements.txt
python cmo/run.py --out evaluation/results/<team_name>.csv
```

Deterministic. Two consecutive runs give the same MD5.
