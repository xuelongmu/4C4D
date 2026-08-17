# The held-out metric scored 3 images of 300 — campaign re-derivation

Date: 2026-08-17

Status: complete. Every quality conclusion in this campaign was read off a
3-image average. All decision-critical runs have been re-scored on the full
held-out set from saved checkpoints; two adopted decisions reverse.

## The defect

`Scene.getValidationCameras(scale, tag, num)` treats `num` as a **stride**,
not a count, and every caller used the default `num=100`. The Xuelong
held-out set is 300 images (cam04 + cam06 x 150 frames), so training reported
held-out PSNR from **3 images**. The train-view metric used 12 of 1,200.

This was invisible: the number was plausible, moved in believable directions,
and no log line states the population size.

### Measured cost

Same config, same seed 42, three runs (all 300 images vs the 3-image metric):

| run | 3-image | 300-image | error |
|---|---:|---:|---:|
| fx-ctl-s42 | 19.36 | 19.45 | +0.09 |
| fx-ctl-s43 | 20.35 | 20.03 | −0.32 |
| fx-ctl-s44 | 21.04 | 20.33 | **−0.71** |

Same-config spread: **1.68 dB on 3 images, 0.88 dB on 300**. Roughly half the
apparent run-to-run variance was measurement error. Sub-dB effects — which is
every quality effect this campaign chased — were unreadable.

## Re-derivation

All runs re-scored at `chkpnt7000` (the checkpoint every historical run kept)
on all 300 held-out images with `scripts/evaluate_full_heldout.py`:

| run | 300-image PSNR | vs its baseline | original claim | verdict |
|---|---:|---:|---|---|
| `ab8-control` (pre-fix, 2M) | 20.15 | — | baseline | — |
| `ab8-fix7` (#7 decay) | 20.51 | **+0.37** | +0.6 dB win | **holds**, smaller |
| `ab8-fix5` (#5 temporal densify) | 16.72 | **−3.43** | −1.4 dB | **holds**, far worse |
| `ab8-fix10` (#10 temporal prune) | 18.63 | −1.52 | −1.8 dB | **holds** |
| `ab8-ship` (all kept fixes, 2M) | 20.32 | +0.17 vs control | parity | **holds** |
| `ab8-shipcache` (#15 cache) | 20.18 | −0.14 vs ship | quality-inert | **holds** |
| `ab8-sqrtlr` (#16) | 20.28 | −0.04 vs ship | rejected | **holds** (neutral, not harmful) |
| `ab8-budget1m` (#18 1M cap) | 19.96 | **−0.36 vs ship** | **+0.2 dB win, adopted** | **REVERSED** |
| `ab8-fastprofile` (budget+cache) | 19.28 | −1.04 vs ship | production default | **REVERSED** |
| `ab8-fast-s43` (same, seed 43) | 19.92 | −0.40 vs ship | production default | **REVERSED** |
| `ab8-staticfreeze` (#20 lite, s42) | 20.34 | — | +0.73 dB | **no effect** |
| `ab8-staticfreeze-s43` | 20.41 | — | +0.56 dB | **no effect** |
| `ab8-staticfreeze-s44` | 20.17 | — | −0.29 dB | **no effect** |
| `ab8-nofreeze-s44` (control) | 20.35 | — | control | — |

## What changes

1. **The 1M gaussian budget (#18) is a trade, not a win.** It costs
   **0.36 dB** against the uncapped ship tip, and the production profile
   (budget + cache) sits 0.40-1.04 dB below it. It still halves the model and
   cuts wall time from 32:43 to 18:29 — a legitimate trade to offer, but it
   must be labelled as buying speed and size with quality, not as free.
2. **The campaign headline needs splitting.** "~47 min → 18:29 at held-out
   parity" is false as one claim. Correctly:
   - **Bug fixes alone**: 20.15 → 20.32 at 32:43 vs ~47 min — *faster at
     parity or slightly better*. This holds.
   - **Adding the 1M budget**: 18:29, but 0.4-1.0 dB below the ship tip. A
     trade.
3. **The static freeze (#20-lite) has no effect**, on any implementation.
   The re-scored arms (20.34 / 20.41 / 20.17) bracket the control (20.35).
   This independently confirms the structural finding in
   `2026-08-16-issue-20-static-dynamic-split.md`: the whole-clip criterion is
   unreachable at initialization and densification collapses temporal support
   geometrically, so almost nothing is ever classified static.
4. **Rejections all hold, and #5 was much worse than recorded** (−3.43 dB, not
   −1.4). Large effects survived the bad metric; only sub-dB claims were
   corrupted — which is exactly the expected failure pattern.

## Fixes landed

- `--val_stride` now defaults to **1** (every held-out image, ~25 s per
  evaluation, ~3 min over a 7,500-iteration run). The coarse stride survives
  for the train metric as `--val_stride_train`.
- `scripts/evaluate_full_heldout.py` re-scores saved checkpoints on the full
  held-out set, so past experiments can be re-compared without retraining
  (~1 minute per checkpoint). This is how the table above was produced.
- The docstring on `getValidationCameras` now states that `num` is a stride
  and what the default did.

## Process lessons

1. **State the population size next to every metric.** "PSNR 20.4" and
   "PSNR 20.4 (n=3)" are different claims, and only the second is auditable.
   The reporter should log `n` alongside the value.
2. **Measure the instrument before trusting it.** Repeating one configuration
   under a fixed seed and reporting the spread costs one extra run and bounds
   what any experiment can resolve. Doing that first would have exposed this
   before a single adoption decision.
3. **Re-scoring beats re-running.** Checkpoints made the entire campaign
   recoverable for ~15 minutes of GPU time. Keeping a final checkpoint per run
   is cheap insurance against exactly this class of error.
4. This is the third instance of the same root cause in this campaign, after
   the batch-dependent opacity decay and the freeze that never froze:
   **a mechanism was assumed to do what its name implied and was never
   verified.** The other two were caught by review; this one was caught only
   because a variance anomaly forced an audit.
