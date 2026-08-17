# Experiment methodology

The protocol used for the 2026-08 training speed and quality campaign
(`docs/roadmap-2026-08-training-speed-quality.md`), revised as it was
validated. It exists because the first bundle of "obviously correct" bug
fixes shipped a 1.9 dB held-out regression that the fast gate could not see;
every rule below is the response to a specific way that campaign was misled.
The lessons that motivate the rules, with numbers, are in
`docs/LESSONS.md`.

Read this before running an experiment on this repo. It is a protocol, not a
tutorial: it assumes the training commands in `README.md` already work.

## Summary

| Stage | Cost | Gates | Cannot see |
| --- | --- | --- | --- |
| Mechanism check (assertion / unit test) | minutes | *the change does what its name says* | whether it helps |
| Smoke (`scripts/smoke_test.sh`) | ~2 min | crashes, collapses, obvious perf cliffs | anything under ~0.4 dB; all late-training dynamics |
| Full A/B (`scripts/ab_launch.sh`) | ~20-35 min/side | held-out quality, wall time, model size | whether the arm measured the intended feature |
| Multi-seed replication | 2-3x full A/B | sampling luck | a treatment arm that is broken in every seed |

No quality claim may rest on a smoke run. No sub-1 dB adoption may rest on a
single seed. **And no A/B is interpretable until the mechanism check has
passed** — the campaign's headline quality win turned out to be a feature that
was only ~8% effective plus a random per-seed perturbation, replicated across
two seeds before anyone checked that it did anything (`LESSONS.md` §7).

## The fixed benchmark

Every experiment in the campaign used one benchmark so that runs months apart
remain comparable. Change it only with a documented reason.

- **Scene:** the Xuelong `clip_f300_5s` posefix capture (10 synchronized RGB
  cameras, 150 frames / 5 s, 2560x1440 source). Preparation and calibration
  provenance: `docs/experiments/2026-08-08-xuelong-depthkit-rgb.md`.
- **Split:** train `0,1,2,3,5,7,8,9`, held out `4,6` — the "8-cam split".
  The secondary stress split is train `0,1,2,5,8,9`, held out `3,4,6,7`.
  The 10-camera all-train configuration has **no held-out metric** and is a
  production fit, never an experiment.
- **Resolution:** res 2. This used to require passing `--res 2` explicitly on
  every command — the flag defaults to 1 and was applied *after* the YAML
  merge, so omitting it silently trained at full resolution and invalidated two
  early ablations. `train.py` now gives an explicitly typed flag precedence over
  the config and only applies alias defaults (`--res`, `--initial_num_pts`,
  `--weight_decay`) when the config leaves that key alone, so a config with
  `resolution: 2` is safe on its own. The older experiment docs still carry the
  old instruction.
- **Length:** 7,500 iterations, batch 4, seed 42 (seed 43 for replication).
- **Primary metric: held-out PSNR at iteration 7,500.** Train PSNR is
  secondary and is reported alongside, never in place of it — the two move
  independently and in several experiments moved in opposite directions.
- **Also report:** wall clock, final gaussian count, peak VRAM, checkpoint
  size. Extract SSIM/LPIPS from TensorBoard; the text log prints PSNR only.

## One variable per experiment

Each experiment changes exactly one thing against a named baseline, and the
baseline is re-stated in the results table by run name (`ab8-ship`,
`ab8-fastprofile`, ...) rather than by description.

Combining two independently validated wins is itself an experiment. The
1M-budget cap and sqrt-batch LR were each measured positive alone and
regressed 0.7-1.0 dB when stacked; profiles are validated as whole
configurations, not assembled from per-flag deltas.

Speed-only changes that are intended to be numerically inert (sync removal,
caching, checkpoint hygiene, export vectorization) are exempt from the full
quality A/B and verified by loss-curve overlay against the control instead —
but they still get a full-length run, because the wall-time claim is the
point of them.

## Stage 0 — the mechanism check

Before any training run, prove that the change does the thing it is named
after, as a direct assertion on the mechanism rather than on an outcome:

- freezing something → assert the frozen values are bit-identical across an
  optimizer step, *with warmed-up optimizer state*;
- pruning/masking by index → assert the selected rows are the intended ones
  after a densification and a prune, not merely that the mask length matches;
- a rate or schedule change → assert the effective value at a few iterations;
- a caching or hoisting change → assert numerical equivalence against the
  unhoisted path on one batch.

Two properties matter. It must exercise the **stateful** path — zeroing a
gradient looks like a perfect freeze on a fresh optimizer and does nothing once
Adam carries momentum, which is precisely how the static freeze passed every
inspection for a week. And it must survive the **identity-changing** operations
in the loop: densification appends rows, pruning compacts them, so any
index-based structure has to be re-derived or invalidated at those points.

Keep the check as a regression test, not a scratch script. This stage costs
minutes and, in the one case where it was skipped, would have saved three
7,500-iteration runs, a two-seed replication, a production-config change, and
a round of downstream sessions briefed on void control numbers.

## Stage 1 — the smoke gate

```bash
scripts/smoke_test.sh <run_name> [gpu_index] [extra train.py args...]
```

700 iterations at `--res 4` on the 8-camera split, ~2 minutes on one A6000.
Run it from the root of the worktree whose code is under test. It exercises
densification (from iter 200), the neural opacity-decay path (from iter 500),
SH degree increases, and held-out evaluation, then prints a one-line summary
parsed from the log.

**Measured no-op-change spread on held-out PSNR at this scale is about
±0.4 dB.** The gate therefore answers exactly one question: *did this change
break or collapse training?* Read it as pass/fail, not as a measurement.

Two properties make it structurally unable to substitute for a full run:

1. At 700 iterations the model is at ~73k gaussians, roughly 1/28th of a full
   run's capacity. Changes whose effect is mediated by capacity (densification
   criteria, budget caps, pruning) have barely begun to act.
2. Smoke wall-time differences are dominated by page-cache warmth, not by
   code. Never quote a smoke wall time as a speed result.

The Tier-0 bundle passed every smoke gate individually and lost 1.9 dB
held-out at full scale. That is the gate working as designed, not failing.

## Stage 2 — the full A/B

```bash
export FOURC4D_DATASET=/path/to/converted-scene
export FOURC4D_OUTPUT=/path/to/output-root
scripts/ab_launch.sh <worktree_dir> <output_name> <gpu> [extra train.py args...]
```

Launches a detached 7,500-iteration run at `--res 2` on the 8-camera split and
copies the log to `<run_dir>/train.log` on completion (see *Logs* below).

Rules:

- **Run control and variant concurrently**, one per GPU, from two worktrees at
  the two commits under comparison. Same config file, same seed, same split.
- **Pass one config path to both sides.** `ab_launch.sh` resolves `--config`
  to an absolute path at launch, so both worktrees train against the config
  the launcher saw, not each worktree's own copy.
- **Note GPU contention.** A GPU carrying the interactive viewer (~3.5 GB)
  inflates wall time; cross-GPU wall-time comparisons under contention are
  indicative only. Quality metrics are unaffected. Mark contended rows.
- `train.py` refuses to start if the output folder already exists. Use a fresh
  `output_dir` per run; never overwrite a run you may want to re-read.
- **Re-run the control from the same base revision as the arm.** Historical
  control numbers cannot be carried across a base-code change. Reusing them is
  how void numbers propagated into downstream sessions during this campaign.

**Same-config seed variance is small: 0.09 dB** — three seeds of the production
profile at 20.48 / 20.39 / 20.44. A ±0.4 dB working band was used for most of
the campaign and appears throughout the older experiment logs; it was assembled
from runs that were not in fact the same configuration, and was then defended
rather than questioned when one arm started swinging by 1 dB. Treat 0.1-0.2 dB
as the honest same-config band and be correspondingly suspicious of a wide one.

**Variance asymmetry between arms is a bug signal.** If the treatment arm is
much noisier than the control, audit the treatment code before averaging more
seeds. A change that merely helps or does not help has no mechanism by which to
multiply run-to-run variance tenfold; one that is randomly perturbing state
does. Widening the error bars is the wrong instinct and cost this campaign a
full extra seed plus a retracted adoption.

## Replication before adoption

- **Delta > 1 dB on the primary metric:** one seed may be enough to adopt,
  with a replication run queued behind it.
- **Delta < 1 dB:** requires **three paired seeds** before adoption — the arm
  and its control at each seed, all from the same base revision. Two is not
  enough: it is enough to reject sampling luck (sqrt-batch LR measured 20.91 /
  20.19 / 19.42 and was correctly rejected on the seed-43 pass) but not enough
  to size the variance you are testing against, and the third seed is what
  exposed the static freeze.
- **Pre-register the interpretation** when a replication run is launched to
  settle a disputed result: write down, before it lands, which outcome means
  adopt and which means revert. This was done for the seed-44 control and is
  the reason that result was acted on rather than argued with.
- **A change that is supposed to be quality-inert** (caching, sync removal):
  parity within the same-config band is a pass, and the wall-time win is the
  result. Prefer proving inertness directly — a loss-curve overlay, or a
  numerical-equivalence assertion at Stage 0 — over inferring it from a
  final-metric delta.

Record the rejected runs. Half the value of the experiment log is the list of
plausible ideas that did not survive.

## Bisect procedure when a bundle regresses

Used when the Tier-0 bundle (fixes #5-#12) lost 1.9 dB held-out relative to
control despite every constituent fix passing its smoke gate.

1. **Do not debug forward from the bundle.** Return to the pre-change base
   commit and apply candidate fixes to *it*, one at a time.
2. **Shortlist by mechanism, not by suspicion.** Partition the bundle into
   changes that can affect the objective (densification, pruning, decay,
   LR, loss) and changes that cannot (I/O, logging, CLI plumbing, export).
   Only the first group needs single-fix runs. In the Tier-0 bundle that cut
   eight fixes to three candidates.
3. **Run each candidate alone against control**, same config/seed/split, full
   length. Round 1 isolated #7 as a genuine +0.6 dB win and #5 as a -1.4 dB
   held-out regression, and showed neither explained the bundle's train-side
   suppression.
4. **Then run bundle-minus-candidate** to catch what single-fix runs miss.
   Round 2 identified #10 as the primary regression (-1.8 dB held-out,
   -0.85 dB train alone) and confirmed that #5's cost stacked on top of it.
5. **Revert, do not patch under pressure.** Both #5 and #10 were reverted from
   the integration branch with the failing evidence written into their issue
   threads and folded into follow-up designs (#24 for the temporal criterion,
   a margin-based redesign for the prune). A reverted change with a recorded
   mechanism is a result; a hastily tuned one is a new unvalidated variable.
6. **Re-validate the ship tip** against the original control before declaring
   the campaign clean.

Two bisect rounds cost about six GPU-hours and turned a discarded bundle into
one confirmed win, two documented regressions with mechanisms, and a shippable
30% speedup.

## Branch, worktree, and PR structure

**One issue, one branch, one worktree.** Each fix is developed in a dedicated
`git worktree` under `~/4C4D-worktrees/` (kept outside the repo), which lets
the control and the variant be trained concurrently on the two GPUs from the
same checkout of history. Worktrees are disposable; the branches are not.

Naming used by the campaign:

| Pattern | Purpose |
| --- | --- |
| `base/harness` | frozen pre-change base commit; the control for every A/B |
| `fix/issue-N-<slug>` | one Tier-0 bug fix, branched from `base/harness` |
| `pr/issue-N-<slug>` | cherry-picked review unit in the stacked chain |
| `agent/quality-fixes` | integration branch: real merge history + docs |

**Stacked PRs.** Each issue gets its own PR so it can be reviewed against its
own evidence, and the PRs are stacked base-on-head so a reviewer sees one
change per diff:

```
agent/depthkit-rgb-calibration
  └── base/harness .................................. #27 (harness + roadmap)
        ├── fix/issue-6-decay-mlp-args ............... #28   independent
        ├── fix/issue-7-decay-once-per-step .......... #29   single-fix PRs,
        ├── fix/issue-8-best-checkpoint .............. #30   each on the
        ├── fix/issue-9-loss-plumbing ................ #31   frozen base
        ├── fix/issue-11-test-iters .................. #32
        ├── fix/issue-12-cli-toggles ................. #33
        ├── pr/issue-13-sync-stalls .................. #37   ordered chain:
        │     └── pr/issue-14-hoist-visibility ....... #34   each PR's base
        │           └── pr/issue-21-color-affine ..... #38   is the previous
        │                 └── pr/issue-26-fast-ply ... #35   PR's head
        │                       └── pr/issue-15-gpu-cache .. #39
        │                             └── pr/issue-20-static-freeze .. #40
        └── agent/quality-fixes ...................... #36 (landing PR)
```

Independent fixes that touch disjoint code sit side by side on the base. Only
changes that genuinely build on each other are chained; the chain order is the
merge order. The integration PR is the landing unit and carries the merge
history, the experiment docs, the reverts, and the validated production
config — the per-issue PRs remain the review units.

**Every PR body carries its measurement**, or states that the change is
quality-inert and why. A PR that says only what it does is not reviewable
against this protocol.

**Every experiment result goes back onto its issue thread** as a comment with
the numbers and the verdict, including rejections. The issue thread, not the
PR, is the durable record of why a change was or was not adopted.

## Logs and reporting

- **Every run's log is preserved as `<run_dir>/train.log`**, where `run_dir` is
  `<ModelParams.model_path>/<output_dir>` from the merged config — for the
  custom configs that is `$FOURC4D_OUTPUT/<output_dir>`, since they interpolate
  `${oc.env:FOURC4D_DATASET}` and `${oc.env:FOURC4D_OUTPUT}` rather than
  committing machine paths. `ab_launch.sh` writes to a temporary file while
  training (the run directory must not exist at launch) and copies it into
  place at exit. If you launch `train.py` by hand, copy the log in yourself —
  the report generator reads nothing else.
- `train.py` also writes `training_params.txt` (the fully merged argument set)
  into the run directory. That file, not the shell history, is the record of
  what a run actually did.
- **Comparison report:**

  ```bash
  python scripts/build_experiment_report.py \
      --output-root "$FOURC4D_OUTPUT" \
      --out /path/outside/the/repo/report.html
  ```

  It scans the run directories listed in its `RUNS` manifest for `train.log`
  and `rendered_images/`, and emits one self-contained HTML file: held-out
  PSNR trajectories, a per-run card with the held-out probe render
  (press-and-hold to flip to ground truth), and a summary table with the
  verdict for each run.

  Adding an experiment means adding a row to the `RUNS` manifest at the top of
  the script, with its phase and one-line verdict. Keep the manifest in commit
  order so the report reads as the campaign's narrative.

  **The report embeds capture imagery whose redistribution licensing is not
  established.** Write it outside the repository and do not host it publicly.

## Checklist

Before launching:

- [ ] A mechanism check asserts the change does what it is named after, with
      warmed-up optimizer state and across a densify/prune cycle.
- [ ] Resolution is res 2 — from the config, or from an explicit `--res 2`.
- [ ] Split is the 8-camera split unless the experiment is about the split.
- [ ] Exactly one variable differs from the named baseline.
- [ ] Control is running concurrently, from the base worktree at the **same
      base revision**, same seed.
- [ ] `output_dir` is fresh and named after the experiment.
- [ ] Contention on either GPU is noted.

Before adopting:

- [ ] Held-out PSNR is the number being claimed.
- [ ] The treatment arm's seed spread is comparable to the control's; if it is
      much wider, audit the code instead of adopting.
- [ ] Sub-1 dB delta replicated across three paired seeds.
- [ ] Result and verdict commented on the issue thread, including the
      mechanism if it regressed.
- [ ] Run added to the `RUNS` manifest in `build_experiment_report.py`.
- [ ] Result appended to the relevant file in `docs/experiments/` — appended,
      not overwritten; superseded entries stay with their correction.
