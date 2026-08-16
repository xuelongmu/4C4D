# Lessons from the 2026-08 quality campaign

What the Xuelong 8-camera experiment series actually taught, with the numbers
that taught it. The protocol these lessons produced is
`docs/EXPERIMENT_METHODOLOGY.md`; the raw run logs and tables are in
`docs/experiments/`.

All figures are held-out PSNR at iteration 7,500 on the 8-camera split (train
`0,1,2,3,5,7,8,9`, held out `4,6`), `--res 2`, batch 4, seed 42 unless a seed
is named. Full-run held-out variance on this benchmark is **±0.4 dB**.

---

## 1. Most "obviously correct" changes did not survive full scale

Five quality-affecting changes were taken to a full A/B on the strength of
being clearly right in the code — a criterion that is dead, a value never
clamped, a rule from the literature, a decay applied the wrong number of
times, a capacity cap. **Three of the five regressed or evaporated.**

| Change | Predicted | Measured (held-out) | Verdict |
| --- | --- | --- | --- |
| [#5](https://github.com/xuelongmu/4C4D/issues/5) temporal densification criterion | biggest quality lever for fast motion | **-1.4 dB** (19.54 vs 20.98 @6000), train **+1.1 dB** | reverted |
| [#10](https://github.com/xuelongmu/4C4D/issues/10) t-clamp + temporal prune | free memory, mild quality gain | **-1.8 dB** (18.56 vs 20.34), train **-0.85 dB** | reverted |
| [#16](https://github.com/xuelongmu/4C4D/issues/16) sqrt-batch LR scaling | faster convergence | 20.91 / 20.19 / 19.42 across seeds | rejected |
| [#7](https://github.com/xuelongmu/4C4D/issues/7) once-per-step opacity decay | correctness fix | **+0.6 dB** (20.92 vs 20.34), train **+1.2 dB** | adopted |
| [#18](https://github.com/xuelongmu/4C4D/issues/18) 1M gaussian budget | speed at some quality cost | **+0.2 dB** at half the gaussians | adopted |

The three failures fail in three different ways, and each is worth naming
separately.

### 1a. #5 — restoring a dead criterion added overfitting capacity

`densify_and_prune` computed `grads_t` and passed `grad_t_threshold` into
clone and split, and neither body ever read them. Restoring the 4DGS
OR-condition (`grads >= threshold OR grads_t >= grad_t_threshold`) did exactly
what it was supposed to do — it densified fast-motion regions — and that made
the model worse on views it had never seen: **train +1.1 dB, held-out
-1.4 dB.**

Under 8 cameras the binding constraint is not representational capacity in
dynamic regions, it is evidence. Added capacity gets spent memorizing the
training views. The criterion is not wrong; the raw threshold is. It moves to
[#24](https://github.com/xuelongmu/4C4D/issues/24) with lifespan-adaptive
thresholds and a held-out acceptance gate.

### 1b. #10 — pruning at exactly the render gate caused prune-refill churn

Clamping `new_t` at split and pruning gaussians whose temporal marginal never
reaches the rasterizer's 0.05 gate cost **-1.8 dB held-out and -0.85 dB
train** on its own — the largest single regression in the campaign, and the
only one that hurt both populations, which is what identified it as suppressed
fitting rather than overfitting.

Mechanism: `t` drifts during optimization, so a gaussian sitting near the gate
crosses it, gets irreversibly pruned, and densification pays to rebuild it.
Pruning exactly at a threshold the optimizer is actively wandering across
converts a stable population into churn. Any redesign needs a safety margin
well beyond the gate, an opacity precondition, or a final-cleanup-pass-only
policy — and each variant needs its own held-out gate.

### 1c. #16 — sqrt-batch LR was variance, not signal

The Grendel-GS sqrt(batch) rule measured **+0.5 dB** on first run (20.91 vs
20.39) and was recommended for adoption. It then measured 20.19 with the GPU
cache, and 19.42 on seed 43: **20.91 / 20.19 / 19.42**, a 1.5 dB spread
straddling a baseline that is itself stable at 20.4-20.6.

Doubled LRs did not add quality to this pipeline; they added variance. The
first number was sampling luck, and it was nearly adopted because it arrived
with a plausible mechanism and a citation. **This is the lesson that set the
two-seed rule**: a sub-1 dB single-run delta on this benchmark carries almost
no information.

---

## 2. A change can work exactly as designed and still be worthless

Per-camera color affine
([#21](https://github.com/xuelongmu/4C4D/issues/21)) learned substantial
corrections — the raw-render train evaluation dropped ~4.8 dB (23.38 vs
28.15), which is direct evidence that the affines were absorbing real
per-camera color mismatch rather than sitting near identity. Held-out PSNR:
**20.06 vs ~20.3 baseline. No gain.**

View-dependent spherical harmonics were already absorbing this rig's color
inconsistency. The affine did not remove an error; it relocated one, and
freed no capacity worth the trade.

The flag was merged off by default rather than deleted — it is the right tool
for a rig with worse color consistency, and the finding is dataset-specific,
not a statement about the method. The production answer for this rig is
chart-based static color calibration at ingest, before SH ever sees the
mismatch.

**Generalization: "the parameter learned something meaningful" is not
evidence of improvement.** Only the held-out metric is.

---

## 3. Fast gates cannot see full-scale dynamics — structurally

Every Tier-0 fix passed its 700-iteration smoke gate. The bundle of them lost
**1.9 dB held-out** at 7,500 iterations.

| Fix | Smoke held-out (baseline 18.03, ±0.4) | Full A/B held-out |
| --- | ---: | ---: |
| #5 temporal densification | 18.22 (pass) | -1.4 dB |
| #7 once-per-step decay | 18.01 (pass) | +0.6 dB |
| #10 t-clamp + temporal prune | 17.78 (pass) | -1.8 dB |

The smoke gate is not badly calibrated — it is measuring a different system.
At 700 iterations the model holds ~73k gaussians against a full run's ~2.1M.
Every regression above is mediated by how capacity is *allocated over the
course of densification*, a process that has barely started at iteration 700.
The ±0.4 dB smoke noise band is wider than every real effect in the campaign
except #10.

So the gate's job is crashes and collapses, and it is good at that and cheap
(~2 min). It is not a small version of the real measurement, and no amount of
tightening its threshold would make it one. The same reasoning applies to any
future cheap proxy: ask what mechanism it is supposed to observe and whether
that mechanism has run yet.

A corollary from the same episode: **bundles hide regressions.** Eight fixes
merged sequentially produced one uninterpretable number. Two bisect rounds
(~6 GPU-hours) recovered one confirmed win, two regressions with mechanisms,
and a shippable 30% speedup from what would otherwise have been a discarded
branch.

---

## 4. Train-view and held-out PSNR move independently

In this sparse-view regime they are close to unrelated, and in several
experiments they moved in opposite directions:

| Run | Train | Held-out |
| --- | ---: | ---: |
| #5 temporal densification | **+1.1** | **-1.4** |
| 1M budget cap | **-0.9** | **+0.2** |
| Tier-0 bundle | -1.3 | -1.9 |
| #10 temporal prune | -0.85 | -1.8 |

The sign disagreement in the first two rows is the whole story of the
campaign: the project's bottleneck is sparse-view generalization, not
train-view fidelity, which was already at 28 dB while held-out sat at 20.

Both directions matter diagnostically. When train and held-out fall *together*
(#10, the bundle) the change is suppressing fitting — a defect. When train
rises and held-out falls (#5), the change is buying memorization. Reporting
only one number makes these indistinguishable.

Baseline for scale: 6-cam train 28.0 vs held-out 15.3 dB; 8-cam 26.9 vs
20.3 dB.

---

## 5. Capacity above ~1M gaussians on this scene is pure overfitting

Capping the budget at 1M (`--max_num_pts 1000000`, i.e.
`densify_until_num_points`) against an uncapped ~2.1M baseline:

| | Uncapped | 1M cap |
| --- | ---: | ---: |
| Gaussians | 2,086,435 | 1,013,587 |
| Train PSNR | 28.15 | 27.23 |
| **Held-out PSNR** | 20.39 | **20.61** |
| Wall | 32:43 | 26:57 |
| Checkpoint | — | half |

The second million gaussians bought 0.9 dB of train-view fidelity and cost
held-out quality, wall time, and model size. It was not capacity the scene
needed; it was capacity the optimizer could only spend on memorization.

A 1.5M point on the curve was not measured and is not needed unless
train-view fidelity becomes a deliverable in its own right. Treat ~1M as the
knee **for this scene at this camera count** — it is a property of the
evidence available, not a universal constant.

---

## 6. What actually worked, and what the wins have in common

| Change | Held-out | Other | Status |
| --- | --- | --- | --- |
| [#7](https://github.com/xuelongmu/4C4D/issues/7) once-per-step batch-compensated opacity decay | **+0.6 dB** | train +1.2 dB, faster | adopted |
| [#18](https://github.com/xuelongmu/4C4D/issues/18) 1M gaussian budget | **+0.2 dB** | half the model, -18% wall | adopted |
| [#15](https://github.com/xuelongmu/4C4D/issues/15) GPU-resident uint8 image cache | parity (quality-inert) | **-17% wall** (understated: contended GPU) | adopted |
| [#20](https://github.com/xuelongmu/4C4D/issues/20) static-temporal freeze | **+0.6 / +0.7 dB** (seeds 42 / 43) | train +1 dB, wall unchanged | adopted |

Net for the campaign: **~47 min → 18:29** for 7,500 iterations at held-out
parity or better, with half the model size — and then +0.6-0.7 dB on top from
the static freeze.

Three things the winners share:

- **They constrain rather than add.** Decaying opacity once per step instead
  of four times, capping the population, freezing temporal parameters of
  gaussians that do not move in time. The losers all *added* degrees of
  freedom (temporal densification capacity, doubled LRs, per-camera affines).
  Under 8 cameras, the scarce resource is evidence, and constraints act as
  regularizers.
- **The #7 subtlety is worth stating twice.** Hoisting the decay out of the
  batch loop is a pure correctness fix, but the naive hoist weakens the
  effective decay from `factor^batch_size` to `factor^1` — the first attempt
  measured *worse* on the smoke gate (17.69) for exactly this reason. The
  adopted version compounds the factor by `batch_size` so the shipped batch-4
  behaviour is preserved. A correctness fix that silently changes a
  hyperparameter is two changes.
- **The static freeze punches above its footprint.** Only ~2.5% of gaussians
  end up frozen, yet it is the largest quality win in the campaign. The gain
  is very unlikely to come from those 2.5% directly; it most plausibly comes
  from stabilizing background geometry early so the rest of the optimization
  is not chasing a moving reference. That reading is what makes the full
  static/dynamic split (#20: 3D-SH background subset, background pretraining)
  worth the larger implementation cost.

---

## 7. Process notes that paid for themselves

- **Absolute-path config and explicit `--res`.** Two full ablations were
  discarded because `--res` defaults to 1 and is applied after the YAML merge.
  Cost: several GPU-hours and one round of wrong conclusions.
- **Preserve every log.** `<run_dir>/train.log` plus `training_params.txt` is
  what made bisecting possible weeks later, and what
  `scripts/build_experiment_report.py` reads. Shell history is not a record.
- **Comment results back onto the issue, including rejections.** #5, #10, #16
  and #21 each carry the failing numbers and a mechanism on their thread. That
  is what stops the next person re-running them, and what turned #5 into a
  concrete design constraint for #24 instead of a lost afternoon.
- **Revert cleanly instead of tuning under pressure.** A reverted change with
  a recorded mechanism is a result. A quickly re-tuned one is a new
  unvalidated variable in a bundle you already do not trust.

---

## Open, with the evidence that motivates them

- [#24](https://github.com/xuelongmu/4C4D/issues/24) temporal densification
  with lifespan-adaptive thresholds — absorbs #5's criterion; must clear a
  held-out gate, since the raw form cost 1.4 dB.
- [#10](https://github.com/xuelongmu/4C4D/issues/10) temporal prune redesign —
  margin beyond the render gate, opacity precondition, or final-pass only.
- [#20](https://github.com/xuelongmu/4C4D/issues/20) full static/dynamic split
  with a 3D-SH background subset — the freeze result raises confidence in its
  ceiling.
- [#19](https://github.com/xuelongmu/4C4D/issues/19) MASt3R / video-depth
  supervision — the largest untested lever against the held-out gap, and the
  literature's +1-2 dB class in sparse regimes.
- [#21](https://github.com/xuelongmu/4C4D/issues/21) SE(3) pose deltas (the
  color-affine half is settled; the pose half needs camera gradients in the
  rasterizer).
- [#17](https://github.com/xuelongmu/4C4D/issues/17) sparse Adam,
  [#23](https://github.com/xuelongmu/4C4D/issues/23) absgrad,
  [#25](https://github.com/xuelongmu/4C4D/issues/25) progressive resolution.

At 18-minute runs, each of these is cheap to iterate — which is itself a
result of the speed work, and the main reason it was done first.
