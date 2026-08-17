# Issue #20 — static/dynamic decomposition, full version (2026-08-16)

Follow-up to the lite slice (`--freeze_static_temporal`, PR #40), which won
+0.6–0.7 dB held-out on two seeds while freezing only ~2.5% of gaussians. This
round answers why the frozen set was that small, and tests what happens when the
background is constructed rather than detected.

Code: branch `enh/issue-20-static-full`, worktree
`~/4C4D-worktrees/issue-20-full`. Diagnostics: `scripts/temporal_stats.py`.

**Result: negative, and consistently so.** The 2.5% frozen fraction was explained
(§1) and the split raises it to 30% (§4), but every arm loses held-out PSNR, and
the loss grows monotonically with how much frozen whole-clip background the model
carries — up to −5.3 dB. A second, unrelated defect turned up on the way: the
clip is twice as long as the config declares (§2), and correcting it *also* costs
held-out quality. Nothing here is recommended for adoption; everything is
flag-gated and off by default so the result is reproducible. See §5.

## 1. Why the frozen set was 2.5%

The lite mask calls a gaussian static when its temporal marginal
`exp(-0.5 (t - tau)^2 / cov_t)` clears the renderer's 0.05 gate at both clip
endpoints. Equivalently it covers `t ± w` with

```
w = sqrt(-2 ln(0.05) * cov_t) ≈ sqrt(5.99 * cov_t)
```

and is static iff `w ≥ max(t - t0, t1 - t)`.

`scripts/temporal_stats.py` on `ab8-staticfreeze/chkpnt7000.pth` (N = 1,115,679):

| quantile | 1% | 25% | 50% | 75% | 90% | 99% |
|---|---|---|---|---|---|---|
| `cov_t` | 0.002 | 0.012 | 0.022 | 0.064 | 0.304 | 4.289 |
| half-width `w` | 0.107 | 0.264 | 0.362 | 0.620 | 1.349 | 5.069 |
| clip coverage | 0.000 | 0.000 | 0.069 | 0.148 | 0.303 | 1.000 |

Static fraction 0.0229 (0.0293 opacity-weighted). Two causes, both structural:

**(a) The criterion is unreachable at initialisation.** `create_from_pcd` sets
`dist_t = duration / 5`, so `cov_t = 1.0` and `w = 2.448` — against the 2.500 a
perfectly centred gaussian needs to span `[0, 5]`. Nothing qualifies at iteration
0; the static set can only ever contain gaussians that *grew* their temporal
extent against the loss. That is why the fraction has to climb (19% at iter
1,000) rather than starting high.

**(b) Densification destroys temporal support geometrically.**
`densify_and_split` sets `new_scaling_t = scaling_t / (0.8N)`, i.e. divides by
1.6 per generation, and for `rot_4d` also jitters the child's `t` by the parent's
temporal std. Median `cov_t` falls 1.0 → 0.022 over training, a ~45× variance
reduction, and the median gaussian ends up covering 7% of the clip. This is
exactly the shape of the logged trajectory:

```
iter 1000 0.189   iter 2500 0.012   iter 4500 0.015   iter 6500 0.022
iter 1500 0.102   iter 3000 0.010   iter 5000 0.016   iter 7000 0.023
iter 2000 0.031   iter 3500 0.010   iter 5500 0.019   iter 7500 0.025
```

The collapse spans iterations 500–3500 — precisely the densification window
(`densify_from_iter 500`, `densify_until_iter 3500`) — and recovery only begins
once densification stops.

A static set re-derived each interval from a quantity that densification is
actively shrinking cannot hold a background. The remedy is a persistent label,
not a better threshold.

## 2. A second defect found on the way: the clip is twice as long as configured

`process_camera_info` (`scene/dataset_readers.py:204`) assigns

```python
time_stamp = int(img.split('.')[0][-4:]) / ((max_timestamp + 1.0) / 10.0)
```

which normalises the frame index to `[0, 10)` **regardless of clip length**. For
`clip_f300_5s_rgb_posefix` (150 frames, `Max timestamp is : 149`) the camera
timestamps span `[0, 9.933]`. The config declares `time_duration: [0.0, 5.0]`.

`time_duration` feeds:

- initial `t` sampling — all 75,000 initial gaussians are placed in the first
  half of the clip;
- `scaling_t` init (`duration / 5`) — half the intended temporal extent;
- the rasterizer's `time_duration` argument, i.e. the temporal SH period;
- the lite static criterion, which therefore tests coverage of `[0, 5]` only.

The checkpoint confirms it: `t` runs from −0.35 to 25.1 with median 4.55 and 75th
percentile 7.50, i.e. most gaussians have migrated past the configured `t1 = 5.0`
to reach the data. Against the *true* `[0, 9.933]`, whole-clip coverage would
require `cov_t ≥ 4.12`, which only ~1% of gaussians reach — so the lite freeze's
"whole-clip" set is really "spans the first half".

`--fix_clip_bounds` re-draws `t` and `scaling_t` from the measured range. The
startup warning fires unconditionally so the mismatch stops being invisible.
Fixing `time_duration` at the source (reader or config) is a separate change —
see "Follow-ups".

## 3. What was built

All flag-gated, all off by default.

- `--fix_clip_bounds` — measure the clip from camera timestamps; re-initialise
  temporal parameters from it.
- `--bg_static_split` (implies the two flags above) — partition gaussians into a
  **background layer** (parked at the clip midpoint, whole-clip temporal support,
  temporal gradients frozen) and a **dynamic layer** (ordinary 4D init, spread
  over the clip). `--bg_static_frac`, default 0.5.
  - the label is persistent: carried through `prune_points`,
    `densify_and_clone`, `densify_and_split`;
  - children of a background gaussian are born **dynamic** with a fresh temporal
    extent — the residual that triggered the split is motion the frozen
    background cannot express;
  - a background gaussian **survives its own split**, so the scaffold is not
    consumed by densification; it is removed only by opacity pruning;
  - background gaussians are **excluded from opacity decay**. They are
    temporally visible at every timestamp, so leaving them in the decay set
    applies the factor ~14× more often than to a typical dynamic gaussian. In
    the first smoke this wiped 22% of the background out inside 500 iterations.
- `--bg_pretrain_iters N` (default 0, off) — the literature's formulation: fit
  the whole model to per-camera temporal-median images for N iterations before
  the split. Note that with an L1 loss a temporally constant gaussian is
  *already* pulled toward the per-pixel temporal median of the frames it covers,
  so the pretrain is largely redundant with simply freezing temporal support and
  training on all frames — and it is supervised by 8 median images instead of
  1,500 real ones. Kept as a flag and measured rather than assumed.

## 4. Smoke gate (700 iters, res 4, 8-cam split — crash gate, ±0.4 dB)

| variant | train PSNR | held-out PSNR | final gs | wall |
|---|---|---|---|---|
| control (`--freeze_static_temporal`) | 19.18 | 17.92 | 72,827 | 135 s |
| `--fix_clip_bounds` | 20.10 | **18.73** | 80,381 | 127 s |
| `--bg_static_split` | 20.31 | 17.56 | 75,696 | 149 s |
| `--bg_static_split --bg_pretrain_iters 300` | 18.86 | 14.32 | 63,595 | 184 s |
| median pretrain + duplicated dynamic seed (rejected design) | 18.70 | 14.84 | 131,195 | 278 s |

The smoke gates crashes, not quality — but the median pretrain lands ~3 dB below
control in two independent runs with different downstream designs, far outside
the ±0.4 dB band. The mechanism is visible in the log: after 300 pretrain
iterations the median opacity is 0.029 and 18.9% of gaussians are already below
the prune threshold. Fitting 75,000 gaussians to 8 median images with no
regularisation is itself a severely under-constrained problem on this rig, and it
hands the main run an overfit starting geometry. It is kept last in the full A/B
queue rather than dropped, since 1,500 iterations at res 2 is a different regime
from 300 at res 4.

The rejected design pretrained against medians and then seeded the dynamic layer
by duplicating the background; it overfits the 8 median images and doubles the
gaussian count for no gain. It was replaced by the partition in §3.

`--fix_clip_bounds` is the surprise: +0.8 dB held-out at no wall-cost, from a
one-line-class initialisation fix.

### The split does what it was built to do

`temporal_stats.py` on the `--bg_static_split` smoke checkpoint, measured against
the true clip `[0, 9.933]`:

| | lite run (iter 7000) | `--bg_static_split` (iter 700) |
|---|---|---|
| whole-clip static fraction | 0.023 | **0.420** |
| …opacity-weighted | 0.029 | **0.591** |
| median clip coverage | 0.069 | **0.689** |
| median `cov_t` | 0.022 | 2.742 |

The background holds at 42% after densification from a 50% start — it is
persisting rather than decaying, which is what the persistent label,
split-survival and decay exclusion were for. `cov_t` piles up at 4.98 (the
whole-clip value for `margin 1.1`), the frozen layer showing as a spike in the
distribution. Issue #20's "raise the static fraction" is answered: 2.3% → 42%,
an 18× increase. Whether that converts into held-out PSNR is §5.

## 5. Full A/B (7,500 iters, res 2, train `0,1,2,3,5,7,8,9`, held out `4,6`)

Base flags: `--max_num_pts 1000000 --gpu_cache --freeze_static_temporal`.

| run | seed | train PSNR | **held-out PSNR** | Δ vs control | final gs | wall |
|---|---|---|---|---|---|---|
| `ab8-staticfreeze` (control) | 42 | 27.36 | **21.21** | — | 1,070,964 | ~19 m |
| `ab8-staticfreeze-s43` (control) | 43 | 27.69 | **20.95** | — | 1,065,485 | ~19 m |
| `ab8-clipfix` `--fix_clip_bounds` | 42 | 28.31 | **20.16** | −1.05 | 1,024,716 | 21 m |
| `ab8-clipfix-s43` `--fix_clip_bounds` | 43 | 28.50 | **19.45** | −1.50 | 1,005,427 | 23 m |
| `ab8-bgsplit-s43` (v1 split) | 43 | 26.52 | **17.81** | −3.14 | 1,037,938 | 22 m |
| `ab8-bgsplit-pre` (v1 + 1,500-iter median pretrain) | 42 | 24.45 | **18.39** | −2.82 | 1,099,101 | 24 m |
| `ab8-bgsplit2` (v2 split, frac 0.5) | 42 | 26.31 | **15.89** | −5.32 | 994,614 | 23 m |
| `ab8-bgsplit2-f20` (v2 split, frac 0.2) | 42 | 26.15 | **16.88** | −4.33 | 1,015,759 | 22 m |
| `ab8-bgsplit` (v1 split) | 42 | — | — | — | — | OOM-killed |

### `--fix_clip_bounds` is a real fix that makes held-out worse

Both seeds move the same way and by more than the ±0.4 dB noise band: **+0.9 dB
train, −1.0 to −1.5 dB held-out**. Correcting the temporal initialisation to the
clip the data actually has widens each gaussian's initial temporal support
(`cov_t` 1.0 → 1.99) and spreads `t` over the full range, and the model converts
that extra freedom straight into train-view fit.

This is worth stating plainly: the `time_duration` mismatch of §2 is a genuine
defect, and the accidental narrow temporal init it produces is *helping* held-out
quality on this rig. The smoke's +0.8 dB pointed the other way — 700 iterations
at res 4 is not a proxy for 7,500 at res 2, exactly as the protocol warns.
**Not adopted.**

### The v1 split diluted itself away

−3.14 dB on seed 43, and the background fraction tells the story:

```
iter 6000 0.024 (24,804 bg / 1,026,887 dyn)   iter 7000 0.022 (24,515 bg / 1,071,946 dyn)
iter 6500 0.024 (24,645 bg /   987,998 dyn)   iter 7500 0.023 (24,416 bg / 1,037,938 dyn)
```

v1 made every child dynamic, on the theory that a residual triggering a split is
motion the frozen background cannot express. Over a 75k → 1M population that
rule diluted a 50% background to 2.3% — numerically where the lite version
landed, by a different route. The run also inherited the `--fix_clip_bounds`
handicap above, which v1 implied, so ~1.5 dB of the loss is not attributable to
the split itself.

v2 (commit `ee6043a`) fixes both: children inherit their parent's layer, a
background child keeps its parent's temporal parameters rather than the usual
`scaling_t / 0.8N` shrinkage, and the dynamic layer's init is decoupled from the
clip fix. At smoke scale the background now holds at 51% instead of diluting.

### The median pretrain fails at full scale too

Issue #20's step 1, run as specified — 1,500 pretrain iterations against
per-camera temporal medians before the split — lands at **18.39 dB held-out,
−2.82 vs its seed-42 control**, and 4 dB below control on train. The pretrain's
own diagnostics say why: after 1,500 iterations **28.4% of gaussians are already
below the prune threshold** (up from 18.9% at 300 iterations — the longer it
converges, the more it kills), and the opacity distribution is bimodal, deciles
`0.004 0.004 0.027 0.609 0.996`.

Fitting 75,000 gaussians to 8 median images is a severely under-constrained
static reconstruction on this rig, and the main run inherits its geometry. Three
independent measurements now agree (two smokes at ~−3 dB, one full run at
−2.82 dB). Since an L1 loss already pulls a temporally constant gaussian toward
the per-pixel temporal median of the frames it covers, the pretrain buys a worse
version of something the main objective does for free, at the cost of 3 minutes
and an overfit initialisation. **Not adopted.**

### v2 works as designed, and that is what makes it worse

v2 holds the background at 29.8% of a 995k population (296,420 background /
698,194 dynamic) instead of diluting to 2.3%, with none of the `--fix_clip_bounds`
handicap. It is **−5.32 dB held-out** — worse than v1, which was worse than
control.

The background layer is not broken. In the best checkpoint the background carries
*more* weight than the dynamic layer (mean opacity 0.611 vs 0.326, deciles
reaching 1.000). Train PSNR is 26.31, only 1 dB under control. What collapses is
generalization:

| | train | held-out | gap |
|---|---|---|---|
| control | 27.36 | 21.21 | **6.15** |
| v2 split | 26.31 | 15.89 | **10.42** |

`chkpnt_best` for v2 is from **iteration 1,500** — held-out quality peaked at a
fifth of the run and fell for the remaining 6,000 iterations.

### `bg_static_frac` does not control the background share

Starting the split at 20% instead of 50% ends at **28.5% background** against
v2's 29.8% — the same equilibrium from a 2.5× smaller start. Background
gaussians are temporally visible in every frame, so they are selected by
densification consistently while dynamic gaussians only compete when their
timestamps come up; the final share is set by densification dynamics, not by the
initial split. Held-out is −4.33 vs −5.32, so the knob moves quality a little
(via the early trajectory) while barely moving the thing it names.

This matters for the conclusion: the approach cannot be rescued by tuning
`bg_static_frac` down. Reaching a small background would require capping the
background's share of densification, which is a further mechanism to justify on
top of one that is already losing.

### Conclusion: the hypothesis is falsified on this rig

Ordering every configuration by how much whole-clip frozen background it carries:

| configuration | final background share | held-out Δ vs its seed control |
|---|---|---|
| control (lite freeze, incidental) | 2.5% | — |
| v1 + median pretrain | 1.2% | −2.82 |
| v1 split | 2.3% | −3.14 |
| v2 split, frac 0.2 | 28.5% | −4.33 |
| v2 split, frac 0.5 | 29.8% | −5.32 |

The relationship is monotone in the wrong direction, and every arm sits far
outside the ±0.4 dB noise band. Issue #20's premise — that the lite freeze's
+0.6–0.7 dB is a down-payment on a larger ceiling reachable by *growing* the
static set — does not hold here. The evidence says the opposite: the lite result
worked **because** the set was small and self-selected. Freezing gaussians that
had already earned whole-clip support through the loss is a mild regularizer on
genuinely static content; *imposing* whole-clip support on a large share of the
model forces broad, temporally constant primitives to explain a 5-second clip
from 8 viewpoints, and they fit the training views' time-averaged appearance in a
way that does not transfer.

`--fix_clip_bounds` points the same way: widening initial temporal support
(`cov_t` 1.0 → 1.99) bought +0.9 dB train and −1.0/−1.5 dB held-out. Every
change in this round that gave gaussians more temporal reach cost held-out
quality.

**Recommendation: do not adopt any arm of the full split.** Keep the lite
`--freeze_static_temporal` as shipped. The code lands flag-gated and off by
default so the negative result is reproducible rather than folklore, and so the
diagnostics stay available.

### Which base these numbers were measured on

All eight runs (both controls and all six variants) were measured on
`enh/issue-20-static-freeze` @ `958d98a`, i.e. **before** `e52a797` ("Make the
static temporal freeze actually hold") landed on `pr/issue-20-static-freeze`.
That commit fixes two real defects in the lite path: a zeroed gradient does not
hold a row still because Adam keeps stepping on stale momentum, and the mask was
reused across densification when lengths happened to match.

That does not affect the comparisons here — every arm was run on the same base,
so they are internally consistent — but two things follow:

- The control's absolute 21.21/20.95 dB was measured with a freeze that only
  partly held. Re-running the lite A/B on the fixed base may move the +0.6–0.7 dB
  figure, and that is worth doing independently of this result.
- The `--bg_static_split` path never had the Adam-drift bug. Its background rows
  are assigned once and never receive a nonzero gradient, and the moments are
  reset at labelling, so `exp_avg` stays exactly zero and Adam's step is exactly
  zero. Verified on `ab8-bgsplit2/chkpnt7000.pth`: 302,970 background rows have
  `_t` and `_scaling_t` **bit-exact** at their assigned values, zero drift. The
  negative result is therefore about a genuinely frozen background.

The implementation in this PR is rebased onto `e52a797` and adopts its
snapshot-and-restore for both the label-driven and re-derived paths. The rebased
code is smoke-verified only; the measured numbers above come from the pre-rebase
tree.

### Note on the OOM

`ab8-bgsplit` seed 42 was killed by the host OOM-killer during the GPU image
cache build (`anon-rss 27.5 GB`). Two concurrent `--gpu_cache` runs at res 2 need
~50 GB of the box's 62 GB, and a third session's run pushed it over. The A/B
queue now runs one job at a time. Not a code fault.

## Follow-ups

0. **Where the held-out gap probably goes next.** Three separate levers here
   (grow the static set, correct the temporal init, pretrain a background) all
   traded held-out quality for train fit, which is the signature of a
   capacity/regularization problem rather than a representation problem. That
   argues for Tier 2.2 (depth supervision from MASt3R + video depth) and Tier
   2.3 (pose deltas / colour affine) ahead of further structural work on the 4D
   representation — they add constraints instead of freedom.
1. **Fix `time_duration` at the source.** `process_camera_info` should derive the
   timestamp scale from the clip length, or the configs should declare the range
   the reader actually produces. `--fix_clip_bounds` only patches the gaussian
   side; the rasterizer still receives `time_duration[1] - time_duration[0]` as
   its temporal SH period, which remains wrong by 2× on this dataset. Worth its
   own issue.
2. **Per-gaussian SH routing (issue #20, stretch).** Not attempted. `force_sh_3d`
   exists but is global; routing 16-channel SH for the background and 48 for the
   dynamic layer needs a per-gaussian branch inside the rasterizer's SH
   evaluation and a ragged feature buffer. Scope it against the rot_4d kernel
   before committing to it.
3. **Per-layer learning rates** were considered and dropped. Adam is scale
   invariant, so scaling the gradient of a masked subset is a no-op; a real
   per-layer LR needs separate parameter groups, which means splitting every
   tensor in the optimizer. Not worth it before the split itself is validated.
