# Sparse (selective) Adam over batch-visible gaussians

Date: 2026-08-16

Issue: [#17](https://github.com/xuelongmu/4C4D/issues/17). Branch:
`enh/issue-17-sparse-adam`, stacked on `pr/issue-20-static-freeze`.

Status: complete. On a certified-idle box, `--sparse_adam` is **9.0% faster end
to end** (13.5% per iteration at the 1M-point plateau), with no measurable change
in full-set held-out quality (mean +0.02 dB over three A/B pairs, sign flipping).

## Objective

Taming 3DGS (SIGGRAPH Asia 2024), upstreamed in graphdeco's gaussian-splatting
and available as gsplat's `SelectiveAdam`, restricts the Adam step to the
gaussians a batch actually rendered. The issue estimated 20-40% wall-clock
saving at 2M gaussians. Measure what the technique is actually worth on the
4C4D 4D representation and the Xuelong rig, and land it if it holds quality.

## Environment

- WSL2 Ubuntu-22.04, Python 3.10, PyTorch 2.1.2+cu118, CUDA 11.8.
- Two NVIDIA RTX A6000, shared with other sessions. Microbenchmarks below were
  taken under contention and are only compared within a table, never across.
  The A/B pairs of record were gated on an idle box and sampled throughout to
  prove it stayed idle.
- Quality is scored with `scripts/evaluate_full_heldout.py --stride 1` over all
  300 held-out images. The training log's own held-out line is
  `test_cameras[::100]` = 3 images and is not used for any result here.
- Config `configs/custom/xuelong_clip_f300_5s_rgb_posefix_7500.yaml`, `--res 2`,
  training views `0,1,2,3,5,7,8,9`, `--max_num_pts 1000000 --gpu_cache
  --freeze_static_temporal`, seed 42.

## What the dense step actually costs

The 4D model carries 161 floats per gaussian (xyz 3, opacity 1, scaling 3,
rotation 4, t 1, scaling_t 1, rotation_r 4, and 4D SH 3 x 48 = 144). An Adam
step reads param/grad/exp_avg/exp_avg_sq and writes three of them, so its cost
is set by N alone.

Instrumenting a full run (CUDA-synchronised timer around `optimizer.step()`,
averaged over 100-iteration windows) gives:

| iteration | N | visible fraction | Adam step | iteration wall |
|---|---|---|---|---|
| 100 | 75,000 | 0.710 | 3.4 ms | ~140 ms |
| 500 | 75,000 | 0.670 | 3.0 ms | ~140 ms |
| 2,000 | 242,241 | 0.221 | 6.0 ms | ~110 ms |
| 2,300 | 489,526 | 0.162 | 10.1 ms | ~125 ms |
| 2,700 | 1,080,732 | 0.118 | 42.7 ms | ~380 ms |
| 3,000 | 1,059,883 | 0.121 | 19.8 ms | ~200 ms |

Two findings matter.

First, **the visible fraction collapses as densification proceeds**: 0.71 at
75k points but 0.12 at the 1M cap. The early figure is the misleading one. A
batch of four views out of eight covers most of a coarse 75k cloud, but once
the model is fine-grained, four views touch barely an eighth of it. Since the
run spends roughly two thirds of its iterations at the 1M plateau, the mask is
worth far more than the opening iterations suggest.

Second, **the Adam step is about 10% of a plateau iteration** (19.8 ms of
~200 ms), not the 20-40% of wall clock the issue projected. The projection
assumed the step was a larger share than it is.

## Why a CUDA kernel and not a torch masked step

A pure-torch masked step (`index_select` the visible rows, run Adam on them,
`index_copy_` back) moves ~11 row-widths of data where the dense step moves 7,
so it only wins below roughly 64% visible. Measured at N=1M:

| implementation | time |
|---|---|
| `torch.optim.Adam`, dense | 19.5 ms |
| gather/scatter, 25% visible | 9.6 ms |
| gather/scatter, 50% visible | 18.1 ms |
| gather/scatter, 75% visible | 27.6 ms |
| gather/scatter, 100% visible | 36.5 ms |

At the observed 0.71 visible fraction of the early iterations this approach
would have been a slowdown. The kernel instead launches over all rows and
returns before touching memory for masked-out gaussians, so its traffic scales
with the visible count with no gather overhead.

Measured against torch's optimizer at N=1M on one A6000 (contended; compare
within the table only):

| visible | sparse kernel | speedup vs dense |
|---|---|---|
| 100% | 21.9 ms | 2.28x |
| 50% | 11.7 ms | 4.28x |
| 25% | 6.3 ms | 7.88x |
| 12% | 3.7 ms | 13.40x |

The 2.28x at 100% visible is not from masking — it is from fusing what torch's
`foreach` path does in several passes over the state into one.

## Semantics

The kernel reproduces `torch.optim.Adam`'s single-tensor update term for term,
including global-step bias correction, so an all-true mask matches the dense
step to float32 rounding (~1 ulp; `tests/test_sparse_adam.py` asserts this over
20 steps). Upstream's kernel omits bias correction, which makes it a different
optimizer and would confound a speed A/B with an optimizer change; that choice
was not copied.

The one genuine behavioural difference is inherent to the method: **a dense
Adam step still moves a gaussian that has no gradient**, because stale momentum
keeps being applied while `exp_avg`/`exp_avg_sq` decay toward zero. Masking
freezes those gaussians and their moments instead. This is what Taming 3DGS
accepts, but it is a real change, not purely an optimisation.

### Interaction with opacity decay

The neural opacity decay writes `_opacity.data` in place once per step. It no
longer uses a boolean mask: `decay_visibility` counts, per gaussian, how many of
the batch's viewpoints see it (`markVisible` ∧ temporal marginal), and that count
is the exponent on the decay factor, so a gaussian seen by no viewpoint gets
`factor ** 0 = 1` and is left alone.

The decay's visibility and the Adam mask are still not nested. `markVisible` is
a frustum test while the Adam mask is the union of `radii > 0`, so a gaussian can
be counted visible (decayed) yet have `radii == 0` in every view of the batch
(not Adam-stepped). Those gaussians have their opacity changed while their
moments freeze, where the dense path would have applied a momentum-only step.
They are by construction sub-pixel or degenerate gaussians, so no correction was
made; recorded because it is not obvious from the code.

Note also that `_opacity.data` is rewritten for every row, including rows the
count leaves undecayed, so all gaussians take a `sigmoid`/`inverse_sigmoid`
round-trip each step. That perturbation is identical in both arms and predates
this change.

### Interaction with the static temporal freeze

`--freeze_static_temporal` snapshots the frozen rows before the step and restores
them afterwards, which is exact regardless of what the optimizer did internally.
It therefore composes with the masked step without modification: a static row the
batch never touched was not stepped at all, and the restore is a no-op for it.
Both arms of the A/B below run with the freeze on.

## Implementation

- `utils/csrc/sparse_adam.cu` — the masked step. Parameters are `[N, ...]` with
  the gaussian index leading, so the mask selects contiguous rows of
  `M = numel/N` floats.
- `utils/sparse_adam.py` — `SparseGaussianAdam`, a `torch.optim.Adam` subclass
  that keeps torch's state layout so `GaussianModel`'s densification and
  pruning surgery on `optimizer.state`, and existing checkpoints, keep working.
  The extension is JIT-built on first use and cached, so `--sparse_adam` costs
  nothing when off.
- Two fallbacks to a dense step, both covered by the tests:
  - **Stale mask.** `densify_and_prune` runs *before* `optimizer.step()`, so on
    densification iterations the mask is shorter than N. In practice this
    fallback never fires during training: those parameters were just rebuilt by
    `torch.cat` and carry no gradient, so the earlier `p.grad is None` check
    short-circuits first. It is defensive, and the unit test drives it directly.
  - **Non-row-major parameters.** `fetchPly` builds positions as
    `np.vstack([x, y, z]).T`, so `_xyz` arrives F-contiguous (stride `(1, N)`)
    and stays that way until the first densification `cat` replaces it. The
    optimizer relays such parameters out at construction. This is a pre-existing
    quirk worth a separate look: for the first ~500 iterations every rasterizer
    call has to re-lay-out `_xyz` as well.
- `SPARSE_ADAM_STATS=1` reports kernel / dense-fallback / skipped step counts at
  exit. A masked step that quietly fell back to dense for every parameter would
  still train and still pass every quality check, so confirming the kernel did
  the work is worth a counter. Over a 700-iteration smoke: **6,246 kernel steps,
  zero fallbacks**, and 54 skips (six densification events x nine parameters).
- `ninja` added to `environment.yml`; the JIT build needs it, and the failure
  is otherwise reported as an opaque torch error.

## Smoke results

`scripts/smoke_test.sh` (700 iterations, res 4, held-out cameras 4 and 6),
noise band ±0.4 dB:

| arm | run | train PSNR | held-out PSNR |
|---|---|---|---|
| dense | 1 | 19.16 | 18.36 |
| dense | 2 | 19.18 | 17.97 |
| sparse | 1 | 19.09 | 17.42 |
| sparse | 2 | 19.08 | 17.77 |

Held-out means 18.16 (dense) against 17.60 (sparse), with a within-arm spread
of about 0.4 dB in both arms — two samples per arm cannot separate a real
0.5 dB regression from noise. Train PSNR is consistently 0.09 dB lower for
sparse, which is small but present in both replicates and is the expected sign:
freezing invisible gaussians slows convergence.

The smoke is also the least favourable operating point for this change. It
never leaves the regime where the visible fraction is ~0.7, so it pays the full
convergence cost of masking while collecting almost none of the speedup that
only appears near the 1M-point cap. The full-scale A/B is the decisive test.

The full-scale runs settled it: the apparent 0.5 dB smoke deficit was noise.
**The smoke test cannot gate this flag** — it never reaches the point where the
change does anything, and its held-out number is `test_cameras[::100]`, which on
this split is a handful of images (see the metric warning below).

## Full-scale A/B

Three A/B pairs were run, all 7,500 iterations at seed 42 with the flags in
"Environment", each pair back to back on one GPU:

| pair | base revision | box | use |
|---|---|---|---|
| `ab17-*` | pre-merge | contended by another session through the dense arm | quality sample only; timings discarded |
| `ab17c-*` | pre-merge | gated idle, `others=0` sampled throughout | pre-merge timing of record |
| `ab17m-*` | merged (this branch) | gated idle, `others=0` sampled throughout | **timing of record** |

The third pair exists because the base branch advanced 16 commits mid-experiment,
including a corrected static-temporal freeze, a rewritten once-per-step opacity
decay, and `4e324a1`, which hoists the temporal covariance out of the
per-viewpoint loop. That last one makes every iteration cheaper, so it changes
the denominator of any percentage claim. Carrying a number across a base change
like that is not sound, so both arms were re-run on the merged revision.

### Speed: 9.0% end to end, 13.5% at the plateau

Merged base, certified idle: **dense 19:36 (1,176 s) against sparse 17:50
(1,070 s) — 9.0% faster.** Per-1,000-iteration segments:

| segment | dense | sparse | delta |
|---|---|---|---|
| 0-1,000 | 147 s | 151 s | +4 s |
| 1,000-2,000 | 109 s | 107 s | -2 s |
| 2,000-3,000 | 156 s | 135 s | -21 s |
| 3,000-4,000 | 173 s | 148 s | -25 s |
| 4,000-5,000 | 168 s | 149 s | -19 s |
| 5,000-6,000 | 160 s | 140 s | -20 s |
| 6,000-7,000 | 167 s | 147 s | -20 s |

Over the 1M-point plateau (iterations 4,000-7,000) the arms take 495 s against
436 s, **13.5% faster per iteration**.

Three things make this attributable to the masked step rather than to anything
else:

1. **The first two thousand iterations are a wash** (256 s against 258 s). N is
   small there and the Adam step is ~2% of an iteration, so the mechanism
   predicts no saving, and none appears. The gap opens only as densification
   drives N up and the visible fraction down.
2. **The sparse arm carries a slightly larger model, not a smaller one.** At
   iteration 7,000 the checkpoints hold 995,421 gaussians (dense) against
   1,035,672 (sparse) — sparse is 4.0% *bigger*, so the win is not a
   fewer-gaussians artifact; if anything that works against it.
3. **The kernel really did the work.** `SPARSE_ADAM_STATS=1` over the full run:
   66,861 kernel steps, **zero dense fallbacks**, 621 skips (parameters rebuilt
   by densification, which carry no gradient).

The pre-merge pair gave 7.7% end to end and 11.1% at the plateau. Both figures
are lower than the merged base's because the base's own optimisations had not
landed yet: cheaper iterations make the Adam step a larger share of what is
left, so removing it buys more.

This is still well short of the issue's 20-40% estimate, for the reason
established above — the Adam step is ~10% of a plateau iteration, so ~10% was
always the ceiling on total wall clock. The 13.5% plateau figure exceeds that
ceiling slightly because the fused kernel also beats torch's multi-pass
`foreach` step on the visible rows, not only on the masked-out ones.

### Quality: no measurable difference

**Metric warning.** The `Evaluating test` line in a training log comes from
`getValidationCameras(tag='test', num=100)`, i.e. `test_cameras[::100]`, which on
the Xuelong 8-camera split is **3 images out of 300**. Every held-out number in
an earlier revision of this document came from there and none of them should be
trusted. The figures below re-score the saved iteration-7,000 checkpoints against
all 300 held-out images with `scripts/evaluate_full_heldout.py --stride 1`.

Full held-out set, 300 images, iteration 7,000:

| pair | dense PSNR | sparse PSNR | delta | dense SSIM | sparse SSIM |
|---|---|---|---|---|---|
| `ab17m` (merged) | 20.090 | 20.264 | **+0.173** | 0.7770 | 0.7801 |
| `ab17c` (pre-merge) | 20.336 | 20.077 | **-0.260** | 0.7819 | 0.7774 |
| `ab17` (pre-merge) | 20.516 | 20.662 | **+0.147** | 0.7797 | 0.7875 |
| mean | 20.314 | 20.334 | **+0.020** | 0.7795 | 0.7817 |

**The sign flips across pairs, so there is no measurable quality difference.**
Mean delta is +0.02 dB across three pairs. For scale, two runs of the *same* code
at the *same* seed (`ab17-dense` and `ab17c-dense`) differ by 0.18 dB on this
metric, because the rasterizer backward accumulates with nondeterministic
atomics. Three pairs cannot resolve a difference smaller than roughly 0.2 dB, and
the observed spread is inside that.

For contrast, the 3-image metric reported +0.24 dB for the same `ab17m` pair
where the full set says +0.17, and it reported sparse *ahead* in the `ab17c` pair
where the full set says it is 0.26 dB *behind*. The 3-image estimator inverted a
sign.

Held-out quality is preserved. No seed-43 replication was run: with the effect
this far inside the noise floor, more seeds would buy precision on a quantity
that is not distinguishable from zero.

### Retracted: the "slower mid-run convergence" finding

An earlier revision of this document reported that sparse trails dense by
0.27-0.39 dB from iteration 3,000 through 6,000, and concluded that
`--sparse_adam` is not a free speedup on a shortened schedule. **That is
withdrawn on two independent grounds:**

1. Every number in it was the 3-image training-log metric, which above is shown
   to invert signs on differences of this size.
2. It came from the pair whose dense arm was contended, and the trend reverses in
   the clean pairs.

It also cannot be re-measured from what was saved: only `chkpnt7000.pth` and
`chkpnt_best.pth` are written, so there are no mid-run checkpoints to re-score
against the full held-out set. **The mid-run trajectory is simply unmeasured.**

The mechanism-level prediction behind the claim — that freezing gaussians the
batch did not touch should slow convergence — remains reasonable, and the unit
test confirms those gaussians really are frozen. It is just not established by
anything here. Anyone wanting to early-stop should save mid-run checkpoints and
score them on the full set.
