# Sparse (selective) Adam over batch-visible gaussians

Date: 2026-08-16

Issue: [#17](https://github.com/xuelongmu/4C4D/issues/17). Branch:
`enh/issue-17-sparse-adam`, stacked on `pr/issue-20-static-freeze`.

Status: implementation, microbenchmarks and the full-scale A/B are complete.
Held-out quality is unchanged and the plateau speedup is ~11% per iteration
(~6% of total wall clock). A clean re-run of the timing arms is queued behind
another session's jobs; see "Full-scale A/B".

## Objective

Taming 3DGS (SIGGRAPH Asia 2024), upstreamed in graphdeco's gaussian-splatting
and available as gsplat's `SelectiveAdam`, restricts the Adam step to the
gaussians a batch actually rendered. The issue estimated 20-40% wall-clock
saving at 2M gaussians. Measure what the technique is actually worth on the
4C4D 4D representation and the Xuelong rig, and land it if it holds quality.

## Environment

- WSL2 Ubuntu-22.04, Python 3.10, PyTorch 2.1.2+cu118, CUDA 11.8.
- Two NVIDIA RTX A6000. Both cards were shared with another session throughout;
  absolute wall times below are contended and are only compared within a
  measurement, never across.
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

The neural opacity decay writes `_opacity.data` in place once per step, gated
by `torch.where(mask, decayed, old)` where the mask is the frustum-and-temporal
visibility of batch item 0 only. Most decayed rows are therefore also inside
the union `radii > 0` mask that drives the Adam step, and their optimizer state
stays consistent.

The two masks are not nested, though: `markVisible` is a frustum test, so a
gaussian can be in frustum for view 0 (decayed) yet have `radii == 0` in every
view of the batch (not Adam-stepped). Those gaussians have their opacity
changed while their moments freeze, where the dense path would have applied a
momentum-only step. They are by construction sub-pixel or degenerate gaussians,
so no correction was made; recorded here because it is not obvious from the
code.

## Implementation

- `utils/csrc/sparse_adam.cu` — the masked step. Parameters are `[N, ...]` with
  the gaussian index leading, so the mask selects contiguous rows of
  `M = numel/N` floats.
- `utils/sparse_adam.py` — `SparseGaussianAdam`, a `torch.optim.Adam` subclass
  that keeps torch's state layout so `GaussianModel`'s densification and
  pruning surgery on `optimizer.state`, and existing checkpoints, keep working.
  The extension is JIT-built on first use and cached, so `--sparse_adam` costs
  nothing when off.
- Two fallbacks to a dense step, both exercised by the tests:
  - **Stale mask.** `densify_and_prune` runs *before* `optimizer.step()`, so on
    densification iterations the mask is shorter than N. Those parameters were
    just rebuilt by `torch.cat` and carry no gradient, so the step is a no-op
    either way.
  - **Non-row-major parameters.** `fetchPly` builds positions as
    `np.vstack([x, y, z]).T`, so `_xyz` arrives F-contiguous (stride `(1, N)`)
    and stays that way until the first densification `cat` replaces it. The
    optimizer relays such parameters out at construction. This is a pre-existing
    quirk worth a separate look: for the first ~500 iterations every rasterizer
    call has to re-lay-out `_xyz` as well.
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

In hindsight the full run explains the smoke: sparse Adam converges more slowly
in the middle of a schedule and catches up only at the end (see below), and a
700-iteration smoke measures nothing but that lag. **The smoke test should not
be used to gate this flag.**

## Full-scale A/B

Both arms from this branch, 7,500 iterations, seed 42, flags as in
"Environment": `ab17-dense` (no flag) against `ab17-sparse` (`--sparse_adam`),
run back to back on one GPU.

### Quality: unchanged

| | dense | sparse | delta |
|---|---|---|---|
| held-out PSNR (iter 7000) | 20.919 | 20.954 | +0.035 |
| train PSNR (iter 7000) | 27.292 | 26.608 | -0.684 |
| final gaussians | 1,016,038 | 1,003,717 | -1.2% |

Held-out quality is flat — 0.035 dB against a ±0.4 dB noise band. No `--seed 43`
replication was run, because held-out did not move.

Train PSNR is 0.68 dB lower while held-out is level or marginally better, so
sparse Adam fits the training views less tightly without costing generalisation.
That is consistent with masking acting as a mild regulariser, but with one
sample per arm it is an observation, not a claim.

### Convergence is slower mid-run

Held-out PSNR by iteration:

| iteration | dense | sparse |
|---|---|---|
| 1,500 | 19.730 | 19.716 |
| 3,000 | 20.481 | 20.148 |
| 4,500 | 20.819 | 20.553 |
| 6,000 | 21.005 | 20.620 |
| 7,000 | 20.919 | 20.954 |

Sparse trails by 0.27-0.39 dB from iteration 3,000 through 6,000 and only
catches up at the end. This is the predicted cost of freezing gaussians that
the batch did not touch, and it matters practically: **`--sparse_adam` is not a
free speedup for a shortened schedule.** Anyone early-stopping at 6,000
iterations would pay 0.39 dB for it. At the full 7,500 the deficit closes.

### Speed

Per-1,000-iteration segment times:

| segment | dense | sparse |
|---|---|---|
| 0-1,000 | 438 s | 164 s |
| 1,000-2,000 | 293 s | 96 s |
| 2,000-3,000 | 359 s | 130 s |
| 3,000-4,000 | 325 s | 144 s |
| 4,000-5,000 | 159 s | 143 s |
| 5,000-6,000 | 152 s | 135 s |
| 6,000-7,000 | 158 s | 138 s |

**The early segments of the dense arm are contended, not slow.** Reaching
iteration 4,000 took 23:35 for dense against 8:54 for sparse — a 2.65x gap over
a stretch where N is small and the Adam step is about 2% of an iteration, so it
cannot be the optimizer. Another session was training on both cards during that
window. Total wall clock (32:54 dense against 17:44 sparse) is therefore not a
usable number and is excluded.

The plateau segments, where both arms appear to have run under comparable load,
give the usable comparison: iterations 4,000-7,000 took 469 s dense against
416 s sparse, so **the sparse arm is 11.3% faster per iteration at the 1M-point
plateau**. Independently, the sparse arm's 17:44 total sits 6.0% under the
established clean control `ab8-staticfreeze` (18:52, same flags and seed).

Those two figures are consistent: the saving only applies once densification
has driven the visible fraction down, which is roughly the last two thirds of
the run, so a per-plateau-iteration saving of ~11% lands near ~6% of total wall
clock. Both are well short of the issue's 20-40% estimate, for the reason given
above — the Adam step was never more than ~10% of an iteration here.

A clean back-to-back re-run of both arms on an idle box is queued to replace the
contended dense arm; results will be appended.
