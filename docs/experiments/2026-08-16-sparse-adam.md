# Sparse (selective) Adam over batch-visible gaussians

Date: 2026-08-16

Issue: [#17](https://github.com/xuelongmu/4C4D/issues/17). Branch:
`enh/issue-17-sparse-adam`, stacked on `pr/issue-20-static-freeze`.

Status: complete. On a certified-idle box, `--sparse_adam` is 7.7% faster end to
end (11.1% per iteration at the 1M-point plateau) with held-out quality
unchanged.

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

The full-scale runs settled it: the apparent 0.5 dB smoke deficit was noise.
Across four full runs the same-seed held-out spread is 0.38 dB on its own, and
sparse ends level or ahead of dense in both pairs. **The smoke test cannot gate
this flag** — it is under-powered against its own noise, and it never reaches
the point where the change does anything.

## Full-scale A/B

Both arms from this branch, 7,500 iterations, seed 42, flags as in
"Environment": `ab17-dense` (no flag) against `ab17-sparse` (`--sparse_adam`),
run back to back on one GPU.

Two batches were run, because the first one's dense arm was contended:

- **Batch 1** (`ab17-dense` / `ab17-sparse`) — another session was training on
  both cards through the dense arm. Its timings are unusable; its quality
  numbers are kept only as a second sample.
- **Batch 2** (`ab17c-dense` / `ab17c-sparse`) — gated on `pgrep -af train.py`
  returning no foreign process, with a 30-second sampler that recorded
  `others=0` for the whole of both arms. This is the measurement of record.

### Speed: 7.7% total, 11.1% at the plateau

Batch 2, certified idle: **dense 19:10 (1,150 s), sparse 17:42 (1,062 s) — 7.7%
faster.** Per-1,000-iteration segments:

| segment | dense | sparse | delta |
|---|---|---|---|
| 0-1,000 | 154 s | 155 s | +1 s |
| 1,000-2,000 | 106 s | 103 s | -3 s |
| 2,000-3,000 | 144 s | 131 s | -13 s |
| 3,000-4,000 | 163 s | 145 s | -18 s |
| 4,000-5,000 | 160 s | 150 s | -10 s |
| 5,000-6,000 | 157 s | 139 s | -18 s |
| 6,000-7,000 | 165 s | 145 s | -20 s |

**The saving appears exactly where the mechanism predicts it.** The first two
segments are identical to within a second or two — N is small there and the
Adam step is ~2% of an iteration — and the gap opens as densification drives N
up and the visible fraction down. Over the 1M-point plateau (iterations
4,000-7,000) the arms take 482 s against 434 s, **11.1% faster per iteration**.
Batch 1's plateau window gave 11.3% independently, so the two batches agree on
the part of batch 1 that was not contended.

For reference, batch 2's dense arm (19:10) lands 1.6% off the historical
`ab8-staticfreeze` control (18:52), but the same-revision arm above is the
control this claim rests on, not that historical number.

The result is well short of the issue's 20-40% estimate, for the reason given
above: the Adam step was never more than ~10% of an iteration here, so ~10% was
the ceiling.

### Quality: unchanged

Held-out PSNR at iteration 7,000, all four runs:

| | dense | sparse |
|---|---|---|
| batch 1 | 20.919 | 20.954 |
| batch 2 (clean) | 20.539 | 20.805 |
| mean | 20.729 | 20.880 |

Sparse is ahead in both pairs, by +0.035 and +0.27 dB, mean +0.15 dB — inside
the ±0.4 dB band. Held-out quality is preserved. No `--seed 43` replication was
run, because it did not move.

Note the same-seed, same-code spread: the dense arm returned 20.919 and 20.539
in two runs, 0.38 dB apart. **The ±0.4 dB noise band applies at fixed seed**,
because the rasterizer backward accumulates with nondeterministic atomics. Final
gaussian counts agree within ~1% in both batches.

Train PSNR is consistently lower for sparse (27.292 against 26.608 in batch 1,
27.691 against 26.778 in batch 2), by 0.68-0.91 dB, while held-out is level or
better. Sparse Adam fits the training views less tightly without costing
generalisation, which is consistent with masking acting as a mild regulariser.
Two samples per arm make that an observation, not a claim.

### Retracted: the "slower mid-run convergence" finding

An earlier revision of this document reported that sparse trails dense by
0.27-0.39 dB from iteration 3,000 through 6,000, and concluded that
`--sparse_adam` is not a free speedup on a shortened schedule. **That does not
reproduce and is withdrawn.** Held-out PSNR by iteration, both batches:

| iteration | dense b1 | sparse b1 | dense b2 | sparse b2 |
|---|---|---|---|---|
| 1,500 | 19.730 | 19.716 | 19.322 | 19.311 |
| 3,000 | 20.481 | 20.148 | 19.868 | 19.914 |
| 4,500 | 20.819 | 20.553 | 20.229 | 20.428 |
| 6,000 | 21.005 | 20.620 | 20.496 | 20.540 |
| 7,000 | 20.919 | 20.954 | 20.539 | 20.805 |

In batch 1 sparse trails through the middle; in the clean batch 2 it leads at
every checkpoint. The deficit was one contended pair, not a property of the
method — plausibly because contention changes the scheduling that drives the
rasterizer's nondeterministic accumulation order. Given a 0.38 dB same-seed
spread, mid-run differences of this size are not resolvable with two samples per
arm either way.

The mechanism-level prediction that motivated the claim (freezing untouched
gaussians should slow convergence) remains reasonable, and the unit test
confirms those gaussians really are frozen. It is simply not visible above the
noise at this scale. Anyone wanting to early-stop should measure it directly
rather than trust either direction reported here.
