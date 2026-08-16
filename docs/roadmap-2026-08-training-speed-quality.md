# 4C4D training speed & quality roadmap — 2026-08-08

Synthesis of three research passes: a code audit of this repo's hot loop, a
survey of 2024–2026 3DGS/4DGS acceleration and sparse-view literature, and a
survey of production volumetric-capture calibration workflows. Sources are
linked inline; the experiment protocol below is designed so each change is
tested one variable at a time.

## Where we stand (measured)

- 10-cam Xuelong posefix run: 7,500 iters in ~43 min on one A6000, 2.3–2.5 it/s
  at ~2.1M gaussians, train PSNR 28.6 dB.
- **Held-out novel-view quality is the real problem, not train-view fidelity:**
  6-cam run: train 28.0 dB vs held-out **15.3 dB**; 8-cam: 26.9 vs **20.3 dB**.
- Throughput degrades ~2× as gaussian count grows 75k → 2.1M; periodic dips to
  1.2–1.7 it/s coincide with ~14 CPU–GPU sync points bunched on every 100th
  iteration.
- ~24 GB of checkpoint I/O per run caused by a broken best-checkpoint test.

## Benchmark protocol (fixed for all experiments)

- **Scene:** Xuelong clip_f300_5s posefix dataset, seed 42, `--res 2`, 7,500
  iters, batch 4 — identical to `configs/custom/xuelong_clip_f300_5s_rgb_posefix_7500.yaml`.
- **Standard split:** train `0,1,2,3,5,7,8,9`, held out `4,6` (the existing
  ablation8 split). Secondary stress split: train `0,1,2,5,8,9`, held out
  `3,4,6,7` (ablation6). The 10-cam all-train config is for production fits,
  not for experiments — it has no held-out metric.
- **Report per run:** held-out PSNR/SSIM/LPIPS (primary), train PSNR
  (secondary), wall-clock to 7,500 iters, peak VRAM, final gaussian count,
  checkpoint size. Extract SSIM from TensorBoard, not the text log.
- **One change per experiment.** Run control and variant concurrently on the
  two A6000s with the same seed. Log every run in `docs/experiments/`.
- **E0 first: noise floor.** Repeat the 8-cam baseline with seeds 42/43/44.
  Any claimed improvement must exceed the observed seed spread. (Budget: one
  evening, both GPUs.)
- Speed-only changes that should be numerically inert (sync removal, caching,
  checkpoint fixes) are verified by loss-curve overlay against control, not by
  full metric A/B.

---

## Tier 0 — Defects to fix before any experiment (they confound everything)

These came out of the code audit. Several silently change what the method does
relative to the paper; fix and A/B them first because every later experiment
inherits them.

1. **Temporal densification criterion is dead code.** `densify_and_prune`
   computes `grads_t` and passes `grad_t_threshold` into clone/split
   (`gaussian_model.py:676,723`) but neither body ever reads them —
   `densify_grad_t_threshold` in the config has zero effect, and the per-iter
   `t_gradient_accum` bookkeeping (`train.py:180-186`) is thrown away.
   Restoring the `grads_t >= threshold` OR-condition is likely the single
   biggest quality lever for fast-motion regions. (A/B experiment.)
2. **Neural Decaying Function runs with Dropout(0.1) permanently on.**
   `Coefficient` is never put in `.eval()` (`module/__init__.py:20`,
   `train.py:58`), so opacity decay is stochastic per render. Also
   `--hidden_dim`/`--dropout_rate` flags are dead (constructor called with no
   args). (A/B experiment: deterministic decay.)
3. **Opacity decay is applied 4× per iteration** — once per batch item inside
   the `train.py:133` loop, each seeing the previous item's already-decayed
   opacities, so the effective decay rate depends on `batch_size`. Hoist to
   once per optimizer step. (A/B experiment; also a speed win, see Tier 1.)
4. **`chkpnt_best.pth` is rewritten at every test iteration** when no test
   cameras exist (`0.0 >= 0.0`, `train.py:363-364`) — ~24 GB of blocking I/O
   per 10-cam run, and "best" is meaningless. Guard on a non-empty test set.
   (Speed-only.)
5. **Broken `vars()` loss plumbing:** `Lopa_mask`/`Lrigid`/`Lmotion` are never
   computed anywhere; enabling any `lambda_*` in YAML would raise `KeyError`
   (`train.py:84-86,200-213`). No regularizer of any kind is currently active —
   the only losses are L1 + fused-SSIM. Fix the plumbing before Tier 2 adds
   regularizers.
6. **Batch LR scaling never applied:** LRs are the upstream batch-1 defaults
   but we train at batch 4. Grendel-GS's rule is scale by sqrt(batch)
   ([arXiv 2406.18533](https://arxiv.org/abs/2406.18533)). Zero-cost A/B.
7. Small dead work: unused `flow_2d` allocated + differentiated every render
   (`gaussian_renderer/__init__.py:129`); `max_radii2D` maintained every iter
   but never read because `size_threshold` is forced `None` under opacity
   decay (`train.py:247-249`); `new_t` never clamped to `time_duration` in
   split and no time-based prune exists, so out-of-range gaussians live
   forever; `t` LR schedule commented out (`gaussian_model.py:518-521`);
   `--opacity_decay`/`--time_aware` are `store_true` with `default=True` so
   they can't be disabled from the CLI; `train.py:445` uses the class-default
   30,000 iterations rather than the config value when building test
   iterations (happens to work at 7,500 — latent bug).

## Tier 1 — Speed (target: 43 min → ~15 min at equal quality)

Ordered by expected (gain × ease). Literature anchors: Taming 3DGS
([SIGGRAPH Asia 2024](https://humansensinglab.github.io/taming-3dgs/)),
gsplat ([arXiv 2409.06765](https://arxiv.org/pdf/2409.06765)), Faster-GS
([arXiv 2602.09999](https://arxiv.org/abs/2602.09999), demonstrated on 4D),
3DGS-MCMC ([arXiv 2404.09591](https://arxiv.org/abs/2404.09591)).

1. **Hoist the opacity-decay/visibility block** (top audit bottleneck,
   `gaussian_renderer/__init__.py:64-75`). The full 4D covariance chain
   (3 batched 4×4 GEMMs, ~0.5–1 GB transient allocs, autograd graph) runs 4×
   per iteration to produce a boolean mask, though it depends only on
   parameters that change once per step. Compute once per iteration under
   `torch.no_grad()`. This is the main reason throughput halves as N grows.
2. **Kill the periodic stalls:** drop both `empty_cache()` calls
   (`train.py:366`, `gaussian_model.py:768`), the 2.1M-point TensorBoard
   histogram every 100 iters (`train.py:312`), sync-printing in
   clone/split (`gaussian_model.py:684,728`), and batch the `.item()` calls in
   the progress logging. Speed-only; verify by loss overlay.
3. **GPU-resident uint8 image cache.** Whole dataset at res 2 is 4.15 GB as
   uint8 — fits on the A6000 beside ~5.4 GB of gaussian+Adam state. Pre-decode
   once, index on GPU, fuse uint8→float. Eliminates the 12-worker DataLoader,
   44 MB/iter of pageable H2D, per-iter `Camera` deepcopy
   (`scene/cameras.py:85-90`), and 20× worker respawn (no
   `persistent_workers`). Precompute all camera matrices on GPU once.
4. **Sparse/selective Adam** (Taming; already upstreamed in graphdeco): update
   only gaussians visible in the current batch. Near drop-in — the visibility
   mask exists (`radii > 0`) — and representation-agnostic. Upstream reports
   2.7× combined with the accelerated rasterizer.
5. **Gaussian budget.** The 75k→2.1M blowup drives backward, Adam, and sort
   cost. Test hard caps at 1.0M and 1.5M via Taming-style score-based
   densification (easy port — reuses existing gradient stats) or gsplat's MCMC
   config (relocation needs a 4D re-derivation of the split-preservation
   rule). If 1M holds held-out PSNR, this alone approaches 2×.
6. **Progressive resolution:** first ~30–40% of iters at res 4, then res 2.
   Proportional raster savings plus a sparse-view regularization effect.
7. **PLY export and checkpoint hygiene:** replace
   `list(map(tuple, attributes))` (`gaussian_model.py:394`) with structured
   array field assignment; save fp16 or pruned checkpoints (SH features are
   89% of the 644 B/gaussian payload).
8. **Later / bigger:** port Taming's per-splat backward or the Faster-GS
   kernel recipe into the rot_4d rasterizer (CUDA surgery, bounded); enable
   TF32 matmuls; `torch.compile` + bf16 on the MLP/loss stack (custom kernels
   dominate, so expect only 5–15%); gsplat backend migration with 4D→3D
   conditioning in PyTorch — worth it only as a platform play (brings sparse
   Adam, MCMC, absgrad, pose optimization, batching for free).

## Tier 2 — Quality (target: close the 8-cam held-out gap 20.3 → 24+ dB)

The held-out collapse is a sparse-view generalization problem; train-view PSNR
is already fine. Ordered by expected impact:

1. **Static/dynamic decomposition with a frozen (or slow-LR) 3D background.**
   For a fixed rig this is the biggest structural win: pretrain background as
   pure 3D gaussians with 3D SH (16 vs 48 SH channels — 3× on the dominant
   memory term), spend 4D capacity on the performer only. Literature:
   Splatography ([arXiv 2511.05152](https://arxiv.org/abs/2511.05152), up to
   +3 dB at half the size on sparse film rigs), Ex4DGS, QUEEN's
   gradient-difference split. Natural hook: the temporal mask already built at
   `gaussian_renderer/__init__.py:66-68` — `cov_t` spanning the whole clip ≈
   static. Also fixes background flicker. Speed and quality together.
2. **Depth supervision from priors we already have.** Rendered-depth loss
   against MASt3R multi-view depth (static structure) and a video-depth model
   per camera (dynamic frames), using scale-invariant Pearson/ranking loss per
   patch (DNGaussian/FSGS lineage; +1–2 dB class in sparse regimes, kills
   floaters). We already run MASt3R for init — the marginal cost is small.
3. **Learnable SE(3) pose deltas + per-camera exposure/color affine.** Our
   1.2 px Sampson residual is above what splatting can exploit at 1280×720.
   10 shared cameras × 1,500 images is heavily overconstrained — small-LR
   pose deltas after ~1k iters, plus a 3×4 per-camera color affine (upstream
   graphdeco has exposure compensation). Expect a few tenths of a dB and
   visibly crisper detail; also the production insurance policy for bumped
   rigs.
4. **Learnable per-camera time offsets.** Sub-frame desync creates cm-scale
   inconsistency for fast motion (33 mm/frame at 1 m/s, 30 fps) — larger than
   our calibration error. Unsynchronized-4DGS literature
   ([arXiv 2511.11175](https://arxiv.org/html/2511.11175v1)) recovers ~1 dB
   with learnable time shifts. Cheap parameter, big insurance.
5. **Temporally aware densification** ([arXiv 2606.23212](https://arxiv.org/pdf/2606.23212)):
   lifespan-adaptive thresholds fix under-densification of short-lived
   gaussians — precisely our fast-motion failure mode. Pairs with restoring
   the dead `grads_t` criterion (Tier 0.1).
6. **absgrad densification criterion** (AbsGS,
   [ACM MM 2024](https://dl.acm.org/doi/10.1145/3664647.3681361)): accumulate
   absolute screen-space gradients; a few lines in the backward + threshold
   retune (~2×). Best quality-per-line-of-code; fully 4D-compatible.
7. **Cheap regularizers** (after fixing the loss plumbing): temporal-opacity
   entropy, local rigidity/motion smoothness. ~0.2–0.5 dB each in sparse
   setups; tune carefully.
8. **Aspirational:** FreeTimeGS-style per-gaussian motion functions + 4D
   relocation ([arXiv 2506.05348](https://arxiv.org/abs/2506.05348); best
   published quality-per-training-hour on N3DV, 33.2 dB in ~1 h);
   Diffuman4D diffusion pseudo-views for 4–8-cam human capture
   ([arXiv 2507.13344](https://arxiv.org/abs/2507.13344)) if held-out quality
   remains the binding metric and we accept GPU-hours per capture; streaming/
   incremental training (HiCoM, QUEEN, Instant Gaussian Stream; 2–3 s/frame)
   once sequences outgrow the 5 s global-4D regime.

## Suggested experiment order

| # | Change (one variable) | Primary metric | Expect |
|---|---|---|---|
| E0 | Baseline ×3 seeds (8-cam split) | seed spread | noise floor |
| E1 | Tier 0 speed-only bundle (sync/checkpoint/histogram/flow_2d/dead work) | wall time; loss overlay | −15–25% time, identical curve |
| E2 | Hoist decay+visibility block to once/step, no_grad | wall time; held-out PSNR | −20–30% time; small quality Δ (decay-rate fix) |
| E3 | GPU-resident image cache | wall time | −5–10% time, flat early-iter profile |
| E4 | Deterministic decay MLP (eval mode) | held-out PSNR | small +, less variance |
| E5 | Restore temporal densification criterion | held-out PSNR, dynamic-region crops | + in fast motion |
| E6 | sqrt(batch) LR scaling | both | convergence speed |
| E7 | Sparse Adam | wall time | −20–40% time at 2M |
| E8 | Budget cap 1.0M / 1.5M | time vs held-out PSNR | find the knee |
| E9 | Pose deltas + color affine | held-out PSNR, sharpness | +0.2–0.5 dB |
| E10 | Depth supervision (MASt3R + video depth) | held-out PSNR/LPIPS | +1–2 dB class |
| E11 | Static/dynamic split, frozen bg | time AND held-out PSNR | large combined |
| E12 | absgrad + threshold retune | quality at fixed budget | sharper detail |

E1–E3 are compounding speed wins with no intended quality change; land them
first so every later A/B is cheaper. E4–E12 each get a control run on the
sibling GPU.

---

## Calibration: current state and upgrade path

Current: Depthkit/Scatter factory calibration → corrected handedness
conversion → RGB-only BA → **1.2 px median Sampson** (COLMAP control 1.0 px).
The production research verdict: this is already near the practical ceiling
for 1440p; the upgrade path is *independence from Depthkit* and *validation
speed*, not a better solver.

1. **Board-based primary flow:** rigid ChArUco board (~150 mm squares),
   multical-style multi-board bundle adjustment
   ([multical](https://github.com/oliver-batchelor/multical)), seeded from the
   previous day's calibration. Floor board sets world origin/up (EasyMocap
   pattern). Gates: per-camera reprojection RMS < 0.5 px, pairwise Sampson
   median < 1.5 px, and a delta-vs-yesterday report that flags bumped cameras.
2. **In-training refinement as insurance:** SE(3) deltas + time offsets
   (Tier 2.3/2.4) absorb residual drift; MASt3R/VGGT markerless recovery is
   the fallback for a bumped camera mid-take, not the primary source
   (markerless is ~10× less accurate than board+BA on wide baselines,
   [arXiv 2507.14798](https://arxiv.org/abs/2507.14798)).
3. **Sync measurement:** hardware trigger if available; regardless, record a
   RocSync-style LED clock (~$20 PCB, 1.3 ms RMSE, also measures exposure
   windows and drift — [arXiv 2511.14948](https://arxiv.org/html/2511.14948))
   or at minimum flash+clap at every take head/tail.
4. **Color:** ColorChecker seen by every camera under show lighting;
   per-camera linear correction against a reference camera applied at ingest
   (view-inconsistent color otherwise poisons SH fitting).

## Production on-set workflow (time-budgeted)

Pre-talent tech: **~45 min total, ~20 min human-active.**

1. **T-90/T-60 — warm-up (0 min active):** rig powered on arrival; thermal
   equilibrium improves calibration stability 2–10×. Push locked
   exposure/WB/focus profile to all cameras.
2. **T-60 — sync check (5 min):** trigger/genlock confirm; 10 s LED-clock or
   flash+clap capture; automated decode gates at < 0.1 frame offset.
3. **T-55 — calibration capture (10–15 min):** floor board (origin), then
   board through the volume: ~3 heights × 8 positions facing each adjacent
   pair, 150–200 detections/camera, slow movement.
4. **T-40 — solve (5–10 min, automated):** BA + outlier rejection + scorecard
   with pass/fail gates and delta report.
5. **T-30 — color pass (5 min):** chart to each camera; auto-solve; ΔE gate.
6. **T-25 — smoke test (10 min):** stand-in performs 15 s; automated fast
   3DGS/Instant-NGP single-frame fit (FastGS-class recipes: 100–200 s) renders
   held-out views to a dashboard. **This is the go/no-go gate before talent.**
   Bad extrinsics show as fog/doubling immediately.
7. **During takes (0 min):** LED clock at take heads; QC daemon on offload —
   frame counts, timestamp monotonicity, sync decode, metadata diff vs
   calibration profile, calibration ID stamped into take metadata.
8. **After bumps / end of day (5–10 min):** re-solve (delta report localizes
   which camera moved); closing verification pass brackets the day.

## Software build-out priorities (maps to this repo)

1. `preprocessing/calibration/` — one-command board solve + scorecard
   (detections → BA → per-camera/per-pair dashboard, gates, delta-vs-previous).
   Reuse the existing `validate_rgb_calibration.py` Sampson machinery. Keep
   the Depthkit adapter as one importer among several.
2. `preprocessing/qc/` — per-take QC daemon (counts, sync decode, metadata
   diff) emitting red/green per take.
3. `scripts/smoke_fit.py` — scripted single-frame fast-3DGS fit + held-out
   renders for the on-set gate (<5 min end-to-end).
4. Trainer: pose deltas, time offsets, color affine (Tier 2) — the same code
   is the production tolerance-absorption layer.
5. Ingest: standard layout `take_id/camNN/frame_%06d.png`, undistortion +
   color LUT at ingest, conversion manifest with calibration ID (extends the
   existing Depthkit converter manifest).

## Key sources

Taming 3DGS · gsplat · Faster-GS (4D-demonstrated) · 3DGS-MCMC · AbsGS ·
Temporally-Aware Densification (2606.23212) · FreeTimeGS · Ex4DGS ·
Splatography · Diffuman4D · HiCoM/QUEEN/Instant Gaussian Stream ·
Grendel-GS sqrt-batch LR · unsynchronized-4DGS (2511.11175) · RocSync ·
multical · EasyMocap calibration · Evercoast/MetriCal · Depthkit calibration
docs · MRCS Collet et al. (URLs inline above.)
