# Tier-0 bug fixes: smoke gates and full A/B

Date: 2026-08-10

Status: all eight bug fixes merged to `agent/quality-fixes` after smoke gates;
full 7,500-iteration A/B (pre-fix vs post-fix) running. Append the A/B results
when complete.

## Process

Issues [#5](https://github.com/xuelongmu/4C4D/issues/5)–[#12](https://github.com/xuelongmu/4C4D/issues/12)
(bugs) and [#13](https://github.com/xuelongmu/4C4D/issues/13) (sync-stall
bundle) were each fixed on their own branch in a dedicated git worktree,
gated by `scripts/smoke_test.sh` (700 iterations, res 4, Xuelong posefix
scene, train cams 0,1,2,3,5,7,8,9, held out 4,6; ~2 min on one A6000), then
merged sequentially into `agent/quality-fixes`.

The smoke run exercises densification (from iter 200), the neural opacity
decay path (from iter 500), SH degree increases, and held-out evaluation.
Observed no-op-change spread on held-out PSNR at this scale is roughly
±0.4 dB; smoke gates catch crashes and collapses, not sub-dB quality shifts.

## Smoke results (700 iters, res 4)

| Run | wall s | train PSNR | held-out PSNR | gaussians |
|---|---:|---:|---:|---:|
| baseline (pre-fix) | 123 | 19.15 | 18.03 | 73,358 |
| #5 temporal densification restored | 93 | 19.17 | 18.22 | 73,907 |
| #6 decay MLP args passthrough | 96 | 19.13 | 18.19 | 73,325 |
| #7 once-per-step decay, power-compensated | 90 | 19.17 | 18.01 | 72,792 |
| #7 (first attempt, uncompensated) | 89 | 19.17 | 17.69 | 77,215 |
| #8 best-checkpoint guard | 97 | 19.17 | 17.73 | 73,228 |
| #9 loss-plumbing fail-fast | 94 | 19.19 | 18.21 | 73,258 |
| #10 t-clamp + temporal prune | 94 | 19.06 | 17.78 | 72,654 |
| #11 exhaust_test from merged config | 93 | 19.14 | 18.47 | 73,106 |
| #12 CLI toggles | 118 | 19.18 | 18.22 | 73,117 |
| #13 sync-stall bundle | 108 | 19.06 | 17.85 | 72,960 |

Notes:

- The uncompensated once-per-step decay weakened the effective decay rate to
  factor^1 instead of the released configs' factor^batch_size and raised the
  gaussian count; the merged fix compounds the factor by batch_size, which
  matches the expected per-step decay of the shipped batch-4 behavior.
- Smoke wall-time differences at this scale are dominated by page-cache
  warmth, not code changes; speed conclusions come from the full A/B.

## Full A/B (7,500 iters, res 2, 8-cam split)

- Control: commit `29bc344` (pre-fix code + harness), GPU 1,
  `output_dir ab8-control`.
- Fixed: commit `5122c2d` (fixes #5–#12; excludes #13), GPU 0,
  `output_dir ab8-fixed`.
- Same config (`xuelong_clip_f300_5s_rgb_posefix_7500.yaml`), seed 42,
  `--res 2`, training view 0,1,2,3,5,7,8,9.
- Caveat: GPU 1 carries a ~3.5 GB viewer process, so cross-GPU wall-time is
  indicative only; quality metrics are unaffected.

Control results (complete, ~47 min wall on GPU 1):

| Iter | train PSNR | held-out PSNR |
|---:|---:|---:|
| 4500 | 26.43 | 20.94 |
| 6000 | 27.22 | **20.98** |
| 7000 | 27.65 | 20.11 |
| 7500 | 26.85 | 20.34 |

Final gaussian count 2,158,774. Held-out PSNR peaks near iter 6000 and
declines afterward — late-run overfitting after the densification cap; the
best checkpoint (20.98) is what chkpnt_best.pth captured. Reproduces the
2026-08-08 ablation8 result (20.26 final) within run-to-run variance.

Fixed-side results (complete, **32:54 wall** on GPU 0 vs ~47 min control):

| Iter | control train / test | fixed train / test |
|---:|---|---|
| 4500 | 26.43 / 20.94 | 26.02 / 18.99 |
| 6000 | 27.22 / **20.98** | 26.59 / 19.08 |
| 7000 | 27.65 / 20.11 | 26.34 / 19.27 |
| 7500 | 26.85 / 20.34 | (not evaluated — see below) |

Final gaussians 2,103,615 (control 2,158,774).

**Findings:**

1. **Speed: ~30% faster** (32:54 vs ~47 min; control ran on the GPU with a
   ~3.5 GB viewer process, so treat the exact ratio as indicative).
2. **Quality regression: −1.3 dB train, −1.9 dB held-out at iter 6000.**
   Both populations degrade, so this is suppressed fitting, not just
   overfitting. The ±0.4 dB smoke gate could not see it. Candidate causes are
   the three quality-affecting fixes: #5 (temporal densification OR), #7
   (once-per-step decay, power-compensated — lumpier decay, higher variance),
   #10 (t-clamp + temporal prune). Densification counts in the logs do not
   show a clone explosion from #5.
3. **#11 follow-up bug:** the corrected `range(0, iterations, test_per_iter)`
   excludes the final iteration, so the fixed run never evaluated at 7500.
   Fixed by appending `args.iterations` to `test_iterations`.

**Bisect round 1 (complete):** single-fix runs on the pre-fix base, same
config/seed/split:

| Run | train @7500 | held-out @6000 | held-out @7500 |
|---|---:|---:|---:|
| control | 26.85 | 20.98 | 20.34 |
| #7 only (once-per-step decay) | **28.07** | **21.06** | **20.92** |
| #5 only (temporal densification) | 27.91 | 19.54 | 20.00 |

- **#7 is a genuine improvement**: +1.2 dB train, +0.6 dB held-out at final,
  and faster (~36 min on the contended GPU). Keep.
- **#5 is the held-out regression**: train improves while held-out drops
  ~1.4 dB at iter 6000 — the restored temporal criterion adds overfitting
  capacity in dynamic regions under sparse views. This matches the
  temporally-aware-densification literature (issue #24): the raw criterion
  needs lifespan-adaptive thresholds. Plan: revert #5 from the integration
  branch and fold the criterion into the #24 work with tuning + a held-out
  gate.
- Neither run reproduces the bundle's train-side suppression, implicating
  #10 or an interaction.

**Bisect round 2 (running):** `ab8-fix10` = #10 only (GPU 0);
`ab8-no5` = bundle through #12 with #5 reverted (GPU 1). The second run
doubles as the candidate ship configuration.
