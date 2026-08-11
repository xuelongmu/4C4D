# Enhancement experiments: E-series on the ship tip

Date: 2026-08-10

Baseline for all rows: `ab8-ship` (branch `agent/quality-fixes` after the #5
and #10 reverts) — train 28.15 / held-out 20.39 dB @7500, 32:43 wall on an
uncontended A6000, 2.09M gaussians. Same scene, seed 42, `--res 2`, 8-cam
split (held out: 4,6) throughout. GPU 1 rows carry viewer contention
(~3.5 GB, wall times inflated).

| Experiment | Flags | wall | train @7500 | held-out @7500 | gaussians |
|---|---|---|---:|---:|---:|
| ship baseline | — | 32:43 | 28.15 | 20.39 | 2,086,435 |
| ship + cache | `--gpu_cache` | 27:02† | 28.03 | 20.23 | ~2.1M |
| budget 1M (#18) | `--max_num_pts 1000000` | **26:57** | 27.23 | 20.61 | 1,013,587 |
| sqrt-batch LR (#16) | all LRs ×2 | 34:49† | 28.01 | **20.91** | 2,062,058 |

† contended GPU 1.

## Findings

1. **Budget cap 1M (#18): adopt.** Held-out +0.2 dB with **half the
   gaussians**, ~18% faster, half the model/checkpoint size. The second
   million gaussians was pure train-view overfitting capacity (train −0.9 dB,
   held-out up). A 1.5M point on the curve is not needed unless train-view
   fidelity becomes a deliverable.
2. **Sqrt-batch LR (#16): adopt.** Held-out +0.5 dB at equal train quality —
   the Grendel-GS sqrt(batch) rule holds for this pipeline at batch 4.
3. Both wins are held-out-first, consistent with the project's actual
   bottleneck being sparse-view generalization.

## Running

- `ab8-combo`: budget 1M + sqrt LR + cache together (interaction check;
  GPU 0). If clean, this becomes the recommended production config
  (~expected: sub-25 min, held-out ≥ 20.9).
- `ab8-coloraffine`: `--color_affine` full-length A/B (GPU 1).

## Next

Full-length `--color_affine` verdict, then the implementation-heavy quality
items: MASt3R depth supervision (#19) and static/dynamic background split
(#20), each behind its own held-out-gated A/B.
