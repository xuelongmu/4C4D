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

## Round 2 results

| Experiment | Flags | wall | train @7500 | held-out @7500 | gaussians |
|---|---|---|---:|---:|---:|
| combo | budget 1M + sqrt LR + cache | **18:04** | 27.42 | 19.88 | 1,003,715 |
| color affine (#21) | `--color_affine` | 35:38† | 23.38 (raw) | 20.06 | ~2.1M |

4. **Combo: negative interaction.** 2.6x faster than the original code but
   held-out (19.88) underperforms budget-only (20.61) and sqrt-only (20.91):
   doubled LRs on half the capacity overshoot. Do not stack naively; the
   production profiles below validate each pairing separately.
5. **Color affine: not adopted for this rig.** No held-out gain (20.06 vs
   ~20.3), and raw-render train eval drops ~4.8 dB, meaning the affines
   learned substantial per-camera corrections that view-dependent SH was
   evidently already absorbing. Keep the flag for rigs with worse color
   consistency; consider chart-based static calibration at ingest instead
   (production roadmap item).

## Profile validation

| Profile | Flags | wall | train @7500 | held-out @7500 |
|---|---|---|---:|---:|
| fast | budget 1M + cache | **18:29** | 26.52 | 20.48 |
| quality | sqrt LR + cache | 27:31† | 26.83 | 20.19 |

6. **Fast profile adopted as production default**
   (`configs/custom/xuelong_posefix_production.yaml`): three runs at the 1M
   budget landed at 20.61/20.48 held-out vs ship 20.39 — parity or better,
   2.6x faster than the pre-fix code, half the model size.
7. **Sqrt-LR gain not replicated with the cache** (20.19 vs the 20.91
   sqrt-only run) — the effect is inside run-to-run variance until a
   multi-seed pass says otherwise. Seed-43 replication pair running
   (`ab8-fast-s43`, `ab8-sqrtlr-s43`). Full-run held-out variance should be
   treated as ~±0.4 dB, not ±0.3.

## Seed-43 replication

| Profile | s42 held-out | s43 held-out |
|---|---:|---:|
| fast (budget 1M + cache) | 20.48 | 20.39 |
| sqrt LR + cache | 20.19 | 19.42 |

8. **Fast profile confirmed**: 20.61 / 20.48 / 20.39 across three runs —
   stable at or above the ship baseline. Production config stands.
9. **Sqrt-LR rejected**: 20.91 / 20.19 / 19.42 — a 1.5 dB spread straddling
   baseline. The initial +0.5 dB was sampling luck; doubled LRs add variance,
   not quality, on this pipeline. Config keeps stock LRs.

## Next

Implementation-heavy quality items targeting the sparse-view held-out gap:
MASt3R depth supervision (#19) and static/dynamic background split (#20),
each behind its own held-out-gated A/B against the production profile
(18-minute runs make these cheap to iterate).
