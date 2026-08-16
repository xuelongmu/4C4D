# Xuelong Depthkit RGB calibration and camera-count ablation

Date: 2026-08-08

Status: the 6-, 8-, and 10-camera runs are complete. A 2026-08-11 follow-up
found that the median-only calibration acceptance was invalid; results below
remain useful diagnostics but do not establish a fully calibrated rig.

## Objective

Prepare a short calibrated multi-camera Depthkit/Scatter capture for 4C4D
without using recorded depth during training, validate the camera convention,
inspect the reconstruction interactively, and compare 6 and 8 input cameras
against the full 10-camera baseline.

## Data and access

- Input: a locally supplied Xuelong Depthkit/Scatter capture.
- Pilot interval: 150 frames (approximately five seconds) beginning near source
  frame 300.
- Cameras: ten synchronized RGB streams, `cam00` through `cam09`.
- RGB raster: 2560 x 1440 before 4C4D's `--res 2` training reduction.
- The cameras were physically rolled, so the stored RGB raster appears sideways.
- The dataset, extracted frames, calibration database, point cloud, logs,
  checkpoints, and rendered media are not committed. Their redistribution and
  licensing were not established.

Recorded depth images were not used by the accepted conversion, RGB point
initialization, training, evaluation, or viewer. Factory RGB/depth calibration
metadata was still required to transform Scatter's saved depth-camera rig pose
to the RGB optical center. For an RGB-only camera rig, the equivalent inputs are
per-camera RGB intrinsics and distortion plus synchronized RGB extrinsics; if
extrinsics are unavailable, recover and validate them with an RGB SfM/bundle
adjustment workflow.

## Revision and environment

- Starting revision: `5896e0a` on `agent/depthkit-rgb-calibration`.
- Base stack: `codex/cinematic-camera-sequencer` (PR #1).
- Runtime: WSL2, Python 3.10.20, PyTorch 2.1.2+cu118, CUDA 11.8.
- GPUs: two NVIDIA RTX A6000 cards, 49,140 MiB each; driver 582.16.
- Training seed: 42.
- Sanitized matched config: `configs/custom/depthkit_rgb_5s_7500.yaml`.

## Calibration findings

The first imported convention passed a coarse rig look-at check but disagreed
with RGB feature geometry. A look-at score is therefore not sufficient evidence
that the camera orientation is correct.

For saved Scatter world pose `W`, saved color extrinsic `E`, and
`H = diag(1, 1, -1, 1)`, the accepted conversion is:

```text
world_from_depth = H @ W @ H
world_from_color = world_from_depth @ inverse(E)
```

Conjugating with `H` changes handedness on both sides while preserving a proper
rotation (`det(R) = +1`). The saved color extrinsic was determined to represent
depth-to-color, so it must be inverted to obtain the RGB camera pose above.

RGB validation results on the ten-camera take:

| Pose source | Median Sampson error | Interpretation |
| --- | ---: | --- |
| superseded imported convention | about 311 px | invalid |
| corrected Depthkit convention | 2.00 px | accepted fixed-pose import |
| RGB-only bundle-adjusted poses | 1.2037 px | used for the accepted pilot |
| COLMAP pairwise control | 1.00 px | independent comparison |

The bundle-adjusted result used 4,626 verified RGB matches. These numbers are
dataset-specific diagnostics, not general thresholds.

## RGB-only preparation workflow

1. Convert and undistort the synchronized RGB frames with the corrected fixed
   pose convention. Do not pass `--depth-points`.
2. Validate supplied poses against verified RGB matches with
   `validate_rgb_calibration.py`.
3. Run RGB-only bundle adjustment when the supplied-pose epipolar error is
   materially worse than the RGB control.
4. Initialize 75,000 points from filtered RGB triangulation plus calibrated
   rig-volume seeds with `initialize_rgb_points.py`.
5. Verify the conversion manifest reports `depthDataUsed: false` before
   training.

See `preprocessing/depthkit/README.md` for complete converter and validator
commands. Keep the generated scene outside Git.

## Matched training protocol

Set sanitized paths through environment variables:

```bash
export FOURC4D_DATASET=/path/to/converted-scene
export FOURC4D_OUTPUT=/path/to/output-root
```

The full baseline command is:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/custom/depthkit_rgb_5s_7500.yaml \
  --res 2 \
  --output_dir all10 \
  --training_view 0,1,2,3,4,5,6,7,8,9
```

The nested camera subsets were chosen to cover the calibrated ring while the
8-camera set omits two near-duplicate center positions:

```bash
# Six balanced cameras; held out: 3,4,6,7
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/custom/depthkit_rgb_5s_7500.yaml \
  --res 2 \
  --output_dir ablation6 \
  --training_view 0,1,2,5,8,9

# Eight balanced cameras; held out: 4,6
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/custom/depthkit_rgb_5s_7500.yaml \
  --res 2 \
  --output_dir ablation8 \
  --training_view 0,1,2,3,5,7,8,9
```

Always pass `--res 2` explicitly. `train.py` currently defaults `--res` to 1
and applies it after loading the YAML, so omitting the flag silently changes the
experiment to full resolution. Two partial ablation runs were invalidated and
discarded after this was detected; they are not results.

## Results and observations

| Cameras | Training frames | Held-out frames | Final L1 | Final PSNR | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 900 | 600 | 0.1162212 held-out | 15.3338 dB held-out | complete |
| 8 | 1,200 | 300 | 0.0610094 held-out | 20.2633 dB held-out | complete |
| 10 | 1,500 | 0 | 0.0172833 | 28.5980 dB | complete |

First scheduled evaluation at iteration 1,500:

| Cameras | Population | L1 | PSNR |
| ---: | --- | ---: | ---: |
| 6 | train | 0.0362267 | 21.5416 dB |
| 6 | held-out test | 0.1147999 | 15.9162 dB |
| 8 | train | 0.0433458 | 20.5087 dB |
| 8 | held-out test | 0.0661743 | 20.2036 dB |

These are checkpoints on different held-out populations, not final rankings.
SSIM is computed by the training reporter and written to TensorBoard but is not
printed in the text log; extract it with the final metrics rather than
estimating it from PSNR.

The 10-camera baseline completed 7,500 iterations in approximately 43 minutes,
saved 1,989,480 Gaussians, and produced a coherent reconstruction. A
superseded calibration run measured roughly 26.65 dB but is invalid for the
camera-count comparison.

The 6- and 8-camera jobs use the same input scene, point initialization, seed,
resolution, batch size, optimization schedule, and iteration count. Only the
training-camera subset and resulting held-out set change. Built-in test metrics
for 6 and 8 cameras measure unseen views; the 10-camera train metric measures
seen views, so label those populations explicitly rather than treating them as
directly interchangeable.

Two resolution-2 jobs ran concurrently at roughly 4 to 5 iterations/s each
during the early phase with substantial VRAM headroom. The invalid full-
resolution attempt was slower and drove one GPU to 84 C, demonstrating why the
resolution check must precede conclusions about multi-GPU safety.

## 2026-08-11 calibration audit correction

The earlier 1.2037 px aggregate median concealed catastrophic residual tails.
Revalidation of the accepted pose model measured:

| Metric | Supplied pose | COLMAP pairwise control |
| --- | ---: | ---: |
| Median Sampson error | 1.2037 px | 0.9958 px |
| p90 Sampson error | 269.9592 px | 3.3256 px |
| Matches below 4 px | 69.15% | 99.31% |

The verified RGB match graph connected all ten cameras, but the strict
calibration-consistent graph split into six components. `cam09` had no reliable
neighbors and a 65.66 px per-camera median across its verified matches. Several
other camera regions also had large tails, so this is a global pose-graph
problem rather than proof of one translation sign or scale error.

Container timestamp and content-motion checks did not support a synchronization
offset: source frame 300 was exactly 10.000 seconds in every MP4, frame cadence
was 30 Hz without gaps across the pilot, and motion correlation for `cam05` and
`cam09` peaked at zero lag. The source videos differed by one frame only at the
end of the recording.

Static-background tests at source frames 300, 386, and 449 retained severe pose
tails while their independent pairwise controls remained accurate. In
particular, `cam09`-`cam07` measured 72.34 px median at source frame 449. In
contrast, `cam05`-`cam07` remained approximately 0.37-0.40 px median across the
three timestamps, so the evidence does not justify a wholesale `cam05` pose or
identity correction.

A controlled 1,500-iteration diagnostic excluded `cam05` and `cam09` and used
training views `0,1,2,3,4,6,7,8`. It completed in 356 seconds at resolution 2:

| Population | L1 | PSNR |
| --- | ---: | ---: |
| trained views | 0.0297272 | 23.9856 dB |
| held-out `cam05`,`cam09` | 0.1990717 | 11.3703 dB |

The trained-view PSNR was 3.48 dB above the original eight-camera run at the
same iteration, while the held-out pair failed severely. Because the held-out
camera identities and coverage differ, this is a localization diagnostic rather
than a comparable ablation score. It supports prioritizing `cam09` and the
`cam01`/`cam06`/`cam07` region in a static RGB-only global recalibration.

Reusable outcome: calibration acceptance must include residual tails,
per-camera support, reliable-neighbor counts, graph connectivity, multi-frame
static observations, and an independent synchronization audit. Dataset-specific
poses or camera exclusions must not be embedded in the generic converter.

## Viewer orientation

The learned scene was geometrically coherent but initially appeared rotated by
90 degrees because the source RGB raster retained the physical sensor roll.
This is a presentation issue, not evidence that recorded depth entered
training. The viewer now supports a non-destructive display correction:

```bash
python viewer_4c4d.py \
  --config configs/custom/depthkit_rgb_5s_7500.yaml \
  --checkpoint "$FOURC4D_OUTPUT/all10/chkpnt_best.pth" \
  --training-views 0,1,2,3,4,5,6,7,8,9 \
  --camera-rotation-ccw 90 \
  --host 0.0.0.0 \
  --port 8080
```

For a future dataset that should be stored upright, rotate the RGB raster and
calibration together. For a 90-degree counter-clockwise image rotation:

```text
width' = height
height' = width
fx' = fy
fy' = fx
cx' = cy
cy' = width - 1 - cx
c2w' = c2w @ Rz(+90 degrees)
```

The present completed checkpoint does not require retraining; its viewer-only
correction swaps the focal axes and dimensions and applies the corresponding
camera roll. The viewer also chooses the closest shot gate, which is 9:16 for
this capture.

## Remaining work

1. Recover a new RGB-only global pose model from static multi-frame background
   or calibration-target tracks, prioritizing `cam09` and the weak graph region.
2. Rerun the calibration and synchronization quality gates before training.
3. Extract final SSIM and Gaussian counts if the invalid calibration runs are
   retained for optimization diagnostics.
4. Decide whether upright raster conversion belongs in the Depthkit converter;
   if added, cover principal-point and pose rotation with tests.
