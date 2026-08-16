# Depthkit / Scatter preprocessing

`convert_depthkit_to_4c4d.py` converts a calibrated multi-sensor Depthkit project
into the image layout and COLMAP text model expected by 4C4D.

## Install

```bash
python3 -m venv .venv-depthkit
source .venv-depthkit/bin/activate
pip install -r preprocessing/depthkit/requirements.txt
```

Timestamp validation also requires `ffprobe` from FFmpeg to be available on
`PATH`.

On Windows PowerShell, activate the environment with
`.venv-depthkit\Scripts\Activate.ps1` instead.

## Validate a recording

```bash
python preprocessing/depthkit/convert_depthkit_to_4c4d.py \
  /path/to/depthkit-project \
  TAKE_NAME \
  --validate-only
```

## Convert a trimmed pilot

```bash
python preprocessing/depthkit/convert_depthkit_to_4c4d.py \
  /path/to/depthkit-project \
  TAKE_NAME \
  /path/to/output-scene \
  --start-frame 200 \
  --max-frames 60
```

The output contains `images/`, `sparse/0/cameras.txt`, `images.txt`,
`points3D.txt`, and `conversion_manifest.json`. The converter undistorts RGB
frames and converts the fixed rig poses to COLMAP convention. RGB-only output is
the default: `points3D.txt` is empty and must be populated with an RGB-based
initializer such as COLMAP, MASt3R, or MAtCha before training.

Depth PNGs are never read unless depth initialization is explicitly requested:

```bash
python preprocessing/depthkit/convert_depthkit_to_4c4d.py \
  /path/to/depthkit-project TAKE_NAME /path/to/output-scene \
  --depth-points
```

The fixed color-camera poses still use the factory color/depth calibration to
relate Scatter's rig pose to the RGB optical center. This uses calibration
metadata only, not recorded depth-camera images.

The default pose conversion was validated against COLMAP-verified RGB feature
matches on the Xuelong rig. Depthkit's saved world transform requires a
left-to-right-handed conversion, and the stored color extrinsic is interpreted
as depth-to-color. Override `--scatter-basis` or
`--color-extrinsics-direction` only for projects whose metadata uses a known
different convention.

The conversion applied by the default is, for the saved world pose `W`, saved
color extrinsic `E`, and reflection `H = diag(1, 1, -1, 1)`:

```text
world_from_depth = H @ W @ H
world_from_color = world_from_depth @ inverse(E)
```

Conjugating with `H` is important: the resulting rotations remain proper
(`det(R) = +1`) and can be represented by COLMAP quaternions. Do not rotate
physically rolled camera images just to make them appear upright unless the
intrinsics and poses are transformed with them.

## Verify RGB synchronization

Before comparing camera poses, verify that equal frame numbers refer to equal
presentation times in every RGB stream:

```bash
python preprocessing/depthkit/validate_rgb_sync.py \
  --manifest /path/to/scene/conversion_manifest.json \
  --frames 0,300,449 \
  --output /path/to/rgb-sync-report.json \
  --fail-on-quality-gate
```

This checks PTS alignment, cadence, gaps, and the requested common frame range.
Matching container timestamps do not prove hardware shutter synchronization.
For moving captures, also validate image content with a flash, timecode display,
or cross-camera motion correlation.

## Verify poses from RGB matches

A rig look-at score is only a coarse sanity check. Before a long training run,
extract and match features from one synchronized RGB frame per camera, then
compare the fixed model against COLMAP's verified matches:

```bash
python preprocessing/depthkit/validate_rgb_calibration.py \
  --database /path/to/database.db \
  --model /path/to/scene/sparse/0 \
  --output /path/to/calibration-report.json \
  --fail-on-quality-gate
```

The report includes aggregate median, p90, and p95 Sampson errors; per-pair and
per-camera support; reliable-neighbor counts; and connected components for the
verified-match and calibration-consistent graphs. A low aggregate median is not
sufficient if the tail is large or some cameras have no reliable neighbors.

The default gate requires at least 30 verified matches per edge, COLMAP control
p90 at most 4 px, supplied-pose median at most 2 px, supplied-pose p90 at most
4 px, two reliable neighbors per camera, and a connected reliable graph. These
are conservative starting values, not universal physical tolerances; expose any
project-specific changes in the saved command and JSON report.

Use multiple timestamps with static scene content. When a person or another
non-rigid subject dominates the frame, mask it before feature extraction and
prefer background, calibration-target, or empty-stage features. A pair should
only drive a pose correction when its failure repeats across timestamps and its
independent COLMAP control remains accurate.

On the investigated ten-camera Xuelong take, the previous convention measured
about 311 px median Sampson error. The corrected Depthkit convention measured
2.00 px and RGB-only bundle adjustment reduced the aggregate median to 1.20 px,
but the improved validator found a 269.96 px p90 and a disconnected
calibration-consistent graph. The median-only acceptance was therefore
superseded; see the dated experiment report for the dataset-specific evidence.

Treat validation and pose refinement as separate operations. The validators do
not rewrite calibration. A future refinement step should use static multi-frame
tracks, anchor one camera and the rig scale, write a new model directory, and
rerun both quality gates before training.

## Initialize without recorded depth

After fixed-pose RGB triangulation, retain reliable points and fill the remaining
initialization budget inside volumes inferred from the calibrated rig:

```bash
python preprocessing/depthkit/initialize_rgb_points.py \
  /path/to/scene \
  --triangulated-points /path/to/triangulated/points3D.txt \
  --num-points 75000
```

This reads RGB imagery, the COLMAP camera model, and optional RGB-triangulated
points. It does not read recorded depth frames. The output report and updated
conversion manifest explicitly record `depthDataUsed: false`.

Run the tests with:

```bash
python -m unittest discover -s preprocessing/depthkit/tests -v
```

The dated Xuelong calibration, RGB-only training, viewer, and camera-count
ablation handoff is recorded in
[`docs/experiments/2026-08-08-xuelong-depthkit-rgb.md`](../../docs/experiments/2026-08-08-xuelong-depthkit-rgb.md).
