# Depthkit / Scatter preprocessing

`convert_depthkit_to_4c4d.py` converts a calibrated multi-sensor Depthkit project
into the image layout and COLMAP text model expected by 4C4D.

## Install

```bash
python3 -m venv .venv-depthkit
source .venv-depthkit/bin/activate
pip install -r preprocessing/depthkit/requirements.txt
```

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

## Verify poses from RGB matches

A rig look-at score is only a coarse sanity check. Before a long training run,
extract and match features from one synchronized RGB frame per camera, then
compare the fixed model against COLMAP's verified matches:

```bash
python preprocessing/depthkit/validate_rgb_calibration.py \
  --database /path/to/database.db \
  --model /path/to/scene/sparse/0 \
  --output /path/to/calibration-report.json
```

The supplied-pose error and COLMAP pairwise control should be of the same order.
On the investigated ten-camera Xuelong take, the previous convention measured
about 311 px median Sampson error, while the corrected Depthkit convention
measured 2.00 px. RGB-only bundle adjustment reduced it to 1.20 px; COLMAP's
independent pairwise control was 1.00 px.

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
