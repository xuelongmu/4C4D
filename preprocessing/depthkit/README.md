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
frames, converts the fixed rig poses to COLMAP convention, and optionally seeds
the point cloud from synchronized depth images.

Run the tests with:

```bash
python -m unittest discover -s preprocessing/depthkit/tests -v
```
