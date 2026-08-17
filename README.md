<div align="center">

# 4C4D: 4 Camera 4D Gaussian Splatting

### CVPR 2026

[Junsheng Zhou](https://junshengzhou.github.io/)<sup>#</sup>, Zhifan Yang<sup>#</sup>, Liang Han, [Wenyuan Zhang](https://wen-yuan-zhang.github.io/), Kanle Shi, Shenkun Xu, [Yu-Shen Liu](https://yushen-liu.github.io/)<sup>*</sup>

**Tsinghua University**

<sup>#</sup> Equal contribution &nbsp;&nbsp; <sup>*</sup> Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://junshengzhou.github.io/4C4D/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2604.04063)

</div>

---

## Abstract

This paper tackles the challenge of recovering 4D dynamic scenes from videos captured by as few as **four portable cameras**. Learning to model scene dynamics for temporally consistent novel-view rendering is a foundational task in computer graphics, where previous works often require dense multi-view captures using camera arrays of dozens or even hundreds of views. We propose **4C4D**, a novel framework that enables high-fidelity 4D Gaussian Splatting from video captures of extremely sparse cameras. Our key insight is that geometric learning under sparse settings is substantially more difficult than modeling appearance. Driven by this observation, we introduce a **Neural Decaying Function** on Gaussian opacities for enhancing the geometric modeling capability of 4D Gaussians. This design mitigates the inherent imbalance between geometry and appearance modeling in 4DGS by encouraging the 4DGS gradients to focus more on geometric learning. Extensive experiments across sparse-view datasets with varying camera overlaps show that 4C4D achieves **superior performance** over prior art.

## Pipeline

<div align="center">
  <img src="assets/pipeline.png" width="90%"/>
</div>

<p align="center"><em>Figure 1. Overview of the 4C4D framework.</em></p>

We introduce a **Neural Decaying Function** $f_\theta$, implemented as a lightweight neural network, to adaptively control the opacity decay of Gaussians. Given key Gaussian attributes as input, $f_\theta$ predicts a factor that controls the decay of Gaussian opacities. During training, both the Neural Decaying Function and the 4D Gaussians are jointly optimized via gradient backpropagation under a photometric rendering loss.

## Getting Started

### 1. Installation

**Clone and set up the 4C4D environment:**

```bash
git clone https://github.com/yangzf-1023/4C4D
cd 4C4D
conda env create --file environment.yml
conda activate 4c4d
```

**Set up [MASt3R](https://github.com/anttwo/MAtCha) for dense point cloud initialization:**

> Since COLMAP produces extremely sparse point clouds with few input views, we use MASt3R-based reconstruction instead.

```bash
cd ..
git clone https://github.com/anttwo/MAtCha.git
cd MAtCha
python install.py
python download_checkpoints.py
conda activate matcha
```

### 2. Data Preparation

#### Dataset Structure

Whether you use the provided pre-processed data or prepare your own custom dataset, please organize the data directory as follows:

```
data/
├── N3V/                              # or your custom dataset name
│   ├── flame_steak/                  # scene directory
│   │   ├── images/                   # input frames
│   │   │   ├── cam00_0000.png        # format: cam{XX}_{YYYY}.png
│   │   │   ├── cam00_0001.png        #   XX   = camera index (zero-padded)
│   │   │   ├── cam01_0000.png        #   YYYY = frame index  (zero-padded)
│   │   │   └── ...
│   │   └── sparse/
│   │       └── 0/
│   │           ├── cameras.bin       # camera intrinsics  (COLMAP format)
│   │           ├── images.bin        # camera extrinsics  (COLMAP format)
│   │           └── points3D.bin      # reconstructed 3D points
│   ├── cook_spinach/
│   │   ├── images/
│   │   └── sparse/
│   │       └── 0/
│   │           ├── cameras.bin
│   │           ├── images.bin
│   │           └── points3D.bin
│   └── ...                           # additional scenes
```

> **Format note:** Both `.bin` (binary) and `.txt` (text) COLMAP formats are supported for all files under `sparse/0/`.

> **Important — how `sparse/0/` files are generated:**
> - `points3D.*` is always reconstructed from **sparse (training) views only**, since it serves as the point cloud initialization for training.
> - `images.*` and `cameras.*` can be generated from either **sparse views** or **all (dense) views**, depending on whether you need to render/evaluate on held-out test views. If you only train without evaluation, sparse views are sufficient; if you need test-view evaluation, generate them from all views so that test camera poses are included.

#### Pre-processed Data

We provide pre-processed data for all scenes in the Neural 3D Video (N3V) dataset (first 300 frames, using training views `1, 10, 13, 20`). You can download it directly and skip to [Training](#3-training):

> **Download:** [google drive](https://drive.google.com/drive/folders/1bKEMaXSSr7j_awlX3miEHe3-9JvrKIxy?usp=drive_link)

#### Preparing the N3V Dataset from Scratch

If you prefer to process the raw data yourself:

1. Download the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and extract each scene to `data/N3V/`.

2. Preprocess the raw video:

```bash
cd ../4C4D
conda activate 4dgs
python scripts/n3v2blender.py data/N3V/$SCENE --training_view $TRAIN_VIEW
```

3. *(Recommended)* Generate dense point clouds with MASt3R for best results:

```bash
# Convert to COLMAP format
python scripts/n3v2colmap.py data/N3V/$SCENE --training_view $TRAIN_VIEW
python scripts/n3v2colmap.py data/N3V/$SCENE

# Run MASt3R reconstruction
cd ../MAtCha
conda activate matcha
python train.py \
  -s ../4C4D/data/N3V/$SCENE/mast3r_${N_SPARSE} \
  -o ../4C4D/data/N3V/$SCENE/mast3r_${N_SPARSE} \
  --sfm_config posed --sfm_only

# Copy reconstructed point cloud
cd ../4C4D
conda activate 4dgs
cp -r data/N3V/$SCENE/mast3r_${N_DENSE}/sparse data/N3V/$SCENE/
cp data/N3V/$SCENE/mast3r_${N_DENSE}/mast3r_sfm/sparse/0/points3D.* \
   data/N3V/$SCENE/sparse/0/
```

#### Preparing Custom Datasets

To use your own data, organize it according to the [Dataset Structure](#dataset-structure) above. Ensure that:

- **`images/`** contains the extracted video frames named as `cam{XX}_{YYYY}.png`, where `XX` is the zero-padded camera index and `YYYY` is the zero-padded frame index.
- **`sparse/0/`** contains valid COLMAP-format camera parameters and point cloud files. You may obtain these via COLMAP, MASt3R, or any other SfM pipeline. Refer to the generation notes above for guidance on which views to use.

For calibrated multi-sensor Depthkit/Scatter captures, see the
[`preprocessing/depthkit`](preprocessing/depthkit/) converter. It undistorts the
RGB streams, converts the fixed rig calibration to COLMAP convention, and can
initialize `points3D.txt` from synchronized depth frames.
[`docs/experiments/2026-08-08-xuelong-depthkit-rgb.md`](docs/experiments/2026-08-08-xuelong-depthkit-rgb.md)
records the RGB-only preparation workflow end to end — handedness conversion,
epipolar validation, bundle adjustment, and the camera-count ablation — and
[Production profile for a calibrated custom rig](#production-profile-for-a-calibrated-custom-rig)
is the training profile validated on it.

<details>
<summary><b>Variable Reference</b></summary>

| Variable       | Description                                          | Example        |
|:---------------|:-----------------------------------------------------|:---------------|
| `$SCENE`       | Scene name from the N3V dataset                      | `flame_steak`  |
| `$TRAIN_VIEW`  | Training view indices (comma-separated)              | `1,10,13,20`   |
| `$N_SPARSE`    | Number of sparse views, equal to `len($TRAIN_VIEW)`  | `4`            |
| `$N_DENSE`     | Total number of views in the scene                   | `21`           |

</details>

### 3. Training

```bash
python train.py \
  --config $CONFIG_PATH \
  --training_view $TRAIN_VIEW \
  --output_dir $OUTPUT_DIR
```

The run directory is `ModelParams.model_path` from the config joined with
`--output_dir`. `train.py` refuses to start if it already exists, and writes the
fully merged argument set to `<run_dir>/training_params.txt`. Any top-level YAML
key that names a `train.py` argument overrides that argument's default.

#### Production profile for a calibrated custom rig

`configs/custom/xuelong_posefix_production.yaml` is the validated profile for a
sparse calibrated multi-camera capture, tuned on a 10-camera Depthkit/Scatter
rig with two cameras held out. Against the pre-tuning code it trains 7,500
iterations in **18:29 instead of ~47 min** on one A6000, at equal or better
held-out quality and half the model size.

The custom configs keep machine paths out of Git by interpolating two
environment variables into `ModelParams`, so set those rather than editing the
config:

```bash
export FOURC4D_DATASET=/path/to/converted-scene   # -> ModelParams.source_path
export FOURC4D_OUTPUT=/path/to/output-root        # -> ModelParams.model_path

python train.py \
  --config configs/custom/xuelong_posefix_production.yaml \
  --res 2 \
  --output_dir production \
  --training_view 0,1,2,3,5,7,8,9
```

The run lands in `$FOURC4D_OUTPUT/production`.

**Always pass `--res` explicitly.** It defaults to 1 and is applied *after* the
YAML merge, so omitting it silently trains at full resolution.

Because top-level YAML keys override `train.py` arguments, each tuned behaviour
below is config-settable as well as available on the CLI:

| Config key / flag | Production value | Effect |
| --- | --- | --- |
| `gpu_cache` / `--gpu_cache` | `true` | decode the whole training set once and keep it as uint8 on the GPU; drops the DataLoader, per-iteration H2D copies and camera deepcopies. Quality-inert, ~17% less wall time. |
| `freeze_static_temporal` / `--freeze_static_temporal` | `true` | zero the temporal gradients (`t`, `scaling_t`, `rotation_r`) of gaussians whose support spans the whole clip, so background geometry stops churning in time. +0.6-0.7 dB held-out on two seeds. |
| `densify_until_num_points` / `--max_num_pts` | `1000000` | hard gaussian budget. Halves model size and wall time; above ~1M this scene only gains train-view fidelity. |
| `exhaust_test` | `true` | evaluate held-out views on a regular schedule including the final iteration. |
| `color_affine` / `--color_affine` | off | per-camera 3x4 color affine on the training loss. No gain on this rig (view-dependent SH already absorbed the mismatch); keep for rigs with worse color consistency. |

Both boolean flags accept a negation on the CLI (`--no-gpu_cache`,
`--no-freeze_static_temporal`) to override the config for an experiment.
Deliberately *not* enabled: sqrt-batch LR scaling, rejected across three seeds.

#### Running experiments on the trainer

If you are changing the training loop rather than fitting a scene, follow
[`docs/EXPERIMENT_METHODOLOGY.md`](docs/EXPERIMENT_METHODOLOGY.md) — the
two-stage protocol (a ~2-minute smoke gate, then a full held-out A/B) and the
branch/PR structure used to validate this profile.
[`docs/LESSONS.md`](docs/LESSONS.md) records what was learned and, more
usefully, which plausible changes failed and why.

```bash
# ~2 min: does this change break training?
scripts/smoke_test.sh my-change 0

# ~20 min/side: does it improve held-out quality? (one per GPU)
scripts/ab_launch.sh /path/to/control-worktree ab8-control 1 && sleep 10
scripts/ab_launch.sh /path/to/variant-worktree ab8-variant 0 && sleep 10
```

Both read `FOURC4D_DATASET` / `FOURC4D_OUTPUT` through the config and honour
`FOURC4D_PYTHON` when the environment's interpreter is not on `PATH`.
`ab_launch.sh` additionally honours `FOURC4D_AB_CONFIG` and `FOURC4D_AB_VIEWS`,
preflights the interpreter and config before backgrounding anything, and copies
each run's log to `<run_dir>/train.log`.

#### Regenerating the experiment comparison report

`scripts/build_experiment_report.py` scans the run directories under an output
root for `train.log` and `rendered_images/`, and writes one self-contained HTML
file: held-out PSNR trajectories, a card per run with the held-out probe render
(press and hold to flip to ground truth), and a summary table of verdicts.

```bash
python scripts/build_experiment_report.py \
  --output-root "$FOURC4D_OUTPUT" \
  --out /path/outside/the/repo/report.html
```

Add new experiments as rows in the `RUNS` manifest at the top of the script,
in commit order, each with its phase and one-line verdict. The report embeds
capture imagery whose redistribution licensing is not established — write it
outside the repository and do not host it publicly.

### 4. Visualization

Render a novel-view trajectory after training:

```bash
python render.py \
  --config $CONFIG_PATH \
  --training_view $TRAIN_VIEW \
  --output_dir $OUTPUT_DIR \
  --traj arc \
  --start_checkpoint output/N3V/$SCENE/chkpnt30000.pth
```

#### Interactive 4D viewer

`viewer_4c4d.py` uses the native 4C4D CUDA rasterizer behind a Viser browser UI,
so camera motion and the frame slider preserve the learned temporal Gaussians.

```bash
pip install -r requirements-viewer.txt

python viewer_4c4d.py \
  --config configs/dynerf/flame_steak.yaml \
  --checkpoint output/N3V/flame_steak/run-4view/chkpnt_best.pth \
  --training-views 1,10,13,20 \
  --host 0.0.0.0 \
  --port 8080
```

For captures whose RGB sensors were physically rolled but whose stored rasters
were intentionally left unchanged, add `--camera-rotation-ccw 90` (or 180/270).
The viewer applies the matching camera roll, swaps the focal axes and image
dimensions for quarter turns, and chooses the closest cinematic shot gate. This
is a display correction for an existing checkpoint; preprocessing an upright
dataset requires rotating the raster, intrinsics, and camera pose together.

Open `http://localhost:8080` on the same machine. This also works when the
server runs in WSL2 and the browser runs on Windows. Use `--width` to trade
interactive resolution for frame rate. Space toggles playback. The bundled
viewer controls provide persistent look and move sensitivity plus independent
X/Y orbit and move inversion settings.

The **Cinematic Shot** panel turns the free camera into a shot camera. Set the
shot cursor, navigate or enter an exact XYZ/Euler pose, choose the lens, then
select **Add / update keyframe**. Scrubbing or enabling **Preview camera move**
evaluates the interpolated camera path while the dynamic splat advances with
the same elapsed timeline time.
The viewport can show the 3D path, key-camera frustums, rule-of-thirds guides,
an action-safe frame, and common cinematic aspect ratios.
The docked cinematic sequencer provides a frame ruler, draggable playhead,
transport and key-navigation controls, plus camera-transform and lens/FOV key
tracks. Dynamic scene time is shown as a readout, not a keyframable track. Its
light/dark theme toggle is persistent.
Realtime preview resolution matches the browser viewport by default. The
selected shot aspect is shown with a configurable translucent matte over any
visible overscan; wider gates are letterboxed without
stretching. The translucent matte preserves scene context on every side of the
gate. Final output height is derived from the shot aspect, and that
aspect is embedded in JSON, glTF, and USD camera exports.

The **Camera & Lens** panel keeps filmback width, filmback height, and focal
length on one compact row, with vertical and horizontal FOV paired beneath it;
editing any value keeps the lens fields synchronized.
Sensor-gate presets include Super 35 4-perf, Super 35 3-perf, full frame,
APS-C, Micro Four Thirds, and a custom gate. A full-frame-equivalent focal
length is editable for lens matching; changing either physical focal length or
35mm equivalent updates the other value and both FOV readouts.

The **Render & Export** panel renders the current shot through the native CUDA
rasterizer and downloads an H.264 MP4. It can also download the animated camera
as glTF 2.0, USD ASCII, or a lossless 4C4D JSON sidecar containing shot-frame
camera position, WXYZ rotation, and focal length.
MP4 rendering keeps dynamic playback at its native timing relative to the
shot/output FPS. MP4 export requires `ffmpeg` on `PATH` (for Ubuntu/WSL:
`sudo apt install ffmpeg`). Use `--shot-frames` to change the default shot
length.

### 5. Evaluation

Evaluate on held-out test views:

```bash
python render.py \
  --config $CONFIG_PATH \
  --training_view $TRAIN_VIEW \
  --output_dir $OUTPUT_DIR \
  --test \
  --start_checkpoint output/N3V/$SCENE/chkpnt30000.pth
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{zhou20264c4d,
  title     = {4C4D: 4 Camera 4D Gaussian Splatting},
  author    = {Zhou, Junsheng and Yang, Zhifan and Han, Liang and Zhang, Wenyuan and Shi, Kanle and Xu, Shenkun and Liu, Yushen},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

Our codebase builds upon [4DGS](https://fudan-zvg.github.io/4d-gaussian-splatting/) and [MASt3R](https://github.com/naver/mast3r). We thank the authors for their excellent work.
