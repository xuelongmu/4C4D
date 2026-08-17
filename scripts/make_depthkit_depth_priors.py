"""Per-frame depth priors from the recorded Depthkit sensor depth (issue #19).

The MASt3R prior in `make_mast3r_depth_priors.py` is static structure only and
deliberately excludes the moving subject. This one covers the subject, because
the capture already contains real per-frame depth: each Azure Kinect wrote a
640x576 uint16 millimetre frame at 30 fps alongside the colour video, and the
conversion manifest records `recordedDepthFramesRead: false` -- it has never
been used.

Registration reuses `preprocessing/depthkit/convert_depthkit_to_4c4d.py`, the
same module that built this scene's COLMAP model, so the conventions cannot
drift apart: unproject with the depth intrinsics and distortion, apply the
factory depth->colour extrinsic, then project with the *undistorted* colour
intrinsics read from the scene's own `cameras.txt`. The value stored is z in
the colour camera's frame -- exactly what the rasterizer renders, in metres, on
the same scale as the reconstruction.

Usage:
  # check registration on one frame before generating 1200
  python scripts/make_depthkit_depth_priors.py --project ~/4C4D/test_data/Xuelong \
      --scene ~/4C4D/data/Xuelong/clip_f300_5s_rgb_posefix --num_frames 1

  # full clip
  python scripts/make_depthkit_depth_priors.py --project ~/4C4D/test_data/Xuelong \
      --scene ~/4C4D/data/Xuelong/clip_f300_5s_rgb_posefix
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "preprocessing"))

from scene.colmap_loader import read_intrinsics_text, read_extrinsics_text  # noqa: E402


def _conv():
    """Import the conversion module from the main repo checkout.

    Worktrees of this branch do not necessarily carry `preprocessing/`, and it
    is not part of this change -- it is the module that produced the scene.
    """
    for root in (REPO, os.path.expanduser("~/4C4D")):
        path = os.path.join(root, "preprocessing")
        if os.path.isdir(os.path.join(path, "depthkit")):
            if path not in sys.path:
                sys.path.insert(0, path)
            from depthkit import convert_depthkit_to_4c4d as m
            return m
    raise SystemExit("could not locate preprocessing/depthkit")


def drop_discontinuities(depth, valid, threshold, window, erode=0):
    """Invalidate occlusion boundaries.

    The depth and colour cameras sit a few cm apart, so depth samples near an
    occluding edge project onto colour pixels belonging to a different surface.
    Unfiltered this is a small fraction of pixels but a huge error: on this rig
    it put 16% of sampled points more than 50 cm out, with a tail past 4 m,
    while the median stayed at 2 cm. Correlating against that tail teaches the
    model to put geometry in empty space.
    """
    if threshold <= 0:
        return valid
    k = np.ones((window, window), np.uint8)
    far = depth.max() + 1.0
    local_max = cv2.dilate(np.where(valid, depth, 0.0), k)
    local_min = -cv2.dilate(np.where(valid, -depth, -far), k)
    edge = cv2.dilate(((local_max - local_min) > threshold).astype(np.uint8), k) > 0
    valid = valid & ~edge
    if erode > 0:
        # Optional: also drop hole borders. The forward splat is speckled, so
        # this costs a lot of coverage -- measure before enabling.
        ke = np.ones((erode, erode), np.uint8)
        valid &= cv2.erode(valid.astype(np.uint8), ke) > 0
    return valid


def register_frame(conv, camera, depth_index, color_k, out_w, out_h, max_depth,
                   edge_threshold=0.05, edge_window=5):
    """Project one recorded depth frame into the colour camera.

    Returns (depth, valid) at (out_h, out_w), depth in metres as z in the colour
    camera frame -- the same quantity the rasterizer accumulates.
    """
    depth_raw = cv2.imread(str(camera.depth_paths[depth_index]), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or depth_raw.ndim != 2:
        raise RuntimeError(f"unreadable depth frame {camera.depth_paths[depth_index]}")

    di = camera.depth_calibration["intrinsics"]
    ys, xs = np.mgrid[0:depth_raw.shape[0], 0:depth_raw.shape[1]]
    z = depth_raw.astype(np.float64) * 0.001  # Depthkit PNG depth is mm
    keep = (z > 0.0) & (z <= max_depth)
    if not keep.any():
        return np.zeros((out_h, out_w), np.float32), np.zeros((out_h, out_w), bool)

    pixels = np.stack([xs[keep], ys[keep]], -1).reshape(-1, 1, 2).astype(np.float64)
    normalized = cv2.undistortPoints(
        pixels, conv.intrinsic_matrix(di), conv.distortion_vector(di)).reshape(-1, 2)
    zk = z[keep]
    pts_depth = np.column_stack([normalized[:, 0] * zk, normalized[:, 1] * zk, zk])

    color_from_depth, _ = conv.color_depth_transforms(
        camera.color_calibration["extrinsics"], "depth-to-color")
    pts_color = (color_from_depth @ np.column_stack(
        [pts_depth, np.ones(len(pts_depth))]).T).T[:, :3]

    front = pts_color[:, 2] > 1e-6
    pts_color = pts_color[front]
    uvw = (color_k @ pts_color.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    zc = pts_color[:, 2]

    inside = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, zc = u[inside], v[inside], zc[inside]

    # z-buffer: several depth samples land on one colour pixel, and behind an
    # occluding edge the far one is the wrong answer, so keep the nearest.
    buf = np.full(out_h * out_w, np.inf)
    np.minimum.at(buf, v * out_w + u, zc)
    depth = buf.reshape(out_h, out_w)
    valid = np.isfinite(depth)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    valid = drop_discontinuities(depth, valid, edge_threshold, edge_window)
    return depth, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Depthkit project root (has dkproject.json)")
    ap.add_argument("--scene", required=True, help="converted 4C4D scene (for cameras.txt)")
    ap.add_argument("--recording", default="")
    ap.add_argument("--cameras", default="0,1,2,3,5,7,8,9")
    ap.add_argument("--out", default="")
    ap.add_argument("--start_frame", type=int, default=-1,
                    help="source frame of clip frame 0 (default: from conversion_manifest.json)")
    ap.add_argument("--num_frames", type=int, default=-1)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--max_depth", type=float, default=10.0)
    # Mild by design. A stricter filter was measured and rejected: it cost more
    # than half the coverage without reducing the gross-error rate, because the
    # apparent outliers were unreliable 2-view triangulations in the reference,
    # not registration errors. This setting only removes genuine flying pixels
    # at silhouette edges, for ~2% of coverage.
    ap.add_argument("--edge_threshold", type=float, default=0.10,
                    help="invalidate pixels whose local depth range exceeds this (m); 0 disables")
    ap.add_argument("--edge_window", type=int, default=3)
    args = ap.parse_args()

    project_root = os.path.expanduser(args.project)
    scene = os.path.expanduser(args.scene)
    out_dir = os.path.expanduser(args.out) if args.out else os.path.join(scene, "depth_priors_depthkit")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(scene, "conversion_manifest.json")) as f:
        cm = json.load(f)
    recording = args.recording or cm["recording"]
    start_frame = args.start_frame if args.start_frame >= 0 else cm["sourceStartFrame"]
    num_frames = args.num_frames if args.num_frames >= 0 else cm["frameCount"]

    conv = _conv()
    project = conv.load_json(conv.Path(os.path.join(project_root, "dkproject.json")))
    # No require_depth kwarg: it does not exist in every checkout of this
    # module, and checking depth_paths here is equivalent and version-proof.
    cameras, problems = conv.gather_cameras(conv.Path(project_root), project, recording)
    for p in problems:
        print(f"  [warn] {p}")
    by_index = {c.camera_index: c for c in cameras}
    missing = [i for i, c in by_index.items() if not c.depth_paths or c.depth_calibration is None]
    if missing:
        print(f"  [warn] no recorded depth for camera indices {sorted(missing)}")

    # Undistorted colour intrinsics straight from the scene's COLMAP model, so
    # the prior lands on the same pixels as the training images.
    intr = read_intrinsics_text(os.path.join(scene, "sparse/0/cameras.txt"))
    extr = read_extrinsics_text(os.path.join(scene, "sparse/0/images.txt"))
    cam_id_of = {os.path.splitext(im.name)[0].split("_")[0]: im.camera_id for im in extr.values()}

    wanted = [int(c) for c in args.cameras.split(",")]
    per_camera = {}
    for ci in wanted:
        name = f"cam{ci:02d}"
        cam = by_index[ci]
        colmap_cam = intr[cam_id_of[name]]
        fx, fy, cx, cy = colmap_cam.params
        scale = args.width / colmap_cam.width
        out_w = args.width
        out_h = int(round(colmap_cam.height * scale))
        color_k = np.array([[fx * scale, 0, cx * scale],
                            [0, fy * scale, cy * scale],
                            [0, 0, 1]], dtype=np.float64)

        cov = []
        for f in range(num_frames):
            depth, valid = register_frame(
                conv, cam, start_frame + f, color_k, out_w, out_h, args.max_depth,
                edge_threshold=args.edge_threshold, edge_window=args.edge_window)
            np.savez_compressed(os.path.join(out_dir, f"{name}_{f:04d}.npz"),
                                depth=depth, valid=valid)
            cov.append(valid.mean())
        per_camera[name] = {
            "sensor": cam.sensor_number,
            "device": cam.device_id,
            "frames": num_frames,
            "mean_valid_fraction": round(float(np.mean(cov)), 4),
            "resolution": [out_w, out_h],
        }
        print(f"  {name} (Sensor{cam.sensor_number:02d}): {num_frames} frames, "
              f"{100 * np.mean(cov):.1f}% valid")

    manifest = {
        "source": "recorded Azure Kinect depth, registered into the colour camera",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene": scene,
        "project": project_root,
        "recording": recording,
        "note": ("Per-frame and metric: depth is z in the colour camera frame, in metres, "
                 "on the same scale as the reconstruction. Registration reuses "
                 "preprocessing/depthkit/convert_depthkit_to_4c4d.py, the module that built "
                 "this scene's COLMAP model. Unlike the MASt3R prior this covers the moving "
                 "subject."),
        "params": {
            "start_frame": start_frame, "num_frames": num_frames,
            "width": args.width, "max_depth": args.max_depth,
            "edge_threshold": args.edge_threshold, "edge_window": args.edge_window,
        },
        "per_camera": per_camera,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote priors for {len(wanted)} cameras to {out_dir}")


if __name__ == "__main__":
    main()
