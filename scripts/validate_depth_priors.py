"""Score depth priors against the scene's triangulated COLMAP points (issue #19).

A prior only has to be right up to a per-patch affine for the training loss to
use it, so the metric is Spearman rank correlation between the prior's depth and
the true triangulated depth at the points COLMAP actually observed in that view.
Pearson is reported too, but rank correlation is what matches the loss's
invariance and is not dragged around by a few far outliers.

Ground truth is the RGB bundle-adjusted triangulation under rgb_init/refined,
NOT sparse/0. sparse/0/points3D.txt is the training initialization: 75k
synthetic "calibrated rig volume" points with empty TRACK entries, whose
keypoint associations in images.txt are placeholders (some project behind the
camera). Scoring against it measures nothing -- the reproj_px control below
exists to catch exactly that mistake.

Usage:
  python scripts/validate_depth_priors.py \
      --scene ~/4C4D/data/Xuelong/clip_f300_5s_rgb_posefix \
      --priors ~/4C4D/data/Xuelong/clip_f300_5s_rgb_posefix/depth_priors
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_points3D_binary)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--priors", default="")
    ap.add_argument("--frame", default="0000", help="frame whose COLMAP observations are used")
    ap.add_argument("--colmap", default="rgb_init/refined",
                    help="reconstruction to score against, relative to --scene")
    ap.add_argument("--min_track", type=int, default=3,
                    help="ignore points seen in fewer views than this; 2-view tracks are "
                         "20%% gross-error and drag every prior's score down (see below)")
    args = ap.parse_args()

    scene = os.path.expanduser(args.scene)
    prior_dir = os.path.expanduser(args.priors) if args.priors else os.path.join(scene, "depth_priors")
    model_dir = os.path.join(scene, args.colmap)

    extr = read_extrinsics_binary(os.path.join(model_dir, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(model_dir, "cameras.bin"))
    xyz, _, _ = read_points3D_binary(os.path.join(model_dir, "points3D.bin"))

    # read_points3D_binary drops the COLMAP ids but preserves file order, and
    # the binary format is written in ascending id order, so recover the ids the
    # same way COLMAP wrote them.
    import struct
    with open(os.path.join(model_dir, "points3D.bin"), "rb") as f:
        (num_points,) = struct.unpack("<Q", f.read(8))
        ids, track_lens = [], []
        for _ in range(num_points):
            pid = struct.unpack("<Q", f.read(8))[0]
            f.read(35)  # xyz (3 doubles) + rgb (3 bytes) + error (1 double)
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)
            ids.append(pid)
            track_lens.append(track_len)

    # Two-view tracks are the minimum COLMAP will emit and are badly
    # unreliable here: measured against the recorded sensor depth their
    # gross-error rate is 20%, versus 3.0% at three views, 0.4% at four and
    # 0.0% at five or more. Two independent priors (MASt3R and the Kinect)
    # disagree with the *same* points, so it is the ground truth that is
    # wrong. Leaving them in caps every prior's apparent quality at ~0.7
    # correlation regardless of how good it is.
    pt_xyz = {pid: xyz[i] for i, pid in enumerate(ids)
              if track_lens[i] >= args.min_track}
    print(f"{len(pt_xyz)} of {len(ids)} triangulated points in {args.colmap} "
          f"have track length >= {args.min_track}")

    per_cam = defaultdict(lambda: ([], []))
    for img in extr.values():
        name = os.path.splitext(img.name)[0]
        cam_id, frame = name.split("_")
        if frame != args.frame:
            continue
        # Same precedence as utils/depth_priors.py: per-frame beats per-camera.
        for stem in (f"{cam_id}_{frame}", cam_id):
            prior_path = os.path.join(prior_dir, f"{stem}.npz")
            if os.path.exists(prior_path):
                break
        else:
            continue

        with np.load(prior_path) as z:
            depth, valid = z["depth"], z["valid"]
        ph, pw = depth.shape
        cam = intr[img.camera_id]
        sx, sy = pw / cam.width, ph / cam.height

        R = img.qvec2rotmat()
        obs_ids = img.point3D_ids
        keep = np.array([i in pt_xyz for i in obs_ids])
        if not keep.any():
            continue
        xy = img.xys[keep]
        pts = np.stack([pt_xyz[i] for i in obs_ids[keep]])
        # COLMAP images.txt stores world-to-camera (R, t); depth is the z of the
        # point in camera coordinates.
        z_cam = (pts @ R.T + img.tvec)[:, 2]

        # Control: reproject each observed point with the stored pose and
        # intrinsics and compare against the stored 2D observation. If this is
        # not sub-pixel, the fault is in this script's geometry (or the
        # reconstruction), not in the prior being scored.
        p_cam = pts @ R.T + img.tvec
        front = p_cam[:, 2] > 0
        f = cam.params
        fx, fy, ccx, ccy = (f[0], f[0], f[1], f[2]) if len(f) == 3 else (f[0], f[1], f[2], f[3])
        proj = np.stack([fx * p_cam[:, 0] / p_cam[:, 2] + ccx,
                         fy * p_cam[:, 1] / p_cam[:, 2] + ccy], -1)
        reproj_px = np.linalg.norm(proj[front] - xy[front], axis=1)
        reproj_med = float(np.median(reproj_px)) if front.any() else float("nan")

        u = np.round(xy[:, 0] * sx).astype(int)
        v = np.round(xy[:, 1] * sy).astype(int)
        ok = (u >= 0) & (u < pw) & (v >= 0) & (v < ph) & (z_cam > 0)
        ok &= valid[np.clip(v, 0, ph - 1), np.clip(u, 0, pw - 1)]
        if ok.sum() < 20:
            print(f"{cam_id}: only {ok.sum()} usable observations, skipping")
            continue
        per_cam[cam_id] = (depth[v[ok], u[ok]], z_cam[ok], reproj_med)

    print(f"\n{'cam':8} {'n':>6} {'spearman':>10} {'pearson':>9} {'med|err|m':>10} {'reproj_px':>10}")
    sps, prs, errs = [], [], []
    for cam_id in sorted(per_cam):
        d_prior, d_true, reproj = per_cam[cam_id]
        sp, pr = spearman(d_prior, d_true), pearson(d_prior, d_true)
        # Only meaningful for a metric prior (the Depthkit one); a MASt3R prior
        # is in an arbitrary scale and this column is noise there.
        err = float(np.median(np.abs(d_prior - d_true)))
        sps.append(sp); prs.append(pr); errs.append(err)
        print(f"{cam_id:8} {len(d_prior):6d} {sp:10.3f} {pr:9.3f} {err:10.3f} {reproj:10.2f}")
    if sps:
        print(f"{'MEAN':8} {'':6} {np.mean(sps):10.3f} {np.mean(prs):9.3f} {np.mean(errs):10.3f}")
        print("\nreproj_px is the control: sub-pixel means the poses, intrinsics and "
              "depth convention used here are correct, so a near-zero correlation is a "
              "real property of the prior and not a bug in this script.")
        print("\nRank correlation is what the patch-Pearson loss can exploit. "
              "Near 1.0 = the prior orders depth like the real geometry; "
              "near 0 = the prior carries no usable signal.")


if __name__ == "__main__":
    main()
