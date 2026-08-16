"""Generate static-structure depth priors for --depth_supervision (issue #19).

The scene is dynamic but its *structure* is not: the room is static and only the
subject moves. So we build one depth prior per camera rather than per frame:

  1. Reduce each camera's clip to a temporal-median background plate. The moving
     subject is a minority of the samples at any pixel, so the median is the
     empty room.
  2. Mark every pixel the subject ever touches as invalid (temporal MAD above a
     threshold, dilated). A single static prior is then correct at every frame of
     that camera, because the pixels where it would have been wrong carry no
     supervision at all.
  3. Run MASt3R multi-view inference over the plates and globally align them, so
     all cameras share one depth scale.

The loss is scale-invariant per patch, so MASt3R's arbitrary global scale and
its disagreement with the COLMAP frame do not matter — only local structure does.

Usage:
  python scripts/make_mast3r_depth_priors.py \
      --scene ~/4C4D/data/Xuelong/clip_f300_5s_rgb_posefix \
      --cameras 0,1,2,3,5,7,8,9
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np


def build_plates(scene, cameras, stride, mad_threshold, dilate, size, verbose=True):
    """Per camera: (median plate BGR uint8, static mask bool) at `size`."""
    from concurrent.futures import ThreadPoolExecutor

    img_dir = os.path.join(scene, "images")
    out = {}
    for cam in cameras:
        frames = sorted(f for f in os.listdir(img_dir) if f.startswith(cam + "_"))
        frames = frames[::stride]

        def read(name):
            im = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
            # INTER_AREA is a box filter: the right way down, and it averages
            # away sensor noise that would otherwise inflate the motion mask.
            return cv2.resize(im, size, interpolation=cv2.INTER_AREA)

        with ThreadPoolExecutor(max_workers=8) as pool:
            stack = np.stack(list(pool.map(read, frames)))

        plate = np.median(stack, axis=0)
        # MAD, not std: a pixel the subject crosses briefly has a huge std but a
        # small MAD, and we want to flag it either way -- so take the max over
        # channels and threshold low.
        mad = np.median(np.abs(stack.astype(np.float32) - plate), axis=0).max(axis=-1)
        moving = (mad > mad_threshold).astype(np.uint8)
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
            moving = cv2.dilate(moving, k)
        static = moving == 0

        out[cam] = (plate.astype(np.uint8), static)
        if verbose:
            print(f"  {cam}: {len(frames)} frames -> plate {size[0]}x{size[1]}, "
                  f"{100.0 * static.mean():.1f}% static")
        del stack
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cameras", default="0,1,2,3,5,7,8,9")
    ap.add_argument("--out", default="")
    ap.add_argument("--matcha", default=os.path.expanduser("~/MAtCha"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=2, help="frame stride for the median plate")
    ap.add_argument("--mad_threshold", type=float, default=6.0,
                    help="temporal MAD (0-255) above which a pixel counts as moving")
    ap.add_argument("--dilate", type=int, default=9, help="dilation of the motion mask, in pixels")
    ap.add_argument("--niter", type=int, default=500, help="global alignment iterations")
    ap.add_argument("--min_conf", type=float, default=3.0, help="MASt3R confidence threshold")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reuse_plates", action="store_true",
                    help="reuse cached median plates/masks instead of re-reading the clip")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate plates CCW by this much before MASt3R, and rotate the depth "
                         "back after. This rig films sideways, and MASt3R is trained on upright "
                         "imagery -- feeding it rotated frames yields a featureless depth ramp.")
    args = ap.parse_args()

    scene = os.path.expanduser(args.scene)
    out_dir = os.path.expanduser(args.out) if args.out else os.path.join(scene, "depth_priors")
    matcha = os.path.expanduser(args.matcha)
    cameras = [f"cam{int(c):02d}" for c in args.cameras.split(",")]
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(matcha, "mast3r"))
    import mast3r.utils.path_to_dust3r  # noqa: F401  (puts dust3r on sys.path)
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    import torch

    # 2560x1440 -> 512x288, which is already a multiple of 16, so load_images
    # will not crop and the prior is a pure rescale of the full frame. Anything
    # else would silently misalign the prior against the render.
    probe = cv2.imread(os.path.join(scene, "images", f"{cameras[0]}_0000.png"))
    full_h, full_w = probe.shape[:2]
    size = (args.image_size, int(round(args.image_size * full_h / full_w)))
    assert size[1] % 16 == 0, f"{size} is not a multiple of 16; load_images would crop"

    plate_dir = os.path.join(out_dir, "plates")
    os.makedirs(plate_dir, exist_ok=True)
    cached = os.path.join(plate_dir, "plates.npz")

    if args.reuse_plates and os.path.exists(cached):
        print(f"Reusing cached plates from {cached}")
        with np.load(cached) as z:
            plates = {cam: (z[f"{cam}_plate"], z[f"{cam}_static"]) for cam in cameras}
    else:
        print(f"Building background plates from {full_w}x{full_h} frames (stride {args.stride})...")
        plates = build_plates(scene, cameras, args.stride, args.mad_threshold, args.dilate, size)
        np.savez_compressed(cached, **{
            f"{cam}_{k}": v for cam in cameras
            for k, v in zip(("plate", "static"), plates[cam])})

    fwd = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_CLOCKWISE}.get(args.rotate)
    inv = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(args.rotate)
    mast3r_size = size if args.rotate in (0, 180) else (size[1], size[0])
    assert mast3r_size[0] % 16 == 0 and mast3r_size[1] % 16 == 0, \
        f"{mast3r_size} is not a multiple of 16; load_images would crop"

    plate_paths = []
    for cam in cameras:
        p = os.path.join(plate_dir, f"{cam}.png")
        plate = plates[cam][0]
        cv2.imwrite(p, cv2.rotate(plate, fwd) if fwd is not None else plate)
        plate_paths.append(p)

    ckpt = os.path.join(matcha, "mast3r", "checkpoints",
                        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    print(f"Loading MASt3R from {ckpt}...")
    model = AsymmetricMASt3R.from_pretrained(ckpt).to(args.device)

    imgs = load_images(plate_paths, size=args.image_size, verbose=True)
    for im in imgs:
        h, w = im["true_shape"][0]
        assert (int(w), int(h)) == mast3r_size, \
            f"load_images returned {w}x{h}, expected {mast3r_size}"

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"Running MASt3R on {len(pairs)} pairs over {len(imgs)} views...")
    output = inference(pairs, model, args.device, batch_size=1, verbose=True)

    net = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer,
                         min_conf_thr=args.min_conf)
    loss = net.compute_global_alignment(init="mst", niter=args.niter, schedule="cosine", lr=args.lr)
    print(f"Global alignment converged, loss={float(loss):.6f}")

    depths = [d.detach().cpu().numpy() for d in net.get_depthmaps()]
    # net.get_conf() applies the aligner's conf transform (log by default);
    # min_conf_thr is defined against the *raw* confidence, which is what
    # get_masks() compares. Mixing the two silently rejects every pixel.
    conf_masks = [m.detach().cpu().numpy() for m in net.get_masks()]
    raw_confs = [c.detach().cpu().numpy() for c in net.im_conf]
    pooled = np.concatenate([c.ravel() for c in raw_confs])
    print(f"Raw MASt3R confidence: p5={np.percentile(pooled, 5):.2f} "
          f"median={np.median(pooled):.2f} p95={np.percentile(pooled, 95):.2f} "
          f"(threshold {args.min_conf})")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"

    preview_dir = os.path.join(out_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    per_camera = {}
    for i, cam in enumerate(cameras):
        depth = depths[i].astype(np.float32)
        conf_mask = conf_masks[i]
        if inv is not None:  # back to the orientation the training frames are in
            depth = cv2.rotate(depth, inv)
            conf_mask = cv2.rotate(conf_mask.astype(np.uint8), inv).astype(bool)
        static = plates[cam][1]
        assert depth.shape == static.shape, f"{depth.shape} vs mask {static.shape}"
        valid = static & conf_mask & (depth > 0)
        np.savez_compressed(os.path.join(out_dir, f"{cam}.npz"), depth=depth, valid=valid)

        # Inverse depth colormap, invalid pixels blacked out -- the cheapest way
        # to catch a prior that is upside down or misaligned before spending a
        # training run on it.
        vis = np.zeros(depth.shape, np.float32)
        if valid.any():
            d = depth[valid]
            lo, hi = np.percentile(d, 2), np.percentile(d, 98)
            vis[valid] = 1.0 - np.clip((depth[valid] - lo) / max(hi - lo, 1e-8), 0, 1)
        vis = (cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
               * valid[..., None])
        cv2.imwrite(os.path.join(preview_dir, f"{cam}_depth.png"), vis)

        per_camera[cam] = {
            "valid_fraction": round(float(valid.mean()), 4),
            "static_fraction": round(float(static.mean()), 4),
            "confident_fraction": round(float(conf_masks[i].mean()), 4),
            "depth_min": round(float(depth[valid].min()), 4) if valid.any() else None,
            "depth_max": round(float(depth[valid].max()), 4) if valid.any() else None,
        }
        print(f"  {cam}: {100 * valid.mean():.1f}% valid "
              f"(static {100 * static.mean():.1f}%, confident {100 * conf_masks[i].mean():.1f}%)")

    manifest = {
        "source": "MASt3R multi-view inference over temporal-median background plates",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene": scene,
        "cameras": cameras,
        "prior_resolution": [size[0], size[1]],
        "source_resolution": [full_w, full_h],
        "note": ("Static-structure depth only: one prior per camera, valid at every frame. "
                 "Pixels the subject ever occupies are marked invalid. Depth is in MASt3R's "
                 "arbitrary global scale; the training loss is scale-invariant per patch."),
        "model_checkpoint": os.path.basename(ckpt),
        "mast3r_commit": "anttwo/MAtCha vendored mast3r",
        "generator_commit": commit,
        "params": {
            "image_size": args.image_size, "stride": args.stride,
            "mad_threshold": args.mad_threshold, "dilate": args.dilate,
            "niter": args.niter, "min_conf": args.min_conf, "lr": args.lr,
            "rotate": args.rotate,
            "scene_graph": "complete", "alignment_loss": float(loss),
        },
        "per_camera": per_camera,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(cameras)} priors + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
