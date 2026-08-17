#!/usr/bin/env python
"""Evaluate a checkpoint on the COMPLETE held-out set.

Why this exists: training reports held-out PSNR from
`Scene.getValidationCameras(tag='test')`, which takes `[::100]` — on the
Xuelong 8-camera split that is **3 images out of 300**. A 3-image average is
a high-variance estimator, and it is the number every A/B in this campaign was
judged on. Three same-config runs spread 1.68 dB on it.

This script re-scores saved checkpoints against every held-out image, so
existing experiments can be re-compared at much lower variance without
retraining. Costs about a minute per checkpoint.

Usage:
  python scripts/evaluate_full_heldout.py \
      --config configs/custom/xuelong_clip_f300_5s_rgb_posefix_7500.yaml \
      --training_view 0,1,2,3,5,7,8,9 --res 2 \
      --checkpoints /path/run-a/chkpnt7500.pth /path/run-b/chkpnt7500.pth \
      --stride 1 --json out.json
"""
import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402
from fused_ssim import fused_ssim  # noqa: E402


@torch.no_grad()
def score(scene, gaussians, pipe, background, args, stride):
    cams = scene.getTestCameras()
    idx = range(0, len(cams), stride)
    l1_sum = psnr_sum = ssim_sum = 0.0
    n = 0
    for i in tqdm(idx, desc="held-out", ncols=80):
        gt_image, viewpoint = cams[i]
        gt_image = gt_image.cuda()
        viewpoint = viewpoint.cuda()
        image = torch.clamp(render(viewpoint, gaussians, pipe, background,
                                   args=args, iteration=10 ** 9)["render"], 0.0, 1.0)
        l1_sum += l1_loss(image, gt_image).mean().double().item()
        psnr_sum += psnr(image, gt_image).mean().double().item()
        ssim_sum += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().item()
        n += 1
    return {"n_images": n, "l1": l1_sum / n, "psnr": psnr_sum / n, "ssim": ssim_sum / n}


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    op = OptimizationParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--training_view", type=str, default="")
    ap.add_argument("--testing_view", type=str, default="")
    ap.add_argument("--res", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1,
                    help="1 = every held-out image (default); raise to sample")
    ap.add_argument("--json", type=str, default="")
    # Flags render()/Scene read off the args namespace.
    ap.add_argument("--opacity_decay", action="store_true", default=False)
    ap.add_argument("--time_aware", action="store_true", default=False)
    ap.add_argument("--gaussian_dim", type=int, default=4)
    ap.add_argument("--time_duration", nargs=2, type=float, default=[0, 5.0])
    ap.add_argument("--rot_4d", action="store_true", default=True)
    ap.add_argument("--force_sh_3d", action="store_true", default=False)
    ap.add_argument("--num_pts", type=int, default=75000)
    ap.add_argument("--num_pts_ratio", type=float, default=1.0)
    ap.add_argument("--redundant_ratio", type=float, default=0.0)
    ap.add_argument("--downsample_method", type=str, default="random")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    def merge(key, host):
        if isinstance(host[key], DictConfig):
            for k in host[key].keys():
                merge(k, host[key])
        elif hasattr(args, key):
            setattr(args, key, host[key])
    for k in cfg.keys():
        merge(k, cfg)
    args.resolution = args.res
    args.eval = True
    if args.training_view:
        args.training_view = [f"cam{str(int(c)).zfill(2)}" for c in sorted(args.training_view.split(','))]
    if args.testing_view:
        args.testing_view = [f"cam{str(int(c)).zfill(2)}" for c in sorted(args.testing_view.split(','))]

    dataset, pipe = lp.extract(args), pp.extract(args)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")

    results = {}
    scene = gaussians = None
    for ckpt in args.checkpoints:
        if gaussians is None:
            gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim,
                                      time_duration=args.time_duration, rot_4d=args.rot_4d,
                                      force_sh_3d=args.force_sh_3d,
                                      sh_degree_t=2 if pipe.eval_shfs_4d else 0)
            # Scene is built once: the held-out camera list is identical across
            # runs, and rebuilding it per checkpoint would dominate runtime.
            scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts,
                          num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration,
                          training_view=args.training_view, testing_view=args.testing_view,
                          redundant_ratio=args.redundant_ratio,
                          downsample_method=args.downsample_method)
        model_params, it = torch.load(ckpt)
        gaussians.restore(model_params, None)
        name = os.path.basename(os.path.dirname(ckpt))
        r = score(scene, gaussians, pipe, background, args, args.stride)
        r["iteration"] = it
        results[name] = r
        print(f"{name:24s} n={r['n_images']:4d}  PSNR {r['psnr']:.4f}  "
              f"SSIM {r['ssim']:.4f}  L1 {r['l1']:.5f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
