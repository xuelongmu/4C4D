"""Report the temporal-support distribution of a trained 4D gaussian checkpoint.

The static/dynamic split keys off whether a gaussian's temporal marginal
    exp(-0.5 (t - tau)^2 / cov_t)
stays above the renderer's 0.05 gate across the whole clip. That is equivalent
to a half-width

    w = sqrt(-2 ln(0.05) * cov_t) ~= sqrt(5.99 * cov_t)

so a gaussian covers [t - w, t + w] and is "whole-clip static" iff
w >= max(t - t0, t1 - t).

Usage: scripts/temporal_stats.py <checkpoint.pth> [more.pth ...]
Run from the repo root (imports utils.general_utils for the 4D covariance).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general_utils import build_scaling_rotation_4d

GATE = 0.05
TIME_DURATION = tuple(float(x) for x in os.environ.get("CLIP_BOUNDS", "0,5").split(","))


def stats(path):
    params, iteration = torch.load(path, map_location="cuda", weights_only=False)
    xyz, opacity = params[1], params[6]
    scaling, rotation = params[4], params[5]
    t, scaling_t, rotation_r = params[14], params[15], params[16]
    rot_4d = params[17]
    n = xyz.shape[0]

    if rot_4d:
        scaling_xyzt = torch.exp(torch.cat([scaling, scaling_t], dim=1))
        L = build_scaling_rotation_4d(scaling_xyzt, rotation, rotation_r)
        cov_t = (L @ L.transpose(1, 2))[:, 3, 3].unsqueeze(1)
    else:
        cov_t = torch.exp(scaling_t)

    w = torch.sqrt(-2.0 * torch.log(torch.tensor(GATE, device=cov_t.device)) * cov_t)[:, 0]
    t0, t1 = TIME_DURATION
    duration = t1 - t0
    tt = t[:, 0]
    static = w >= torch.maximum(tt - t0, t1 - tt)
    covered = (torch.minimum(tt + w, torch.tensor(t1, device=w.device))
               - torch.maximum(tt - w, torch.tensor(t0, device=w.device))).clamp_min(0) / duration
    alpha = torch.sigmoid(opacity)[:, 0]

    w_init = (-2.0 * torch.log(torch.tensor(GATE)) * (duration / 5.0) ** 2).sqrt().item()
    print(f"\n=== {path}  (iter {iteration}, N={n:,}) ===")
    print(f"static fraction (whole-clip):        {static.float().mean():.4f}")
    print(f"static fraction, opacity-weighted:   {(static.float() * alpha).sum() / alpha.sum():.4f}")
    print(f"half-width needed at clip midpoint:  {duration / 2:.3f}")
    print(f"half-width at init (cov_t=(D/5)^2):  {w_init:.3f}   <-- below the {duration / 2:.1f} needed")

    qs = torch.tensor([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0], device=w.device)
    for name, v in (("t", tt), ("cov_t", cov_t[:, 0]), ("half-width", w),
                    ("clip coverage", covered), ("opacity", alpha)):
        vals = torch.quantile(v.float(), qs).tolist()
        print(f"  {name:>14} " + " ".join(f"{q * 100:>3.0f}%={x:<8.3f}"
                                          for q, x in zip(qs.tolist(), vals)))
    for thr in (0.10, 0.25, 0.50, 0.75, 0.90, 1.00):
        print(f"  clip coverage >= {thr:>4.0%}: {(covered >= thr).float().mean():.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
