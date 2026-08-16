"""Unit checks for the scale-invariant depth loss (issue #19).

Run: python scripts/test_depth_loss.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils.loss_utils import patchwise_pearson_depth_loss as ploss

torch.manual_seed(0)
H = W = 128
prior = torch.rand(H, W) * 5.0 + 1.0
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


# 1. Identical depth -> zero loss
v = ploss(prior.clone(), prior).item()
check("identical -> 0", abs(v) < 1e-4, f"loss={v:.2e}")

# 2. Affine remap (unknown scale AND offset) -> still zero. This is the whole
#    point: the prior need not share the reconstruction's scale.
v = ploss(prior * 7.3 + 12.0, prior).item()
check("affine-invariant", abs(v) < 1e-4, f"loss={v:.2e}")

# 3. Anti-correlated -> 2 (worst case of 1 - corr)
v = ploss(-prior, prior).item()
check("anti-correlated -> 2", abs(v - 2.0) < 1e-3, f"loss={v:.4f}")

# 4. Independent noise -> ~1
v = ploss(torch.rand(H, W) + 1.0, prior).item()
check("uncorrelated -> ~1", abs(v - 1.0) < 0.15, f"loss={v:.4f}")

# 5. Gradient reaches the rendered depth and points somewhere useful
d = (torch.rand(H, W) + 1.0).requires_grad_(True)
ploss(d, prior).backward()
check("gradient flows", d.grad is not None and torch.isfinite(d.grad).all()
      and d.grad.abs().sum() > 0, f"|grad|={d.grad.abs().sum():.3e}")

# 6. Descending the loss actually improves correlation
d = (torch.rand(H, W) + 1.0).requires_grad_(True)
opt = torch.optim.Adam([d], lr=0.05)
first = None
for _ in range(200):
    opt.zero_grad(); l = ploss(d, prior); l.backward(); opt.step()
    first = l.item() if first is None else first
check("optimizable", l.item() < first - 0.1, f"{first:.4f} -> {l.item():.4f}")

# 7. Masked-out region must not influence the result
prior_v = prior.clone()
mask = torch.ones(H, W, dtype=torch.bool)
mask[:, :64] = False
junk = prior.clone(); junk[:, :64] = torch.rand(H, 64) * 100
a = ploss(junk, prior_v, mask).item()
b = ploss(prior.clone(), prior_v, mask).item()
check("mask excludes region", abs(a - b) < 1e-4, f"{a:.2e} vs {b:.2e}")

# 8. Flat prior patches are dropped rather than producing NaN
v = ploss(torch.rand(H, W) + 1.0, torch.full((H, W), 3.0))
check("flat prior -> 0, no NaN", torch.isfinite(v) and abs(v.item()) < 1e-6, f"loss={v.item():.2e}")

# 9. All-invalid mask -> zero, no NaN
v = ploss(torch.rand(H, W) + 1.0, prior, torch.zeros(H, W, dtype=torch.bool))
check("empty mask -> 0", torch.isfinite(v) and abs(v.item()) < 1e-6, f"loss={v.item():.2e}")

# 10. Non-divisible resolution (720/32 = 22.5) must not crash
v = ploss(torch.rand(720, 1280) + 1.0, torch.rand(720, 1280) + 1.0)
check("ragged resolution", torch.isfinite(v), f"loss={v.item():.4f}")

# 11. Channel-first (1,H,W) inputs, as the rasterizer returns
v = ploss(prior.clone().unsqueeze(0), prior.unsqueeze(0),
          torch.ones(1, H, W, dtype=torch.bool)).item()
check("(1,H,W) inputs", abs(v) < 1e-4, f"loss={v:.2e}")

# 12. Patch-local, not global: a prior that is globally correlated but locally
#     scrambled must score worse than the true one.
scrambled = prior.clone()
for i in range(0, H, 32):
    for j in range(0, W, 32):
        scrambled[i:i+32, j:j+32] = scrambled[i:i+32, j:j+32].flatten()[
            torch.randperm(32 * 32)].reshape(32, 32)
v = ploss(scrambled, prior).item()
check("penalizes local scramble", v > 0.8, f"loss={v:.4f}")

print()
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
