"""Correctness checks for SparseGaussianAdam against torch.optim.Adam.

Run from the repo root on a CUDA machine:

    python tests/test_sparse_adam.py

The parameter shapes mirror a 4D GaussianModel with sh_degree=3 and
sh_degree_t=2 (161 floats per gaussian). Values are compared to float32
rounding, not bitwise: the kernel does the same arithmetic in the same order as
torch's single-tensor Adam, but fuses it into one pass, so results differ by
about one ulp.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sparse_adam import SparseGaussianAdam  # noqa: E402

N = 5000
SHAPES = [(3,), (1, 3), (47, 3), (1,), (3,), (4,), (1,), (1,), (4,)]
DEV = "cuda"


def make(seed):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn((N,) + s, device=DEV)) for s in SHAPES]


def dense(params):
    return torch.optim.Adam([{"params": [p], "lr": 1e-3} for p in params], lr=0.0, eps=1e-15)


def sparse(params):
    return SparseGaussianAdam([{"params": [p], "lr": 1e-3} for p in params], lr=0.0, eps=1e-15)


def drive(a, b, oa, ob, visibility, steps, seed):
    """Feed both optimizers identical gradients for `steps` iterations."""
    for i in range(steps):
        torch.manual_seed(seed + i)
        grads = [torch.randn_like(p) for p in a]
        for p, g in zip(a, grads):
            p.grad = g.clone()
        for p, g in zip(b, grads):
            p.grad = g.clone()
        oa.step()
        ob.step(visibility=visibility)


def test_all_visible_matches_dense():
    a, b = make(1), make(1)
    oa, ob = dense(a), sparse(b)
    vis = torch.ones(N, dtype=torch.bool, device=DEV)
    drive(a, b, oa, ob, vis, steps=20, seed=100)

    worst = max((x - y).abs().max().item() for x, y in zip(a, b))
    worst_m = max(
        (oa.state[x]["exp_avg"] - ob.state[y]["exp_avg"]).abs().max().item() for x, y in zip(a, b)
    )
    print(f"[1] all-visible vs dense Adam, 20 steps: max|dparam|={worst:.3e} max|dexp_avg|={worst_m:.3e}")
    assert worst < 1e-5, "an all-true mask must reproduce the dense step"
    assert worst_m < 1e-6, "moments must track the dense step"


def test_masked_freezes_invisible():
    a, b = make(2), make(2)
    oa, ob = dense(a), sparse(b)
    torch.manual_seed(7)
    vis = torch.rand(N, device=DEV) < 0.4
    before = [p.detach().clone() for p in b]
    drive(a, b, oa, ob, vis, steps=5, seed=200)

    frozen = max((p[~vis] - p0[~vis]).abs().max().item() for p, p0 in zip(b, before))
    moved = max((x[vis] - y[vis]).abs().max().item() for x, y in zip(a, b))
    print(f"[2] masked (40% visible), 5 steps: max|dinvisible|={frozen:.3e} max|visible-dense|={moved:.3e}")
    assert frozen == 0.0, "invisible gaussians must not move"
    assert moved < 1e-6, "visible gaussians must match the dense update"

    for p in b:
        st = ob.state[p]
        assert st["exp_avg"][~vis].abs().max().item() == 0.0, "invisible moments must stay untouched"
        assert st["exp_avg_sq"][~vis].abs().max().item() == 0.0


def test_stale_mask_falls_back_to_dense():
    """Densification changes N before the step; the stale mask must be ignored."""
    a, b = make(3), make(3)
    oa, ob = dense(a), sparse(b)
    short = torch.ones(N // 2, dtype=torch.bool, device=DEV)
    drive(a, b, oa, ob, short, steps=3, seed=300)

    worst = max((x - y).abs().max().item() for x, y in zip(a, b))
    print(f"[3] stale-mask fallback vs dense Adam, 3 steps: max|dparam|={worst:.3e}")
    assert worst < 1e-6, "the fallback must reproduce a dense step"


def test_f_contiguous_param_is_relaid_out():
    """fetchPly hands _xyz over as np.vstack([x, y, z]).T, i.e. F-contiguous."""
    ref = torch.randn(3, N, device=DEV).T
    assert not ref.is_contiguous() and ref.stride() == (1, N)
    a = [torch.nn.Parameter(ref.clone().contiguous())]
    b = [torch.nn.Parameter(ref.clone())]
    oa, ob = dense(a), sparse(b)
    assert b[0].is_contiguous(), "the optimizer must relay out F-contiguous params"
    assert (a[0] - b[0]).abs().max().item() == 0.0, "relayout must preserve values"

    torch.manual_seed(11)
    vis = torch.rand(N, device=DEV) < 0.4
    drive(a, b, oa, ob, vis, steps=3, seed=400)
    moved = (a[0][vis] - b[0][vis]).abs().max().item()
    print(f"[5] F-contiguous param, 3 masked steps: max|visible-dense|={moved:.3e}")
    assert moved < 1e-6, "a relaid-out param must still match the dense update"


def test_missing_grad_is_skipped():
    """Freshly cat-ed parameters carry no gradient on densification iterations."""
    p = torch.nn.Parameter(torch.randn(N, 3, device=DEV))
    o = sparse([p])
    p0 = p.detach().clone()
    o.step(visibility=torch.ones(N, dtype=torch.bool, device=DEV))
    assert (p - p0).abs().max().item() == 0.0
    print("[4] params with grad=None are skipped: ok")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "these tests need a CUDA device"
    test_all_visible_matches_dense()
    test_masked_freezes_invisible()
    test_stale_mask_falls_back_to_dense()
    test_f_contiguous_param_is_relaid_out()
    test_missing_grad_is_skipped()
    print("ALL SPARSE ADAM TESTS PASSED")
