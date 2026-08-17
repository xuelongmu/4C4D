"""Sparse (selective) Adam over gaussians visible in the current batch.

From Taming 3DGS (SIGGRAPH Asia 2024), upstreamed in graphdeco's
gaussian-splatting and available as gsplat's ``SelectiveAdam``. A dense Adam
step reads and writes every one of the ~161 floats per gaussian on every
iteration; with a per-gaussian visibility mask only the touched rows move, so
the step's DRAM traffic scales with the visible count instead of N.

Two behavioural notes:

* Bias correction here uses the global step count, exactly as
  ``torch.optim.Adam`` does, so an all-true mask reproduces the dense step.
  (Upstream's kernel drops bias correction entirely, which is a different
  optimizer; we keep it so the flag is a pure speed change.)
* A dense Adam step still moves a gaussian with a zero gradient, because stale
  momentum keeps being applied and decayed. Masking freezes those gaussians
  instead. That is the intended semantics of the method, but it is a real
  behavioural difference and not only an optimisation.

The CUDA kernel is compiled on first use and cached under
``~/.cache/torch_extensions``; compilation is only attempted when the optimizer
is actually constructed, i.e. when ``--sparse_adam`` is passed.
"""

import atexit
import os
import sys
import threading

import torch

_EXT = None
_EXT_LOCK = threading.Lock()

# Set SPARSE_ADAM_STATS=1 to have the per-parameter step counts reported at exit.
# A masked step that silently fell back to the dense path for every parameter
# would still train and still pass every quality check, so it is worth being
# able to confirm the kernel is the thing doing the work.
_STATS = {"kernel": 0, "dense_fallback": 0, "no_grad": 0} if os.environ.get("SPARSE_ADAM_STATS") else None
if _STATS is not None:
    atexit.register(lambda: print(f"[sparse_adam] step counts: {_STATS}", flush=True))


def _load_extension():
    """JIT-build utils/csrc/sparse_adam.cu, memoised."""
    global _EXT
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load

        # nvcc and ninja both ship in the conda env's bin directory, but that
        # directory is not on PATH when python is invoked by absolute path
        # rather than through an activated shell.
        env_bin = os.path.join(sys.prefix, "bin")
        if os.path.isdir(env_bin) and env_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
        if cpp_extension.CUDA_HOME is None and os.path.exists(os.path.join(env_bin, "nvcc")):
            # cpp_extension resolves CUDA_HOME once at import time, so setting
            # the environment variable alone is not enough.
            os.environ.setdefault("CUDA_HOME", sys.prefix)
            cpp_extension.CUDA_HOME = sys.prefix

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "sparse_adam.cu")
        try:
            _EXT = load(
                name="sparse_adam_4c4d",
                sources=[src],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"--sparse_adam needs to build {src}, and the build failed ({exc}). "
                "It requires ninja (`pip install ninja`) and a working nvcc. "
                "Drop --sparse_adam to train with the dense torch optimizer."
            ) from exc
        return _EXT


class SparseGaussianAdam(torch.optim.Adam):
    """Adam whose step can be restricted to a subset of gaussians.

    Drop-in for ``torch.optim.Adam``: the state layout (``step``, ``exp_avg``,
    ``exp_avg_sq``) is unchanged, so ``GaussianModel``'s densification and
    pruning surgery on ``optimizer.state`` keeps working, and checkpoints stay
    interchangeable with dense runs.
    """

    def __init__(self, params, lr, eps):
        super().__init__(params, lr=lr, eps=eps)
        _load_extension()
        # fetchPly builds positions as np.vstack([x, y, z]).T, so _xyz arrives
        # F-contiguous and stays that way until the first densification cat.
        # Relaying it out is a pure layout change — Adam is elementwise — and
        # the kernel needs row-major gaussians.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.is_contiguous():
                        p.data = p.data.contiguous()

    @torch.no_grad()
    def _dense_param_step(self, p, state, beta1, beta2, eps, step_size, bias_corr2_sqrt):
        """torch.optim.Adam's single-tensor update, for params the kernel skips."""
        state["exp_avg"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
        denom = (state["exp_avg_sq"].sqrt() / bias_corr2_sqrt).add_(eps)
        p.addcdiv_(state["exp_avg"], denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None, visibility=None):
        """Adam step; with ``visibility`` set, only those gaussians are updated.

        ``visibility`` is a bool tensor with one entry per gaussian. A parameter
        falls back to the dense update when the mask cannot apply to it: when N
        changed this iteration (densification runs before the step, and the
        freshly ``cat``-ed parameters carry no gradient, so the step is a no-op
        anyway) or when its layout is not row-major over gaussians.
        """
        if visibility is None:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ext = _load_extension()
        visibility = visibility.contiguous()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            assert not group["amsgrad"] and group["weight_decay"] == 0, (
                "SparseGaussianAdam only implements plain Adam"
            )

            for p in group["params"]:
                if p.grad is None:
                    if _STATS is not None:
                        _STATS["no_grad"] += 1
                    continue
                assert not p.grad.is_sparse, "SparseGaussianAdam does not support sparse gradients"

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                t = state["step"].item()
                step_size = lr / (1.0 - beta1**t)
                bias_corr2_sqrt = (1.0 - beta2**t) ** 0.5

                if (p.shape[0] != visibility.numel() or not p.is_contiguous()
                        or not state["exp_avg"].is_contiguous()
                        or not state["exp_avg_sq"].is_contiguous()):
                    self._dense_param_step(p, state, beta1, beta2, eps, step_size, bias_corr2_sqrt)
                    if _STATS is not None:
                        _STATS["dense_fallback"] += 1
                    continue
                if _STATS is not None:
                    _STATS["kernel"] += 1

                ext.sparse_adam_step(
                    p.data,
                    p.grad.contiguous(),
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    visibility,
                    beta1,
                    beta2,
                    eps,
                    step_size,
                    bias_corr2_sqrt,
                )

        return loss
