#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from math import exp
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import math

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

def msssim(rgb, gts):
    # assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    # assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(rgb, gts).item()


def _to_patches(t, patch_size):
    """(H, W) -> (num_patches, patch_size * patch_size), non-overlapping."""
    h, w = t.shape
    return (t.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
             .permute(0, 2, 1, 3)
             .reshape(-1, patch_size * patch_size))


def patchwise_pearson_depth_loss(render_depth, prior_depth, valid_mask=None,
                                 patch_size=32, min_valid_frac=0.5, var_floor=1e-6):
    """Scale-invariant patch-wise depth loss (DNGaussian / FSGS / SparseGS style).

    Pearson correlation is invariant to an affine remap of depth, so a prior of
    unknown scale and offset still constrains geometry. It is computed over
    non-overlapping patches rather than the whole image: a global correlation is
    dominated by the coarse foreground/background split and barely constrains
    local structure, which is where sparse-view floaters live.

    Both depth maps are divided by their masked median first. Pearson is already
    scale-free per patch, but the shared normalization puts `var_floor` in
    dimensionless units so it means "this patch is flat" for either input.

    Patches that are mostly invalid, or where the prior carries no depth
    variation, are dropped: correlation is undefined there and the gradient
    would be noise. Returns a 0-d tensor (detached zero if nothing survives).
    """
    if render_depth.dim() == 3:
        render_depth = render_depth.squeeze(0)
    if prior_depth.dim() == 3:
        prior_depth = prior_depth.squeeze(0)

    h, w = render_depth.shape
    ph, pw = h - h % patch_size, w - w % patch_size
    if ph == 0 or pw == 0:
        return render_depth.new_zeros(())

    x = render_depth[:ph, :pw]
    y = prior_depth[:ph, :pw]
    if valid_mask is None:
        m = torch.ones_like(y)
    else:
        m = valid_mask.squeeze(0)[:ph, :pw].to(y.dtype)

    if m.sum() < patch_size * patch_size:
        return render_depth.new_zeros(())

    # Shared normalization so var_floor is scale-free (see docstring).
    sel = m > 0
    x = x / x.detach()[sel].median().clamp_min(1e-8)
    y = y / y[sel].median().clamp_min(1e-8)

    x, y, m = (_to_patches(t, patch_size) for t in (x, y, m))

    n = m.sum(1)
    keep = n >= min_valid_frac * patch_size * patch_size
    if not keep.any():
        return render_depth.new_zeros(())
    x, y, m, n = x[keep], y[keep], m[keep], n[keep]

    xc = (x - (x * m).sum(1, keepdim=True) / n[:, None]) * m
    yc = (y - (y * m).sum(1, keepdim=True) / n[:, None]) * m
    var_x = (xc * xc).sum(1) / n
    var_y = (yc * yc).sum(1) / n
    cov = (xc * yc).sum(1) / n

    # A flat prior patch has no ordering to transfer; a flat render patch would
    # otherwise divide by ~0. Floor the render variance instead of dropping it,
    # so empty regions still feel a pull toward the prior's structure.
    ok = var_y > var_floor
    if not ok.any():
        return render_depth.new_zeros(())
    corr = cov[ok] / (var_x[ok].clamp_min(var_floor) * var_y[ok]).sqrt()
    return (1.0 - corr.clamp(-1.0, 1.0)).mean()
