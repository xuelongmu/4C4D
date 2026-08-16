#
# Loading of precomputed depth priors used by --depth_supervision.
#
# Priors live outside the repo (they are hundreds of MB and are regenerated
# from the scene, not versioned). Each file is an .npz holding 'depth'
# (H, W float32, any scale) and optionally 'valid' (H, W bool). A sibling
# manifest.json records how they were produced; it is only logged, never
# required.
#

import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def _resize(t, height, width):
    # Nearest, not bilinear: interpolating depth across an occlusion boundary
    # invents surfaces that exist in neither the near nor the far object.
    return F.interpolate(t[None, None], size=(height, width), mode="nearest")[0, 0]


def load_depth_priors(prior_dir, cameras, device="cuda", verbose=True):
    """Map image_name -> (depth, valid) for every camera that has a prior.

    A per-frame prior (``cam00_0042.npz``) takes precedence over a per-camera
    one (``cam00.npz``), so a static multi-view prior and a per-frame video
    prior load through the same path. Cameras with no prior are simply absent
    from the returned dict and are skipped by the loss.

    Priors are resized to each camera's training resolution and cached per
    (file, resolution), so the 8 static maps here are held once no matter how
    many frames reference them.
    """
    if not os.path.isdir(prior_dir):
        raise FileNotFoundError(
            f"--depth_supervision is on but no depth priors at {prior_dir}. "
            f"Generate them with scripts/make_mast3r_depth_priors.py or pass "
            f"--depth_prior_dir.")

    manifest_path = os.path.join(prior_dir, "manifest.json")
    if verbose and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Depth priors: {manifest.get('source', 'unknown source')} "
              f"({manifest.get('generated_at', 'undated')})")

    cache = {}
    priors = {}
    missing = set()
    for cam in cameras:
        name = cam.image_name
        cam_id = name.split('_')[0]
        for stem in (name, cam_id):
            path = os.path.join(prior_dir, f"{stem}.npz")
            if os.path.exists(path):
                break
        else:
            missing.add(cam_id)
            continue

        key = (path, cam.image_height, cam.image_width)
        if key not in cache:
            with np.load(path) as data:
                depth = torch.from_numpy(data["depth"].astype(np.float32))
                if "valid" in data:
                    valid = torch.from_numpy(data["valid"]).to(torch.float32)
                else:
                    valid = (depth > 0).to(torch.float32)
            depth = _resize(depth.to(device), cam.image_height, cam.image_width)
            valid = _resize(valid.to(device), cam.image_height, cam.image_width) > 0.5
            # A prior pixel of zero/negative depth is "no measurement", not "at
            # the camera"; the median normalization in the loss would be
            # dragged toward zero if these leaked through.
            valid &= depth > 0
            cache[key] = (depth, valid)
        priors[name] = cache[key]

    if verbose:
        covered = sorted({n.split('_')[0] for n in priors})
        print(f"Depth priors: {len(priors)} views over cameras {covered}"
              + (f"; no prior for {sorted(missing)}" if missing else ""))
        for (path, h, w), (_, valid) in cache.items():
            print(f"  {os.path.basename(path)} -> {w}x{h}, "
                  f"{100.0 * valid.float().mean().item():.1f}% valid")
    if not priors:
        raise RuntimeError(
            f"--depth_supervision is on but none of the {len(cameras)} training "
            f"views matched a prior in {prior_dir}")
    return priors
