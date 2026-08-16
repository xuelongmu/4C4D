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

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss #, ssim, msssim
from gaussian_renderer import render, decay_visibility
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from module import Coefficient

import imageio
import math
from fused_ssim import fused_ssim as fast_ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from datetime import datetime


def color_affine_key(image_name):
    """Camera identity behind a frame's image name, for per-camera color affine.

    Rig captures name frames <camera>_<frame> (cam03_0117), so the trailing
    frame number is dropped; anything else is used whole.
    """
    head, sep, tail = image_name.rpartition('_')
    return head if sep and tail.isdigit() else image_name


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    

    # The regularizer losses behind lambda_opa_mask/lambda_rigid/lambda_motion
    # are not implemented; the previous vars()-based EMA plumbing silently did
    # nothing and would raise KeyError if a lambda were nonzero. Fail fast
    # instead of training something other than what the config claims, and do
    # it before the logger, model, and scene are built so an invalid config
    # does not pay the dataset load and CUDA allocations first.
    unimplemented_lambdas = [key for key in opt.__dict__.keys()
                             if key.startswith('lambda') and key != 'lambda_dssim'
                             and opt.__dict__[key] != 0]
    if unimplemented_lambdas:
        raise NotImplementedError(
            f"Losses for {unimplemented_lambdas} are not implemented; set them to 0 "
            "or implement the corresponding regularizers.")

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    if args.opacity_decay:
        coefficient = Coefficient(hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).cuda()
    else:
        coefficient = None
        
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, 
                              rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0, coefficient=coefficient)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, 
                  time_duration=time_duration, training_view=args.training_view, testing_view=args.testing_view,
                  redundant_ratio=args.redundant_ratio, downsample_method=args.downsample_method)
    gaussians.training_setup(opt)  
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print("Restored gaussians from checkpoint: {}".format(checkpoint))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    best_psnr = 0.0
    has_test_views = len(scene.getValidationCameras(tag='test')) > 0
    if not has_test_views:
        print("No held-out test views: chkpnt_best.pth will not be written")
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training", ncols=110)
    first_iter += 1
        
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),
                                           dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        
    gaussians.env_map = env_map
    training_dataset = scene.getTrainCameras()

    # Per-camera learnable color affine (3x4: linear color mix + offset),
    # parameterized as a delta from identity. Multi-camera rigs have per-ISP
    # color mismatch that otherwise gets absorbed into view-dependent SH.
    # Applied to the rendered image only when computing the training loss;
    # evaluation and held-out cameras always use the raw render. The first
    # training camera is anchored to identity and the rest are weight-decayed
    # toward it, otherwise the affines form a global color gauge the model
    # drifts into (raw renders and held-out cameras then mismatch).
    color_affine_delta = None
    color_affine_optimizer = None
    color_affine_index = {}
    if args.color_affine:
        # Index by the cameras actually in the training set, not by
        # args.training_view: the Blender loader ignores that list entirely, and
        # the Colmap loader only honours it when eval is on, so with the default
        # eval=False every camera outside the list missed the lookup, fell back
        # to the never-optimized anchor at index 0, and the feature silently did
        # nothing.
        cam_names = sorted({color_affine_key(c.image_name)
                            for c in training_dataset.viewpoint_stack})
        color_affine_index = {name: i for i, name in enumerate(cam_names)}
        if len(cam_names) < 2:
            print(f"Warning: --color_affine found {len(cam_names)} distinct camera(s) "
                  f"({cam_names}); every view maps to the identity anchor, so "
                  "compensation is a no-op for this dataset")
        color_affine_delta = nn.Parameter(torch.zeros(len(cam_names), 3, 4, device="cuda"))
        color_affine_optimizer = torch.optim.AdamW(
            [color_affine_delta], lr=args.color_affine_lr, weight_decay=args.color_affine_weight_decay)
        color_affine_eye = torch.eye(3, 4, device="cuda")

        # Resuming must continue from the saved compensation, not from identity
        # with a fresh optimizer, which would abruptly change the training
        # objective mid-run.
        if checkpoint:
            affine_state_path = os.path.join(
                os.path.dirname(checkpoint) or ".", "color_affine_resume.pth")
            if os.path.exists(affine_state_path):
                affine_state = torch.load(affine_state_path)
                if affine_state['index'] == color_affine_index:
                    color_affine_delta.data.copy_(affine_state['affine_delta'].cuda())
                    color_affine_optimizer.load_state_dict(affine_state['optimizer'])
                    print(f"Restored color affine state from {affine_state_path}")
                else:
                    print(f"Warning: {affine_state_path} indexes different cameras "
                          "than this run; starting color affine from identity")
            else:
                print(f"Warning: --color_affine resumed from {checkpoint} but no "
                      f"{os.path.basename(affine_state_path)} beside it; "
                      "starting color affine from identity")
    if dataset.dataloader:
        print("\nUsing DataLoader for training dataset")
    else:
        print("\nNot using DataLoader for training dataset")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
    
    img_dir = os.path.join(scene.model_path, "rendered_images")
    os.makedirs(img_dir, exist_ok=True)
    
    iteration = first_iter
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            if iteration % 100 == 0:
                iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []

            batch_cams = [batch_data[i][1].cuda() for i in range(batch_size)]

            # Opacity decay, applied exactly once per optimizer step rather than
            # once per batch item. Each gaussian's decay exponent is the number
            # of the step's viewpoints that can see it, so the result does not
            # depend on which camera happens to come first in the shuffled
            # batch.
            #
            # Every render in the step shares the decayed opacity, but each one
            # backpropagates separately, so the shared tensor is handed to them
            # as a detached leaf: the per-item backward passes accumulate into
            # its .grad instead of trying to walk (and free) the one decay
            # subgraph four times. That subgraph is backpropagated once after
            # the loop, which is also what carries the render gradients back to
            # _opacity and the coefficient network.
            decayed_opacity = None
            shared_opacity = None
            if args.opacity_decay and iteration > args.decay_from_iter:
                if args.time_aware:
                    visibility_counts = torch.zeros(gaussians.get_xyz.shape[0], 1, device="cuda")
                    # The temporal covariance is the expensive part of the
                    # visibility test (build_scaling_rotation_4d plus two
                    # batched Nx4x4 GEMMs) and is the same for every viewpoint
                    # in the step, so build it once instead of per camera.
                    with torch.no_grad():
                        cov_t = gaussians.get_cov_t()
                    for cam in batch_cams:
                        visibility_counts += decay_visibility(
                            cam, gaussians, pipe, background, cov_t=cov_t).view(-1, 1).float()
                else:
                    visibility_counts = batch_size
                decayed_opacity = gaussians.opacity_decay(
                    f_min=args.f_min, f_max=args.f_max, power=visibility_counts)
                shared_opacity = decayed_opacity.detach().requires_grad_(True)

            for batch_idx in range(batch_size):
                gt_image, _ = batch_data[batch_idx]
                viewpoint_cam = batch_cams[batch_idx]
                gt_image = gt_image.cuda()

                render_pkg = render(viewpoint_cam, gaussians, pipe, background, args=args, iteration=iteration,
                                    decayed_opacity=shared_opacity)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # depth, alpha = render_pkg["depth"], render_pkg["alpha"]

                # Loss (per-camera color affine applies only to the training loss;
                # eval and held-out renders stay raw; camera 0 is the anchor)
                affine_idx = color_affine_index.get(color_affine_key(viewpoint_cam.image_name), 0) \
                    if color_affine_delta is not None else 0
                if color_affine_delta is not None and affine_idx != 0:
                    affine = color_affine_eye + color_affine_delta[affine_idx]
                    image_for_loss = torch.einsum('dc,chw->dhw', affine[:, :3], image) + affine[:, 3][:, None, None]
                else:
                    image_for_loss = image
                Ll1 = l1_loss(image_for_loss, gt_image)
                Lssim = 1.0 - fast_ssim(image_for_loss.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                    
                loss = loss / batch_size
                loss.backward()
                
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

            # Chain the batch's accumulated opacity gradient through the decay
            # subgraph exactly once, reaching _opacity and the coefficient.
            if shared_opacity is not None and shared_opacity.grad is not None:
                decayed_opacity.backward(shared_opacity.grad)

            if (iteration % 1500 == 0 or iteration == 2):  # Save every 100 iterations
                # Convert rendered image tensor to numpy and save
                image = torch.clamp(image, 0.0, 1.0)
                img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Convert ground truth image tensor to numpy and save
                gt_img_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
                gt_img_np = (gt_img_np * 255).astype(np.uint8)
                
                # Create filenames with iteration number and camera ID
                img_filename = f"iter_{iteration}_cam_{viewpoint_cam.image_name}.png"
                gt_img_filename = f"iter_{iteration}_cam_{viewpoint_cam.image_name}_gt.png"
                
                imageio.imwrite(os.path.join(img_dir, img_filename), img_np)
                imageio.imwrite(os.path.join(img_dir, gt_img_filename), gt_img_np)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
            
            if iteration % 100 == 0:
                iter_end.record()
            loss_dict = {"Ll1": Ll1, "Lssim": Lssim}

            with torch.no_grad():
                if iteration % 10 == 0:
                    # One fused D2H transfer instead of four separate syncs
                    loss_val, l1_val, ssim_val, psnr_val = torch.stack(
                        [loss.detach(), Ll1.detach(), Lssim.detach(),
                         psnr(image, gt_image).mean()]).tolist()
                    ema_loss_for_log = 0.4 * loss_val + 0.6 * ema_loss_for_log
                    ema_l1loss_for_log = 0.4 * l1_val + 0.6 * ema_l1loss_for_log
                    ema_ssimloss_for_log = 0.4 * ssim_val + 0.6 * ema_ssimloss_for_log

                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                "PSNR": f"{psnr_val:.{2}f}",
                                "gs_num":f"{gaussians.get_xyz.shape[0]}"}

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                    
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log 
                if iteration % 100 == 0 or iteration in testing_iterations:
                    elapsed = 0.0
                    if iteration % 100 == 0:
                        torch.cuda.synchronize()
                        elapsed = iter_start.elapsed_time(iter_end)
                    
                    test_psnr = training_report(
                        tb_writer, iteration, Ll1, Lssim, loss, 
                        l1_loss, elapsed, 
                        testing_iterations, scene, render, 
                        (pipe, background), loss_dict, img_dir=img_dir)

                # Densification
                if iteration < opt.densify_until_iter:
                    # max_radii2D feeds only the size_threshold prune, which is
                    # disabled under opacity_decay; skip the masked max then.
                    if args.add_size_threshold and not args.opacity_decay:
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, 
                                    visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, 
                                    visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                        
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval and args.add_size_threshold else None
                        if args.opacity_decay:
                            size_threshold = None
                        prune_only = opt.densify_until_num_points > 0 and gaussians.get_xyz.shape[0] >= opt.densify_until_num_points
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, 
                                                    size_threshold, opt.densify_grad_t_threshold, prune_only=prune_only)
                    
                    if ((iteration % opt.opacity_reset_interval == 0 and not args.opacity_decay) or (
                        dataset.white_background and iteration == opt.densify_from_iter)) and args.reset_opacity:
                        gaussians.reset_opacity()
                        
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    if gaussians.coefficient is not None:
                        gaussians.coef_optimizer.step()
                        gaussians.coef_optimizer.zero_grad(set_to_none = True)

                    if color_affine_optimizer is not None:
                        color_affine_optimizer.step()
                        color_affine_optimizer.zero_grad(set_to_none = True)
                        
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)
                        
                # Save chkpnt. Without held-out test views there is no metric to
                # rank checkpoints by, and test_psnr stays 0.0, which would
                # rewrite chkpnt_best.pth at every test iteration.
                if (iteration in testing_iterations) and has_test_views:
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")
                        
                # Save Gaussians
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    if color_affine_delta is not None:
                        affine_state = {'index': color_affine_index,
                                        'affine_delta': color_affine_delta.detach().cpu(),
                                        'optimizer': color_affine_optimizer.state_dict(),
                                        'iteration': iteration}
                        torch.save(affine_state,
                                   os.path.join(scene.model_path, f"color_affine_{iteration}.pth"))
                        # Fixed name next to the checkpoints so --start_checkpoint
                        # can find the matching affine state without being told
                        # which iteration to look for.
                        torch.save(affine_state,
                                   os.path.join(scene.model_path, "color_affine_resume.pth"))
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Lssim, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None, img_dir=""):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Lssim.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = (
            {'name': 'train', 'cameras': scene.getValidationCameras(tag='train')},
            {'name': 'test', 'cameras': scene.getValidationCameras(tag='test')},
        )
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'], ncols=80)):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                         
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    
                    if config['name'] == 'test' and idx % 5 == 0:
                        # Convert rendered image tensor to numpy and save
                        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        # Convert ground truth image tensor to numpy and save
                        gt_img_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
                        gt_img_np = (gt_img_np * 255).astype(np.uint8)
                        
                        # Create filenames with iteration number and camera ID
                        img_filename = f"test_iter_{iteration}_cam_{viewpoint.image_name}.png"
                        gt_img_filename = f"test_iter_{iteration}_cam_{viewpoint.image_name}_gt.png"
                        
                        imageio.imwrite(os.path.join(img_dir, img_filename), img_np)
                        imageio.imwrite(os.path.join(img_dir, gt_img_filename), gt_img_np)
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
        # Only release cached blocks after a full evaluation pass; doing this
        # every 100 iterations forced a device sync + allocator flush.
        torch.cuda.empty_cache()
    return psnr_test_iter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0, 10.0])
    parser.add_argument('--initial_num_pts', type=int, default=-1)
    parser.add_argument('--num_pts', type=int, default=100000)
    parser.add_argument('--max_num_pts', type=int, default=None)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exhaust_test", action="store_true")
    
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--training_view", type=str, default="1,10,13,20",
                        help="Comma-separated list of cameras to use for validation, \
                            e.g. '0,1,2'. If not specified, all cameras will be used.")
    
    parser.add_argument("--testing_view", type=str, default="",
                        help="Comma-separated list of cameras to use for validation, \
                            e.g. '0,1,2'. If not specified, all cameras will be used.")
    
    # opacity decay
    parser.add_argument("--opacity_decay", action=BooleanOptionalAction, default=True)
    parser.add_argument('--f_max', default=0.998, type=float, help='max factor')
    parser.add_argument("--f_min", type=float, default=0.996, help='min factor')
    
    parser.add_argument("--dropout_rate", type=float, default=0.1, help='dropout_rate')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay')
    parser.add_argument("--hidden_dim", type=int, default=32, help='dim of mlp')
    parser.add_argument("--decay_from_iter", type=int, default=500, help='decay from iter')
    
    parser.add_argument('--redundant_ratio', default=0.0, type=float)
    parser.add_argument('--res', default=1, type=int)
    parser.add_argument('--downsample_method', default='random', type=str, choices=['fps', 'random'])
    
    parser.add_argument('--test_per_iter', default=1500, type=int)
    
    parser.add_argument('--time_aware', action=BooleanOptionalAction, default=True)
    parser.add_argument('--color_affine', action=BooleanOptionalAction, default=False,
                        help='learn a per-training-camera 3x4 color affine applied to the training loss')
    parser.add_argument('--color_affine_lr', type=float, default=1e-4)
    parser.add_argument('--color_affine_weight_decay', type=float, default=1e-2)
    parser.add_argument("--reset_opacity", action="store_true", default=False)
    parser.add_argument("--add_size_threshold", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])

    # Which options were actually typed on the command line. argparse only
    # fills in a default for a dest that is not already present on the
    # namespace, so parsing a second time into a namespace pre-seeded with
    # sentinels leaves every option the command line did not set as the
    # sentinel. The YAML merge below assigns every configured leaf onto args,
    # which would otherwise silently discard an explicit flag: with
    # opacity_decay or time_aware present in the config, --no-opacity_decay
    # and --no-time_aware had no effect at all.
    _unset = object()
    cli_probe = Namespace(**{a.dest: _unset for a in parser._actions if a.dest != "help"})
    parser.parse_args(sys.argv[1:], namespace=cli_probe)
    cli_explicit = {dest for dest, value in vars(cli_probe).items() if value is not _unset}

    args.save_iterations.append(args.iterations)

    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            if key in cli_explicit:
                return  # an explicit command-line value outranks the config
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:
        # args.iterations reflects the merged config; op.iterations is the
        # OptimizationParams class default (30000) regardless of config.
        args.test_iterations = args.test_iterations + [i for i in range(0, args.iterations, args.test_per_iter)]
        
    if args.initial_num_pts is not None:
        args.num_pts = args.initial_num_pts
                
    if args.max_num_pts is not None:
        args.densify_until_num_points = args.max_num_pts
        
    if args.res is not None:
        args.resolution = args.res
        
    if args.output_dir:
        args.model_path = os.path.join(args.model_path, args.output_dir)
        
    if args.weight_decay:
        args.coefficient_weight_decay = args.weight_decay
    
    if os.path.exists(args.model_path):
        raise AssertionError(f"Output folder {args.model_path} already exists")
    os.makedirs(args.model_path, exist_ok=True)
        
    if args.training_view: 
        args.training_view = [f"cam{str(int(cam)).zfill(2)}" for cam in sorted(args.training_view.split(','))]
        
    if args.testing_view: 
        args.testing_view = [f"cam{str(int(cam)).zfill(2)}" for cam in sorted(args.testing_view.split(','))]
        
    if args.opacity_decay:
        args.densify_until_iter = args.iterations
    
    params_file = os.path.join(args.model_path, "training_params.txt")
    
    with open(params_file, "w") as f:
        f.write(str(args))
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, 
             args.rot_4d, args.force_sh_3d, args.batch_size)

    # All done
    print("\nTraining complete.")
