# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch
import json 
import uuid
import os
import lpips
from .util import unsqueeze_xdim

from ipdb import set_trace as debug
from i2sb.util import clear_color, clear
import matplotlib.pyplot as plt


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)
        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)
    
        self.perceptual_loss = lpips.LPIPS(net='vgg')
        self.perceptual_loss = self.perceptual_loss.to(self.device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)
        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def q_sample_plus_noise(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 and return the noise generated """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape
        # TODO: check if these are calculated correctly
        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)
        noise = torch.randn_like(x0)
        xt = mu_x0 * x0 + mu_x1 * x1
        xt = xt + std_sb * noise
        return xt.detach(), noise.detach()


    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False, verbose=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        if verbose:
            return xt_prev, mu_x0
        else:
            return xt_prev
    
    def p_posterior_ddim(self, nprev, n, x_n, x0, pred_eps, eta=1.0):
        """ Posterior sampling for ddim. OT-ODE disabled. """

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        
        c1 = var.sqrt() * eta
        c2 = var.sqrt() * np.sqrt(1 - eta**2)
        
        xt_prev = xt_prev + c1 * torch.randn_like(xt_prev) + c2 * pred_eps

        return xt_prev
    
    def ddpm_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []

        #log_steps = steps
        assert steps[0] == log_steps[0] == 0

        steps = list(steps)[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        cnt = 0
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            cpu_pred = pred_x0.detach().cpu()
            cpu_xt   = xt.detach().cpu()

            del pred_x0
            cnt += 1
            if prev_step in log_steps:
                pred_x0s.append(cpu_pred)
                xs.append(cpu_xt)

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    def create_lambdas(self, steps, lambda_schedule):
        if lambda_schedule == "const":
            lambdas = np.ones((1000, ))
        elif lambda_schedule == "sqrt":
            lambdas = self.std_sb/(1-self.std_sb**2).sqrt()
            print(lambdas)
        elif lambda_schedule == "sqrt_m1":
            lambdas = (1 - self.mu_x0**2).sqrt()/self.mu_x0
        elif lambda_schedule == "linear_m1":
            lambdas = (1 - self.mu_x0**2)/self.mu_x0
        elif lambda_schedule == "clip_m1":
            lambdas = (1 - self.mu_x0**2)/self.mu_x0
            lambdas = torch.maximum(lambdas, torch.ones_like(lambdas))
        elif lambda_schedule == "linear":
            lambdas = self.std_sb/(1-self.std_sb)
        elif lambda_schedule == "reverse_sqrt_m1":
            lambdas = self.mu_x0/(1-self.mu_x0**2).sqrt()
        elif lambda_schedule == "double_sqrt_m1":
            lambdas = (1 - self.mu_x0**2).sqrt()/self.mu_x0
            lambdas = lambdas.sqrt()
        elif lambda_schedule == "increasing":
            lambdas = np.linspace(0, 2, 1000)
        elif lambda_schedule == "symmetric":
            lambdas = (1 - self.mu_x0**2).sqrt()/self.mu_x0
            lambdas_flip = (1 - self.mu_x0**2).sqrt()/self.mu_x0
            lambdas_flip = lambdas_flip.flip(dims=(0,))
            lambdas = torch.cat((lambdas_flip[:500], lambdas[500:]), dim=0)
        elif lambda_schedule == "real_sqrt":
            lambdas = (1/self.std_fwd)
        elif lambda_schedule == "real_linear":
            lambdas = (1/(self.std_fwd**2))
        elif lambda_schedule == "real_double_sqrt":
            lambdas = (1/(self.std_fwd)).sqrt()
            lambdas = lambdas.sqrt()
        else:
            raise ValueError(f"Unknown lambda schedule: {lambda_schedule}")
        
        return lambdas
    
    def variational_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, mask=None, log_steps=None, verbose=True, corrupt_method=None, lambda_schedule = "const", lambda_scale=0.25, adam_lr = 0.1, adam_betas=(0.9, 0.99), consistency_scale=1.0,  clip_denoise=True):
        xs = []
        pred_x0s = []
        # Ignore log steps since that is meant for other iterative methods
        # Instead log the mean_approximation at maximum len(log_steps) even intervals
        steps = list(steps)[::-1]
        log_steps = torch.arange(0, len(steps), 1).long() if log_steps is None else log_steps
        print(log_steps)
        lambdas = lambda_scale * self.create_lambdas(steps, lambda_schedule)
        count = 0
        mean_approximation = x1.clone().detach().to(self.device).requires_grad_()
        viz_steps = tqdm(steps, desc='Variational sampling', total=len(steps)) if verbose else steps
        with torch.enable_grad():
            optimizer = torch.optim.Adam([mean_approximation], lr=adam_lr, betas=adam_betas, weight_decay=0.0)
            for optimization_step in viz_steps:
                # Sample random time step and obtain a sample from our variational distribution
                xt = self.q_sample(optimization_step, mean_approximation, x1)
                pred_x0 = pred_x0_fn(xt, optimization_step)
                 
                #_, measurement_approximation = corrupt_method(pred_x0)
                _ , measurement_approximation = corrupt_method(mean_approximation)
                measurement_error = (((x1_forw.detach() - measurement_approximation)**2).mean())/2
                variational_error = torch.mul((mean_approximation - pred_x0).detach(), mean_approximation).mean()
                loss =  consistency_scale * measurement_error + lambdas[optimization_step] * (variational_error) #+ 0.000 * perceptional_error)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                if count in log_steps:
                    # Note that this here is not exactly correct and we save our best prediction in xs and not pred_x0s
                    # as the xs is what is saved as recon afterwards
                    # so the pred_x0s is actually very wrong
                    pred_x0s.append(xt.detach().cpu())
                    distribution_sample = self.q_sample(0, mean_approximation, x1)
                    xs.append(distribution_sample.detach().cpu())
                    #xs.append(mean_approximation.detach().cpu())
                count += 1
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    def get_uniform_timesteps(self, start_step, end_step, n_samples):
        """ Get n_samples uniformly spaced from [0, n_timesteps) """
        n_timesteps = len(self.betas)
        assert start_step < end_step, f"{start_step=}, {end_step=}"
        assert end_step <= n_timesteps, f"{end_step=}, {n_timesteps=}"
        steps = torch.linspace(start_step, end_step-1, n_samples).long()
        return steps


    def get_all_timestep_losses(self, steps, mean_approximation, x1, x1_forw, pred_x0_fn, mask=None, corrupt_method=None, lambda_schedule = "const", lambda_scale=0.25, adam_lr = 0.1, adam_betas=(0.9, 0.99), consistency_scale=1.0, targets=None):
        loss_over_all_timesteps = []
        psnr_over_all_timesteps = []
        lambdas = lambda_scale * self.create_lambdas(steps, lambda_schedule)
        # Sample random time step and obtain a sample from our variational distribution
        for current_step in range(999, 0, -100):
            xt = self.q_sample(current_step, mean_approximation, x1)
            pred_x0 = pred_x0_fn(xt, current_step)
            #pred_x0 = pred_x0[:, 0, :].cuda()
            _, measurement_approximation = corrupt_method(mean_approximation)
            measurement_error = (((x1_forw.detach() - measurement_approximation)**2).mean())/2
            variational_error = torch.mul((mean_approximation - pred_x0).detach(), mean_approximation).mean()
            if targets is not None:
                psnr = -10 * torch.log10(((mean_approximation.detach().clamp(-1, 1) - targets)**2).mean())
            loss =  consistency_scale * measurement_error + lambdas[current_step] * (variational_error)
            loss_over_all_timesteps.append(loss.item())
            psnr_over_all_timesteps.append(psnr.item() if targets is not None else 0.0)
        return loss_over_all_timesteps, psnr_over_all_timesteps

    def augmented_variational_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, mask=None, log_steps=None, verbose=True, corrupt_method=None, lambda_schedule = "const", lambda_scale=0.25, adam_lr = 0.1, adam_betas=(0.9, 0.99), consistency_scale=1.0, clip_denoise=True, inner_sampling_steps = 4, inner_optimization_steps = 4, perception_strength = 0.00001, inner_consistency_steps = 1, targets = None):
        xs = []
        pred_x0s = []
        # Ignore log steps since that is meant for other iterative methods
        # Instead log the mean_approximation at maximum len(log_steps) even intervals
        # log loss history
        loss_history = []
        loss_over_all_timesteps_history = []
        psnr_over_all_timesteps_history = []
        steps_to_json = lambda x: [int(s) for s in x]
        steps = list(steps)[::-1]
        json.dump(steps_to_json(steps), open(f"steps.json", "w"))
        log_steps = torch.arange(0, len(steps), 1).long() if log_steps is None else log_steps
        lambdas = lambda_scale * self.create_lambdas(steps, lambda_schedule)
        #print(lambdas)
        count = 0
        mean_approximation = x1.clone().detach().to(self.device).requires_grad_()
        viz_steps = tqdm(steps, desc='Variational sampling', total=len(steps)) if verbose else steps
        with torch.enable_grad():
            optimizer = torch.optim.Adam([mean_approximation], lr=adam_lr, betas=adam_betas, weight_decay=0.0)
            internal_optimizer = torch.optim.Adam([mean_approximation], lr=adam_lr, betas=adam_betas, weight_decay=0.0)
            for optimization_step in viz_steps:
                # Sample random time step and obtain a sample from our variational distribution
                internal_steps = self.get_uniform_timesteps(0, optimization_step, inner_sampling_steps + 1)
                xt = self.q_sample(optimization_step, mean_approximation, x1)
                with torch.no_grad():
                    pred_x0 = self.ddpm_sampling(internal_steps, pred_x0_fn, xt, x1_pinv, x1_forw, mask=mask, ot_ode=False, log_steps=internal_steps, verbose=False)[0]
                    #pred_x0 = self.dds_sampling(internal_steps, pred_x0_fn, xt, x1_pinv, x1_forw, mask=mask, ot_ode=False, log_steps=internal_steps, verbose=False)[0]
                pred_x0 = pred_x0[:, 0, :].cuda()
                #loss_over_all_timesteps, psnr_over_all_timesteps = self.get_all_timestep_losses(steps, mean_approximation, x1, x1_forw, pred_x0_fn, mask=mask, corrupt_method=corrupt_method, lambda_schedule=lambda_schedule, lambda_scale=lambda_scale, adam_lr=adam_lr, adam_betas=adam_betas, consistency_scale=consistency_scale, targets=targets)
                #loss_over_all_timesteps_history.append(loss_over_all_timesteps)
                #psnr_over_all_timesteps_history.append(psnr_over_all_timesteps)
                for _ in range(inner_optimization_steps): 
                    _ , measurement_approximation = corrupt_method(mean_approximation)
                    #_, measurement_approximation = corrupt_method(pred_x0)
                    measurement_error = (((x1_forw.detach() - measurement_approximation)**2).mean())/2
                    variational_error = torch.mul((mean_approximation - pred_x0).detach(), mean_approximation).mean()
                    #perception_error = self.perceptual_loss(pred_x0, mean_approximation).mean()
                    #loss =  consistency_scale * measurement_error + lambdas[optimization_step] * (variational_error + perception_strength * perception_error)
                    assert perception_strength == 0, f"{perception_strength=}"
                    loss = consistency_scale * measurement_error + lambdas[optimization_step] * (variational_error)
                    #loss = loss + perception_strength * perception_error
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    assert inner_consistency_steps == 0, f"{inner_consistency_steps=}"
                    if targets is not None:
                        mean_approximation_detached = mean_approximation.detach().clamp(-1, 1)
                        psnr = -10 * torch.log10(((mean_approximation_detached - targets)**2).mean())
                        lpips_loss = self.perceptual_loss(mean_approximation_detached, targets).mean()
                        loss_history.append([loss.item(), variational_error.item(), measurement_error.item(), psnr.item(), lpips_loss.item()]) #, perception_error.item()])
                    else:
                        loss_history.append([loss.item(), variational_error.item(), measurement_error.item()]) #, perception_error.item()])
                    #for _ in range(inner_consistency_steps):
                    #    _, measurement_approximation = corrupt_method(mean_approximation)
                    #    measurement_error = (((x1_forw.detach() - measurement_approximation)**2).mean())/2
                    #    measurement_error.backward() 
                    #    internal_optimizer.step()
                    #    internal_optimizer.zero_grad()
                if count in log_steps:
                    pred_x0s.append(xt.detach().cpu())
                    distribution_sample = self.q_sample(0, mean_approximation, x1)
                    xs.append(distribution_sample.detach().cpu())
                count += 1
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        #json.dump(loss_over_all_timesteps_history, open(f"loss_over_all_timesteps.json", "w"))
        #if targets is not None:
        #    json.dump(psnr_over_all_timesteps_history, open(f"psnr_over_all_timesteps.json", "w"))
        
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s), loss_history
    
    def ddnm_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, mask=None, 
                      corrupt_type=None, corrupt_method=None, step_size=1.0,
                      ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            pred_x0 = pred_x0_fn(xt, step)
            
            # projection data consistency - useless for inpainting. Might as well drop
            if mask is not None:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            else:
                _, corrupt_x0_pinv = corrupt_method(pred_x0)
            pred_x0 = pred_x0 - corrupt_x0_pinv + x1_pinv
            
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def ddpm_dps_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, 
                          mask=None, corrupt_type=None, corrupt_method=None, step_size=1.0,
                          ot_ode=False, log_steps=None, verbose=True, results_dir=None):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]
        
        if results_dir is not None:
            for t in ["x0_before", "x0_after", "x0_diff", "x0_diff_mean"]:
                (results_dir / t).mkdir(exist_ok=True, parents=True)

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            # turn on gradient
            xt.requires_grad_()
            pred_x0 = pred_x0_fn(xt, step)
            
            # DPS
            # for inpainting, corrupt_method returns a tuple
            if mask is not None:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                _, corrupt_x0_forw = corrupt_method(pred_x0)
            else:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
                
                
            residual = corrupt_x0_forw - x1_forw
            residual_norm = torch.linalg.norm(residual) ** 2
            # residual_norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xt)[0]
            
            xt, mu_x0 = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, verbose=True)
            if results_dir is not None:
                plt.imsave(str(results_dir / "x0_before" / f"{step}.png"), clear_color(pred_x0))
            #xt = xt - mu_x0 * step_size * norm_grad
            pair_steps.set_postfix({"mu_x0": mu_x0.item()})
            xt = xt - step_size * norm_grad
            pred_x0_correct = pred_x0 - step_size * norm_grad
            if results_dir is not None:
                plt.imsave(str(results_dir / "x0_after" / f"{step}.png"), clear_color(pred_x0_correct), cmap='gray')
                plt.imsave(str(results_dir / "x0_diff" / f"{step}.png"), clear_color(norm_grad), cmap='gray')
                plt.imsave(str(results_dir / "x0_diff_mean" / f"{step}.png"), clear(norm_grad.mean(dim=1).unsqueeze(dim=1)))
            # xt = xt - step_size * norm_grad
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def dds_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw,
                     mask=None, corrupt_type=None, corrupt_method=None, step_size=1.0,
                     ot_ode=False, log_steps=None, verbose=True, results_dir=None):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]
        
        if results_dir is not None:
            for t in ["x0_before", "x0_after"]:
                (results_dir / t).mkdir(exist_ok=True, parents=True)
 
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDS sampling', total=len(steps)-1) if verbose else pair_steps
        cnt = 0
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            # turn on gradient
            xt.requires_grad_()
            pred_x0 = pred_x0_fn(xt, step)
            
            # DPS
            # for inpainting, corrupt_method returns a tuple
            if mask is not None:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                _, corrupt_x0_forw = corrupt_method(pred_x0)
            else:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
            # residual = corrupt_x0 - x1_meas
            residual = corrupt_x0_forw - x1_forw
            residual_norm = torch.linalg.norm(residual) ** 2
            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=pred_x0)[0]
            
            xt, mu_x0 = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, verbose=True)
            #xt = xt - mu_x0 * step_size * norm_grad
            xt = xt - step_size * norm_grad
            # take multiple gradient steps
            if results_dir is not None:
                if cnt == 5:
                    plt.imsave(str(results_dir / "x0_before" / f"{step}.png"), clear_color(pred_x0))
                    for k in range(5):
                        pred_x0 = pred_x0 - step_size * norm_grad
                        _, corrupt_x0_forw = corrupt_method(pred_x0)
                        residual = corrupt_x0_forw - x1_forw
                        residual_norm = torch.linalg.norm(residual) ** 2
                        norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=pred_x0)[0]
                        plt.imsave(str(results_dir / "x0_after" / f"{step}_{k}.png"), clear_color(pred_x0))
            
            xt.detach_()
            pred_x0.detach_()
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
            cnt += 1

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def pigdm_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw,
                       mask=None, corrupt_type=None, corrupt_method=None, step_size=1.0,
                       ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            # turn on gradient
            xt.requires_grad_()
            pred_x0 = pred_x0_fn(xt, step)
            
            # for inpainting, corrupt_method returns a tuple
            if mask is not None:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            else:
                _, corrupt_x0_pinv = corrupt_method(pred_x0)
            
            # pigdm
            mat = x1_pinv - corrupt_x0_pinv
            mat_rs = mat.detach().reshape(mat.shape[0], -1)
            mat_x = (mat_rs * pred_x0.reshape(mat.shape[0], -1)).sum()
            guidance = torch.autograd.grad(outputs=mat_x, inputs=pred_x0)[0]
            
            xt, mu_x0 = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, verbose=True)
            xt = xt + mu_x0 * step_size * guidance
            
            # free memory
            xt.detach_()
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def ddim_sampling(self, steps, pred_x0_eps_fn, x1, eta=1.0, eps=1.0, mask=None, log_steps=None, verbose=True):
        """
        (pred_x0_fn) for ddim_sampling returns both pred_x0, model_output
        >> pred_x0, pred_eps = pred_x0_eps_fn(xt, step)
        """
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0, pred_eps = pred_x0_eps_fn(xt, step)
            xt = self.p_posterior_ddim(prev_step, step, xt, pred_x0, pred_eps, eta=eta)

            if mask is not None:
                xt_true = x1
                _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
