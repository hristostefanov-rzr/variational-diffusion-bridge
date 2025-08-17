import os
from pathlib import Path
from CDDB.logger import Logger
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torchvision.utils as tu
from torch_ema import ExponentialMovingAverage
from CDDB.i2sb import Runner, ckpt_util
from CDDB.corruption import build_corruption
from CDDB.sample import build_val_dataset, build_subset_per_gpu, get_recon_imgs_fn
from CDDB.evaluation.fid_util import NumpyResizeDataset, collect_features
import torch.distributed as dist
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import wandb
RESULT_DIR = Path("/hristo/results")


def build_ckpt_option(opt, log, ckpt_path):
    ckpt_path = Path(ckpt_path)
    opt_pkl_path = ckpt_path / "options.pkl"
    assert opt_pkl_path.exists()
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)
    log.info(f"Loaded options from {opt_pkl_path=}!")

    overwrite_keys = ["use_fp16", "device", "timestep_sampling", "lambda_scheduling", "lambda_scale", "adam_lr", "adam_betas", "consistency_scale", "end_step", "start_step",
                      "inner_sampling_steps", "inner_optimization_steps", "perception_strength", "inner_consistency_steps"]
    for k in overwrite_keys:
        assert hasattr(opt, k)
        setattr(ckpt_opt, k, getattr(opt, k))

    ckpt_opt.load = ckpt_path / "latest.pt"
    return ckpt_opt


@torch.no_grad()
def compute_fid_features_from_numpy(numpy_arr, batch_size=256, mode="legacy_pytorch"):

    dataset = NumpyResizeDataset(numpy_arr, mode=mode)
    mu, sigma = collect_features(dataset, mode,
        num_workers=1, batch_size=batch_size, use_dataparallel=False, verbose=False,
    )
    return mu, sigma

def log_images_into_wandb(xs_to_save, pred_x0_to_save):
    #xs_grid = tu.make_grid(xs_to_save, nrow=2)
    #pred_x0_to_save_grid = tu.make_grid(pred_x0_to_save, nrow=2)

    #xs_grid_np = xs_grid.cpu().permute(1, 2, 0).numpy()
    #pred_x0_grid_np = pred_x0_to_save_grid.cpu().permute(1, 2, 0).numpy()
    
    #wandb.log({
    #"x_t_image_panel": wandb.Image(xs_grid_np, caption="X_t through the timesteps")
    #})
    #wandb.log({
    #"mode_approximation_image_panel": wandb.Image(pred_x0_grid_np, caption="Mode approximation through the timesteps")
    #})
    #print(xs_to_save.shape)
    #print(pred_x0_to_save.shape)
    # Loop through each sample in the batch (4 samples)
    for i in range(xs_to_save.shape[0]):
        # Extract the sequence of images for the current batch (of shape [10, 3, 256, 256])
        images = xs_to_save[i]
        # Flip the image order backwards
        images = images.flip(0)
        # Rescale to [0, 255] and convert to uint8
        images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        npy_frames = np.array(images)
        wandb.log({f"diffusion_gif_{i}": wandb.Video(npy_frames, fps=2, format="mp4")})
        # Make a grid of the final results
        grid = tu.make_grid(images, nrow=3)  # 4 images per row
        # Convert the grid to a numpy array
        grid = grid.permute(1, 2, 0).cpu().numpy()
        # Log the grid to wandb
        wandb.log({"diffuison_grid": wandb.Image(grid)})
    print(f"Logged {xs_to_save.shape} diffusion samples to wandb.")
    wandb.log({"single_image": wandb.Image(xs_to_save[2, 0, ...].cpu().permute(1, 2, 0).numpy())})
"""
def create_loss_plots(loss_history):

    if loss_history is None:
        return
    loss_history = np.array(loss_history)

    # Create figure and left axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot the loss-related curves on ax1
    ax1.plot(loss_history[:, 0], color='tab:blue', label='Total Loss')
    ax1.plot(loss_history[:, 1], color='tab:orange', label='Regularization Term')
    ax1.plot(loss_history[:, 2], color='tab:green', label='Consistency Term')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # If PSNR is present, plot it on the right axis
    if loss_history.shape[1] > 3:
        ax2 = ax1.twinx()
        ax2.plot(loss_history[:, 3], color='tab:red', linestyle='--', label='PSNR')
        ax2.set_ylabel('PSNR')

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    ax1.set_title('Loss History')

    # Save the plot
    loss_plot_path = "loss_history_plot.png"
    fig.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close(fig)

    # Log the plot to wandb
    wandb.log({"loss_history": wandb.Image(loss_plot_path)})
"""
def create_loss_plots(loss_history):
    if loss_history is None:
        return
    loss_history = np.array(loss_history)

    # Create figure and left axis (ax1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 1) Plot the loss‐related curves on ax1
    ax1.plot(loss_history[:, 0], label='Total Loss')
    ax1.plot(loss_history[:, 1], label='Regularization Term')
    ax1.plot(loss_history[:, 2], label='Consistency Term')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    n_cols = loss_history.shape[1]

    # Only create right‐hand axes if we have at least 4 columns (PSNR)
    if n_cols >= 4:
        # 2) First right axis for PSNR
        ax2 = ax1.twinx()
        ax2.plot(
            loss_history[:, 3],
            linestyle='--',
            label='PSNR'
        )
        ax2.set_ylabel('PSNR')

        # 3) Second right axis for LPIPS if available (column 4)
        if n_cols >= 5:
            ax3 = ax1.twinx()
            # Move this second right axis slightly to the right:
            ax3.spines["right"].set_position(("axes", 1.15))
            # Make sure the spine is visible
            ax3.spines["right"].set_visible(True)

            ax3.plot(
                loss_history[:, 4],
                color='tab:purple',
                linestyle='-.',
                label='LPIPS'
            )
            ax3.set_ylabel('LPIPS')

            # Combine legends from all three axes:
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            lines_3, labels_3 = ax3.get_legend_handles_labels()
            ax1.legend(
                lines_1 + lines_2 + lines_3,
                labels_1 + labels_2 + labels_3,
                loc='upper right'
            )
        else:
            # Only PSNR (n_cols == 4)
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    else:
        # Fewer than 4 columns: only loss curves
        ax1.legend(loc='upper right')

    ax1.set_title('Loss & Metrics History')

    # Save and log
    loss_plot_path = "loss_history_plot.png"
    fig.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close(fig)

    wandb.log({"loss_history": wandb.Image(loss_plot_path)})

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out, opt):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
        x1_pinv = corrupt_img.to(opt.device)
        x1_forw = corrupt_img.to(opt.device)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
        x1_pinv = None
        x1_forw = None
    elif "blur" in corrupt_type:
        clean_img, y = out
        mask = None
        corrupt_img_y, corrupt_img_pinv = corrupt_method(clean_img.to(opt.device))
        corrupt_img = corrupt_img_y
        x1 = corrupt_img_y.to(opt.device)
        x1_pinv = corrupt_img_pinv.to(opt.device)
        x1_forw = x1
    else: # sr, jpeg case
        clean_img, y = out
        mask = None
        corrupt_img_pinv, corrupt_img_y = corrupt_method(clean_img.to(opt.device))
        corrupt_img = corrupt_img_pinv
        x1 = corrupt_img.to(opt.device)
        x1_pinv = x1
        x1_forw = corrupt_img_y.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None

    return corrupt_img, x1, mask, cond, y, clean_img, x1_pinv, x1_forw

def sample_images(opt, run):
    log = Logger(0, ".log")
    #opt.dataset_dir = "/home/hristo/Code/thesis/CDDB/imagenet_data/imagenet-mini/val"
    # get (default) ckpt option
    ckpt_opt = build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    #ckpt_opt.cond_x1 = True
    ckpt_opt.ckpt_dir = opt.ckpt_dir
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log, corrupt_type)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )
    #
    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn, sample_dir = get_recon_imgs_fn(opt, nfe)
    
    for t in ["input", "recon", "label", "extra"]:
        (sample_dir / t).mkdir(exist_ok=True, parents=True)
    log.info(f"Recon images will be saved to {sample_dir}!")

    recon_imgs = []
    clean_imgs = []
    ys = []
    num = 0
    
    log_count = 10
    loss_history = None
    for loader_itr, out in enumerate(val_loader):
        corrupt_img, x1, mask, cond, y, clean_img, x1_pinv, x1_forw = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out, opt)

        if opt.use_cddb_deep:
            sv_idx = str(loader_itr).zfill(3)
            #results_dir = sample_dir / f"{sv_idx}"
            results_dir = None
            xs, pred_x0s = runner.cddb_deep_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, corrupt_type=corrupt_type, 
                corrupt_method=corrupt_method, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, step_size=opt.step_size,
                results_dir=results_dir,
            )
        elif opt.use_cddb:
            sv_idx = str(loader_itr).zfill(3)
            #results_dir = sample_dir / f"{sv_idx}"
            results_dir = None
            xs, pred_x0s = runner.cddb_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, corrupt_type=corrupt_type, 
                corrupt_method=corrupt_method, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, step_size=opt.step_size,
                results_dir=results_dir,
            )
        elif opt.use_variational:
            xs, pred_x0s = runner.variational_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, corrupt_method=corrupt_method,
            )
        elif opt.use_augmented_variational:
            xs, pred_x0s, loss_history = runner.augmented_variational_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, corrupt_method=corrupt_method,
                targets = clean_img.to(opt.device).detach(),
            )
        else:
            xs, pred_x0s = runner.ddpm_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count
            )
        
        if loader_itr <= 2:
            create_loss_plots(loss_history)
            log_images_into_wandb(xs, pred_x0s)

        recon_img = xs[:, 0, ...].to(opt.device)
        pred_x0s = pred_x0s[:, 0, ...].to(opt.device)
   
        assert recon_img.shape == corrupt_img.shape
        if loader_itr == 0 and opt.global_rank == 0: # debug
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png")
            tu.save_image((recon_img+1)/2, ".debug/recon.png")
            log.info("Saved debug images!")

        # [-1,1]
        #print(recon_img.shape)
        #gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(recon_img)
        clean_imgs.append(clean_img)
        y = y.to(opt.device)
        #gathered_y = collect_all_subset(y, log)
        ys.append(y)

        #num += len(gathered_recon_img)
        num += len(recon_img)
        log.info(f"Collected {num} recon images!")
        
        # save input, recon, label also as image files
        for idx in range(len(corrupt_img)):
            sv_idx = str(opt.batch_size * loader_itr + idx).zfill(3)
         
            input_idx = (corrupt_img[idx:idx+1, ...] + 1) / 2
            recon_idx = (recon_img[idx:idx+1, ...] + 1) / 2
            label_idx = (clean_img[idx:idx+1, ...] + 1) / 2
            tu.save_image(input_idx, str(sample_dir / f"input" / f"{sv_idx}.png"))
            tu.save_image(recon_idx, str(sample_dir / f"recon" / f"{sv_idx}.png"))
            tu.save_image(label_idx, str(sample_dir / f"label" / f"{sv_idx}.png"))  
        
        #dist.barrier()

    del runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]
    label_arr = torch.cat(ys, axis=0)[:n_samples]
    clean_arr = torch.cat(clean_imgs, axis=0)[:n_samples]
    #if opt.global_rank == 0:
    #    torch.save({"arr": arr, "label_arr": label_arr}, recon_imgs_fn)
    #    log.info(f"Save at {recon_imgs_fn}")

    #dist.barrier()
    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}, ys={label_arr.shape}")
    return arr, label_arr, clean_arr