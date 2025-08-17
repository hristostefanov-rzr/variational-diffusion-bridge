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

def sample_images(opt):
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