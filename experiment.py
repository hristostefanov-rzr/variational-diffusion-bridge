

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys
# Add the CDDB folder to sys.path
root_path = os.path.dirname(os.path.abspath(__file__))
cddb_path = os.path.join(root_path, "CDDB")
sys.path.insert(0, cddb_path) 

import torch
import torchvision.utils as tu
from adapter import sample_images, compute_fid_features_from_numpy
from CDDB.i2sb import Runner, download_ckpt
from easydict import EasyDict as edict
from pathlib import Path
import numpy as np
import math
from CDDB.evaluation.fid_util import compute_fid_ref_stat, compute_fid_from_numpy
from CDDB.logger import Logger
from CDDB.i2sb import download_ckpt
from CDDB.sample import get_recon_imgs_fn
import torchvision.utils as tu
from cleanfid.fid import frechet_distance
from fid_score import get_fid_value_of_folders

def calculate_psnr_score(gt_batch, gen_batch, max_val=255.0):
    """
    Compute the average PSNR per image in the batch.

    Parameters:
        gt_batch (torch.Tensor): Ground truth images of shape (N, C, H, W).
        gen_batch (torch.Tensor): Generated images of shape (N, C, H, W).
        max_val (float): Maximum possible pixel value.
    
    Returns:
        float: Average PSNR value (in dB) over the batch.
    """
    gt_batch = gt_batch.float()
    gen_batch = gen_batch.float()
    
    # Compute MSE for each image
    mse_per_image = torch.mean((gt_batch - gen_batch) ** 2, dim=[1, 2, 3])
    
    # Compute PSNR for each image
    psnr_per_image = 10 * torch.log10((max_val ** 2) / mse_per_image)
    
    return psnr_per_image.mean().item()

def compute_fid_from_arrays(arr1, arr2, batch_size=256, mode="legacy_pytorch"):
    mu1, sigma1 = compute_fid_features_from_numpy(arr1, batch_size=batch_size, mode=mode)
    mu2, sigma2 = compute_fid_features_from_numpy(arr2, batch_size=batch_size, mode=mode)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


opt = edict({'distributed': False, 
        'device': 'cuda', 
        'ckpt': 'sr4x-bicubic',
        'seed': 42, 
        'n_gpu_per_node': 1, 
        'master_address': 'localhost', 
        'node_rank': 0, 
        'num_proc_node': 1, 
        'image_size': 256, 
        #'dataset_dir': Path('/home/hristo/Code/thesis/small_image_net_data'), 
        'dataset_dir':Path('/hristo/imagenet/CLS-LOC'),
        'lmdb_dir': Path('/hristo/data/lmdb'),
        'ckpt_dir': Path('/hristo/data'),
        #'datapoints_folder': 'CDDB/dataset/',
        'partition': None, 
        'add_noise': False, 
        'batch_size': 4, 
        'ckpt': 'sr4x-bicubic', 
        'nfe': 10, 
        'clip_denoise': True, 
        'use_fp16': True, 
        'eta': 1.0, 
        'use_cddb_deep': False, 
        'use_cddb': False,
        'use_variational': False,
        'use_augmented_variational': True,
        'step_size': 1.0, 
        'prob_mask': 0.35,
        'global_size': 1,
        'global_rank': 0,  
        'lambda_scale': 0.2,
        "consistency_scale": 1.0,
        "lambda_scheduling": "real_sqrt",
        "timestep_sampling": "uniform_random",
        "adam_lr": 0.02,
        "adam_betas": (0.9, 0.99),
        "start_step" : 900,
        "end_step": 1000,
        "inner_sampling_steps": 10,
        "inner_optimization_steps": 10,
        "inner_consistency_steps": 0,
        "perception_strength": 0.0,
        })
models_specified = opt.use_cddb_deep + opt.use_cddb + opt.use_variational
assert models_specified <= 1, "Please use only one of the models: cddb_deep, cddb or variational"
model_used = "cddb_deep" if opt.use_cddb_deep else "ddb"
model_used = "cddb" if opt.use_cddb else model_used
model_used = "variational" if opt.use_variational else model_used
model_used = "augmented_variational" if opt.use_augmented_variational else model_used

download_ckpt(opt.ckpt_dir)
arr, label_arr, clean_arr = sample_images(opt)
log = Logger(rank = 1)
arr = ((arr + 1)/2) * 255
clean_arr = ((clean_arr + 1)/2)  * 255

arr = arr.to(opt.device)
clean_arr = clean_arr.to(opt.device)
psnr_score = calculate_psnr_score(arr, clean_arr)
print("PSNR score: ", psnr_score)

_, sample_dir = get_recon_imgs_fn(opt, opt.nfe)
fake_images_path = os.path.join(sample_dir, "recon")
true_images_path = os.path.join(sample_dir, "label")
fid_score = get_fid_value_of_folders(true_images_path, fake_images_path)
print("FID score: ", fid_score)




