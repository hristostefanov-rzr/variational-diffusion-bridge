

import os
import sys
# Add the CDDB folder to sys.path
root_path = os.path.dirname(os.path.abspath(__file__))
cddb_path = os.path.join(root_path, "CDDB")
sys.path.insert(0, cddb_path) 

import torchvision.utils as tu
from adapter import sample_images, compute_fid_features_from_numpy
from easydict import EasyDict as edict
from pathlib import Path
import numpy as np
from CDDB.evaluation.fid_util import compute_fid_ref_stat, compute_fid_from_numpy
from CDDB.logger import Logger
import torchvision.utils as tu
from cleanfid.fid import frechet_distance
import torch
from torchvision import datasets, transforms

# Download and load CIFAR-10
transform = transforms.Compose([
    transforms.Resize(299),  # Inception-v3 requires 299x299
    transforms.ToTensor(),
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Convert to numpy arrays (N, H, W, C)
train_images = trainset.data  # (50000, 32, 32, 3)
test_images = testset.data    # (10000, 32, 32, 3)

from torchvision.transforms.functional import resize


def compute_fid_from_arrays(arr1, arr2, batch_size=256, mode="legacy_pytorch"):
    mu1, sigma1 = compute_fid_features_from_numpy(arr1, batch_size=batch_size, mode=mode)
    mu2, sigma2 = compute_fid_features_from_numpy(arr2, batch_size=batch_size, mode=mode)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def preprocess(images):
    # Resize to 299x299 and convert to float32
    resized = torch.stack([resize(torch.tensor(img).permute(2,0,1), (299, 299)) for img in images])
    # Normalize to [-1, 1]
    #return (resized / 127.5 - 1).numpy().transpose(0, 2, 3, 1)  # (N, 299, 299, 3)
    return resized.numpy().transpose(0, 2, 3, 1)  # (N, 299, 299, 3)
train_images = preprocess(train_images[:10000])  # Use subset for faster testing
test_images = preprocess(test_images)
print(train_images[0])
print(train_images[0].mean(0).mean(0))
fid_identical = compute_fid_from_arrays(train_images, train_images)
print(f"FID identical: {fid_identical}")
fid_score = compute_fid_from_arrays(train_images, test_images)
print(f"FID score: {fid_score}")
noise = np.random.randint(0, 255, (10000, 299, 299, 3), dtype=np.uint8)
noise = preprocess(noise)
fid_noise = compute_fid_from_arrays(train_images, noise)
print(f"FID noise: {fid_noise}")


