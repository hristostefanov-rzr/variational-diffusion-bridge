from pytorch_fid import fid_score
import torch

def get_fid_value_of_folders(true_images_path, fake_images_path):
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[true_images_path, fake_images_path],
        batch_size=8,  # Reduce if OOM errors occur
        device = "cpu",
        dims=2048
    )  # Inception-v3 feature dimension
    return fid_value
if __name__ == "__main__":
    true_images_path = "/hristo/diffusion-tradeoffs/results/sr4x-bicubic/samples_cddb_deep_nfe50_step1.0/label"
    fake_images_path = "/hristo/diffusion-tradeoffs/results/sr4x-bicubic/samples_cddb_deep_nfe50_step1.0/recon"
    # Compute FID between PNG folders
    fid_value = get_fid_value_of_folders(true_images_path, fake_images_path)
    print(f"FID: {fid_value:.2f}")