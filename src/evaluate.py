import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import DocumentDataset
from src.model import EnhancementUNet
from src.pipeline import GPUDegradationPipeline
from src.train import SSIM # Reuse SSIM metric from train.py

def calculate_psnr(pred, target):
    """Calculates Peak Signal-to-Noise Ratio for [0, 1] normalized images."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    # MAX_I is 1.0 since tensors are in [0, 1]
    return -10 * torch.log10(mse).item()

def save_comparison_plot(degraded, enhanced, clean, output_path):
    """Saves a side-by-side comparison of the images."""
    # Convert from CHW tensor to HWC numpy array for matplotlib
    deg_img = degraded.cpu().squeeze(0).permute(1, 2, 0).numpy()
    enh_img = enhanced.cpu().squeeze(0).permute(1, 2, 0).numpy()
    cln_img = clean.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Clip to [0, 1] just in case
    deg_img = np.clip(deg_img, 0, 1)
    enh_img = np.clip(enh_img, 0, 1)
    cln_img = np.clip(cln_img, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(deg_img)
    axes[0].set_title("1. Degraded (Input)")
    axes[0].axis('off')
    
    axes[1].imshow(enh_img)
    axes[1].set_title("2. Enhanced (Output)")
    axes[1].axis('off')
    
    axes[2].imshow(cln_img)
    axes[2].set_title("3. Clean (Ground Truth)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1) # Use 1 for evaluation plots
    parser.add_argument('--img_size', type=int, default=1024)
    args = parser.parse_args()

    # Directories
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_enhancement.pth")
    eval_dir = os.path.join(base_dir, "results", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Model weights not found at {ckpt_path}.")
        print("Please train the model first.")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # 1. Dataset Preparation (Recreating the exact 80/10/10 split)
    clean_paths = sorted(glob.glob(os.path.join(base_dir, "data", "clean_scans", "*.*")))
    bg_paths = sorted(glob.glob(os.path.join(base_dir, "data", "backgrounds", "*.*")))
    
    # Must use same seed as train.py to get the exact same splits
    random.seed(42)
    random.shuffle(clean_paths)
    
    train_split = int(0.8 * len(clean_paths))
    val_split = int(0.9 * len(clean_paths))
    
    # Extract the final 10% for pure testing
    test_scans = clean_paths[val_split:]
    print(f"Found {len(test_scans)} images in the test set.")
    
    if len(test_scans) == 0:
        print("Test set is empty! You need more images in data/clean_scans/.")
        return
        
    test_dataset = DocumentDataset(test_scans, bg_paths, split='test', target_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Setup Model, Pipeline, and Metrics
    model = EnhancementUNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    gpu_pipeline = GPUDegradationPipeline(target_size=args.img_size).to(device)
    ssim_metric = SSIM(size_average=True).to(device)
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(test_loader)
    
    print("Starting Evaluation...")
    
    with torch.no_grad():
        for i, (clean_batch, bg_batch) in enumerate(test_loader):
            clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
            
            # Generate the degraded inputs using the deterministic test split
            inputs, targets = gpu_pipeline(clean_batch, bg_batch, task='enhancement')
            
            # Model Inference
            outputs = model(inputs)
            
            # Compute Metrics
            batch_psnr = calculate_psnr(outputs, targets)
            batch_ssim = ssim_metric(outputs, targets).item()
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            
            # Save visual plots for the first 5 images
            if i < 5:
                out_file = os.path.join(eval_dir, f"test_result_{i+1}.png")
                # Take the first image in the batch for plotting
                save_comparison_plot(inputs[0:1], outputs[0:1], targets[0:1], out_file)
                print(f"Saved visual comparison: {out_file}")

    # Calculate and Print Final Metrics
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print("\n" + "="*40)
    print("        EVALUATION RESULTS")
    print("="*40)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Saved {min(5, num_batches)} comparison plots to: {eval_dir}")
    print("="*40)

if __name__ == "__main__":
    main()
