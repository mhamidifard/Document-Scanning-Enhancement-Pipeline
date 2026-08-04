import os
import sys
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Dynamically get the base directory of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Ensure the parent directory is in sys.path
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import DocumentDataset
from src.pipeline import GPUDegradationPipeline

def main():
    artifact_dir = os.path.join(base_dir, "results", "visualize-degradation")
    os.makedirs(artifact_dir, exist_ok=True)

    clean_paths = sorted(glob.glob(os.path.join(base_dir, "data", "clean_scans", "*.*")))
    bg_paths = sorted(glob.glob(os.path.join(base_dir, "data", "backgrounds", "*.*")))

    if not clean_paths or not bg_paths:
        print("Data directories are empty! Cannot visualize.")
        return

    # 1. Use the dataset to load and compress images (batch_size 3 for 3 visualizations)
    dataset = DocumentDataset(clean_paths, bg_paths, split='train', target_size=1024)
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    
    clean_batch, bg_batch = next(iter(loader))
    
    # 2. Use the GPU Pipeline to degrade them
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
    pipeline = GPUDegradationPipeline(target_size=1024).to(device)
    
    print("Running GPU degradation pipeline...")
    with torch.no_grad():
        # Use task='corner' so we get the fully degraded image (not the rectified one)
        degraded_batch, corners_batch = pipeline(clean_batch, bg_batch, task='corner')

    # 3. Save the visualizations
    for i in range(3):
        # Convert tensor (C, H, W) to numpy (H, W, C)
        deg_img = degraded_batch[i].cpu().permute(1, 2, 0).numpy()
        deg_img = np.clip(deg_img * 255.0, 0, 255).astype(np.uint8)
        
        # Draw corners
        corners = corners_batch[i].cpu().numpy() * 1024 # Scale back from normalized [0,1]
        
        # Convert RGB to BGR for OpenCV saving
        deg_img_bgr = cv2.cvtColor(deg_img, cv2.COLOR_RGB2BGR)
        
        # Draw green lines between the 4 corners
        for j, pt in enumerate(corners):
            cv2.circle(deg_img_bgr, (int(pt[0]), int(pt[1])), 15, (255, 0, 0), -1)
            next_pt = corners[(j+1)%4]
            cv2.line(deg_img_bgr, (int(pt[0]), int(pt[1])), (int(next_pt[0]), int(next_pt[1])), (0, 255, 0), 5)

        out_path = os.path.join(artifact_dir, f"sample_degraded_{i+1}.jpg")
        cv2.imwrite(out_path, deg_img_bgr)
        print(f"Saved visualization {i+1} to {out_path}")

if __name__ == "__main__":
    main()
