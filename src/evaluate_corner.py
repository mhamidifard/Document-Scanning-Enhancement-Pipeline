import os
import sys
import glob
import argparse
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import DocumentDataset
from src.model import CornerRegressionNet, CornerHeatmapNet
from src.pipeline import GPUDegradationPipeline

def extract_heatmap_coordinates(heatmaps):
    """
    Extracts (x, y) coordinates from heatmaps using argmax.
    heatmaps: (B, 4, H, W)
    Returns: (B, 4, 2) tensor of normalized coordinates in [0, 1]
    """
    B, C, H, W = heatmaps.shape
    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(B, C, -1)
    
    # Get index of maximum value
    max_idx = torch.argmax(heatmaps_flat, dim=2)
    
    # Convert flat index back to 2D coordinates
    y = (max_idx // W).float()
    x = (max_idx % W).float()
    
    # Normalize back to [0, 1]
    coords = torch.stack([x / (W - 1), y / (H - 1)], dim=2)
    return coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--canvas_size', type=int, default=1536)
    parser.add_argument('--threshold', type=float, default=0.05, help="Normalized distance threshold for success")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    reg_path = os.path.join(ckpt_dir, "best_corner_regression.pth")
    heat_path = os.path.join(ckpt_dir, "best_corner_heatmap.pth")

    # 1. Setup Test Dataset (Final 10%)
    clean_paths = sorted(glob.glob(os.path.join(base_dir, "data", "clean_scans", "*.*")))
    bg_paths = sorted(glob.glob(os.path.join(base_dir, "data", "backgrounds", "*.*")))
    
    random.seed(42)
    random.shuffle(clean_paths)
    val_split = int(0.9 * len(clean_paths))
    
    test_scans = clean_paths[val_split:]
    if len(test_scans) == 0:
        print("Test set is empty! You need more images.")
        return
        
    test_dataset = DocumentDataset(test_scans, bg_paths, split='test', canvas_size=args.canvas_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Setup GPU Pipeline
    gpu_pipeline = GPUDegradationPipeline(target_size=args.img_size, canvas_size=args.canvas_size).to(device)
    
    # 3. Load Models
    models_to_test = {}
    
    if os.path.exists(reg_path):
        model = CornerRegressionNet(in_channels=3).to(device)
        model.load_state_dict(torch.load(reg_path, map_location=device, weights_only=True))
        model.eval()
        models_to_test['Regression'] = model
    
    if os.path.exists(heat_path):
        model = CornerHeatmapNet(in_channels=3, out_channels=4).to(device)
        model.load_state_dict(torch.load(heat_path, map_location=device, weights_only=True))
        model.eval()
        models_to_test['Heatmap'] = model
        
    if not models_to_test:
        print("No trained models found! Run train_corner.py first.")
        return

    print("\n" + "="*60)
    print("        CORNER DETECTION EVALUATION (SYNTHETIC TEST SET)")
    print("="*60)
    
    for approach_name, model in models_to_test.items():
        total_error = 0.0
        total_corners = 0
        success_count = 0
        total_images = 0
        
        saved_images = 0
        import matplotlib.pyplot as plt
        import cv2
        
        with torch.no_grad():
            for clean_batch, bg_batch in test_loader:
                B = clean_batch.size(0)
                clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
                
                # Get the degraded image and true corners
                inputs, true_corners = gpu_pipeline(clean_batch, bg_batch, task='corner')
                
                # Model Prediction
                outputs = model(inputs)
                
                # Extract predicted coordinates
                if approach_name == 'Regression':
                    pred_corners = outputs.view(B, 4, 2)
                else:
                    pred_corners = extract_heatmap_coordinates(outputs)
                
                # Calculate Euclidean Distance
                diff = (pred_corners - true_corners) * args.img_size
                distances = torch.sqrt(torch.sum(diff**2, dim=2)) # Shape (B, 4)
                
                total_error += torch.sum(distances).item()
                total_corners += (B * 4)
                
                # Check accuracy threshold
                max_dist_per_image, _ = torch.max(distances, dim=1) # Shape (B)
                success_mask = max_dist_per_image <= (args.threshold * args.img_size)
                success_count += torch.sum(success_mask).item()
                total_images += B
                
                # Save visual comparisons for the first 5 images
                for b in range(B):
                    if saved_images < 5:
                        # Convert tensor to numpy image
                        img_np = inputs[b].cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        # OpenCV uses BGR, but we'll just plot with matplotlib directly in RGB
                        
                        tc = (true_corners[b].cpu().numpy() * args.img_size).astype(int)
                        pc = (pred_corners[b].cpu().numpy() * args.img_size).astype(int)
                        
                        # Draw True Corners (Green) and Predicted (Red)
                        img_draw = img_np.copy()
                        for i in range(4):
                            cv2.circle(img_draw, tuple(tc[i]), 4, (0, 255, 0), -1) # Green
                            cv2.circle(img_draw, tuple(pc[i]), 3, (255, 0, 0), -1) # Red (actually Blue in BGR, but matplotlib sees RGB so Red)
                            
                            # Draw lines connecting the 4 corners
                            cv2.line(img_draw, tuple(tc[i]), tuple(tc[(i+1)%4]), (0, 255, 0), 2)
                            cv2.line(img_draw, tuple(pc[i]), tuple(pc[(i+1)%4]), (255, 0, 0), 2)
                            
                        plt.figure()
                        plt.imshow(img_draw)
                        plt.title(f"{approach_name} - True (Green) vs Pred (Red)")
                        plt.axis('off')
                        
                        results_dir = os.path.join(base_dir, "results", "corner_eval")
                        os.makedirs(results_dir, exist_ok=True)
                        save_path = os.path.join(results_dir, f"{approach_name.lower()}_{saved_images + 1}.jpg")
                        
                        plt.savefig(save_path)
                        plt.close()
                        
                        saved_images += 1
                
        mean_error = total_error / total_corners
        success_rate = (success_count / total_images) * 100
        
        print(f"\nModel: {approach_name}")
        print(f"Mean Localization Error : {mean_error:.2f} pixels (at {args.img_size}x{args.img_size})")
        print(f"Success Rate (< {args.threshold*100}% err) : {success_rate:.1f}% of images perfect")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
