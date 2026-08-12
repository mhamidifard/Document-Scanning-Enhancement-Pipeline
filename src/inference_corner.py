import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.model import CornerRegressionNet, CornerHeatmapNet
from src.evaluate_corner import extract_heatmap_coordinates

def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for Corner Detection")
    parser.add_argument('--input', type=str, required=True, help="Path to input raw photo")
    parser.add_argument('--output', type=str, default="corner_output.jpg", help="Path to save annotated image")
    parser.add_argument('--approach', type=str, required=True, choices=['regression', 'heatmap'], help="Model to use")
    parser.add_argument('--img_size', type=int, default=256, help="Size to resize for model inference")
    parser.add_argument('--model_dir', type=str, default="checkpoints")

    args = parser.parse_args()

    ckpt_path = os.path.join(base_dir, args.model_dir, f"best_corner_{args.approach}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Model weights not found at {ckpt_path}.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running {args.approach.upper()} inference on {device}")

    # Load Model
    if args.approach == 'regression':
        model = CornerRegressionNet(in_channels=3).to(device)
    else:
        model = CornerHeatmapNet(in_channels=3, out_channels=4).to(device)
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # ========================================================
    # 1. Preprocess the image
    # ========================================================
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read image at {args.input}")
        return
        
    orig_h, orig_w = img.shape[:2]
    print(f"Original image dimensions: {orig_w}x{orig_h}")
    
    # Convert BGR to RGB and resize to model's expected size
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))
    
    # Normalize to [0, 1] and convert to tensor (1, C, H, W)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # ========================================================
    # 2. Predict the four corners
    # ========================================================
    print("Running model prediction...")
    with torch.no_grad():
        outputs = model(img_tensor)
        
        if args.approach == 'regression':
            pred_corners = outputs.view(1, 4, 2)
        else:
            pred_corners = extract_heatmap_coordinates(outputs)
            
        # pred_corners is [1, 4, 2] containing normalized coordinates [0, 1]
        corners_normalized = pred_corners[0].cpu().numpy()

    # ========================================================
    # 3. Map coordinates
    # ========================================================
    # Scale coordinates back to the original image resolution
    corners_original = corners_normalized.copy()
    corners_original[:, 0] *= orig_w
    corners_original[:, 1] *= orig_h
    
    print("Predicted Corners (x, y):")
    for i, pt in enumerate(corners_original):
        print(f"  Corner {i+1}: ({int(pt[0])}, {int(pt[1])})")

    # ========================================================
    # 4. Visualize the corners
    # ========================================================
    # Draw on a copy of the original BGR image for saving
    vis_bgr = img.copy()
    for j, pt in enumerate(corners_original):
        # Draw Corner point
        cv2.circle(vis_bgr, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1) # Red dots
        # Draw line to next corner
        next_pt = corners_original[(j+1)%4]
        cv2.line(vis_bgr, (int(pt[0]), int(pt[1])), (int(next_pt[0]), int(next_pt[1])), (0, 255, 0), 4) # Green lines
        
    cv2.imwrite(args.output, vis_bgr)
    print(f"Saved annotated image to {args.output}")

    # Generate Matplotlib side-by-side plot
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Raw Input")
    axes[0].axis('off')
    
    axes[1].imshow(vis_rgb)
    axes[1].set_title(f"Predicted Corners ({args.approach})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = args.output.replace('.jpg', '_plot.png').replace('.png', '_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved visualization plot to {plot_path}")

if __name__ == "__main__":
    main()
