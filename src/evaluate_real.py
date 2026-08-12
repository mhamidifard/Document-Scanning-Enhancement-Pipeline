import os
import sys
import glob
import json
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.model import CornerRegressionNet, CornerHeatmapNet
from src.evaluate_corner import extract_heatmap_coordinates

def order_points(pts):
    """
    Orders 4 points into: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    pts: numpy array of shape (4, 2)
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-Left has smallest sum, Bottom-Right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    
    # Top-Right has smallest diff (y - x), Bottom-Left has largest diff (y - x)
    diff = np.diff(pts, axis=1) # y - x
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    
    return rect

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to Roboflow COCO export folder (e.g. data/real_test/test)")
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.05, help="Normalized distance threshold for success")
    parser.add_argument('--model_dir', type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Parse COCO JSON
    json_paths = glob.glob(os.path.join(args.dataset_dir, "*.json"))
    if not json_paths:
        print(f"Error: No JSON file found in {args.dataset_dir}")
        return
        
    coco_file = json_paths[0]
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
        
    # Map image ID to filename
    images_info = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']} for img in coco_data['images']}
    
    # Map image ID to annotation (Polygon points)
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        seg = ann['segmentation'][0] # List of [x1, y1, x2, y2, x3, y3, x4, y4]
        
        # Convert to shape (4, 2)
        pts = np.array(seg).reshape(4, 2)
        
        # Order the points to match our network's expected output (TL, TR, BR, BL)
        ordered_pts = order_points(pts)
        annotations[img_id] = ordered_pts
        
    print(f"Loaded {len(annotations)} labeled images from real dataset.")

    # 2. Load Models
    ckpt_dir = os.path.join(base_dir,args.model_dir )
    reg_path = os.path.join(ckpt_dir, "best_corner_regression.pth")
    heat_path = os.path.join(ckpt_dir, "best_corner_heatmap.pth")
    
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
    print("        CORNER DETECTION EVALUATION (REAL DATASET)")
    print("="*60)
    
    # Results directory for images
    results_dir = os.path.join(base_dir, "results", "real_eval")
    os.makedirs(results_dir, exist_ok=True)
    
    for approach_name, model in models_to_test.items():
        total_error = 0.0
        total_corners = 0
        success_count = 0
        total_images = 0
        saved_images = 0
        
        with torch.no_grad():
            for img_id, true_corners in annotations.items():
                img_info = images_info[img_id]
                img_path = os.path.join(args.dataset_dir, img_info['file_name'])
                
                if not os.path.exists(img_path):
                    continue
                    
                # Load and preprocess image
                img_bgr = cv2.imread(img_path)
                orig_h, orig_w = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Resize to target size for network
                img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device) # Shape (1, 3, 256, 256)
                
                # Normalize True Corners to [0, 1] range
                true_corners_norm = true_corners.copy()
                true_corners_norm[:, 0] /= orig_w
                true_corners_norm[:, 1] /= orig_h
                true_corners_tensor = torch.tensor(true_corners_norm, dtype=torch.float32, device=device).unsqueeze(0) # (1, 4, 2)
                
                # Model Prediction
                outputs = model(img_tensor)
                
                if approach_name == 'Regression':
                    pred_corners_tensor = outputs.view(1, 4, 2)
                else:
                    pred_corners_tensor = extract_heatmap_coordinates(outputs)
                
                # Calculate Error using normalized coordinates scaled to network space (pixels)
                diff = (pred_corners_tensor - true_corners_tensor) * args.img_size
                distances = torch.sqrt(torch.sum(diff**2, dim=2)) # Shape (1, 4)
                
                total_error += torch.sum(distances).item()
                total_corners += 4
                
                max_dist = torch.max(distances).item()
                if max_dist <= (args.threshold * args.img_size):
                    success_count += 1
                    
                total_images += 1
                
                # Save visual comparisons for all images
                # Create a side-by-side comparison
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # 1. Original Image with Overlays (draw on the original high-res image for clarity)
                img_draw = img_rgb.copy()
                
                # Scale coordinates to the high-res image
                tc_orig = (true_corners_norm * np.array([orig_w, orig_h])).astype(int)
                pc_orig = (pred_corners_tensor[0].cpu().numpy() * np.array([orig_w, orig_h])).astype(int)
                
                for i in range(4):
                    # True Corners (Green)
                    cv2.circle(img_draw, tuple(tc_orig[i]), 15, (0, 255, 0), -1) 
                    cv2.line(img_draw, tuple(tc_orig[i]), tuple(tc_orig[(i+1)%4]), (0, 255, 0), 5)
                    
                    # Predicted Corners (Red)
                    cv2.circle(img_draw, tuple(pc_orig[i]), 10, (255, 0, 0), -1)
                    cv2.line(img_draw, tuple(pc_orig[i]), tuple(pc_orig[(i+1)%4]), (255, 0, 0), 5)
                
                axes[0].imshow(img_draw)
                axes[0].set_title(f"{approach_name} - True (Green) vs Pred (Red)")
                axes[0].axis('off')
                
                # 2. Extracted (Warped) Document
                # We use cv2.warpPerspective to extract the document using the Predicted Corners
                # Target dimensions for the extracted document (A4 ratio approx)
                doc_w, doc_h = 800, 1131
                dst_pts = np.array([
                    [0, 0],
                    [doc_w - 1, 0],
                    [doc_w - 1, doc_h - 1],
                    [0, doc_h - 1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(pc_orig.astype("float32"), dst_pts)
                warped = cv2.warpPerspective(img_rgb, M, (doc_w, doc_h))
                
                axes[1].imshow(warped)
                axes[1].set_title("Extracted Document")
                axes[1].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(results_dir, f"{approach_name.lower()}_real_{saved_images + 1}.jpg")
                plt.savefig(save_path)
                plt.close()
                
                saved_images += 1
                    
        if total_corners > 0:
            mean_error = total_error / total_corners
            success_rate = (success_count / total_images) * 100
            
            print(f"\nModel: {approach_name}")
            print(f"Mean Localization Error : {mean_error:.2f} pixels (at {args.img_size}x{args.img_size})")
            print(f"Success Rate (< {args.threshold*100}% err) : {success_rate:.1f}% of images perfect")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
