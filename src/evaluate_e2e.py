import os
import sys
import glob
import json
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.model import CornerRegressionNet, CornerHeatmapNet, EnhancementUNet
from src.evaluate_corner import extract_heatmap_coordinates
from src.evaluate_real import order_points

def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # image must be uint8
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image.astype(float) - blurred.astype(float)) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def run_ocr_and_get_confidence(img_rgb):
    """
    Runs Tesseract OCR on an RGB image and returns the average confidence score
    of the detected words. Returns 0 if no words are found.
    """
    try:
        # We convert to grayscale for Tesseract for slightly better accuracy
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)
        
        confidences = []
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            conf = data['conf'][i]
            
            # Tesseract uses '-1' for non-word boxes (blocks, lines, etc.)
            if conf != '-1' and int(conf) != -1 and len(word) > 0:
                confidences.append(float(conf))
                
        if len(confidences) == 0:
            return 0.0
        return sum(confidences) / len(confidences)
    except Exception as e:
        print(f"OCR failed (is tesseract installed?): {e}")
        return 0.0

def warp_document(img_bgr, corners_pixel, target_size=768):
    """
    Warps a quadrilateral into a square target_size x target_size crop.
    """
    dst_pts = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners_pixel.astype("float32"), dst_pts)
    warped_bgr = cv2.warpPerspective(img_bgr, M, (target_size, target_size))
    return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

def tensor_to_rgb(tensor):
    """ Converts a CHW [0, 1] tensor to HWC [0, 255] uint8 RGB image """
    img_np = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to Roboflow COCO export folder")
    parser.add_argument('--corner_model', type=str, default='heatmap', choices=['regression', 'heatmap'])
    parser.add_argument('--enhancement_size', type=int, default=768, help="Size of the enhanced document")
    parser.add_argument('--corner_model_dir', type=str, default="checkpoints", help="Directory containing the corner model")
    parser.add_argument('--enh_model_dir', type=str, default="checkpoints", help="Directory containing the enhancement model")
    parser.add_argument('--sharpen', action='store_true', help="Apply unsharp mask to intensify text")
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
        
    images_info = {img['id']: {'file_name': img['file_name'], 'orig_name': img.get('extra', {}).get('name', img['file_name']), 'width': img['width'], 'height': img['height']} for img in coco_data['images']}
    
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        seg = ann['segmentation'][0]
        pts = np.array(seg).reshape(4, 2)
        annotations[img_id] = order_points(pts)
        
    print(f"Loaded {len(annotations)} labeled images for End-to-End evaluation.")

    # 2. Load Models
    corner_ckpt_dir = os.path.join(base_dir, args.corner_model_dir)
    enh_ckpt_dir = os.path.join(base_dir, args.enh_model_dir)
    
    corner_path = os.path.join(corner_ckpt_dir, f"best_corner_{args.corner_model}.pth")
    enh_path = os.path.join(enh_ckpt_dir, "best_enhancement.pth")
    
    if not os.path.exists(corner_path):
        print(f"Error: Corner model {corner_path} not found.")
        return
    if not os.path.exists(enh_path):
        print(f"Error: Enhancement model {enh_path} not found.")
        return
        
    if args.corner_model == 'regression':
        corner_net = CornerRegressionNet(in_channels=3).to(device)
    else:
        corner_net = CornerHeatmapNet(in_channels=3, out_channels=4).to(device)
        
    corner_net.load_state_dict(torch.load(corner_path, map_location=device, weights_only=True))
    corner_net.eval()
    
    enh_net = EnhancementUNet(in_channels=3, out_channels=3).to(device)
    enh_net.load_state_dict(torch.load(enh_path, map_location=device, weights_only=True))
    enh_net.eval()

    print("\n" + "="*60)
    print("        END-TO-END PIPELINE EVALUATION (OCR)")
    print("="*60)
    
    results_dir = os.path.join(base_dir, "results", "e2e_eval")
    os.makedirs(results_dir, exist_ok=True)
    ref_dir = os.path.join(base_dir, "data", "reference_scans")
    
    stats = {
        "Raw_TrueCorners": [],
        "Enhanced_TrueCorners": [],
        "Enhanced_PredCorners": [],
        "ReferenceScan": []
    }
    
    with torch.no_grad():
        for img_id, true_corners_orig in annotations.items():
            img_info = images_info[img_id]
            img_path = os.path.join(args.dataset_dir, img_info['file_name'])
            
            # Find matching reference scan
            orig_name = img_info['orig_name']
            ref_path = os.path.join(ref_dir, orig_name)
            
            if not os.path.exists(img_path):
                continue
                
            img_bgr = cv2.imread(img_path)
            orig_h, orig_w = img_bgr.shape[:2]
            
            # --- CORNER DETECTION STAGE ---
            img_rgb_256 = cv2.cvtColor(cv2.resize(img_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
            corner_in = torch.from_numpy(img_rgb_256).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            
            c_out = corner_net(corner_in)
            if args.corner_model == 'regression':
                pred_corners_norm = c_out.view(1, 4, 2)[0].cpu().numpy()
            else:
                pred_corners_norm = extract_heatmap_coordinates(c_out)[0].cpu().numpy()
                
            # Scale predicted corners to high-res image
            pred_corners_orig = pred_corners_norm.copy()
            pred_corners_orig[:, 0] *= orig_w
            pred_corners_orig[:, 1] *= orig_h
            
            # --- WARP STAGE ---
            raw_crop_true = warp_document(img_bgr, true_corners_orig, args.enhancement_size)
            raw_crop_pred = warp_document(img_bgr, pred_corners_orig, args.enhancement_size)
            
            # --- ENHANCEMENT STAGE ---
            def enhance_crop(crop_rgb):
                tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                out_tensor = enh_net(tensor)
                out_rgb = tensor_to_rgb(out_tensor)
                if args.sharpen:
                    out_rgb = apply_unsharp_mask(out_rgb)
                return out_rgb
                
            enh_crop_true = enhance_crop(raw_crop_true)
            enh_crop_pred = enhance_crop(raw_crop_pred)
            
            # --- REFERENCE SCAN ---
            ref_rgb = None
            if os.path.exists(ref_path):
                ref_bgr = cv2.imread(ref_path)
                ref_rgb = cv2.cvtColor(cv2.resize(ref_bgr, (args.enhancement_size, args.enhancement_size)), cv2.COLOR_BGR2RGB)
            else:
                # Fallback blank image if reference scan is missing
                ref_rgb = np.zeros_like(raw_crop_true)
                
            # --- OCR EVALUATION ---
            conf_raw_true = run_ocr_and_get_confidence(raw_crop_true)
            conf_enh_true = run_ocr_and_get_confidence(enh_crop_true)
            conf_enh_pred = run_ocr_and_get_confidence(enh_crop_pred)
            conf_ref      = run_ocr_and_get_confidence(ref_rgb)
            
            stats["Raw_TrueCorners"].append(conf_raw_true)
            stats["Enhanced_TrueCorners"].append(conf_enh_true)
            stats["Enhanced_PredCorners"].append(conf_enh_pred)
            if os.path.exists(ref_path):
                stats["ReferenceScan"].append(conf_ref)
            
            # --- VISUALIZATION ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            
            axes[0, 0].imshow(raw_crop_true)
            axes[0, 0].set_title(f"Raw Crop (True Corners)\nOCR Confidence: {conf_raw_true:.1f}%", color='black')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(ref_rgb)
            axes[0, 1].set_title(f"CamScanner Reference\nOCR Confidence: {conf_ref:.1f}%", color='black' if os.path.exists(ref_path) else 'red')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(enh_crop_true)
            axes[1, 0].set_title(f"Enhanced (True Corners)\nOCR Confidence: {conf_enh_true:.1f}%", color='green')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(enh_crop_pred)
            axes[1, 1].set_title(f"Enhanced (Pred Corners)\nOCR Confidence: {conf_enh_pred:.1f}%", color='blue')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(results_dir, f"e2e_{orig_name}")
            plt.savefig(save_path)
            plt.close()
            
    print("\nFINAL PIPELINE OCR AVERAGES:")
    print("-" * 30)
    for k, v in stats.items():
        if len(v) > 0:
            print(f"{k:22s}: {np.mean(v):.1f}%")
        
    print("\nVisualizations saved to results/e2e_eval/")

if __name__ == "__main__":
    main()
