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

from src.model import EnhancementUNet

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

def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for Document Enhancement")
    parser.add_argument('--input', type=str, required=True, help="Path to input rectified image")
    parser.add_argument('--output', type=str, default="enhanced_output.jpg", help="Path to save enhanced image")
    parser.add_argument('--model_dir', type=str, default="checkpoints")
    parser.add_argument('--img_size', type=int, default=768, help="Size to resize for model inference")
    parser.add_argument('--sharpen', action='store_true', help="Apply unsharp mask to intensify text and contrast")
    args = parser.parse_args()

    ckpt_path = os.path.join(base_dir, args.model_dir, "best_enhancement.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Model weights not found at {ckpt_path}.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")

    # Load Model
    model = EnhancementUNet().to(device)
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
    # 2. Predict the enhanced image
    # ========================================================
    print("Running model prediction...")
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    # ========================================================
    # 3. Post-process the output
    # ========================================================
    # Convert from tensor to numpy array [H, W, C]
    enhanced_img = enhanced_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Clamp values just in case and convert to 8-bit image [0, 255]
    enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
    
    # Resize the enhanced image back to the original dimensions
    enhanced_img_orig_size = cv2.resize(enhanced_img, (orig_w, orig_h))
    
    # Optional Post-processing: Sharpening (Unsharp Mask)
    if args.sharpen:
        print("Applying Unsharp Masking to intensify text...")
        enhanced_img_orig_size = apply_unsharp_mask(enhanced_img_orig_size)
    
    # Save the output image (Convert RGB back to BGR for OpenCV)
    enhanced_bgr = cv2.cvtColor(enhanced_img_orig_size, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, enhanced_bgr)
    print(f"Saved 8-bit enhanced image to {args.output}")

    # ========================================================
    # 4. Visualize the model's output
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Input ({orig_w}x{orig_h})")
    axes[0].axis('off')
    
    axes[1].imshow(enhanced_img_orig_size)
    axes[1].set_title(f"Enhanced Output ({orig_w}x{orig_h})")
    axes[1].axis('off')
    
    plt.tight_layout()
    # Save plot instead of plt.show() since this runs on Colab
    plot_path = args.output.replace('.jpg', '_plot.png').replace('.png', '_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved visualization plot to {plot_path}")

if __name__ == "__main__":
    main()
