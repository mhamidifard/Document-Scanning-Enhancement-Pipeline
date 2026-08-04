import os
import sys
import cv2
import numpy as np

# Dynamically get the base directory of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Ensure the parent directory is in sys.path so we can import src.dataset
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import DegradationPipeline

def main():
    # Set artifact directory relative to the base directory
    artifact_dir = os.path.join(base_dir, "result", "visualize-degradation")
    os.makedirs(artifact_dir, exist_ok=True)

    # Set paths dynamically based on base_dir
    clean_path = os.path.join(base_dir, "data", "clean_scans", "1.jpg")
    bg_path = os.path.join(base_dir, "data", "backgrounds", "1.jpg")

    clean_img = cv2.imread(clean_path)
    if clean_img is None:
        raise FileNotFoundError(f"Could not load {clean_path}")
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        raise FileNotFoundError(f"Could not load {bg_path}")
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

    print("Running degradation pipeline...")
    pipeline = DegradationPipeline()
    
    # Generate 3 distinct degraded versions
    for i in range(1, 4):
        degraded_img, corners = pipeline.forward(clean_img, bg_img)

        # Draw corners on degraded image
        vis_img = degraded_img.copy()
        # for j, pt in enumerate(corners):
        #     cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)
        #     next_pt = corners[(j+1)%4]
        #     cv2.line(vis_img, (int(pt[0]), int(pt[1])), (int(next_pt[0]), int(next_pt[1])), (0, 255, 0), 4)

        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(artifact_dir, f"sample_degraded_{i}.jpg")
        cv2.imwrite(out_path, vis_img_bgr)
        print(f"Saved visualization {i} to {out_path}")

if __name__ == "__main__":
    main()
