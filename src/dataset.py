import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import os

class DegradationPipeline:
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng if rng is not None else np.random.RandomState()

    def apply_perspective_warp(self, clean_img: np.ndarray, bg_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bg_h, bg_w = bg_img.shape[:2]
        
        # Define base rectangle inside background (e.g. 15% margin)
        margin_x = int(bg_w * 0.15)
        margin_y = int(bg_h * 0.15)
        
        # Max perturbation (e.g. 20% of width/height)
        max_dx = int(bg_w * 0.20)
        max_dy = int(bg_h * 0.20)

        # Base corners: TL, TR, BR, BL
        base_corners = np.array([
            [margin_x, margin_y],
            [bg_w - margin_x, margin_y],
            [bg_w - margin_x, bg_h - margin_y],
            [margin_x, bg_h - margin_y]
        ], dtype=np.float32)

        # Apply random perturbations within quadrants
        perturbed_corners = np.zeros_like(base_corners)
        
        # TL (moves right/down mostly, but bounded by margin and center)
        perturbed_corners[0, 0] = base_corners[0, 0] + self.rng.uniform(-margin_x*0.5, max_dx)
        perturbed_corners[0, 1] = base_corners[0, 1] + self.rng.uniform(-margin_y*0.5, max_dy)
        
        # TR (moves left/down)
        perturbed_corners[1, 0] = base_corners[1, 0] + self.rng.uniform(-max_dx, margin_x*0.5)
        perturbed_corners[1, 1] = base_corners[1, 1] + self.rng.uniform(-margin_y*0.5, max_dy)
        
        # BR (moves left/up)
        perturbed_corners[2, 0] = base_corners[2, 0] + self.rng.uniform(-max_dx, margin_x*0.5)
        perturbed_corners[2, 1] = base_corners[2, 1] + self.rng.uniform(-max_dy, margin_y*0.5)
        
        # BL (moves right/up)
        perturbed_corners[3, 0] = base_corners[3, 0] + self.rng.uniform(-margin_x*0.5, max_dx)
        perturbed_corners[3, 1] = base_corners[3, 1] + self.rng.uniform(-max_dy, margin_y*0.5)

        # Ensure corners are within image bounds
        perturbed_corners[:, 0] = np.clip(perturbed_corners[:, 0], 0, bg_w - 1)
        perturbed_corners[:, 1] = np.clip(perturbed_corners[:, 1], 0, bg_h - 1)

        clean_h, clean_w = clean_img.shape[:2]
        src_corners = np.array([
            [0, 0],
            [clean_w - 1, 0],
            [clean_w - 1, clean_h - 1],
            [0, clean_h - 1]
        ], dtype=np.float32)

        # Compute Homography
        H_mat = cv2.getPerspectiveTransform(src_corners, perturbed_corners)

        # Warp clean scan onto background
        warped_doc = cv2.warpPerspective(clean_img, H_mat, (bg_w, bg_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # Create a mask to overlay onto background
        mask = np.zeros((clean_h, clean_w), dtype=np.uint8)
        mask.fill(255)
        warped_mask = cv2.warpPerspective(mask, H_mat, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
        
        # Composite
        composited = bg_img.copy()
        mask_indices = warped_mask > 0
        composited[mask_indices] = warped_doc[mask_indices]

        return composited, perturbed_corners

    def apply_scale(self, img: np.ndarray) -> np.ndarray:
        scale_factor = self.rng.uniform(0.25, 0.5) # Downscale by 2 to 4
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored

    def apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        
        # Brightness & Contrast
        alpha = self.rng.uniform(0.8, 1.2) # Contrast
        beta = self.rng.uniform(-30, 30)   # Brightness
        img_float = img_float * alpha + beta
        
        # Color cast (assuming RGB order)
        r_scale = self.rng.uniform(0.9, 1.1)
        b_scale = self.rng.uniform(0.9, 1.1)
        img_float[:, :, 0] *= r_scale
        img_float[:, :, 2] *= b_scale
        
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_illumination_and_shadows(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        
        # Simple linear gradient for illumination
        start_val = self.rng.uniform(0.5, 1.0)
        end_val = self.rng.uniform(0.5, 1.0)
        
        if self.rng.rand() > 0.5:
            # Horizontal
            grad = np.linspace(start_val, end_val, w).astype(np.float32)
            grad = np.tile(grad, (h, 1))
        else:
            # Vertical
            grad = np.linspace(start_val, end_val, h).astype(np.float32)
            grad = np.tile(grad[:, None], (1, w))
            
        grad = np.stack([grad, grad, grad], axis=-1)
        img_float = img.astype(np.float32) * grad
        
        # Soft shadows via polygons
        if self.rng.rand() > 0.3: # 70% chance of shadow
            shadow_mask = np.ones((h, w), dtype=np.float32)
            num_pts = self.rng.randint(3, 6)
            pts = np.zeros((num_pts, 2), dtype=np.int32)
            pts[:, 0] = self.rng.randint(0, w, size=num_pts)
            pts[:, 1] = self.rng.randint(0, h, size=num_pts)
            
            # Draw dark polygon
            cv2.fillPoly(shadow_mask, [pts], 0.4) # drop intensity inside shadow
            
            # Blur the shadow mask heavily
            ksize = self.rng.choice([31, 51, 71, 91])
            shadow_mask = cv2.GaussianBlur(shadow_mask, (ksize, ksize), 0)
            shadow_mask = np.stack([shadow_mask]*3, axis=-1)
            
            img_float = img_float * shadow_mask

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_blur_and_noise(self, img: np.ndarray) -> np.ndarray:
        # Blur
        if self.rng.rand() > 0.5:
            ksize = self.rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
        # Noise
        noise = self.rng.normal(0, self.rng.uniform(2, 10), img.shape)
        img_float = img.astype(np.float32) + noise
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        quality = self.rng.randint(30, 81)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        # RGB to BGR for cv2 imencode
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        # BGR to RGB
        return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    def forward(self, clean_img: np.ndarray, bg_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Perspective Warp (creates labels)
        img, corners = self.apply_perspective_warp(clean_img, bg_img)
        
        # 2. Scale
        img = self.apply_scale(img)
        
        # 3. Color Jitter
        img = self.apply_color_jitter(img)
        
        # 4. Shadows
        img = self.apply_illumination_and_shadows(img)
        
        # 5. Blur & Noise
        img = self.apply_blur_and_noise(img)
        
        # 6. JPEG Compression
        img = self.apply_jpeg_compression(img)
        
        return img, corners

class DocumentDataset(Dataset):
    def __init__(self, clean_scans: List[str], backgrounds: List[str], split: str = 'train', target_size: int = 256):
        self.clean_scans = clean_scans
        self.backgrounds = backgrounds
        self.split = split
        self.target_size = target_size
        
        if len(self.clean_scans) == 0 or len(self.backgrounds) == 0:
            raise ValueError("Scan and background lists cannot be empty.")

    def __len__(self):
        # Arbitrary epoch size; typically multiple of clean scans
        return len(self.clean_scans) * 10 

    def __getitem__(self, idx):
        # Deterministic generation for val/test
        if self.split in ['val', 'test']:
            rng = np.random.RandomState(seed=idx)
        else:
            rng = np.random.RandomState() # Random per sample

        pipeline = DegradationPipeline(rng=rng)
        
        # Select images
        clean_path = self.clean_scans[idx % len(self.clean_scans)]
        bg_path = self.backgrounds[rng.randint(0, len(self.backgrounds))]
        
        clean_img = cv2.imread(clean_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        
        bg_img = cv2.imread(bg_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        
        # Generate degradation
        degraded_img, corners = pipeline.forward(clean_img, bg_img)
        
        # ------------------ Preprocessing ------------------
        orig_h, orig_w = degraded_img.shape[:2]
        
        # Resize images to target size
        degraded_img = cv2.resize(degraded_img, (self.target_size, self.target_size))
        clean_img_target = cv2.resize(clean_img, (self.target_size, self.target_size))
        
        # Scale corner coordinates
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y
        
        # Normalize corners to [0, 1]
        corners[:, 0] /= self.target_size
        corners[:, 1] /= self.target_size
        
        # Convert images to tensors and normalize to [0, 1]
        # HWC -> CHW
        degraded_tensor = torch.from_numpy(degraded_img.transpose((2, 0, 1))).float() / 255.0
        clean_tensor = torch.from_numpy(clean_img_target.transpose((2, 0, 1))).float() / 255.0
        corners_tensor = torch.from_numpy(corners).float()
        
        return degraded_tensor, clean_tensor, corners_tensor
