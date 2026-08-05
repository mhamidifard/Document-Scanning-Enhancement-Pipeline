import torch
import torch.nn as nn
import kornia

class GPUDegradationPipeline(nn.Module):
    def __init__(self, target_size=768, canvas_size=1536, max_perturb=0.20, margin=0.15):
        super().__init__()
        self.target_size = target_size
        self.canvas_size = canvas_size
        self.max_perturb = max_perturb
        self.margin = margin
        
        # Define base corner regions based on document rules (normalized 0 to 1)
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.regions = torch.tensor([
            [margin, margin, 0.5 - margin, 0.5 - margin], # TL: x_min, y_min, x_max, y_max
            [0.5 + margin, margin, 1.0 - margin, 0.5 - margin], # TR
            [0.5 + margin, 0.5 + margin, 1.0 - margin, 1.0 - margin], # BR
            [margin, 0.5 + margin, 0.5 - margin, 1.0 - margin]  # BL
        ], dtype=torch.float32)
        
        # Base source corners for homography (0 to canvas_size)
        self.src_corners = torch.tensor([
            [0, 0],
            [canvas_size - 1, 0],
            [canvas_size - 1, canvas_size - 1],
            [0, canvas_size - 1]
        ], dtype=torch.float32)
        
        # Kornia GPU Augmentations - Toned down
        self.color_jitter = kornia.augmentation.ColorJitter(0.15, 0.15, 0.15, 0.05, p=0.8)
        self.blur = kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.3)

    def generate_random_corners(self, batch_size, device):
        """Generates perturbed corners for the entire batch in parallel."""
        corners = torch.zeros((batch_size, 4, 2), device=device)
        for i in range(4):
            x_min, y_min, x_max, y_max = self.regions[i]
            corners[:, i, 0] = torch.empty(batch_size, device=device).uniform_(x_min, x_max)
            corners[:, i, 1] = torch.empty(batch_size, device=device).uniform_(y_min, y_max)
            
        # Scale to canvas dimensions
        corners_scaled = corners * self.canvas_size
        return corners_scaled, corners

    def forward(self, clean_batch, bg_batch, task='corner'):
        B, C, H, W = clean_batch.shape # H, W will be canvas_size
        device = clean_batch.device
        
        # 1. Generate corners
        corners_scaled, corners_normalized = self.generate_random_corners(B, device)
        
        # Expand src_corners for batch
        src_corners_batch = self.src_corners.unsqueeze(0).expand(B, 4, 2).to(device)
        
        # 2. Compute Homography and Warp
        # Homography from flat document to perturbed background corners
        H_mat = kornia.geometry.transform.get_perspective_transform(src_corners_batch, corners_scaled)
        
        # Warp clean image
        warped_clean = kornia.geometry.transform.warp_perspective(clean_batch, H_mat, dsize=(H, W))
        
        # 3. Create Mask and Blend
        ones = torch.ones_like(clean_batch)
        mask = kornia.geometry.transform.warp_perspective(ones, H_mat, dsize=(H, W))
        
        degraded = warped_clean * mask + bg_batch * (1 - mask)
        
        # 4. Shadow Generation (Vectorized)
        shadows = torch.ones_like(degraded)
        if torch.rand(1).item() > 0.5:
            # Random horizontal shadow cutoff (lighter)
            intensity = torch.empty(B, 1, 1, 1, device=device).uniform_(0.6, 0.95)
            start_idx = torch.randint(0, W // 2, (B,), device=device)
            for b in range(B):
                shadows[b, :, :, start_idx[b]:] *= intensity[b].item()
        else:
            # Soft linear gradient (lighter)
            gradient = torch.linspace(1.0, 0.6, W, device=device).unsqueeze(0).unsqueeze(0).expand(B, 3, H, W)
            shadows *= gradient
        
        degraded = degraded * shadows
        
        # 5. Apply Kornia Augmentations (Color, Blur) - Toned down
        degraded = self.color_jitter(degraded)
        degraded = self.blur(degraded)
        
        # 6. Noise - Toned down
        noise = torch.randn_like(degraded) * 0.01
        degraded = torch.clamp(degraded + noise, 0.0, 1.0)
        
        if task == 'corner':
            return degraded, corners_normalized
            
        elif task == 'enhancement':
            # Rectify degraded image back to flat rectangle using inverse homography
            H_inv = kornia.geometry.transform.get_perspective_transform(corners_scaled, src_corners_batch)
            rectified_degraded = kornia.geometry.transform.warp_perspective(degraded, H_inv, dsize=(H, W))
            
            # Interpolate down to target_size to save VRAM and feed model
            import torch.nn.functional as F
            rect_down = F.interpolate(rectified_degraded, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            clean_down = F.interpolate(clean_batch, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            
            return rect_down, clean_down
        
        else:
            raise ValueError("Task must be 'corner' or 'enhancement'")
