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
        
        # Kornia GPU Augmentations - Increased intensity for synthetic-to-real gap
        self.color_jitter = kornia.augmentation.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.9)
        self.blur = kornia.augmentation.RandomGaussianBlur((5, 5), (0.1, 1.5), p=0.5)

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
        
        # 4. Shadow Generation (Vectorized, Diverse & Overlapping)
        shadows = torch.ones_like(degraded)
        
        # Apply 1 to 3 overlapping shadows to simulate complex real-world lighting
        num_shadows = torch.randint(1, 4, (1,)).item()
        
        for _ in range(num_shadows):
            rand_val = torch.rand(1).item()
            
            if rand_val < 0.33:
                # 1. Harsh horizontal/diagonal cutoff (like a desk edge or paper curl)
                # Allow much darker shadows (down to 0.1)
                intensity = torch.empty(B, 1, 1, 1, device=device).uniform_(0.1, 0.6)
                start_idx = torch.randint(0, W // 2, (B,), device=device)
                
                # Randomly decide if shadow is on the left or right
                for b in range(B):
                    if torch.rand(1).item() > 0.5:
                        shadows[b, :, :, start_idx[b]:] *= intensity[b].item()
                    else:
                        shadows[b, :, :, :start_idx[b]] *= intensity[b].item()
                    
            elif rand_val < 0.66:
                # 2. Soft linear gradient (general uneven lighting)
                # Allow gradient to go very dark (0.1 to 0.5)
                gradient = torch.linspace(1.0, torch.empty(1).uniform_(0.1, 0.5).item(), W, device=device)
                if torch.rand(1).item() > 0.5:
                    gradient = torch.flip(gradient, dims=[0])
                gradient = gradient.unsqueeze(0).unsqueeze(0).expand(B, 3, H, W)
                shadows *= gradient
                
            else:
                # 3. Radial/Blob shadow (simulating a hand or phone casting a shadow)
                y = torch.linspace(-1, 1, H, device=device)
                x = torch.linspace(-1, 1, W, device=device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                for b in range(B):
                    cx = torch.empty(1, device=device).uniform_(-0.8, 0.8).item()
                    cy = torch.empty(1, device=device).uniform_(-0.8, 0.8).item()
                    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
                    
                    radius = torch.empty(1, device=device).uniform_(0.5, 1.5).item()
                    # Darker blob shadows
                    intensity = torch.empty(1, device=device).uniform_(0.1, 0.5).item()
                    
                    blob = torch.clamp((dist / radius), 0, 1)
                    blob = intensity + (1.0 - intensity) * blob
                    shadows[b] *= blob.unsqueeze(0)
        
        degraded = degraded * shadows
        
        # 5. Apply Kornia Augmentations (Color, Blur) - Toned down
        degraded = self.color_jitter(degraded)
        degraded = self.blur(degraded)
        
        # 6. Noise (Simulate variable ISO camera noise)
        # Random noise intensity between 0.01 (clean) and 0.08 (very grainy)
        noise_level = torch.empty(B, 1, 1, 1, device=device).uniform_(0.01, 0.08)
        noise = torch.randn_like(degraded) * noise_level
        degraded = torch.clamp(degraded + noise, 0.0, 1.0)
        
        if task == 'corner':
            import torch.nn.functional as F
            deg_down = F.interpolate(degraded, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            # Used for evaluation/visualization, returns normalized coordinates
            return deg_down, corners_normalized
            
        elif task == 'regression':
            import torch.nn.functional as F
            deg_down = F.interpolate(degraded, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            # Flatten to shape (B, 8) for the regression network target
            return deg_down, corners_normalized.view(B, 8)
            
        elif task == 'heatmap':
            # Generate 4 Gaussian heatmaps (one for each corner)
            heatmaps = torch.zeros((B, 4, self.target_size, self.target_size), device=device)
            y = torch.linspace(0, self.target_size - 1, self.target_size, device=device)
            x = torch.linspace(0, self.target_size - 1, self.target_size, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            
            sigma = 0.05 * self.target_size # Standard deviation of Gaussian blob
            
            for b in range(B):
                for i in range(4):
                    # Scale normalized coordinates to target_size (e.g. 256)
                    cx = corners_normalized[b, i, 0] * self.target_size
                    cy = corners_normalized[b, i, 1] * self.target_size
                    
                    dist_sq = (xx - cx)**2 + (yy - cy)**2
                    heatmaps[b, i] = torch.exp(-dist_sq / (2 * sigma**2))
                    
            # We don't need reverse homography for corner detection targets!
            # Resize degraded image down to target_size to feed model
            import torch.nn.functional as F
            deg_down = F.interpolate(degraded, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            
            return deg_down, heatmaps
            
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
            raise ValueError("Task must be 'corner', 'regression', 'heatmap', or 'enhancement'")
