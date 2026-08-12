import torch
import torch.nn as nn
import kornia
import torch.nn.functional as F

class BaseDegradationPipeline(nn.Module):
    def __init__(self, target_size=768, canvas_size=1536, max_perturb=0.20, margin=0.15):
        super().__init__()
        self.target_size = target_size
        self.canvas_size = canvas_size
        self.max_perturb = max_perturb
        self.margin = margin
        
        # Define base corner regions based on document rules (normalized 0 to 1)
        self.regions = torch.tensor([
            [margin, margin, 0.5 - margin, 0.5 - margin], # TL
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

    def generate_random_corners(self, batch_size, device):
        corners = torch.zeros((batch_size, 4, 2), device=device)
        for i in range(4):
            x_min, y_min, x_max, y_max = self.regions[i]
            corners[:, i, 0] = torch.empty(batch_size, device=device).uniform_(x_min, x_max)
            corners[:, i, 1] = torch.empty(batch_size, device=device).uniform_(y_min, y_max)
        corners_scaled = corners * self.canvas_size
        return corners_scaled, corners


class EnhancementDegradationPipeline(BaseDegradationPipeline):
    def __init__(self, target_size=768, canvas_size=1536, max_perturb=0.20, margin=0.15):
        super().__init__(target_size, canvas_size, max_perturb, margin)
        # Kornia GPU Augmentations - Increased intensity for synthetic-to-real gap
        self.color_jitter = kornia.augmentation.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.9)
        self.blur = kornia.augmentation.RandomGaussianBlur((5, 5), (0.1, 1.5), p=0.5)

    def forward(self, clean_batch, bg_batch):
        B, C, H, W = clean_batch.shape
        device = clean_batch.device
        
        corners_scaled, _ = self.generate_random_corners(B, device)
        src_corners_batch = self.src_corners.unsqueeze(0).expand(B, 4, 2).to(device)
        
        H_mat = kornia.geometry.transform.get_perspective_transform(src_corners_batch, corners_scaled)
        warped_clean = kornia.geometry.transform.warp_perspective(clean_batch, H_mat, dsize=(H, W))
        
        ones = torch.ones_like(clean_batch)
        mask = kornia.geometry.transform.warp_perspective(ones, H_mat, dsize=(H, W))
        degraded = warped_clean * mask + bg_batch * (1 - mask)
        
        # Shadow Generation (Brutal)
        shadows = torch.ones_like(degraded)
        num_shadows = torch.randint(1, 4, (1,)).item()
        
        for _ in range(num_shadows):
            rand_val = torch.rand(1).item()
            if rand_val < 0.33:
                intensity = torch.empty(B, 1, 1, 1, device=device).uniform_(0.1, 0.6)
                start_idx = torch.randint(0, W // 2, (B,), device=device)
                for b in range(B):
                    if torch.rand(1).item() > 0.5:
                        shadows[b, :, :, start_idx[b]:] *= intensity[b].item()
                    else:
                        shadows[b, :, :, :start_idx[b]] *= intensity[b].item()
            elif rand_val < 0.66:
                gradient = torch.linspace(1.0, torch.empty(1).uniform_(0.1, 0.5).item(), W, device=device)
                if torch.rand(1).item() > 0.5:
                    gradient = torch.flip(gradient, dims=[0])
                gradient = gradient.unsqueeze(0).unsqueeze(0).expand(B, 3, H, W)
                shadows *= gradient
            else:
                y = torch.linspace(-1, 1, H, device=device)
                x = torch.linspace(-1, 1, W, device=device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                for b in range(B):
                    cx = torch.empty(1, device=device).uniform_(-0.8, 0.8).item()
                    cy = torch.empty(1, device=device).uniform_(-0.8, 0.8).item()
                    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
                    radius = torch.empty(1, device=device).uniform_(0.5, 1.5).item()
                    intensity = torch.empty(1, device=device).uniform_(0.1, 0.5).item()
                    blob = torch.clamp((dist / radius), 0, 1)
                    blob = intensity + (1.0 - intensity) * blob
                    shadows[b] *= blob.unsqueeze(0)
                    
        degraded = degraded * shadows
        degraded = self.color_jitter(degraded)
        degraded = self.blur(degraded)
        
        noise_level = torch.empty(B, 1, 1, 1, device=device).uniform_(0.01, 0.08)
        noise = torch.randn_like(degraded) * noise_level
        degraded = torch.clamp(degraded + noise, 0.0, 1.0)
        
        H_inv = kornia.geometry.transform.get_perspective_transform(corners_scaled, src_corners_batch)
        rectified_degraded = kornia.geometry.transform.warp_perspective(degraded, H_inv, dsize=(H, W))
        
        rect_down = F.interpolate(rectified_degraded, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        clean_down = F.interpolate(clean_batch, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        return rect_down, clean_down


class CornerDegradationPipeline(BaseDegradationPipeline):
    def __init__(self, target_size=256, canvas_size=1536, max_perturb=0.20, margin=0.15):
        super().__init__(target_size, canvas_size, max_perturb, margin)
        # Moderate augmentations for corner detection
        self.color_jitter = kornia.augmentation.ColorJitter(0.15, 0.15, 0.15, 0.05, p=0.8)
        self.blur = kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.3)

    def draw_distractors(self, clean_batch):
        """Draws synthetic thumbs, tabs, and spiral bindings to prevent over-reliance on local edges."""
        B, C, H, W = clean_batch.shape
        device = clean_batch.device
        distracted_batch = clean_batch.clone()
        
        for b in range(B):
            # 50% chance to add distractors
            if torch.rand(1).item() < 0.5:
                num_shapes = torch.randint(2, 8, (1,)).item()
                for _ in range(num_shapes):
                    shape_type = torch.randint(0, 2, (1,)).item() # 0 for rect, 1 for circle
                    # Random color
                    color = torch.rand(3, device=device).view(3, 1, 1)
                    
                    # Random position mostly along edges
                    edge = torch.randint(0, 4, (1,)).item()
                    w_size = torch.randint(W // 40, W // 10, (1,)).item()
                    h_size = torch.randint(H // 40, H // 10, (1,)).item()
                    
                    if edge == 0: # Left
                        x_min, x_max = 0, w_size
                        y_min = torch.randint(0, H - h_size, (1,)).item()
                    elif edge == 1: # Right
                        x_min, x_max = W - w_size, W
                        y_min = torch.randint(0, H - h_size, (1,)).item()
                    elif edge == 2: # Top
                        y_min, y_max = 0, h_size
                        x_min = torch.randint(0, W - w_size, (1,)).item()
                    else: # Bottom
                        y_min, y_max = H - h_size, H
                        x_min = torch.randint(0, W - w_size, (1,)).item()
                        
                    y_max = y_min + h_size
                    x_max = x_min + w_size
                    
                    if shape_type == 0:
                        distracted_batch[b, :, y_min:y_max, x_min:x_max] = color
                    else:
                        # Draw circle using meshgrid
                        y = torch.arange(y_min, y_max, device=device)
                        x = torch.arange(x_min, x_max, device=device)
                        yy, xx = torch.meshgrid(y, x, indexing='ij')
                        cy = y_min + h_size / 2
                        cx = x_min + w_size / 2
                        radius = min(h_size, w_size) / 2
                        dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
                        mask = (dist <= radius).float().unsqueeze(0)
                        distracted_batch[b, :, y_min:y_max, x_min:x_max] = \
                            distracted_batch[b, :, y_min:y_max, x_min:x_max] * (1 - mask) + color * mask
                            
        return distracted_batch

    def forward(self, clean_batch, bg_batch, task='corner'):
        B, C, H, W = clean_batch.shape
        device = clean_batch.device
        
        # 1. Distractors (Tabs, Spiral holes, Thumbs)
        clean_batch = self.draw_distractors(clean_batch)
        
        corners_scaled, corners_normalized = self.generate_random_corners(B, device)
        src_corners_batch = self.src_corners.unsqueeze(0).expand(B, 4, 2).to(device)
        
        H_mat = kornia.geometry.transform.get_perspective_transform(src_corners_batch, corners_scaled)
        warped_clean = kornia.geometry.transform.warp_perspective(clean_batch, H_mat, dsize=(H, W))
        
        ones = torch.ones_like(clean_batch)
        mask = kornia.geometry.transform.warp_perspective(ones, H_mat, dsize=(H, W))
        degraded = warped_clean * mask + bg_batch * (1 - mask)
        
        # Smart Shadows (Lighter than enhancement, but present to teach shadow ignoring)
        shadows = torch.ones_like(degraded)
        
        for _ in range(1): # Just 1 shadow
            rand_val = torch.rand(1).item()
            if rand_val < 0.33:
                intensity = torch.empty(B, 1, 1, 1, device=device).uniform_(0.3, 0.7)
                start_idx = torch.randint(0, W // 2, (B,), device=device)
                for b in range(B):
                    if torch.rand(1).item() > 0.5:
                        shadows[b, :, :, start_idx[b]:] *= intensity[b].item()
                    else:
                        shadows[b, :, :, :start_idx[b]] *= intensity[b].item()
            elif rand_val < 0.66:
                gradient = torch.linspace(1.0, torch.empty(1).uniform_(0.3, 0.7).item(), W, device=device)
                if torch.rand(1).item() > 0.5:
                    gradient = torch.flip(gradient, dims=[0])
                gradient = gradient.unsqueeze(0).unsqueeze(0).expand(B, 3, H, W)
                shadows *= gradient
            else:
                y = torch.linspace(-1, 1, H, device=device)
                x = torch.linspace(-1, 1, W, device=device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                for b in range(B):
                    cx = torch.empty(1, device=device).uniform_(-0.8, 0.8).item()
                    cy = torch.empty(1, device=device).uniform_(-0.8, 0.8).item()
                    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
                    radius = torch.empty(1, device=device).uniform_(0.5, 1.5).item()
                    intensity = torch.empty(1, device=device).uniform_(0.3, 0.7).item()
                    blob = torch.clamp((dist / radius), 0, 1)
                    blob = intensity + (1.0 - intensity) * blob
                    shadows[b] *= blob.unsqueeze(0)
                    
        degraded = degraded * shadows
        degraded = self.color_jitter(degraded)
        degraded = self.blur(degraded)
        
        noise_level = torch.empty(B, 1, 1, 1, device=device).uniform_(0.01, 0.05)
        noise = torch.randn_like(degraded) * noise_level
        degraded = torch.clamp(degraded + noise, 0.0, 1.0)
        
        deg_down = F.interpolate(degraded, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        
        if task == 'corner':
            return deg_down, corners_normalized
        elif task == 'regression':
            return deg_down, corners_normalized.view(B, 8)
        elif task == 'heatmap':
            heatmaps = torch.zeros((B, 4, self.target_size, self.target_size), device=device)
            y = torch.linspace(0, self.target_size - 1, self.target_size, device=device)
            x = torch.linspace(0, self.target_size - 1, self.target_size, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            sigma = 0.05 * self.target_size
            for b in range(B):
                for i in range(4):
                    cx = corners_normalized[b, i, 0] * self.target_size
                    cy = corners_normalized[b, i, 1] * self.target_size
                    dist_sq = (xx - cx)**2 + (yy - cy)**2
                    heatmaps[b, i] = torch.exp(-dist_sq / (2 * sigma**2))
            return deg_down, heatmaps
        else:
            raise ValueError("Task must be 'corner', 'regression', or 'heatmap'")
