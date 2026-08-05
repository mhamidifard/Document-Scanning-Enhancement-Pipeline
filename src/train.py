import os
import sys
import glob
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import DocumentDataset
from src.model import EnhancementUNet
from src.pipeline import GPUDegradationPipeline

# ==========================================
# 1. Custom Loss Functions
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        filter_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3, 1, 3, 3)
        filter_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3, 1, 3, 3)
        self.register_buffer('filter_x', filter_x)
        self.register_buffer('filter_y', filter_y)
        
    def forward(self, pred, target):
        pred_x = F.conv2d(pred, self.filter_x, padding=1, groups=3)
        pred_y = F.conv2d(pred, self.filter_y, padding=1, groups=3)
        target_x = F.conv2d(target, self.filter_x, padding=1, groups=3)
        target_y = F.conv2d(target, self.filter_y, padding=1, groups=3)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)

class EnhancementLoss(nn.Module):
    def __init__(self, w_l1=0.4, w_ssim=0.3, w_sobel=0.3):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_sobel = w_sobel
        self.ssim_loss = SSIM()
        self.sobel_loss = SobelLoss()
        
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = 1 - self.ssim_loss(pred, target)
        sobel = self.sobel_loss(pred, target)
        return self.w_l1 * l1 + self.w_ssim * ssim + self.w_sobel * sobel

# ==========================================
# 2. Main Training Loop
# ==========================================
def plot_losses(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=768)
    parser.add_argument('--canvas_size', type=int, default=1536)
    args = parser.parse_args()

    # Create directories
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Dataset Preparation (Fixed Seed for Deterministic Splits)
    clean_paths = sorted(glob.glob(os.path.join(base_dir, "data", "clean_scans", "*.*")))
    bg_paths = sorted(glob.glob(os.path.join(base_dir, "data", "backgrounds", "*.*")))
    
    if not clean_paths or not bg_paths:
        print("WARNING: Data directories are empty. Please ensure data is loaded for training.")
        return

    random.seed(42)
    random.shuffle(clean_paths)
    
    train_split = int(0.8 * len(clean_paths))
    val_split = int(0.9 * len(clean_paths))
    
    train_scans = clean_paths[:train_split]
    val_scans = clean_paths[train_split:val_split]
    
    print(f"Dataset split: {len(train_scans)} train, {len(val_scans)} val")
    
    # Create datasets (pass canvas_size)
    train_dataset = DocumentDataset(clean_paths[:train_split], bg_paths, split='train', canvas_size=args.canvas_size)
    val_dataset = DocumentDataset(clean_paths[train_split:val_split], bg_paths, split='val', canvas_size=args.canvas_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model, GPU Pipeline, Loss, Optimizer Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 2. Setup GPU Pipeline
    gpu_pipeline = GPUDegradationPipeline(target_size=args.img_size, canvas_size=args.canvas_size).to(device)
    model = EnhancementUNet().to(device)
    criterion = EnhancementLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Checkpointing System
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    latest_ckpt_path = os.path.join(ckpt_dir, "latest_enhancement.pth")
    best_ckpt_path = os.path.join(ckpt_dir, "best_enhancement.pth")
    
    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        # ----------------- Train -----------------
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (clean_batch, bg_batch) in enumerate(train_loader):
            # Move raw tensors to GPU
            clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
            
            # Run the augmentation pipeline fully on the GPU!
            with torch.no_grad():
                inputs, targets = gpu_pipeline(clean_batch, bg_batch, task='enhancement')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # ----------------- Validate -----------------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for clean_batch, bg_batch in val_loader:
                clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
                
                inputs, targets = gpu_pipeline(clean_batch, bg_batch, task='enhancement')
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"=== Epoch {epoch+1} Summary === | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Save Checkpoint after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, latest_ckpt_path)
        
        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}! Saving best model...")
            torch.save(model.state_dict(), best_ckpt_path)
            
        # Update Loss Curve Plot
        plot_losses(train_losses, val_losses, os.path.join(ckpt_dir, "loss_curve_enhancement.png"))

if __name__ == "__main__":
    main()
