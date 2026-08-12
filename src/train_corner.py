import os
import sys
import glob
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Dynamically setup sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import DocumentDataset
from src.model import CornerRegressionNet, CornerHeatmapNet
from src.pipeline import GPUDegradationPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, required=True, choices=['regression', 'heatmap'], help="Which architecture to train")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16) # Smaller image size = bigger batch size!
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--canvas_size', type=int, default=1536)
    parser.add_argument('--use_dropout', action='store_true', help='Enable Dropout layers')
    args = parser.parse_args()

    # Create directories
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_corner_{args.approach}.pth")

    latest_ckpt_path = os.path.join(ckpt_dir, f"latest_corner_{args.approach}.pth")
    plot_path = os.path.join(ckpt_dir, f"loss_curve_{args.approach}.png")
    
    # Setup data...
    clean_paths = sorted(glob.glob(os.path.join(base_dir, "data", "clean_scans", "*.*")))
    bg_paths = sorted(glob.glob(os.path.join(base_dir, "data", "backgrounds", "*.*")))
    
    random.seed(42)
    random.shuffle(clean_paths)
    
    train_split = int(0.8 * len(clean_paths))
    val_split = int(0.9 * len(clean_paths))
    
    train_dataset = DocumentDataset(clean_paths[:train_split], bg_paths, split='train', canvas_size=args.canvas_size)
    val_dataset = DocumentDataset(clean_paths[train_split:val_split], bg_paths, split='val', canvas_size=args.canvas_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    gpu_pipeline = GPUDegradationPipeline(target_size=args.img_size, canvas_size=args.canvas_size).to(device)
    
    if args.approach == 'regression':
        model = CornerRegressionNet(in_channels=3, use_dropout=args.use_dropout).to(device)
        criterion = nn.L1Loss()
    else:
        model = CornerHeatmapNet(in_channels=3, out_channels=4, use_dropout=args.use_dropout).to(device)
        criterion = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Automatic Resume
    if os.path.exists(latest_ckpt_path):
        print(f"Found latest checkpoint! Resuming from {latest_ckpt_path}...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
    
    print(f"Starting {args.approach.upper()} Corner Detection Training...")
    
    import matplotlib.pyplot as plt

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        for clean_batch, bg_batch in train_loader:
            clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
            
            inputs, targets = gpu_pipeline(clean_batch, bg_batch, task=args.approach)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for clean_batch, bg_batch in val_loader:
                clean_batch, bg_batch = clean_batch.to(device), bg_batch.to(device)
                inputs, targets = gpu_pipeline(clean_batch, bg_batch, task=args.approach)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best model to {ckpt_path}")
            
        # Save Latest (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, latest_ckpt_path)
        
        # Plot and save loss curve
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Corner Detection ({args.approach}) Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

if __name__ == "__main__":
    main()
