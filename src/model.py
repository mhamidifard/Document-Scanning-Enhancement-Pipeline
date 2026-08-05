import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [InstanceNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_sigmoid=True):
        super().__init__()
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder (Upsampling)
        # Using Bilinear upsampling + 1x1 Conv2d 
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv4 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_up4 = DoubleConv(128, 64)
        
        # Final Output Layer
        if use_sigmoid:
            self.outc = nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.Sigmoid()  # Bounds output exactly between [0, 1] for image tasks
            )
        else:
            self.outc = nn.Conv2d(64, out_channels, kernel_size=1) # Raw logits for heatmaps

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.down4(x4)
        
        # Decoder with Skip Connections
        # Block 1
        x = self.up1(x5)
        x = self.up_conv1(x)
        x = torch.cat([x4, x], dim=1) # Concatenate along channel dimension
        x = self.conv_up1(x)
        
        # Block 2
        x = self.up2(x)
        x = self.up_conv2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)
        
        # Block 3
        x = self.up3(x)
        x = self.up_conv3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)
        
        # Block 4
        x = self.up4(x)
        x = self.up_conv4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)
        
        # Output
        out = self.outc(x)
        return out

# Aliases for clarity in other scripts
EnhancementUNet = UNet

class CornerHeatmapNet(UNet):
    def __init__(self, in_channels=3, out_channels=4):
        # Heatmaps suffer from vanishing gradients with MSE if a Sigmoid is used.
        # We output raw logits and supervise directly with the Gaussian heatmaps.
        super().__init__(in_channels=in_channels, out_channels=out_channels, use_sigmoid=False)

class CornerRegressionNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder (Downsampling to 1/32 of 256x256 = 8x8)
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2), # 128
            DoubleConv(64, 128),
            nn.MaxPool2d(2), # 64
            DoubleConv(128, 256),
            nn.MaxPool2d(2), # 32
            DoubleConv(256, 512),
            nn.MaxPool2d(2), # 16
            DoubleConv(512, 512),
            nn.MaxPool2d(2)  # 8
        )
        
        # Fully Connected Layers
        # Assuming input size of 256x256, the final feature map is 8x8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8),
            nn.Sigmoid() # Coordinates are normalized [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.fc(x)
        return out

if __name__ == "__main__":
    # Quick verification block
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    enhancement_model = EnhancementUNet(in_channels=3, out_channels=3).to(device)
    heatmap_model = CornerHeatmapNet(in_channels=3, out_channels=4).to(device)
    regression_model = CornerRegressionNet(in_channels=3).to(device)
    
    # Create dummy input tensor (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Min value: {output.min().item():.4f}, Max value: {output.max().item():.4f}")
    
    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
