import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List

def apply_jpeg_compression(img, quality_range=(30, 90), rng=None):
    """Applies random JPEG compression since it cannot easily be done natively on GPU."""
    if rng is None:
        rng = np.random.RandomState()
    quality = rng.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

class DocumentDataset(Dataset):
    def __init__(self, clean_scans: List[str], backgrounds: List[str], split: str = 'train', target_size: int = 1024):
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
        if self.split in ['val', 'test']:
            rng = np.random.RandomState(seed=idx)
        else:
            rng = np.random.RandomState()

        clean_path = self.clean_scans[idx % len(self.clean_scans)]
        bg_path = self.backgrounds[rng.randint(0, len(self.backgrounds))]
        
        # Load and convert to RGB
        clean_img = cv2.imread(clean_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        
        bg_img = cv2.imread(bg_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        
        # Resize on CPU first to save GPU memory and bandwidth
        clean_img = cv2.resize(clean_img, (self.target_size, self.target_size))
        bg_img = cv2.resize(bg_img, (self.target_size, self.target_size))

        # Apply JPEG compression on CPU
        # We apply it to clean_img here (which simulates compressing the final document)
        if rng.rand() > 0.5:
            clean_img = apply_jpeg_compression(clean_img, rng=rng)
        
        # Convert to Tensor (HWC -> CHW) and normalize [0, 1]
        clean_tensor = torch.from_numpy(clean_img.transpose((2, 0, 1))).float() / 255.0
        bg_tensor = torch.from_numpy(bg_img.transpose((2, 0, 1))).float() / 255.0
        
        return clean_tensor, bg_tensor
