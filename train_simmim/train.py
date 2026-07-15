"""
SimMIM Self-Supervised Pre-training Script for EfficientNet-B1
===============================================================
Framework: SimMIM (Simple Masked Image Modeling)
Backbone: EfficientNet-B1
Dataset: ColoTexture-Hybrid (colonoscopy images)

Usage:
    python train.py --dataset_root /path/to/dataset --mask_ratio 0.6 --epochs 300
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import PolypDataset
from model import EfficientNetB1SIMMIM, compute_masked_l1_loss


# ---- Configuration ----
DATA_DIR = './dataset'          # Path to ColoTexture-Hybrid dataset
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 15
NUM_WORKERS = 24
SAVE_PATH = '../pretrained_weights/effb1_simmim_colo.pth'
VIS_DIR = 'visualizations'

os.makedirs(VIS_DIR, exist_ok=True)


class WarmupCosineLRScheduler(LRScheduler):
    """Cosine annealing LR scheduler with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start=0.01, min_lr=1e-6, base_lr=1e-4):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start = warmup_start
        self.min_lr = min_lr
        self.base_lr = base_lr
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = self.warmup_start + (1.0 - self.warmup_start) * epoch / self.warmup_epochs
        else:
            cos_epoch = epoch - self.warmup_epochs
            total_cos = self.total_epochs - self.warmup_epochs
            factor = 0.5 * (1.0 + np.cos(np.pi * cos_epoch / total_cos))
            factor = max(factor, 0)
            factor = self.min_lr / self.base_lr + (1.0 - self.min_lr / self.base_lr) * factor
        return [base_lr * factor for base_lr in self.base_lrs]


def save_batch_visualization(epoch, batch_idx, original, masked, reconstructed, loss):
    """Save a visualization of original, masked, and reconstructed images."""
    vis_path = os.path.join(VIS_DIR, f'epoch{epoch+1}_batch{batch_idx+1}.png')
    
    rand_idx = torch.randint(0, original.size(0), (1,)).item()
    
    with torch.no_grad():
        orig_np = original[rand_idx].cpu().permute(1, 2, 0).numpy()
        masked_np = masked[rand_idx].cpu().permute(1, 2, 0).numpy()
        recon_np = reconstructed[rand_idx].cpu().permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(orig_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(masked_np)
    axes[1].set_title('Masked Image (60%)')
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(recon_np, 0, 1))
    axes[2].set_title(f'Reconstructed (Loss: {loss:.4f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    last_batch_data = None
    
    progress_bar = tqdm(
        dataloader,
        desc='Training',
        leave=False,
        ncols=None,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    for batch_idx, images in enumerate(progress_bar):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            reconstructed, masks, original, masked_images = model(images)
            loss = compute_masked_l1_loss(reconstructed, original, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        current_avg_loss = total_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        progress_bar.set_postfix({'loss': f'{current_avg_loss:.4f}', 'lr': f'{current_lr:.6f}'})
        
        last_batch_data = (original.detach(), masked_images.detach(), reconstructed.detach(), loss.item())
    
    save_batch_visualization(epoch, 0, last_batch_data[0], last_batch_data[1], last_batch_data[2], last_batch_data[3])
    
    return total_loss / num_batches


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    dataset = PolypDataset(DATA_DIR)
    print(f'Dataset size: {len(dataset)} images')
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    model = EfficientNetB1SIMMIM().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scaler = GradScaler('cuda')
    
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        warmup_start=0.01,
        min_lr=MIN_LR,
        base_lr=LR
    )
    
    best_loss = float('inf')
    
    print(f'Starting training for {EPOCHS} epochs...')
    print(f'Warmup epochs: {WARMUP_EPOCHS}, Base LR: {LR}, Min LR: {MIN_LR}')
    print(f'Visualization directory: {VIS_DIR}/')
    
    for epoch in range(EPOCHS):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, scaler, device, epoch)
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} LR: {current_lr:.6f}')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, SAVE_PATH)
            print(f'  -> Best model saved with loss: {best_loss:.4f}')
    
    print(f'\nTraining completed. Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    train()
