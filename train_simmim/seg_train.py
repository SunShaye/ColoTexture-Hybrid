import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation

from unet import EfficientNetB1UNet, DomainAdaptiveNormalization

def replace_bn_with_dan(model):
    replaced = 0
    bn_layers = []
    for name, module in model.named_modules():
        if name.startswith('encoder'):
            continue
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    
    for name, bn in bn_layers:
        dan = DomainAdaptiveNormalization(bn.num_features, bn.eps, bn.momentum)
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], dan)
        replaced += 1
    return replaced


DATA_ROOT = '/home/linux/Dataset/MixDB/ProcessedDataset'
PRETRAINED_PATH = 'retrain/best_colo_model.pth'
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
MIN_LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
FREEZE_ENCODER_EPOCHS = 0
NUM_WORKERS = 24
SAVE_DIR = 'seg_results/colo'
IMAGE_SIZE = 384
EARLY_STOP_PATIENCE = 20
TRAIN_RATIO = 1


os.makedirs(SAVE_DIR, exist_ok=True)


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, enhance=False, ratio=1.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.enhance = enhance

        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        n_samples = int(len(self.image_paths) * ratio)
        self.image_paths = self.image_paths[:n_samples]

        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def _clahe_enhance(self, image):
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_array)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        image = Image.open(img_path).convert('RGB')
        mask_path = None
        for ext in ['.png', '.jpg', '.bmp', '.jpeg', '.PNG', '.JPG', '.BMP', '.JPEG']:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, 0)

        if self.enhance:
            image = self._clahe_enhance(image)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            if torch.rand(1).item() > 0.5:
                angle = torch.randint(-20, 21, (1,)).item()
                image = image.rotate(angle, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, fillcolor=0)

            if torch.rand(1).item() > 0.6:
                radius = torch.randint(1, 5, (1,)).item()
                image = image.filter(ImageFilter.GaussianBlur(radius))

            if torch.rand(1).item() > 0.6:
                img_array = np.array(image).astype(np.float32)
                for c in range(3):
                    shift = np.random.uniform(-20, 20)
                    img_array[:, :, c] = np.clip(img_array[:, :, c] + shift, 0, 255)
                image = Image.fromarray(img_array.astype(np.uint8))

            if torch.rand(1).item() > 0.6:
                factor = np.random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)

            if torch.rand(1).item() > 0.6:
                factor = np.random.uniform(0.5, 1.5)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(factor)

            if torch.rand(1).item() > 0.7:
                img_array = np.array(image).astype(np.float32)
                hsv_shift = np.random.uniform(-0.05, 0.05)
                img_array = np.clip(img_array * (1.0 + hsv_shift), 0, 255)
                image = Image.fromarray(img_array.astype(np.uint8))

            if torch.rand(1).item() > 0.7:
                gamma = np.random.uniform(0.6, 1.6)
                img_array = np.array(image).astype(np.float32) / 255.0
                img_array = np.clip(np.power(img_array, gamma) * 255.0, 0, 255)
                image = Image.fromarray(img_array.astype(np.uint8))

            if torch.rand(1).item() > 0.7:
                factor = np.random.uniform(0.6, 1.4)
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(factor)

            if torch.rand(1).item() > 0.8:
                scale = np.random.uniform(0.85, 1.15)
                new_w = int(image.size[0] * scale)
                new_h = int(image.size[1] * scale)
                image = image.resize((new_w, new_h), Image.BILINEAR)
                mask = mask.resize((new_w, new_h), Image.NEAREST)
                if scale < 1.0:
                    pad_w = (image.size[0] - new_w) // 2
                    pad_h = (image.size[1] - new_h) // 2
                    padded_img = Image.new('RGB', image.size, (0, 0, 0))
                    padded_mask = Image.new('L', image.size, 0)
                    padded_img.paste(image, (pad_w, pad_h))
                    padded_mask.paste(mask, (pad_w, pad_h))
                    image, mask = padded_img, padded_mask
                else:
                    image = image.crop((0, 0, image.size[0], image.size[1]))
                    mask = mask.crop((0, 0, image.size[0], image.size[1]))

        image = self.resize(image)
        mask = self.resize(mask)
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum(dim=(2, 3))
    fp = (pred * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - pred) * target).sum(dim=(2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def boundary_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)

    def sobel_kernels():
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        device = pred.device
        return sobel_x.to(device), sobel_y.to(device)

    kx, ky = sobel_kernels()
    pred_edges = []
    target_edges = []

    for i in range(pred.shape[0]):
        px = F.conv2d(pred[i:i+1], kx, padding=1)
        py = F.conv2d(pred[i:i+1], ky, padding=1)
        pred_edges.append(torch.sqrt(px ** 2 + py ** 2 + 1e-8))

        tx = F.conv2d(target[i:i+1], kx, padding=1)
        ty = F.conv2d(target[i:i+1], ky, padding=1)
        target_edges.append(torch.sqrt(tx ** 2 + ty ** 2 + 1e-8))

    pe = torch.cat(pred_edges, dim=0)
    te = torch.cat(target_edges, dim=0)

    pe_norm = pe / (pe.max() + 1e-8)
    te_norm = te / (te.max() + 1e-8)

    intersection = (pe_norm * te_norm).sum(dim=(2, 3))
    union = pe_norm.sum(dim=(2, 3)) + te_norm.sum(dim=(2, 3))
    b_dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - b_dice.mean()


def edge_supervision_loss(edge_pred, mask):
    mask_np = mask.cpu().numpy()
    edge_gts = []
    for i in range(mask_np.shape[0]):
        m = mask_np[i, 0].astype(bool)
        if m.sum() > 0 and m.sum() < m.size:
            edge = (m ^ binary_erosion(m)).astype(np.float32)
        else:
            edge = np.zeros_like(m, dtype=np.float32)
        edge_gts.append(edge)
    edge_gt = torch.from_numpy(np.stack(edge_gts)).unsqueeze(1).to(mask.device)
    return F.binary_cross_entropy_with_logits(edge_pred, edge_gt, reduction='mean')


def combined_loss(pred, target, epoch, total_epochs):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    tv = tversky_loss(pred, target, alpha=0.3, beta=0.7)

    progress = epoch / total_epochs
    boundary_weight = 0.3 + 0.7 * min(progress, 0.6) / 0.6
    bd = boundary_loss(pred, target)

    return bce + tv + boundary_weight * bd


def deep_supervision_loss(main_pred, edge_pred, d4_pred, d3_pred, target, epoch, total_epochs):
    main = combined_loss(main_pred, target, epoch, total_epochs)
    edge = edge_supervision_loss(edge_pred, target)
    aux4 = combined_loss(d4_pred, target, epoch, total_epochs)
    aux3 = combined_loss(d3_pred, target, epoch, total_epochs)
    return main + 0.5 * edge + 0.3 * aux4 + 0.1 * aux3


def compute_metrics(pred_mask, gt_mask):
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    tp = (pred_mask & gt_mask).sum().float()
    fp = (pred_mask & ~gt_mask).sum().float()
    fn = (~pred_mask & gt_mask).sum().float()

    smooth = 1e-7
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f2 = (5.0 * precision * recall) / (4.0 * precision + recall + smooth)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f2': f2.item(),
    }


def compute_hd95(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 384.0

    pred_boundary = pred_mask ^ binary_erosion(pred_mask)
    gt_boundary = gt_mask ^ binary_erosion(gt_mask)

    pred_dt = distance_transform_edt(~gt_boundary)
    gt_dt = distance_transform_edt(~pred_boundary)

    d_pred_to_gt = pred_dt[pred_boundary]
    d_gt_to_pred = gt_dt[gt_boundary]

    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return 0.0

    return max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    use_deep_sup = epoch < 100

    progress_bar = tqdm(dataloader, desc='Training', leave=False, ncols=None,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    optimizer.zero_grad()
    for step, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            outputs = model(images)
            main_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            if use_deep_sup and isinstance(outputs, tuple) and len(outputs) == 4:
                _, edge_pred, d4_pred, d3_pred = outputs
                loss = deep_supervision_loss(main_pred, edge_pred, d4_pred, d3_pred, masks, epoch, total_epochs)
            else:
                loss = combined_loss(main_pred, masks, epoch, total_epochs)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f2': []}

    for images, masks in tqdm(dataloader, desc='Validating', leave=False, ncols=None):
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            pred = model(images)

        pred_binary = (torch.sigmoid(pred) > 0.5).float()

        for i in range(pred_binary.size(0)):
            m = compute_metrics(pred_binary[i], masks[i])
            for k, v in m.items():
                all_metrics[k].append(v)

    return {k: np.mean(v) for k, v in all_metrics.items()}


@torch.no_grad()
def evaluate_test(model, dataloader, device):
    model.eval()
    all_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f2': [], 'hd95': []}

    for images, masks in tqdm(dataloader, desc='Testing', leave=False, ncols=None):
        images = images.to(device)
        masks_np = masks.numpy()

        with autocast('cuda'):
            pred = sahi_predict(model, images, scales=[0.75, 1.0, 1.25])

        pred_binary = (pred > 0.5).cpu().float()

        for i in range(pred_binary.size(0)):
            m = compute_metrics(pred_binary[i], masks[i])
            for k, v in m.items():
                all_metrics[k].append(v)
            all_metrics['hd95'].append(compute_hd95(pred_binary[i, 0].numpy(), masks_np[i, 0]))

    return {k: np.mean(v) for k, v in all_metrics.items()}


def sahi_predict(model, images, crop_size=384, overlap=96, scales=[1.0]):
    B, C, H, W = images.shape
    
    if len(scales) == 1 and scales[0] == 1.0 and H <= crop_size and W <= crop_size:
        return torch.sigmoid(model(images))

    final_pred = torch.zeros(B, 1, H, W, device=images.device)
    
    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            scaled_images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            scaled_images = images
            new_h, new_w = H, W

        sB, sC, sH, sW = scaled_images.shape

        if sH <= crop_size and sW <= crop_size:
            with autocast('cuda'):
                p = torch.sigmoid(model(scaled_images))
            if scale != 1.0:
                p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            final_pred += p
            continue

        stride = crop_size - overlap
        n_h = max(1, (sH - overlap) // stride + 1)
        n_w = max(1, (sW - overlap) // stride + 1)

        weight_map = torch.zeros(B, 1, sH, sW, device=images.device)
        pred_map = torch.zeros(B, 1, sH, sW, device=images.device)

        for i in range(n_h):
            for j in range(n_w):
                y0 = min(i * stride, max(0, sH - crop_size))
                x0 = min(j * stride, max(0, sW - crop_size))
                y1 = min(y0 + crop_size, sH)
                x1 = min(x0 + crop_size, sW)

                crop = scaled_images[:, :, y0:y1, x0:x1]
                if crop.shape[2] < crop_size or crop.shape[3] < crop_size:
                    padded = torch.zeros(B, C, crop_size, crop_size, device=images.device)
                    padded[:, :, :crop.shape[2], :crop.shape[3]] = crop
                    crop = padded

                with autocast('cuda'):
                    p = torch.sigmoid(model(crop))

                ph, pw = min(crop_size, y1 - y0), min(crop_size, x1 - x0)
                p = p[:, :, :ph, :pw]

                actual_h = min(y0 + ph, sH) - y0
                actual_w = min(x0 + pw, sW) - x0

                cy = min(overlap // 2, actual_h // 2)
                cx = min(overlap // 2, actual_w // 2)

                ry0 = 0 if i == 0 else cy
                ry1 = actual_h if i == n_h - 1 else actual_h - cy
                rx0 = 0 if j == 0 else cx
                rx1 = actual_w if j == n_w - 1 else actual_w - cx

                w = torch.ones(1, 1, ry1 - ry0, rx1 - rx0, device=images.device)
                pred_map[:, :, y0 + ry0:y0 + ry1, x0 + rx0:x0 + rx1] += p[:, :, ry0:ry1, rx0:rx1] * w
                weight_map[:, :, y0 + ry0:y0 + ry1, x0 + rx0:x0 + rx1] += w

        pred_map = pred_map / (weight_map + 1e-8)
        
        if scale != 1.0:
            pred_map = F.interpolate(pred_map, size=(H, W), mode='bilinear', align_corners=False)
        
        final_pred += pred_map

    final_pred = final_pred / len(scales)
    return final_pred


def plot_curves(train_losses, val_dices, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(val_dices) + 1), val_dices, 'r-', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice', fontsize=12)
    ax2.set_title('Validation Dice (single-pass, no TTA)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_img_dir = os.path.join(DATA_ROOT, 'train', 'images')
    train_mask_dir = os.path.join(DATA_ROOT, 'train', 'masks')
    val_img_dir = os.path.join(DATA_ROOT, 'val', 'images')
    val_mask_dir = os.path.join(DATA_ROOT, 'val', 'masks')

    train_dataset = SegDataset(train_img_dir, train_mask_dir, augment=True, enhance=True, ratio=TRAIN_RATIO)
    val_dataset = SegDataset(val_img_dir, val_mask_dir, augment=False, enhance=True, ratio=TRAIN_RATIO)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\nBuilding UNet (EGA + CSEE + CA + SA + ASPP + MSA + STE + EdgeSup + DAN)...")
    model = EfficientNetB1UNet(pretrained_path=PRETRAINED_PATH)
    n_dan = replace_bn_with_dan(model)
    model = model.to(device)
    print(f"  Replaced {n_dan} BN layers with DomainAdaptiveNormalization in decoder")

    total_p = sum(p.numel() for p in model.parameters())
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Total: {total_p:,}  Encoder: {enc_p:,}  Decoder: {total_p - enc_p:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    class WarmupCosineScheduler(LRScheduler):
        def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.min_lr = min_lr
            self.base_lr = base_lr
            super().__init__(optimizer)

        def get_lr(self):
            epoch = self.last_epoch
            if epoch < self.warmup_epochs:
                factor = 0.01 + 0.99 * epoch / self.warmup_epochs
            else:
                cos_epoch = epoch - self.warmup_epochs
                total_cos = self.total_epochs - self.warmup_epochs
                factor = self.min_lr / self.base_lr + (1.0 - self.min_lr / self.base_lr) * 0.5 * (1.0 + np.cos(np.pi * cos_epoch / total_cos))
            return [base_lr * factor for base_lr in self.base_lrs]

    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, LR)
    scaler = GradScaler('cuda')

    train_losses = []
    val_dices = []
    best_dice = 0.0
    early_stop_counter = 0
    best_epoch = 0

    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Batch: {BATCH_SIZE}  Warmup: {WARMUP_EPOCHS}  Freeze enc: {FREEZE_ENCODER_EPOCHS}")
    print(f"Loss: BCE + Tversky(0.3/0.7) + boundary(scheduled) + edge_sup(0.5)")
    print(f"Validation: single-pass, Test: SAHI + multi-scale(0.75/1.0/1.25) + CLAHE")
    print(f"Early stop patience: {EARLY_STOP_PATIENCE} epochs")

    for epoch in range(EPOCHS):
        if epoch < FREEZE_ENCODER_EPOCHS:
            for p in model.encoder.parameters():
                p.requires_grad = False
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True

        epoch_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, EPOCHS)
        train_losses.append(epoch_loss)

        val_metrics = validate(model, val_loader, device)
        val_dices.append(val_metrics['dice'])

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        enc_status = "FROZEN" if epoch < FREEZE_ENCODER_EPOCHS else "UNFROZEN"
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}  "
              f"Val Dice: {val_metrics['dice']:.4f} IoU: {val_metrics['iou']:.4f}  "
              f"LR: {lr:.6f} {enc_status}")

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'dice': best_dice},
                       os.path.join(SAVE_DIR, 'best_unet.pth'))
            print(f"  -> Best model saved (Val Dice: {best_dice:.4f}, single-pass)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}: no improvement for {EARLY_STOP_PATIENCE} epochs")
                print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch+1}")
                break

    plot_curves(train_losses, val_dices, SAVE_DIR)
    print(f"\nCurves saved to {SAVE_DIR}/training_curves.png")

    print(f"\n{'='*60}")
    print("Testing on 5 datasets (SAHI + multi-scale + CLAHE)")
    print(f"{'='*60}")

    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_unet.pth'), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best from epoch {checkpoint['epoch']+1} (Val single-pass Dice: {checkpoint['dice']:.4f})")

    test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    all_results = {}

    for ds_name in test_datasets:
        test_img_dir = os.path.join(DATA_ROOT, 'test', ds_name, 'images')
        test_mask_dir = os.path.join(DATA_ROOT, 'test', ds_name, 'masks')
        if not os.path.exists(test_img_dir):
            print(f"  Skip {ds_name}: not found")
            continue

        test_dataset = SegDataset(test_img_dir, test_mask_dir, augment=False, enhance=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
        print(f"\nTesting {ds_name} ({len(test_dataset)} images)...")
        m = evaluate_test(model, test_loader, device)
        all_results[ds_name] = m
        print(f"  Dice: {m['dice']:.4f}  IoU: {m['iou']:.4f}  Prec: {m['precision']:.4f}  "
              f"Recall: {m['recall']:.4f}  F2: {m['f2']:.4f}  HD95: {m['hd95']:.2f}")

    header = f"{'Dataset':<20} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F2':>8} {'HD95':>8}"
    sep = "-" * 68
    print(f"\n{header}\n{sep}")
    lines = [header, sep]
    for ds, m in all_results.items():
        row = f"{ds:<20} {m['dice']:>8.4f} {m['iou']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f2']:>8.4f} {m['hd95']:>8.2f}"
        print(row)
        lines.append(row)
    if all_results:
        avg = {k: np.mean([r[k] for r in all_results.values()]) for k in all_results[list(all_results.keys())[0]]}
        avg_row = f"{'Average':<20} {avg['dice']:>8.4f} {avg['iou']:>8.4f} {avg['precision']:>8.4f} {avg['recall']:>8.4f} {avg['f2']:>8.4f} {avg['hd95']:>8.2f}"
        print(sep)
        print(avg_row)
        lines.append(sep)
        lines.append(avg_row)

    with open(os.path.join(SAVE_DIR, 'test_results.txt'), 'w') as f:
        f.write("EfficientNet-B1 UNet Results (EGA + CSEE + CA + SA + MSA + STE + EdgeSup + DAN)\n")
        f.write(f"Pretrained: {PRETRAINED_PATH}\n")
        f.write(f"Best epoch: {checkpoint['epoch']+1}  Val single-pass Dice: {checkpoint['dice']:.4f}\n")
        f.write("Loss: BCE + Tversky(0.3/0.7) + boundary(scheduled) + edge_sup(0.5)\n")
        f.write("Test: SAHI + multi-scale(0.75/1.0/1.25) + CLAHE\n\n")
        for line in lines:
            f.write(line + '\n')

    print(f"\nDone. Results saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
