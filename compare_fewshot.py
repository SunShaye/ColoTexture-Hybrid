"""compare_fewshot.py -- comparison of two EfficientNet-B1 backbones under few-shot (training-ratio) settings.

Backbone A : efficientnet_b1_imagenet.pth                     official ImageNet pretrained (baseline)
Backbone B : backbones/ladder/ColoTexture-Hybrid-SIMMIM.pth   weights pretrained on the ColoTexture-Hybrid
                                                              dataset with SIMMIM

Model  : EfficientNetB1UNetStd from unet.py in the project root (standard vanilla U-Net, no extra modules)
Data   : /home/linux/Dataset/MixDB/ProcessedDataset
        validation set fixed at 290 images (val)
        training set sampled from train(1160) at ratios: 100% / 50% / 20% / 10%
        both backbones use exactly the same network, data split, augmentation, loss, optimizer and evaluation protocol

Important : This script **does not use a test set**. For fairness, the final evaluation is
            performed on a fixed number (290) of validation images (the same validation set
            used for early-stopping selection during training).

Flow  :
  outer loop over backbones; middle loop over training-set ratios; inner loop runs 5 full
  "train + validate" cycles (seeds 42..46)
  each run is evaluated on the fixed 290-image validation set and per-image metrics are recorded
  after all ratios for one backbone finish, the summary (mean/std etc. over 5 runs per ratio)
  is written to a txt file in the project root
  after both backbones finish, a per-ratio comparison summary is written to the project root

Outputs  :
  compare_fewshot/<backbone_name>/ratio_XXX/run0X_seedYY/
      train.log              full training log (all stdout redirected)
      epoch_metrics.csv      detailed per-epoch train/val results
      training_curves.png    training curves (loss / val dice / val iou / lr)
      best_unet.pth          best validation weights
      run_config.json        full hyperparameters of this run
      val_per_image.csv      per-image evaluation metrics (split, image, dice, iou, precision, recall, f2, hd95)
      val_per_group.csv      statistics grouped by split
      val_overall.csv        merged overall statistics
      run_summary.txt        human-readable summary of this run
  compare_fewshot/<backbone_name>/runs_summary.json           machine-readable summary of all runs
  compare_fewshot_<backbone_name>.txt                         detailed per-ratio x 5-run summary (project root)
  compare_fewshot_comparison.txt                              per-ratio comparison of both backbones (project root)

Usage  :
  python compare_fewshot.py                              # default: 4 ratios x 5 runs x 2 backbones
  python compare_fewshot.py --ratios 1.0 0.1 --seeds 42 43
"""

import os
import sys
import csv
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy import stats as sp_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unet import EfficientNetB1UNetStd


# ======================================================================
# Configuration
# ======================================================================
DATA_ROOT = '/home/linux/Dataset/MixDB/ProcessedDataset'
METRIC_NAMES = ['dice', 'iou', 'precision', 'recall', 'f2', 'hd95']

# (display name, weight path); the order is the execution order
BACKBONES = [
    ('ImageNet', 'efficientnet_b1_imagenet.pth'),
    ('ColoTexture-Hybrid-SIMMIM',
     'backbones/ladder/ColoTexture-Hybrid-SIMMIM.pth'),
]

DEFAULT_TRAIN_RATIOS = [1.0, 0.5, 0.2, 0.1]   # fraction of the training set used
DEFAULT_SEEDS = [42, 43, 44, 45, 46]          # 5 full train+validate runs per ratio
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP = 5
DEFAULT_WORKERS = 16
IMAGE_SIZE = 384
THRESHOLD = 0.5
OUT_ROOT = os.path.join(PROJECT_ROOT, 'compare_fewshot')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


# ======================================================================
# Logging: write stdout/stderr to a file at the same time
# ======================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
# Data
# ======================================================================
class SegDataset(Dataset):
    """Read image / mask pairs and resize both to IMAGE_SIZE."""

    def __init__(self, image_dir, mask_dir, augment=False):
        self.augment = augment
        self.image_paths = []
        self.mask_paths = []
        for f in sorted(os.listdir(image_dir)) if os.path.isdir(image_dir) else []:
            if not f.lower().endswith(IMAGE_EXTS):
                continue
            base = os.path.splitext(f)[0]
            mask_path = None
            for ext in IMAGE_EXTS:
                c = os.path.join(mask_dir, base + ext)
                if os.path.exists(c):
                    mask_path = c
                    break
            if mask_path is None:
                continue
            self.image_paths.append(os.path.join(image_dir, f))
            self.mask_paths.append(mask_path)
        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                ang = random.uniform(-20, 20)
                image = image.rotate(ang, fillcolor=(0, 0, 0))
                mask = mask.rotate(ang, fillcolor=0)
            if random.random() < 0.4:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.7, 1.3))
            if random.random() < 0.4:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(
                    radius=random.uniform(0.5, 1.5)))
            if random.random() < 0.3:
                arr = np.array(image).astype(np.float32)
                for c in range(3):
                    arr[:, :, c] = np.clip(
                        arr[:, :, c] + random.uniform(-20, 20), 0, 255)
                image = Image.fromarray(arr.astype(np.uint8))

        image = self.resize(image)
        mask = self.resize(mask)
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask


# ======================================================================
# Metrics
# ======================================================================
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
    return {'dice': dice.item(), 'iou': iou.item(),
            'precision': precision.item(), 'recall': recall.item(),
            'f2': f2.item()}


def compute_hd95(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float(IMAGE_SIZE)
    pred_b = pred_mask ^ binary_erosion(pred_mask)
    gt_b = gt_mask ^ binary_erosion(gt_mask)
    pred_dt = distance_transform_edt(~gt_b)
    gt_dt = distance_transform_edt(~pred_b)
    d_pred_to_gt = pred_dt[pred_b]
    d_gt_to_pred = gt_dt[gt_b]
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return 0.0
    return float(max(np.percentile(d_pred_to_gt, 95),
                     np.percentile(d_gt_to_pred, 95)))


def stats_of(values):
    """Descriptive statistics of a list of scalars."""
    keys = ['n', 'mean', 'std', 'median', 'min', 'max', 'q1', 'q3', 'iqr',
            'sem', 'ci95_low', 'ci95_high']
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float('nan') for k in keys}
    n = v.size
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(sp_stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    q1, q3 = np.percentile(v, [25, 75])
    return {
        'n': int(n), 'mean': mean, 'std': std, 'median': float(np.median(v)),
        'min': float(v.min()), 'max': float(v.max()),
        'q1': float(q1), 'q3': float(q3), 'iqr': float(q3 - q1),
        'sem': sem, 'ci95_low': mean - tcrit * sem,
        'ci95_high': mean + tcrit * sem,
    }


# ======================================================================
# Loss / Optimization
# ======================================================================
class WarmupCosine(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        super().__init__(optimizer)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            f = 0.01 + 0.99 * e / max(1, self.warmup_epochs)
        else:
            ce = e - self.warmup_epochs
            tc = max(1, self.total_epochs - self.warmup_epochs)
            f = self.min_lr / self.base_lr + \
                (1.0 - self.min_lr / self.base_lr) * \
                0.5 * (1.0 + np.cos(np.pi * ce / tc))
        return [self.base_lr * f for _ in self.base_lrs]


def dice_loss(logits, target, smooth=1.0):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + smooth
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
    return 1.0 - (num / den).mean()


def criterion(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + \
        dice_loss(logits, target)


# ======================================================================
# Train / Validate / Evaluate
# ======================================================================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc='  train', leave=False, ncols=None)
    for images, masks in bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
        bar.set_postfix(loss=f'{total / n:.4f}')
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    acc = {k: [] for k in ['dice', 'iou', 'precision', 'recall', 'f2']}
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(images)
        pred = (torch.sigmoid(logits) > THRESHOLD).float().cpu()
        for i in range(pred.size(0)):
            m = compute_metrics(pred[i], masks[i])
            for k in acc:
                acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def evaluate_split(model, dataset, split_name, device, batch_size, workers):
    """Evaluate a given dataset image by image and return the record list (with image names).

    This script does not use a test set: split_name is always 'val' (the fixed 290-image validation set).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model.eval()
    records = []
    pos = 0
    for images, masks in tqdm(loader, desc=f'  eval {split_name}', leave=False,
                              ncols=None):
        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(images)
        pred = (torch.sigmoid(logits) > THRESHOLD).float().cpu()
        masks_np = masks.numpy()
        for i in range(pred.size(0)):
            m = compute_metrics(pred[i], masks[i])
            m['hd95'] = compute_hd95(pred[i, 0].numpy(), masks_np[i, 0])
            m['split'] = split_name
            m['image'] = os.path.splitext(
                os.path.basename(dataset.image_paths[pos + i]))[0]
            records.append(m)
        pos += pred.size(0)
    print(f'  [eval] {split_name}: {len(dataset)} images done')
    return records


# ======================================================================
# Persisting results
# ======================================================================
def records_to_group_df(records, group_key):
    """Group by group_key and compute descriptive statistics for every metric."""
    groups = {}
    for r in records:
        groups.setdefault(r[group_key], []).append(r)
    rows = []
    for g, rs in sorted(groups.items()):
        row = {group_key: g, 'n_images': len(rs)}
        for m in METRIC_NAMES:
            s = stats_of([x[m] for x in rs])
            for sk, sv in s.items():
                row[f'{m}_{sk}'] = sv
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        with open(path, 'w') as f:
            f.write('')
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_curves(epoch_rows, path, title):
    ep = [r['epoch'] for r in epoch_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(ep, [r['train_loss'] for r in epoch_rows], 'b-')
    axes[0, 0].set_title('Training Loss'); axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(ep, [r['val_dice'] for r in epoch_rows], 'r-')
    axes[0, 1].set_title('Validation Dice'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(ep, [r['val_iou'] for r in epoch_rows], 'g-')
    axes[1, 0].set_title('Validation IoU'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(ep, [r['lr'] for r in epoch_rows], 'k-')
    axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fmt_row(label, stats, width=14):
    return (f'{label:<10} mean={stats["mean"]:>{width}.4f} '
            f'std={stats["std"]:.4f} median={stats["median"]:.4f} '
            f'min={stats["min"]:.4f} max={stats["max"]:.4f} '
            f'CI95=[{stats["ci95_low"]:.4f},{stats["ci95_high"]:.4f}]')


def write_run_summary_txt(path, cfg, epoch_rows, records, per_group_rows):
    lines = []
    P = lines.append
    P('=' * 90)
    P(f'  Single-run summary  {cfg["backbone_name"]} / train ratio '
      f'{int(round(cfg["ratio"] * 100))}% / seed={cfg["seed"]}')
    P('=' * 90)
    P(f'Backbone weights: {cfg["backbone_path"]}')
    P(f'Model           : {cfg["model"]}')
    P(f'Data root       : {DATA_ROOT}')
    P(f'Train set       : {cfg["n_train"]} images '
      f'(full {cfg["n_train_full"]} images x {int(round(cfg["ratio"] * 100))}%)')
    P(f'Eval set        : {cfg["n_eval"]} images (fixed validation set, no test set)')
    P(f'Hyperparameters : epochs={cfg["epochs"]} patience={cfg["patience"]} '
      f'batch={cfg["batch_size"]} lr={cfg["lr"]} wd={cfg["weight_decay"]} '
      f'warmup={cfg["warmup"]} seed={cfg["seed"]} img={IMAGE_SIZE} '
      f'threshold={THRESHOLD}')
    P('')
    P('-' * 90)
    P('Training process')
    P('-' * 90)
    best = max(epoch_rows, key=lambda r: r['val_dice']) if epoch_rows else None
    P(f'Epochs actually trained: {len(epoch_rows)}')
    if best:
        P(f'Best epoch        : {best["epoch"]}  val_dice={best["val_dice"]:.4f} '
          f'val_iou={best["val_iou"]:.4f}')
    if epoch_rows:
        P(f'Last epoch        : loss={epoch_rows[-1]["train_loss"]:.4f} '
          f'val_dice={epoch_rows[-1]["val_dice"]:.4f}')
    P('Per-epoch metrics: epoch_metrics.csv')
    P('Training curves  : training_curves.png')
    P('')
    P('-' * 90)
    P(f'Evaluation results (fixed validation set, {cfg["n_eval"]} images)')
    P('-' * 90)
    header = f'{"Split":<20}' + ''.join(f'{m:>18}' for m in METRIC_NAMES)
    P(header)
    P('-' * len(header))
    for row in per_group_rows:
        s = f'{row["split"]:<20}'
        for m in METRIC_NAMES:
            s += f'{row[f"{m}_mean"]:>10.4f}±{row[f"{m}_std"]:>6.4f}'
        P(s)
    P('')
    P('-' * 90)
    P('Evaluation results (merged)')
    P('-' * 90)
    for m in METRIC_NAMES:
        s = stats_of([r[m] for r in records])
        P(fmt_row(m.upper(), s))
    P(f'Total images      : {len(records)}')
    P('Per-image metrics: val_per_image.csv')
    P('Per-split stats  : val_per_group.csv')
    P('Merged stats     : val_overall.csv')
    P('=' * 90)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_model_summary_txt(path, backbone_name, backbone_path, ratio_runs, args):
    """Detailed summary for one backbone after all ratios x 5 runs (written to the project root)."""
    lines = []
    P = lines.append
    ratios = sorted(ratio_runs.keys())

    P('=' * 100)
    P(f'  Backbone few-shot comparison summary: {backbone_name}')
    P('=' * 100)
    P(f'Backbone weights: {backbone_path}')
    P(f'Model           : project_root/unet.py :: EfficientNetB1UNetStd (standard vanilla U-Net)')
    P(f'Data root       : {DATA_ROOT}')
    P(f'Eval set        : fixed validation set, no test set')
    P(f'Train ratios    : {[f"{int(round(r * 100))}%" for r in ratios]}')
    P(f'Runs per ratio  : {len(args.seeds)}  (seeds = {args.seeds})')
    P(f'Hyperparameters : epochs={args.epochs} patience={args.patience} '
      f'batch={args.batch_size} lr={args.lr} min_lr={args.min_lr} '
      f'wd={args.weight_decay} warmup={args.warmup} img={IMAGE_SIZE} '
      f'threshold={THRESHOLD}')
    P('')

    # ---- Overview: ratio x metric ----------------------------------
    P('-' * 100)
    P('1. Per-ratio overview (mean of the 5 runs per ratio)')
    P('-' * 100)
    head = (f'{"ratio":<8}{"n_train":<9}{"best_ep":<9}{"val_dice":<10}'
            + ''.join(f'{m:>12}' for m in METRIC_NAMES))
    P(head)
    P('-' * len(head))
    for r in ratios:
        runs = ratio_runs[r]
        n_train = runs[0]['n_train']
        be = np.mean([x['best_epoch'] for x in runs])
        means = {m: float(np.mean([x['overall']['mean'][m] for x in runs]))
                 for m in METRIC_NAMES}
        P(f'{int(round(r * 100)):>3}%{"":<4}{n_train:<9}{be:<9.1f}'
          f'{means["dice"]:<10.4f}'
          + ''.join(f'{means[m]:>12.4f}' for m in METRIC_NAMES))
    P('')

    # ---- Per ratio x per run ----------------------------------------
    P('-' * 100)
    P('2. Per ratio x per run detailed metrics (evaluation set merged)')
    P('-' * 100)
    head = (f'{"ratio":<8}{"run":<5}{"seed":<6}{"best_ep":<9}{"val_dice":<10}'
            + ''.join(f'{m:>12}' for m in METRIC_NAMES))
    P(head)
    P('-' * len(head))
    for r in ratios:
        for x in ratio_runs[r]:
            ov = x['overall']['mean']
            P(f'{int(round(r * 100)):>3}%{"":<4}{x["run_index"]:<5}{x["seed"]:<6}'
              f'{x["best_epoch"]:<9}{x["best_val_dice"]:<10.4f}'
              + ''.join(f'{ov[m]:>12.4f}' for m in METRIC_NAMES))
    P('')

    # ---- Per-ratio aggregation --------------------------------------
    P('-' * 100)
    P('3. Per-ratio aggregate statistics (mean/std/min/max/95%CI over the 5 runs)')
    P('-' * 100)
    for r in ratios:
        runs = ratio_runs[r]
        P(f'\n<Train ratio {int(round(r * 100))}%  {runs[0]["n_train"]} training images, '
          f'{len(runs)} runs>')
        h = (f'  {"Metric":<12}{"runs":>6}{"mean":>12}{"std":>12}'
             f'{"min":>12}{"max":>12}{"median":>12}{"CI95":>26}')
        P(h)
        P('  ' + '-' * (len(h) - 2))
        for m in METRIC_NAMES:
            s = stats_of([x['overall']['mean'][m] for x in runs])
            ci = f'[{s["ci95_low"]:.4f},{s["ci95_high"]:.4f}]'
            P(f'  {m.upper():<12}{s["n"]:>6}{s["mean"]:>12.4f}{s["std"]:>12.4f}'
              f'{s["min"]:>12.4f}{s["max"]:>12.4f}{s["median"]:>12.4f}{ci:>26}')

    # ---- pooled --------------------------------------------------
    P('')
    P('-' * 100)
    P('4. All per-image results from every run merged (pooled over all runs & ratios & images)')
    P('-' * 100)
    all_records = [rec for r in ratios for x in ratio_runs[r]
                   for rec in x['records']]
    h = f'  {"Metric":<12}{"n":>8}' + ''.join(
        f'{k:>12}' for k in ['mean', 'std', 'median', 'min', 'max', 'q1', 'q3', 'iqr'])
    P(h)
    P('  ' + '-' * (len(h) - 2))
    for m in METRIC_NAMES:
        s = stats_of([x[m] for x in all_records])
        P(f'  {m.upper():<12}{s["n"]:>8}{s["mean"]:>12.4f}{s["std"]:>12.4f}'
          f'{s["median"]:>12.4f}{s["min"]:>12.4f}{s["max"]:>12.4f}'
          f'{s["q1"]:>12.4f}{s["q3"]:>12.4f}{s["iqr"]:>12.4f}')
    P(f'\n  total per-image records: {len(all_records)}')
    P('')
    P('-' * 100)
    P('5. Artifact list')
    P('-' * 100)
    P(f'  compare_fewshot/{backbone_name}/ratio_XXX/runYY_seedZZ/  '
      f'per-run train.log / epoch_metrics.csv / training_curves.png / '
      f'best_unet.pth / run_config.json / val_per_image.csv / '
      f'val_per_group.csv / val_overall.csv / run_summary.txt')
    P(f'  {os.path.basename(path)}                             this summary file')
    P('=' * 100)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_comparison_txt(path, results, args):
    """Compare the two backbones across training-set ratios."""
    lines = []
    P = lines.append
    P('=' * 100)
    P('  Backbone few-shot comparison summary: ImageNet  vs  ColoTexture-Hybrid-SIMMIM')
    P('=' * 100)
    P(f'Model    : project_root/unet.py :: EfficientNetB1UNetStd (standard vanilla U-Net)')
    P(f'Data     : {DATA_ROOT}  (evaluation set = fixed validation set, no test set)')
    P(f'Ratios   : {[f"{int(round(r * 100))}%" for r in args.ratios]}')
    P(f'Per ratio: {len(args.seeds)} runs  seeds={args.seeds}')
    P('')
    names = list(results.keys())
    ratios = sorted(args.ratios)

    for m in METRIC_NAMES:
        P('-' * 100)
        extra = ' (lower is better)' if m == 'hd95' else ' (higher is better)'
        P(f'[{m.upper()}] mean±std over 5 runs{extra}')
        P('-' * 100)
        head = f'{"ratio":<8}{"n_train":<9}' + ''.join(f'{n:>34}' for n in names)
        if len(names) == 2:
            head += f'{"Δ(B-A)":>12}'
        P(head)
        P('-' * len(head))
        for r in ratios:
            vals = {}
            cells = ''
            n_train = 0
            for n in names:
                runs = results[n]['ratios'][r]
                n_train = runs[0]['n_train']
                s = stats_of([x['overall']['mean'][m] for x in runs])
                vals[n] = s['mean']
                cells += f'{s["mean"]:>18.4f}±{s["std"]:<8.4f}'
            row = f'{int(round(r * 100)):>3}%{"":<4}{n_train:<9}' + cells
            if len(names) == 2:
                row += f'{vals[names[1]] - vals[names[0]]:>+12.4f}'
            P(row)
        P('')
    P('=' * 100)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ======================================================================
# Single run
# ======================================================================
def run_once(backbone_name, backbone_path, ratio, seed, run_dir, args):
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'train.log')
    log_f = open(log_path, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(old_stdout, log_f)
    sys.stderr = Tee(old_stderr, log_f)

    try:
        print('=' * 90)
        print(f'  RUN START  {backbone_name}  ratio={int(round(ratio * 100))}%  '
              f'seed={seed}')
        print(f'  device={DEVICE}')
        print('=' * 90)

        set_seed(seed)

        # ---- Data: training set subsampled by ratio, validation set fixed ----
        train_full = SegDataset(os.path.join(DATA_ROOT, 'train', 'images'),
                                os.path.join(DATA_ROOT, 'train', 'masks'),
                                augment=True)
        val_ds = SegDataset(os.path.join(DATA_ROOT, 'val', 'images'),
                            os.path.join(DATA_ROOT, 'val', 'masks'),
                            augment=False)
        n_keep = max(1, int(round(len(train_full) * ratio)))
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.permutation(len(train_full))[:n_keep])
        train_ds = Subset(train_full, idx)
        print(f'[data] train={len(train_ds)}/{len(train_full)} '
              f'(ratio={int(round(ratio * 100))}%)  eval(val)={len(val_ds)}')

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=True, drop_last=True,
                                  generator=g, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True, persistent_workers=True)

        # ---- Model: identical unet, only the backbone initialisation differs ----
        backbone = os.path.join(PROJECT_ROOT, backbone_path) \
            if not os.path.isabs(backbone_path) else backbone_path
        print(f'[model] EfficientNetB1UNetStd  backbone={backbone}')
        model = EfficientNetB1UNetStd(backbone_path=backbone).to(DEVICE)
        n_all = sum(p.numel() for p in model.parameters())
        n_enc = sum(p.numel() for p in model.encoder.parameters())
        print(f'[model] params total={n_all:,} encoder={n_enc:,}')

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = WarmupCosine(optimizer, args.warmup, args.epochs,
                                 args.min_lr, args.lr)
        scaler = GradScaler('cuda')

        # ---- Training ----
        epoch_rows = []
        best_dice, best_epoch, bad = -1.0, 0, 0
        ckpt_path = os.path.join(run_dir, 'best_unet.pth')
        epoch_csv = os.path.join(run_dir, 'epoch_metrics.csv')
        csv_f = open(epoch_csv, 'w', newline='')
        csv_w = csv.DictWriter(csv_f, fieldnames=[
            'epoch', 'train_loss', 'val_dice', 'val_iou', 'val_precision',
            'val_recall', 'val_f2', 'lr', 'epoch_seconds'])
        csv_w.writeheader()

        print(f'\n[training] epochs={args.epochs} patience={args.patience} '
              f'batch={args.batch_size} lr={args.lr}')
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         scaler, DEVICE)
            val = validate(model, val_loader, DEVICE)
            lr_now = optimizer.param_groups[0]['lr']
            scheduler.step()
            dt_s = time.time() - t0

            row = {'epoch': epoch, 'train_loss': train_loss,
                   'val_dice': val['dice'], 'val_iou': val['iou'],
                   'val_precision': val['precision'], 'val_recall': val['recall'],
                   'val_f2': val['f2'], 'lr': lr_now,
                   'epoch_seconds': dt_s}
            epoch_rows.append(row)
            csv_w.writerow(row)
            csv_f.flush()

            print(f'[ep {epoch:>3}/{args.epochs}] loss={train_loss:.4f} '
                  f'val_dice={val["dice"]:.4f} val_iou={val["iou"]:.4f} '
                  f'val_prec={val["precision"]:.4f} val_rec={val["recall"]:.4f} '
                  f'val_f2={val["f2"]:.4f} lr={lr_now:.6f} ({dt_s:.1f}s)')

            if val['dice'] > best_dice:
                best_dice, best_epoch, bad = val['dice'], epoch, 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'dice': best_dice, 'backbone': backbone,
                            'seed': seed, 'ratio': ratio}, ckpt_path)
                print(f'    -> best saved (val_dice={best_dice:.4f})')
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'    -> early stop at epoch {epoch} '
                          f'(no improvement for {args.patience} epochs)')
                    break

        csv_f.close()
        save_curves(epoch_rows, os.path.join(run_dir, 'training_curves.png'),
                    f'{backbone_name}  ratio={int(round(ratio * 100))}%  '
                    f'seed={seed}  (best val_dice={best_dice:.4f})')

        # ---- Final evaluation (fixed validation set, no test set) ----
        print(f'\n[eval] load best epoch={best_epoch} val_dice={best_dice:.4f}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        records = evaluate_split(model, val_ds, 'val', DEVICE,
                                 args.batch_size, args.workers)

        # Per-image CSV
        write_csv(os.path.join(run_dir, 'val_per_image.csv'), records,
                  fieldnames=['split', 'image'] + METRIC_NAMES)

        # Statistics per split / merged
        per_group_rows = records_to_group_df(records, 'split')
        overall_rows = records_to_group_df(
            [dict(r, split='ALL') for r in records], 'split')
        write_csv(os.path.join(run_dir, 'val_per_group.csv'), per_group_rows)
        write_csv(os.path.join(run_dir, 'val_overall.csv'), overall_rows)

        per_group_map = {
            row['split']: {
                'n': row['n_images'],
                **{m: {k: row[f'{m}_{k}'] for k in
                       ['n', 'mean', 'std', 'median', 'min', 'max', 'q1',
                        'q3', 'iqr', 'sem', 'ci95_low', 'ci95_high']}
                   for m in METRIC_NAMES},
            } for row in per_group_rows
        }
        overall_map = {
            'n': len(records),
            'mean': {m: float(np.mean([r[m] for r in records])) for m in METRIC_NAMES},
            'std': {m: float(np.std([r[m] for r in records], ddof=1))
                    for m in METRIC_NAMES},
        }

        cfg = {
            'backbone_name': backbone_name, 'backbone_path': backbone,
            'model': 'EfficientNetB1UNetStd (vanilla U-Net)',
            'ratio': ratio, 'seed': seed, 'epochs': args.epochs,
            'patience': args.patience, 'batch_size': args.batch_size,
            'lr': args.lr, 'min_lr': args.min_lr,
            'weight_decay': args.weight_decay, 'warmup': args.warmup,
            'n_train': len(train_ds), 'n_train_full': len(train_full),
            'n_eval': len(val_ds), 'epochs_done': len(epoch_rows),
            'best_epoch': best_epoch, 'best_val_dice': best_dice,
        }
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        write_run_summary_txt(os.path.join(run_dir, 'run_summary.txt'),
                              cfg, epoch_rows, records, per_group_rows)

        run_info = dict(cfg)
        run_info.update({
            'run_index': 0,
            'dir': run_dir,
            'final_train_loss': (epoch_rows[-1]['train_loss']
                                 if epoch_rows else float('nan')),
            'n_eval_images': len(records),
            'per_group': per_group_map,
            'overall': overall_map,
            'records': records,
        })
        print(f'\n[run done] {backbone_name} ratio={int(round(ratio * 100))}% '
              f'seed={seed}  dice={overall_map["mean"]["dice"]:.4f}')
        return run_info, cfg

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


# ======================================================================
# Main flow
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Few-shot training-ratio comparison of two backbone inits.')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR)
    ap.add_argument('--min-lr', type=float, default=DEFAULT_MIN_LR)
    ap.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument('--warmup', type=int, default=DEFAULT_WARMUP)
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--ratios', type=float, nargs='+',
                    default=DEFAULT_TRAIN_RATIOS)
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f'[main] device={DEVICE}  out={OUT_ROOT}')
    print(f'[main] backbones={[b[0] for b in BACKBONES]}')
    print(f'[main] ratios={args.ratios}  seeds={args.seeds}')

    results = {}
    for backbone_name, backbone_path in BACKBONES:
        print('\n' + '#' * 90)
        print(f'#  Backbone: {backbone_name}   ({backbone_path})')
        print('#' * 90)
        bb_dir = os.path.join(OUT_ROOT, backbone_name)
        os.makedirs(bb_dir, exist_ok=True)

        ratio_runs = {}
        for ratio in args.ratios:
            pct = int(round(ratio * 100))
            ratio_dir = os.path.join(bb_dir, f'ratio_{pct:03d}')
            print(f'\n===== Train ratio {pct}%  -> {ratio_dir} =====')
            run_infos = []
            for i, seed in enumerate(args.seeds, start=1):
                run_dir = os.path.join(ratio_dir, f'run{i:02d}_seed{seed}')
                print(f'\n>>> ratio={pct}% run {i}/{len(args.seeds)} '
                      f'seed={seed} -> {run_dir}')
                run_info, _ = run_once(backbone_name, backbone_path, ratio,
                                       seed, run_dir, args)
                run_info['run_index'] = i
                run_infos.append(run_info)
                torch.cuda.empty_cache()
            ratio_runs[ratio] = run_infos

        # ---- All ratios for this backbone finished; write the project-root summary ----
        summary_txt = os.path.join(
            PROJECT_ROOT, f'compare_fewshot_{backbone_name}.txt')
        write_model_summary_txt(summary_txt, backbone_name, backbone_path,
                                ratio_runs, args)
        print(f'\n[summary] {backbone_name}: {summary_txt}')

        with open(os.path.join(bb_dir, 'runs_summary.json'), 'w') as f:
            json.dump({'backbone_name': backbone_name,
                       'backbone_path': backbone_path,
                       'ratios': args.ratios, 'seeds': args.seeds,
                       'runs': {str(r): [{k: v for k, v in x.items()
                                          if k != 'records'}
                                         for x in ratio_runs[r]]
                                for r in ratio_runs}}, f, indent=2,
                      default=float)

        results[backbone_name] = {'path': backbone_path, 'ratios': ratio_runs}

    # ---- Comparison of both backbones ----
    if len(results) >= 2:
        comp = os.path.join(PROJECT_ROOT, 'compare_fewshot_comparison.txt')
        write_comparison_txt(comp, results, args)
        print(f'\n[comparison] {comp}')


if __name__ == '__main__':
    main()
