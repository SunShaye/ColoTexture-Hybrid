"""compare_downstream.py -- Compare the downstream segmentation performance of two
EfficientNet-B1 backbones on the same dataset.

Backbone A : efficientnet_b1_imagenet.pth                     official ImageNet pretrained (baseline)
Backbone B : backbones/ladder/ColoTexture-Hybrid-SIMMIM.pth   weights pretrained on the
                                                              ColoTexture-Hybrid dataset with SIMMIM

Model : EfficientNetB1UNetStd from unet.py in the project root (standard vanilla U-Net, no extra modules)
Data  : /home/linux/Dataset/MixDB/ProcessedDataset
        train(1160) train -> val(290) early-stopping selection -> test(5 datasets, 797 images) test
        Both backbones use exactly the same network, data split, augmentation, loss, optimizer and evaluation protocol

Pipeline :
  For each backbone, repeat the full "train + test" 5 times (seeds 42..46)
  After every run, automatically test on the 5 test sets and record per-image metrics
  After all 5 runs of a backbone finish, write the 5-run summary (mean/std, etc.) to a txt in the project root
  After both backbones finish, write a comparison summary of the two to the project root

Artifacts :
  compare_downstream/<backbone_name>/run0X_seedYY/
      train.log              full training log (all stdout redirected)
      epoch_metrics.csv      detailed train/val results for every epoch
      training_curves.png    training curves (loss / val dice / val iou / lr)
      best_unet.pth          best weights on the validation set
      run_config.json        complete hyperparameters of this run
      test_per_image.csv     per-image test metrics (dataset, image, dice, iou, precision, recall, f2, hd95)
      test_per_dataset.csv   statistics grouped by dataset
      test_overall.csv       pooled statistics over the 5 test sets
      test_summary.txt       human-readable summary of this run
  compare_downstream/<backbone_name>/runs_summary.json        machine-readable summary of the 5 runs
  compare_downstream_<backbone_name>.txt                      detailed summary of the 5 runs (project root)
  compare_downstream_comparison.txt                           comparison summary of the two backbones (project root)

Usage :
  python compare_downstream.py                       # default: 5 runs x 2 backbones
  python compare_downstream.py --epochs 60 --seeds 42 43
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
from torch.utils.data import Dataset, DataLoader
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
TEST_DATASETS = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB',
                 'ETIS-LaribPolypDB', 'Kvasir']
METRIC_NAMES = ['dice', 'iou', 'precision', 'recall', 'f2', 'hd95']
METRIC_DESC = {
    'dice': 'higher is better', 'iou': 'higher is better',
    'precision': 'higher is better',
    'recall': 'higher is better', 'f2': 'higher is better',
    'hd95': 'lower is better (pixels)',
}

# (display name, weight path); the order also defines the execution order
BACKBONES = [
    ('ImageNet', 'efficientnet_b1_imagenet.pth'),
    ('ColoTexture-Hybrid-SIMMIM',
     'backbones/ladder/ColoTexture-Hybrid-SIMMIM.pth'),
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]     # 5 full train+test runs per backbone
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
OUT_ROOT = os.path.join(PROJECT_ROOT, 'compare_downstream')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


# ======================================================================
# Logging: mirror stdout/stderr into a file
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
# Loss / optimization
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
# Training / validation / testing
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
def test_all(model, device, batch_size, workers):
    """Evaluate image by image on the 5 test sets and return a record list (with image names)."""
    model.eval()
    records = []
    for ds_name in TEST_DATASETS:
        img_dir = os.path.join(DATA_ROOT, 'test', ds_name, 'images')
        mask_dir = os.path.join(DATA_ROOT, 'test', ds_name, 'masks')
        if not os.path.isdir(img_dir):
            print(f'  [skip] {ds_name} not found: {img_dir}')
            continue
        dataset = SegDataset(img_dir, mask_dir, augment=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
        pos = 0
        for images, masks in tqdm(loader, desc=f'  test {ds_name}',
                                  leave=False, ncols=None):
            images = images.to(device, non_blocking=True)
            with autocast('cuda'):
                logits = model(images)
            pred = (torch.sigmoid(logits) > THRESHOLD).float().cpu()
            masks_np = masks.numpy()
            for i in range(pred.size(0)):
                m = compute_metrics(pred[i], masks[i])
                m['hd95'] = compute_hd95(pred[i, 0].numpy(), masks_np[i, 0])
                m['dataset'] = ds_name
                m['image'] = os.path.splitext(
                    os.path.basename(dataset.image_paths[pos + i]))[0]
                records.append(m)
            pos += pred.size(0)
        print(f'  [test] {ds_name}: {len(dataset)} images done')
    return records


# ======================================================================
# Writing results to disk
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


def write_run_summary_txt(path, cfg, epoch_rows, records, per_dataset_rows):
    lines = []
    P = lines.append
    P('=' * 90)
    P(f'  Single Run Summary  {cfg["backbone_name"]} / seed={cfg["seed"]}')
    P('=' * 90)
    P(f'Backbone weights : {cfg["backbone_path"]}')
    P(f'Model            : {cfg["model"]}')
    P(f'Data root        : {DATA_ROOT}')
    P(f'Train/Val/Test   : train={cfg["n_train"]} val={cfg["n_val"]} '
      f'test={cfg["n_test"]}')
    P(f'Hyperparameters  : epochs={cfg["epochs"]} patience={cfg["patience"]} '
      f'batch={cfg["batch_size"]} lr={cfg["lr"]} wd={cfg["weight_decay"]} '
      f'warmup={cfg["warmup"]} seed={cfg["seed"]} img={IMAGE_SIZE} '
      f'threshold={THRESHOLD}')
    P('')
    P('-' * 90)
    P('Training Process')
    P('-' * 90)
    best = max(epoch_rows, key=lambda r: r['val_dice']) if epoch_rows else None
    P(f'Epochs actually trained : {len(epoch_rows)}')
    if best:
        P(f'Best epoch              : {best["epoch"]}  val_dice={best["val_dice"]:.4f} '
          f'val_iou={best["val_iou"]:.4f}')
    if epoch_rows:
        P(f'Last epoch              : loss={epoch_rows[-1]["train_loss"]:.4f} '
          f'val_dice={epoch_rows[-1]["val_dice"]:.4f}')
    P('Per-epoch details     : epoch_metrics.csv')
    P('Training curves       : training_curves.png')
    P('')
    P('-' * 90)
    P('Test Results (per dataset)')
    P('-' * 90)
    header = f'{"Dataset":<20}' + ''.join(f'{m:>18}' for m in METRIC_NAMES)
    P(header)
    P('-' * len(header))
    for row in per_dataset_rows:
        s = f'{row["dataset"]:<20}'
        for m in METRIC_NAMES:
            s += f'{row[f"{m}_mean"]:>10.4f}±{row[f"{m}_std"]:>6.4f}'
        P(s)
    P('')
    P('-' * 90)
    P('Test Results (all 5 datasets pooled)')
    P('-' * 90)
    for m in METRIC_NAMES:
        s = stats_of([r[m] for r in records])
        P(fmt_row(m.upper(), s))
    P(f'Total images          : {len(records)}')
    P('Per-image metrics     : test_per_image.csv')
    P('Per-dataset stats     : test_per_dataset.csv')
    P('Pooled stats          : test_overall.csv')
    P('=' * 90)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_backbone_summary_txt(path, backbone_name, backbone_path, run_infos,
                               all_records):
    """Detailed summary after the 5 runs of one backbone (written to the project root)."""
    lines = []
    P = lines.append

    P('=' * 100)
    P(f'  Backbone Summary of 5 Full Training + Test Runs: {backbone_name}')
    P('=' * 100)
    P(f'Backbone weights : {backbone_path}')
    P(f'Model            : project_root/unet.py :: EfficientNetB1UNetStd (standard vanilla U-Net)')
    P(f'Dataset          : {DATA_ROOT}')
    P(f'Test sets        : {", ".join(TEST_DATASETS)}')
    P(f'Number of runs   : {len(run_infos)}  (seeds = {[r["seed"] for r in run_infos]})')
    if run_infos:
        c = run_infos[0]
        P(f'Hyperparameters  : epochs={c["epochs"]} patience={c["patience"]} '
          f'batch={c["batch_size"]} lr={c["lr"]} min_lr={c["min_lr"]} '
          f'wd={c["weight_decay"]} warmup={c["warmup"]} img={IMAGE_SIZE} '
          f'threshold={THRESHOLD}')
        P(f'Data split       : train={c["n_train"]} val={c["n_val"]} test={c["n_test"]}')
    P('')

    # ---- Per-run overview ----------------------------------------
    P('-' * 100)
    P('1. Per-run Overview (best on validation / test pooled)')
    P('-' * 100)
    head = (f'{"run":<6}{"seed":<6}{"best_ep":<9}{"val_dice":<10}'
            + ''.join(f'{m:>12}' for m in METRIC_NAMES))
    P(head)
    P('-' * len(head))
    for r in run_infos:
        ov = r['overall']
        P(f'{r["run_index"]:<6}{r["seed"]:<6}{r["best_epoch"]:<9}'
          f'{r["best_val_dice"]:<10.4f}'
          + ''.join(f'{ov["mean"][m]:>12.4f}' for m in METRIC_NAMES))
    P('')

    # ---- Per-run x per-dataset -----------------------------------
    P('-' * 100)
    P('2. Per-run x Per-dataset Detailed Metrics (mean±std, n=number of images)')
    P('-' * 100)
    for r in run_infos:
        P(f'\n[run{r["run_index"]}  seed={r["seed"]}]  '
          f'best_epoch={r["best_epoch"]}  best_val_dice={r["best_val_dice"]:.4f}  '
          f'trained {r["epochs_done"]} epochs')
        h = f'  {"Dataset":<20}{"n":>6}' + ''.join(f'{m:>18}' for m in METRIC_NAMES)
        P(h)
        P('  ' + '-' * (len(h) - 2))
        for ds in TEST_DATASETS:
            d = r['per_dataset'].get(ds)
            if d is None:
                continue
            s = f'  {ds:<20}{d["n"]:>6}'
            for m in METRIC_NAMES:
                s += f'{d[m]["mean"]:>10.4f}±{d[m]["std"]:>6.4f}'
            P(s)
    P('')

    # ---- Aggregation across the 5 runs ---------------------------
    P('-' * 100)
    P('3. Aggregate Statistics over 5 Runs (mean/std of the per-run metrics)')
    P('-' * 100)
    P('\n[3.1] All 5 datasets pooled -- pooled mean per run, then mean/std over 5 runs')
    h = f'  {"Metric":<12}{"runs":>6}{"mean":>12}{"std":>12}{"min":>12}{"max":>12}{"CI95":>26}'
    P(h)
    P('  ' + '-' * (len(h) - 2))
    for m in METRIC_NAMES:
        vals = [r['overall']['mean'][m] for r in run_infos]
        s = stats_of(vals)
        ci = f'[{s["ci95_low"]:.4f},{s["ci95_high"]:.4f}]'
        P(f'  {m.upper():<12}{s["n"]:>6}{s["mean"]:>12.4f}{s["std"]:>12.4f}'
          f'{s["min"]:>12.4f}{s["max"]:>12.4f}{ci:>26}')

    P('\n[3.2] Per dataset -- mean/std over 5 runs for each test set')
    for ds in TEST_DATASETS:
        P(f'\n  <{ds}>')
        P(f'    {"Metric":<12}{"mean":>12}{"std":>12}{"min":>12}{"max":>12}')
        for m in METRIC_NAMES:
            vals = [r['per_dataset'][ds][m]['mean']
                    for r in run_infos if ds in r['per_dataset']]
            s = stats_of(vals)
            P(f'    {m.upper():<12}{s["mean"]:>12.4f}{s["std"]:>12.4f}'
              f'{s["min"]:>12.4f}{s["max"]:>12.4f}')

    P('\n[3.3] All per-image results of the 5 runs pooled (pooled over all images & runs)')
    h = f'  {"Metric":<12}{"n":>8}' + ''.join(
        f'{k:>12}' for k in ['mean', 'std', 'median', 'min', 'max', 'q1', 'q3', 'iqr'])
    P(h)
    P('  ' + '-' * (len(h) - 2))
    for m in METRIC_NAMES:
        s = stats_of([r[m] for r in all_records])
        P(f'  {m.upper():<12}{s["n"]:>8}{s["mean"]:>12.4f}{s["std"]:>12.4f}'
          f'{s["median"]:>12.4f}{s["min"]:>12.4f}{s["max"]:>12.4f}'
          f'{s["q1"]:>12.4f}{s["q3"]:>12.4f}{s["iqr"]:>12.4f}')
    P(f'\n  Total per-image records: {len(all_records)} '
      f'({len(run_infos)} runs x {len(all_records) // max(1, len(run_infos))} test images)')

    P('')
    P('-' * 100)
    P('4. Artifact List')
    P('-' * 100)
    P(f'  compare_downstream/{backbone_name}/runXX_seedYY/  '
      f'per-run train.log / epoch_metrics.csv / training_curves.png / '
      f'best_unet.pth / run_config.json / test_per_image.csv / '
      f'test_per_dataset.csv / test_overall.csv / test_summary.txt')
    P(f'  {os.path.basename(path)}                             this summary file')
    P('=' * 100)

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_comparison_txt(path, results):
    """Comparison summary of the two backbones."""
    lines = []
    P = lines.append
    P('=' * 100)
    P('  Backbone Comparison Summary: ImageNet  vs  ColoTexture-Hybrid-SIMMIM')
    P('=' * 100)
    P(f'Model    : project_root/unet.py :: EfficientNetB1UNetStd (standard vanilla U-Net)')
    P(f'Dataset  : {DATA_ROOT}')
    P(f'Test sets: {", ".join(TEST_DATASETS)}')
    P('')

    P('-' * 100)
    P('1. All 5 Test Sets Pooled -- mean±std over 5 runs (higher is better, lower HD95 is better)')
    P('-' * 100)
    names = list(results.keys())
    head = f'{"Metric":<12}' + ''.join(f'{n:>34}' for n in names) + f'{"Δ(B-A)":>12}'
    P(head)
    P('-' * len(head))
    for m in METRIC_NAMES:
        row = f'{m.upper():<12}'
        vals = {}
        for n in names:
            s = stats_of([r['overall']['mean'][m] for r in results[n]['runs']])
            vals[n] = s['mean']
            row += f'{s["mean"]:>18.4f}±{s["std"]:<8.4f}'
        if len(names) == 2:
            row += f'{vals[names[1]] - vals[names[0]]:>+12.4f}'
        P(row)
    P('')

    P('-' * 100)
    P('2. Per-dataset Dice / IoU Comparison (mean±std over 5 runs)')
    P('-' * 100)
    for m in ['dice', 'iou']:
        P(f'\n[{m.upper()}]')
        head = f'{"Dataset":<20}' + ''.join(f'{n:>34}' for n in names)
        P(head)
        P('-' * len(head))
        for ds in TEST_DATASETS:
            row = f'{ds:<20}'
            for n in names:
                vals = [r['per_dataset'][ds][m]['mean']
                        for r in results[n]['runs'] if ds in r['per_dataset']]
                s = stats_of(vals)
                row += f'{s["mean"]:>18.4f}±{s["std"]:<8.4f}'
            P(row)
    P('')
    P('=' * 100)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ======================================================================
# Single run
# ======================================================================
def run_once(backbone_name, backbone_path, seed, run_dir, args):
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'train.log')
    log_f = open(log_path, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(old_stdout, log_f)
    sys.stderr = Tee(old_stderr, log_f)

    try:
        print('=' * 90)
        print(f'  RUN START  {backbone_name}  seed={seed}')
        print(f'  device={DEVICE}')
        print('=' * 90)

        set_seed(seed)

        # ---- Data ----
        train_ds = SegDataset(os.path.join(DATA_ROOT, 'train', 'images'),
                              os.path.join(DATA_ROOT, 'train', 'masks'),
                              augment=True)
        val_ds = SegDataset(os.path.join(DATA_ROOT, 'val', 'images'),
                            os.path.join(DATA_ROOT, 'val', 'masks'),
                            augment=False)
        n_test = 0
        for ds in TEST_DATASETS:
            d = os.path.join(DATA_ROOT, 'test', ds, 'images')
            if os.path.isdir(d):
                n_test += len([f for f in os.listdir(d)
                               if f.lower().endswith(IMAGE_EXTS)])
        print(f'[data] train={len(train_ds)} val={len(val_ds)} test={n_test}')

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=True, drop_last=True,
                                  generator=g, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True, persistent_workers=True)

        # ---- Model: exactly the same unet, only the backbone init differs ----
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
                            'seed': seed}, ckpt_path)
                print(f'    -> best saved (val_dice={best_dice:.4f})')
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'    -> early stop at epoch {epoch} '
                          f'(no improvement for {args.patience} epochs)')
                    break

        csv_f.close()
        save_curves(epoch_rows, os.path.join(run_dir, 'training_curves.png'),
                    f'{backbone_name}  seed={seed}  (best val_dice={best_dice:.4f})')

        # ---- Test (load the best validation weights) ----
        print(f'\n[test] load best epoch={best_epoch} val_dice={best_dice:.4f}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        records = test_all(model, DEVICE, args.batch_size, args.workers)

        # Per-image CSV
        per_image_path = os.path.join(run_dir, 'test_per_image.csv')
        write_csv(per_image_path, records,
                  fieldnames=['dataset', 'image'] + METRIC_NAMES)

        # Per-dataset / pooled statistics
        per_dataset_rows = records_to_group_df(records, 'dataset')
        overall_rows = records_to_group_df(
            [dict(r, dataset='ALL') for r in records], 'dataset')
        write_csv(os.path.join(run_dir, 'test_per_dataset.csv'), per_dataset_rows)
        write_csv(os.path.join(run_dir, 'test_overall.csv'), overall_rows)

        per_dataset_map = {
            row['dataset']: {
                'n': row['n_images'],
                **{m: {k: row[f'{m}_{k}'] for k in
                       ['n', 'mean', 'std', 'median', 'min', 'max', 'q1',
                        'q3', 'iqr', 'sem', 'ci95_low', 'ci95_high']}
                   for m in METRIC_NAMES},
            } for row in per_dataset_rows
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
            'seed': seed, 'epochs': args.epochs, 'patience': args.patience,
            'batch_size': args.batch_size, 'lr': args.lr,
            'min_lr': args.min_lr, 'weight_decay': args.weight_decay,
            'warmup': args.warmup, 'n_train': len(train_ds),
            'n_val': len(val_ds), 'n_test': n_test,
            'epochs_done': len(epoch_rows), 'best_epoch': best_epoch,
            'best_val_dice': best_dice,
        }
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        write_run_summary_txt(os.path.join(run_dir, 'test_summary.txt'),
                              cfg, epoch_rows, records, per_dataset_rows)

        run_info = dict(cfg)          # includes hyperparameters such as epochs / patience / lr / n_train ...
        run_info.update({
            'run_index': 0,           # filled in by the caller
            'dir': run_dir,
            'seed': seed,
            'best_epoch': best_epoch,
            'best_val_dice': best_dice,
            'epochs_done': len(epoch_rows),
            'final_train_loss': (epoch_rows[-1]['train_loss']
                                 if epoch_rows else float('nan')),
            'n_test_images': len(records),
            'per_dataset': per_dataset_map,
            'overall': overall_map,
            'records': records,
        })
        print(f'\n[run done] {backbone_name} seed={seed}  '
              f'dice={overall_map["mean"]["dice"]:.4f}')
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
        description='Compare two EfficientNet-B1 backbones on the same dataset.')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR)
    ap.add_argument('--min-lr', type=float, default=DEFAULT_MIN_LR)
    ap.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument('--warmup', type=int, default=DEFAULT_WARMUP)
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f'[main] device={DEVICE}  out={OUT_ROOT}')
    print(f'[main] backbones={[b[0] for b in BACKBONES]}  seeds={args.seeds}')

    results = {}
    for backbone_name, backbone_path in BACKBONES:
        print('\n' + '#' * 90)
        print(f'#  Backbone: {backbone_name}   ({backbone_path})')
        print('#' * 90)
        bb_dir = os.path.join(OUT_ROOT, backbone_name)
        os.makedirs(bb_dir, exist_ok=True)

        run_infos = []
        all_records = []
        for i, seed in enumerate(args.seeds, start=1):
            run_dir = os.path.join(bb_dir, f'run{i:02d}_seed{seed}')
            print(f'\n>>> run {i}/{len(args.seeds)}  seed={seed}  -> {run_dir}')
            run_info, _ = run_once(backbone_name, backbone_path, seed,
                                   run_dir, args)
            run_info['run_index'] = i
            run_infos.append(run_info)
            all_records.extend(run_info['records'])
            torch.cuda.empty_cache()

        # ---- After all 5 runs, write the project-root summary ----
        summary_txt = os.path.join(
            PROJECT_ROOT, f'compare_downstream_{backbone_name}.txt')
        write_backbone_summary_txt(summary_txt, backbone_name, backbone_path,
                                   run_infos, all_records)
        print(f'\n[summary] {backbone_name}: {summary_txt}')

        with open(os.path.join(bb_dir, 'runs_summary.json'), 'w') as f:
            json.dump({'backbone_name': backbone_name,
                       'backbone_path': backbone_path,
                       'seeds': args.seeds,
                       'runs': [{k: v for k, v in r.items() if k != 'records'}
                                for r in run_infos]}, f, indent=2,
                      default=float)

        results[backbone_name] = {'path': backbone_path, 'runs': run_infos}

    # ---- Compare the two backbones ----
    if len(results) >= 2:
        comp = os.path.join(PROJECT_ROOT, 'compare_downstream_comparison.txt')
        write_comparison_txt(comp, results)
        print(f'\n[comparison] {comp}')


if __name__ == '__main__':
    main()
