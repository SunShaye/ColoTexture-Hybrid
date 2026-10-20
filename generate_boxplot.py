"""generate_boxplot.py -- Draw standard box plots from compare_fewshot.py experiment results.

Data source:
  compare_fewshot/<backbone>/ratio_XXX/runYY_seedZZ/val_overall.csv
  For each run, take the overall mean of the metric ({metric}_mean, i.e. the average over 290 evaluation images),
  so **the samples of each box = the 5 run results under that setting** (not per image, and not per epoch).

Plot content:
  Only Dice / IoU / F2 are plotted: one grouped box plot per metric, plus a row combining the three.
  x axis = training set ratio (from 100% down to 10%, from most to least), grouped by backbone at each ratio,
  and each box is built from the results of 5 independent runs:
    - box bottom/top = 25th / 75th percentile (Q1 / Q3)
    - median line    = 50th percentile
    - whiskers       = min and max of the 5 runs in that box
  No mean diamonds, scatter points or outlier markers are drawn; the distribution is always expressed by the box and whiskers.

Outputs (all saved under figure/ in the project root):
  figure/boxplot_<metric>.png      one grouped box plot per metric
  figure/boxplot_all_metrics.png   large figure with Dice / IoU / F2 combined in one row
  figure/boxplot_stats.csv         box statistics used for plotting (for verification and accuracy)

Usage:
  python generate_boxplot.py
  python generate_boxplot.py --data-root compare_fewshot --out-dir figure
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Only generate box plots for these three metrics
METRICS = ['dice', 'iou', 'f2']
METRIC_LABEL = {'dice': 'Dice', 'iou': 'IoU', 'f2': 'F2'}
# Metrics where lower is better (used for the y-axis direction annotation); currently all three are higher-is-better
LOWER_IS_BETTER = {'hd95'}

BACKBONE_COLORS = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']

# Box geometry: boxes in the same group (different backbones at the same ratio) sit flush with no gap; a small gap separates groups
BOX_WIDTH = 1.0
GROUP_GAP = 0.6

# Font sizes (increased substantially overall)
FS_TICK = 24
FS_LABEL = 30
FS_TITLE = 30
FS_LEGEND = 23


# ======================================================================
# Read Data
# ======================================================================
def scan_data(data_root):
    """Return (data, runs_count, backbones, ratios).

    data[(backbone, ratio_pct)][metric] = [overall mean of each run, ...]  # sample = one run
    runs_count[(backbone, ratio_pct)]   = number of runs included in the statistics
    """
    if not os.path.isdir(data_root):
        print(f'[error] Data directory not found: {data_root}')
        print('        Please run compare_fewshot.py first to generate the experiment results.')
        sys.exit(1)

    data = defaultdict(lambda: defaultdict(list))
    runs_count = defaultdict(int)
    backbones, ratios = set(), set()

    for bb in sorted(os.listdir(data_root)):
        bb_dir = os.path.join(data_root, bb)
        if not os.path.isdir(bb_dir):
            continue
        for ratio_dir in sorted(os.listdir(bb_dir)):
            if not ratio_dir.startswith('ratio_'):
                continue
            try:
                ratio = int(ratio_dir.split('_')[1])
            except (IndexError, ValueError):
                continue
            ratio_path = os.path.join(bb_dir, ratio_dir)
            for run_dir in sorted(os.listdir(ratio_path)):
                csv_path = os.path.join(ratio_path, run_dir, 'val_overall.csv')
                if not os.path.isfile(csv_path):
                    continue
                with open(csv_path, newline='') as fh:
                    rows = list(csv.DictReader(fh))
                if not rows:
                    continue
                row = rows[0]                      # val_overall.csv has only one row
                vals, ok = {}, True
                for m in METRICS:
                    try:
                        vals[m] = float(row[f'{m}_mean'])
                    except (KeyError, TypeError, ValueError):
                        ok = False
                        break
                if not ok:
                    continue
                for m in METRICS:
                    data[(bb, ratio)][m].append(vals[m])
                runs_count[(bb, ratio)] += 1
                backbones.add(bb)
                ratios.add(ratio)

    return data, runs_count, sorted(backbones), sorted(ratios, reverse=True)


def preferred_backbone_order(backbones):
    """Put ImageNet first, then sort the rest by name."""
    ordered = [b for b in backbones if 'imagenet' in b.lower()]
    ordered += [b for b in backbones if 'imagenet' not in b.lower()]
    return ordered


# ======================================================================
# Box Statistics (consistent with the matplotlib box plot definition, for verification)
# ======================================================================
def box_stats(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    return {
        'n': int(v.size),
        'mean': float(v.mean()),
        'std': float(v.std(ddof=1)) if v.size > 1 else 0.0,
        'min': float(v.min()),
        'q1': float(q1),
        'median': float(med),
        'q3': float(q3),
        'max': float(v.max()),
        'iqr': float(iqr),
        # Consistent with the figure: the whisker ends are the min/max values
        'whisker_low': float(v.min()),
        'whisker_high': float(v.max()),
    }


# ======================================================================
# Plotting
# ======================================================================
def draw_grouped_boxplot(ax, data, backbones, ratios, metric):
    n_b = len(backbones)
    bw = BOX_WIDTH
    stride = n_b * bw + GROUP_GAP
    centers, positions, series, colors = [], [], [], []

    for gi, r in enumerate(ratios):
        center = gi * stride + n_b * bw / 2.0
        centers.append(center)
        for bi, bb in enumerate(backbones):
            vals = data.get((bb, r), {}).get(metric, [])
            if not vals:
                continue
            positions.append(center + (bi - (n_b - 1) / 2.0) * bw)
            series.append(vals)
            colors.append(BACKBONE_COLORS[bi % len(BACKBONE_COLORS)])

    if not series:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return

    bp = ax.boxplot(
        series, positions=positions, widths=bw,
        patch_artist=True, whis=(0, 100), showfliers=False, showmeans=False,
        medianprops=dict(color='black', linewidth=1.4),
        whiskerprops=dict(color='black', linewidth=1.1),
        capprops=dict(color='black', linewidth=1.1),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.9)

    ax.set_xticks(centers)
    ax.set_xticklabels([f'{r}%' for r in ratios])
    ax.set_xlim(-0.5, len(ratios) * stride - GROUP_GAP + 0.5)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK - 2)
    direction = 'lower is better' if metric in LOWER_IS_BETTER else 'higher is better'
    ax.set_ylabel(f'{METRIC_LABEL[metric]}  ({direction})', fontsize=FS_LABEL)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)


def draw_figure(metric, data, backbones, ratios, out_path):
    fig, ax = plt.subplots(figsize=(17, 10.5))
    draw_grouped_boxplot(ax, data, backbones, ratios, metric)

    handles = [Patch(facecolor=BACKBONE_COLORS[i % len(BACKBONE_COLORS)],
                     edgecolor='black', label=bb, alpha=0.85)
               for i, bb in enumerate(backbones)]
    handles.append(plt.Line2D([], [], color='black', linewidth=1.4,
                              label='median'))
    handles.append(plt.Line2D([], [], color='black', linewidth=1.1,
                              label='min / max'))
    ax.legend(handles=handles, loc='best', fontsize=FS_LEGEND, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def draw_combined(data, backbones, ratios, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(36, 11.5))
    axes = axes.flatten()
    for ax, metric in zip(axes, METRICS):
        draw_grouped_boxplot(ax, data, backbones, ratios, metric)
        ax.set_title(METRIC_LABEL[metric], fontsize=FS_TITLE, fontweight='bold')

    handles = [Patch(facecolor=BACKBONE_COLORS[i % len(BACKBONE_COLORS)],
                     edgecolor='black', label=bb, alpha=0.85)
               for i, bb in enumerate(backbones)]
    handles.append(plt.Line2D([], [], color='black', linewidth=1.4,
                              label='median'))
    handles.append(plt.Line2D([], [], color='black', linewidth=1.1,
                              label='min / max'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               fontsize=FS_LEGEND + 2, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.99))
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_stats_csv(data, backbones, ratios, path):
    keys = ['n', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'iqr',
            'whisker_low', 'whisker_high']
    rows = []
    for metric in METRICS:
        for r in ratios:
            for bb in backbones:
                s = box_stats(data.get((bb, r), {}).get(metric, []))
                if s is None:
                    continue
                row = {'metric': metric, 'backbone': bb, 'ratio_pct': r}
                row.update({k: s[k] for k in keys})
                rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'backbone', 'ratio_pct'] + keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows


# ======================================================================
# Main Flow
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Box plots from compare_fewshot.py results.')
    ap.add_argument('--data-root',
                    default=os.path.join(PROJECT_ROOT, 'compare_fewshot'))
    ap.add_argument('--out-dir', default=os.path.join(PROJECT_ROOT, 'figure'))
    args = ap.parse_args()

    data, runs_count, backbones, ratios = scan_data(args.data_root)
    if not backbones or not ratios:
        print(f'[error] No val_per_image.csv found under {args.data_root}')
        sys.exit(1)
    backbones = preferred_backbone_order(backbones)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[info] Data root  : {args.data_root}')
    print(f'[info] Backbones  : {backbones}')
    print(f'[info] Ratios     : {[f"{r}%" for r in ratios]}')
    print(f'[info] Output dir : {args.out_dir}')

    for metric in METRICS:
        out = os.path.join(args.out_dir, f'boxplot_{metric}.png')
        draw_figure(metric, data, backbones, ratios, out)
        print(f'  -> {out}')

    combined = os.path.join(args.out_dir, 'boxplot_all_metrics.png')
    draw_combined(data, backbones, ratios, combined)
    print(f'  -> {combined}')

    stats_path = os.path.join(args.out_dir, 'boxplot_stats.csv')
    rows = write_stats_csv(data, backbones, ratios, stats_path)
    print(f'  -> {stats_path}  ({len(rows)} boxes)')

    # Print a copy of the box statistics to the console for verification
    print('\n[Box Stats] n per box = number of runs; box = Q1/median/Q3, whiskers = min/max')
    for metric in METRICS:
        print(f'\n<{METRIC_LABEL[metric]}>')
        print(f'  {"backbone":<28}{"ratio":>7}{"n":>7}{"Q1":>10}'
              f'{"median":>10}{"Q3":>10}{"whisk_lo":>11}{"whisk_hi":>11}'
              f'{"mean":>10}')
        for r in ratios:
            for bb in backbones:
                vals = data.get((bb, r), {}).get(metric, [])
                s = box_stats(vals)
                if s is None:
                    continue
                print(f'  {bb:<28}{str(r) + "%":>7}{s["n"]:>7}{s["q1"]:>10.4f}'
                      f'{s["median"]:>10.4f}{s["q3"]:>10.4f}'
                      f'{s["whisker_low"]:>11.4f}{s["whisker_high"]:>11.4f}'
                      f'{s["mean"]:>10.4f}')

    print(f'\n[done] Box plots saved to {args.out_dir}')


if __name__ == '__main__':
    main()
