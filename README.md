<div align="center">

# ColoTexture-Hybrid

A hybrid dataset of unlabeled colonoscopic images capturing diverse mucosal textures

[![lang: English](https://img.shields.io/badge/lang-English-2f81f7?style=flat)](README.md)
[![lang: 简体中文](https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-8b949e?style=flat)](README.zh-CN.md)

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey?style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FJKBW5-1f6feb?style=flat-square)](https://doi.org/10.17605/OSF.IO/JKBW5)
[![SourceForge](https://img.shields.io/badge/download-SourceForge-FC6D26?style=flat-square&logo=sourceforge&logoColor=white)](https://sourceforge.net/projects/colotexture-hybrid/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat-square)](https://www.python.org)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-EE4C2C?style=flat-square)](https://pytorch.org)

</div>

## Contents

[1 Environment](#1-environment) · [2 Repository structure](#2-repository-structure) · [3 Data](#3-data) · [4 Pretrained backbones](#4-pretrained-backbones) · [5 Usage](#5-usage) · [6 Script reference](#6-script-reference) · [7 Outputs and metrics](#7-outputs-and-metrics) · [8 Reported results](#8-reported-results) · [9 Reproduction order](#9-reproduction-order) · [10 Citation](#10-citation) · [11 License](#11-license) · [References](#references)

## 1. Environment

All experiments were conducted in a conda environment named `Unet`:

```bash
conda activate Unet
```

**Table 1.** Runtime environment.

| Item | Requirement |
| :-- | :-- |
| Python | 3.9 or later |
| PyTorch | 2.x, CUDA build (`torch.amp` is used for mixed precision) |
| torchvision | version matching the installed PyTorch |
| Additional packages | `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `seaborn`, `umap-learn`, `natsort`, `tqdm`, `matplotlib`, `psutil`, `numba`, `scipy` |
| Hardware | a single NVIDIA GPU; the recorded runs used 24 CPU workers |

No pinned requirements file is distributed: dependencies were resolved manually in the
environment described above. Install a CUDA-compatible PyTorch first, then the remaining
packages.

The benchmark scripts default to `--workers 16`; the published results were produced with
24. `PolypDataset` preloads all images into RAM, so the worker count and the memory
footprint are coupled.

## 2. Repository structure

```text
ColoTexture-Source/
├── README.md  README.zh-CN.md  LICENSE
├── backbones/ladder/
│   ├── ColoTexture-Hybrid-SIMMIM.pth      # pretrained on the curated hybrid corpus
│   ├── ColoTexture-Stage5-SIMMIM.pth      # pretrained on the five-stage cleaned corpus
│   └── ColoTexture-Raw-SIMMIM.pth         # pretrained on the uncleaned corpus
├── efficientnet_b1_imagenet.pth           # ImageNet baseline backbone
├── ColoTexture-Hybrid/                    # 26,193 curated pretraining frames (jpg)
├── compare_downstream/                    # per-run artifacts, ImageNet vs Hybrid-SIMMIM
├── compare_stage/                         # per-run artifacts, three-backbone ladder
├── compare_fewshot/                       # per-run artifacts, training-ratio sweep
├── eval_results/                          # feature-space figures
├── figure/                                # box plots and box statistics
├── compare_downstream_{ImageNet,ColoTexture-Hybrid-SIMMIM,comparison}.txt
├── compare_stage_{ColoTexture-Hybrid,ColoTexture-Stage5,ColoTexture-Raw}-SIMMIM.txt
├── compare_stage_comparison.txt
├── compare_fewshot_{ImageNet,ColoTexture-Hybrid-SIMMIM,comparison}.txt
├── evaluate_result.txt
├── dataset.py  model.py  unet.py
├── download_pretrained.py
├── SIMMIM-Hybrid.py  SIMMIM-Stage5.py  SIMMIM-Raw.py
├── evaluate.py  generate_boxplot.py
├── compare_downstream.py  compare_stage.py  compare_fewshot.py
├── extract_frames.py  extract_bad_frames.py
├── deduplicate_sequence.py  deduplicate_sequence_parallel.py
├── check_dataset_overlap.py
└── select_images_mouse.py  select_images_keyboard.py
```

The `*.txt` files at the repository root are the five-run aggregate reports written by the
benchmark scripts; the `compare_*/` directories hold the per-run artifacts from which those
reports are computed.

## 3. Data

### 3.1 Downstream segmentation benchmark

```text
/home/linux/Dataset/MixDB/ProcessedDataset/
├── train/{images,masks}       # 1,160 images
├── val/{images,masks}         # 290 images
├── test/{images,masks}        # 797 images: CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir
├── train_list.txt
├── val_list.txt
└── dataset_info.md
```

Images are resized to 384 × 384 and binarised at a threshold of 0.5; `test/` is partitioned by
dataset. The three benchmark scripts do not use these directories in the same way.

**Table 2.** Dataset splits used by the benchmark scripts.

| Script | Training | Model selection | Reported evaluation |
| :-- | :-- | :-- | :-- |
| `compare_downstream.py` | `train/`, 1,160 images | `val/`, 290 images | `test/`, 797 images; five subsets reported individually and pooled |
| `compare_stage.py` | `train/`, 1,160 images | `val/`, 290 images | `test/`, 797 images; five subsets reported individually and pooled |
| `compare_fewshot.py` | `train/`, subsampled to 100 / 50 / 20 / 10 % (1,160 / 580 / 232 / 116 images) | `val/`, 290 images | `val/`, 290 images; no test set |

### 3.2 Pretraining corpora

**Table 3.** Texture corpora used for pretraining.

| Corpus | Default path | Images |
| :-- | :-- | :-- |
| ColoTexture-Hybrid | `/mnt/sda/Dataset/ColoTexture-Hybrid` | all (26,193) |
| ColoTexture-Stage5 | `/mnt/sda/Dataset/stage5` | 26,193 (seeded sample) |
| ColoTexture-Raw | `/mnt/sda/Dataset/stage1` | 26,193 (seeded sample) |

Sampling uses `SAMPLE_SEED = 42`, so the subsets are reproducible. Frames are extracted
from colonoscopy video and de-duplicated against the public polyp benchmarks; the
procedure is implemented in the scripts listed in Section 6. The corpus is distributed
under CC BY 4.0 together with the pretrained checkpoints (Section 4).

## 4. Pretrained backbones

**Distribution.** The ColoTexture-Hybrid corpus and the released checkpoints are available
from OSF under DOI [10.17605/OSF.IO/JKBW5](https://doi.org/10.17605/OSF.IO/JKBW5), with
mirrors on Baidu Netdisk
(<https://pan.baidu.com/s/1_iIi95t7aU_Zjk4VfnhOgA?pwd=2026>, extraction code `2026`) and on
SourceForge (<https://sourceforge.net/projects/colotexture-hybrid/>).

The three ladder checkpoints are plain `torch.save` dictionaries:

```python
{
    'encoder': OrderedDict(...),   # 506 tensors, standard EfficientNet-B1 encoder
    'epoch':   24,                 # snapshot index
    'arch':    'EfficientNetB1Encoder',
}
```

`unet.EfficientNetB1UNetStd.load_backbone()` accepts any of the following:

- the dictionary above (the `'encoder'` entry is unpacked automatically);
- an `encoder.`-prefixed state dict;
- an official torchvision `efficientnet_b1` state dict (`features.*` keys are remapped).

To regenerate the ImageNet baseline checkpoint:

```bash
python download_pretrained.py    # torchvision IMAGENET1K_V1 -> efficientnet_b1_imagenet.pth
```

## 5. Usage

### 5.1 Obtaining the code

```bash
git clone https://github.com/<user>/ColoTexture-Source.git
cd ColoTexture-Source
```

Using the released checkpoints does not require the pretraining corpus or any pretraining
run: the downstream scripts read only `backbones/ladder/*.pth` and
`efficientnet_b1_imagenet.pth`.

### 5.2 Pretraining

```bash
python SIMMIM-Hybrid.py
python SIMMIM-Stage5.py
python SIMMIM-Raw.py
```

**Table 4.** Pretraining configuration (identical for the three scripts).

| Setting | Value |
| :-- | :-- |
| Input resolution | 384 × 384 |
| Masking | 60 % of 16 × 16 patches, random |
| Loss | L1, masked regions only |
| Optimiser | AdamW, lr 1e-4, min lr 1e-6, weight decay 0.05 |
| Schedule | 15 warm-up epochs + cosine decay, 300 epochs |
| Batch size | 64 |
| Initialisation | ImageNet EfficientNet-B1 |

Each script writes `backbones/ladder/<name>.pth` and a visualisation directory
(`*_simmim_visualizations/`) containing reconstruction panels saved every few epochs.
The output path, corpus path and sample count are module-level constants at the top of
each script.

### 5.3 Downstream benchmark

```bash
python compare_downstream.py     # ImageNet vs ColoTexture-Hybrid-SIMMIM, 5 runs each
python compare_stage.py          # three ladder backbones, 5 runs each
python compare_fewshot.py        # ImageNet vs Hybrid-SIMMIM at 100/50/20/10 % training data
```

**Table 5.** Downstream training configuration (identical for the three scripts).

| Setting | Value |
| :-- | :-- |
| Epochs / patience | 100 / 20 |
| Batch size | 16 |
| Learning rate / minimum | 1e-3 / 1e-5 |
| Weight decay / warm-up | 0.01 / 5 epochs |
| Seeds | 42, 43, 44, 45, 46 |
| Image size / threshold | 384 / 0.5 |

The segmentation model is fixed across all runs: `unet.EfficientNetB1UNetStd`, a standard
U-Net decoder (bilinear upsampling, skip concatenation, double convolutions) attached to
the EfficientNet-B1 encoder. No ASPP, attention, edge branch or domain-adaptive
normalisation is used, so differences between runs can be attributed to the backbone
weights.

`compare_fewshot.py` uses a different split from the other two scripts (Table 2): the 290
validation images are fixed and the test set is never used. Only the training subset changes
(100, 50, 20 and 10 % of the 1,160 training images), and every ratio is evaluated on the same 290
images, so the ratios remain comparable with one another.

Common overrides:

```bash
python compare_fewshot.py --epochs 60 --patience 15 --seeds 42 43 44
python compare_fewshot.py --ratios 0.5 0.1
python compare_downstream.py --workers 8
```

Each script trains a newly initialised U-Net, evaluates it on the split listed in Table 2, writes
all artifacts, and proceeds to the next run or backbone. A five-run summary is written to the
repository root before switching backbones. Run the three scripts sequentially on a single GPU.

### 5.4 Feature-space analysis and figures

```bash
python evaluate.py             # writes evaluate_result.txt and eval_results/
python generate_boxplot.py     # reads compare_fewshot/, writes figure/
```

`evaluate.py` compares the two backbones on the same images using Linear Probe, k-NN,
Fisher Discriminant Ratio, Silhouette, Davies-Bouldin, Calinski-Harabasz and CKA, together
with Grad-CAM, multi-layer feature maps, t-SNE / UMAP embeddings, calibration curves and
feature distributions.

`generate_boxplot.py` aggregates the five runs of each setting: box edges are Q1/Q3, the
centre line is the median, and the whiskers span the observed minimum and maximum.

## 6. Script reference

**Data curation.**

| Script | Purpose |
| :-- | :-- |
| `extract_frames.py` | Decode video into frames; NVIDIA hardware decoding with a CPU multiprocessing fallback. |
| `extract_bad_frames.py` | Copy out low-quality frames flagged by an adaptive quality threshold. |
| `deduplicate_sequence.py` | Reference implementation: sequential pHash de-duplication with low-quality frame filtering. |
| `deduplicate_sequence_parallel.py` | 24-core batched version of the same pipeline, with memory monitoring. |
| `check_dataset_overlap.py` | Cross-dataset leakage check: CUDA pHash/dHash candidates, then MD5, descriptor MSE/ZNCC and flip/rotation verification. |
| `select_images_mouse.py` | Manual review by mouse clicks. |
| `select_images_keyboard.py` | Manual review by left/right arrow keys. |

**Model and data.**

| Script | Purpose |
| :-- | :-- |
| `dataset.py` | `PolypDataset` (in-RAM preloading, random resized crop, vertical flip, HSV jitter, motion blur) and the augmentation classes. |
| `model.py` | `EfficientNetB1Encoder`, `EfficientNetB1SIMMIM`, masked L1 loss, torchvision key remapping. |
| `unet.py` | `EfficientNetB1UNetStd`: standard U-Net decoder on the EfficientNet-B1 encoder. |
| `download_pretrained.py` | Download the ImageNet EfficientNet-B1 weights. |

**Training, evaluation and reporting.**

| Script | Purpose |
| :-- | :-- |
| `SIMMIM-Hybrid.py` | SimMIM pretraining on the full ColoTexture-Hybrid corpus. |
| `SIMMIM-Stage5.py` | SimMIM pretraining on a 26,193-image sample of the five-stage cleaned corpus. |
| `SIMMIM-Raw.py` | SimMIM pretraining on a 26,193-image sample of the uncleaned corpus. |
| `evaluate.py` | Backbone feature-space comparison and visualisation. |
| `compare_downstream.py` | Two-backbone downstream benchmark, five runs each. |
| `compare_stage.py` | Three-backbone downstream benchmark, five runs each. |
| `compare_fewshot.py` | Fixed-validation benchmark over four training-set ratios. |
| `generate_boxplot.py` | Box plots of the few-shot results. |

## 7. Outputs and metrics

```text
<experiment>/<backbone>/[ratio_XXX/]runYY_seedZZ/
├── train.log               # per-epoch training log
├── epoch_metrics.csv       # per epoch: train_loss, val dice/iou/precision/recall/f2, lr, epoch_seconds
├── training_curves.png     # loss and validation-metric curves
├── best_unet.pth           # best-on-validation checkpoint
├── run_config.json         # exact configuration of the run
├── test_per_image.csv      # one row per test image, all six metrics   (compare_*)
├── test_per_dataset.csv    # per-dataset statistics                    (compare_*)
├── test_overall.csv        # pooled statistics                         (compare_*)
├── test_summary.txt        # human-readable run report                 (compare_*)
├── val_per_image.csv       # one row per validation image              (compare_fewshot)
├── val_per_group.csv       # per-split statistics                      (compare_fewshot)
├── val_overall.csv         # pooled statistics                         (compare_fewshot)
└── run_summary.txt         # human-readable run report                 (compare_fewshot)
```

Five-run summaries are written to the repository root
(`compare_downstream_<backbone>.txt`, `compare_downstream_comparison.txt`, and the
equivalents for `compare_stage` and `compare_fewshot`). Each summary contains the per-run
overview, the per-run × per-dataset detail, aggregate mean/std/min/max/median/Q1/Q3/IQR/
95 % confidence interval over the five runs, pooled per-image statistics and the artifact
list.

**Table 6.** Reported metrics. All metrics are computed per image and then aggregated.

| Metric | Direction |
| :-- | :-- |
| Dice | higher is better |
| IoU | higher is better |
| Precision | higher is better |
| Recall | higher is better |
| F2 | higher is better |
| HD95 | lower is better |

Per-image values are retained in the `*_per_image.csv` files, so additional statistics can
be recomputed without retraining.

## 8. Reported results

**Table 7.** Pooled over the five test sets; mean ± standard deviation over five runs.
Source: `compare_downstream_comparison.txt`.

| Backbone | Dice | IoU | Precision | Recall | F2 | HD95 |
| :-- | --: | --: | --: | --: | --: | --: |
| ImageNet | 0.7980 ± 0.0211 | 0.7269 ± 0.0223 | 0.8142 ± 0.0077 | 0.8403 ± 0.0256 | 0.8139 ± 0.0223 | 45.186 ± 6.333 |
| ColoTexture-Hybrid-SIMMIM | 0.9011 ± 0.0090 | 0.8367 ± 0.0095 | 0.8929 ± 0.0085 | 0.9285 ± 0.0134 | 0.9147 ± 0.0121 | 19.437 ± 4.043 |

Comparable reports are provided for the three-backbone ladder
(`compare_stage_comparison.txt`), the training-ratio sweep
(`compare_fewshot_comparison.txt`; measured on the validation split, Table 2) and the
feature-space analysis (`evaluate_result.txt`). `compare_downstream.py` and
`compare_stage.py` perform independent seed sweeps, so the Hybrid-SIMMIM aggregate differs
slightly between the two reports (0.9011 and 0.9026 Dice); both use the same recipe.

## 9. Reproduction order

```text
extract_frames.py
  → deduplicate_sequence_parallel.py
  → extract_bad_frames.py
  → check_dataset_overlap.py
  → (optional) select_images_mouse.py / select_images_keyboard.py
download_pretrained.py
SIMMIM-Hybrid.py / SIMMIM-Stage5.py / SIMMIM-Raw.py
evaluate.py
compare_downstream.py → compare_stage.py → compare_fewshot.py
generate_boxplot.py
```

## 10. Citation

The dataset release is registered under DOI
[10.17605/OSF.IO/JKBW5](https://doi.org/10.17605/OSF.IO/JKBW5).

```bibtex
@dataset{colotexture_hybrid_2026,
  title     = {ColoTexture-Hybrid},
  author    = {z z and Sun, Xiaoyan and Xing, Xiaoyu and Yi, Xiaowen},
  year      = {2026},
  publisher = {OSF},
  doi       = {10.17605/OSF.IO/JKBW5},
  url       = {https://osf.io/jkbw5/},
  note      = {Dataset, licensed under CC BY 4.0}
}
```

The code in this repository may be cited as:

```bibtex
@misc{colotexture_source_2026,
  title        = {ColoTexture-Source: Self-Supervised Colonoscopy-Texture Pretraining
                  for Efficient Polyp Segmentation},
  author       = {<author>},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/<user>/ColoTexture-Source}},
  note         = {Licensed under CC BY 4.0}
}
```

## 11. License

This repository is distributed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The full legal code is provided in [`LICENSE`](LICENSE). Redistribution of the material,
in whole or in part, requires attribution to the author, a reference to the licence, and
an indication of any modifications, for example:

```text
Based on "ColoTexture-Source" by <author>
(https://github.com/<user>/ColoTexture-Source), licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Modifications: <description>.
```

The licence covers the source code, the trained checkpoints and the documentation
distributed in this repository. The public polyp benchmarks (Kvasir-SEG, CVC-300,
CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB) and the source endoscopy video are not
relicensed here and remain subject to their original terms; consult those terms before
redistributing images or derived frames.

## References

- Xie, Z., et al. SimMIM: a Simple Framework for Masked Image Modeling. CVPR, 2022.
- Tan, M., Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks. ICML, 2019.
- Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical
  Image Segmentation. MICCAI, 2015.
