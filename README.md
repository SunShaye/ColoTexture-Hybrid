# ColoTexture-Hybrid
A Large-Scale Hybrid Dataset for Intestinal Texture Self-Supervised Pre-training

[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)](https://pytorch.org/)

## Introduction
Colorectal cancer mostly evolves from colon polyps, and deep learning-based analysis is limited by scarce pixel-level annotations and significant intestinal mucosal texture heterogeneity. Besides, ImageNet pre-trained backbones suffer severe domain shift with endoscopic images, failing to capture subtle lesion boundaries.

To address these bottlenecks, we fuse two public endoscopic datasets (LDPolypVideo & HyperKvasir) to build **ColoTexture-Hybrid**, a large-scale unlabeled intestinal endoscopic dataset containing 27,124 clinical static colonoscopy images, designed specifically for domain self-supervised pre-training of gastrointestinal vision models.

### Core Contributions
1. The first large-scale hybrid intestinal texture dataset for self-supervised pre-training on gastrointestinal endoscopy, with 27,124 unlabeled colonoscopy images.
2. A 5-stage cascaded automatic data cleaning pipeline based on Median Absolute Deviation (MAD) to filter blurry frames, duplicate frames, reflective artifacts, ghosting and chromatic aberration.
3. Open-source full dataset resources and matched SimMIM pre-trained EfficientNet-B1 weights under Apache 2.0 license.

## Dataset Overview
### Basic Information
- Total images: 27,124 unlabeled JPG colonoscopy frames
- Total storage size: ~5.1 GB
- Source datasets:
  - HyperKvasir: 19,261 images (71.0%)
  - LDPolypVideo: 7,863 images (29.0%)
- Resolution range: 443x393 ~ 3264x2448
- File naming rule: `colo_1.jpg`, `colo_2.jpg`, ..., `colo_27124.jpg`
- No pixel masks, bounding boxes or category labels, fully adapted to self-supervised representation learning.

### Visual Coverage
1. **Polyp Morphology (Paris Classification)**
    0-IIa, 0-IIb, 0-IIc, 0-Ip, 0-Is, 0-Isp, LST (granular & non-granular subtypes)
2. **Colon Anatomy Segments**
    Transverse colon, descending colon, ascending colon, cecum, sigmoid colon, rectum
3. Common imaging interferences: specular reflection, bubbles, residual fluid, motion blur, color fringe artifacts

### Data Cleaning Pipeline
All raw frames pass a 5-stage MAD-based automatic filtering + manual secondary review:
1. Perceptual hash deduplication for redundant consecutive video frames
2. High-frequency energy filtering to remove motion/out-of-focus blur
3. ROI low-frequency energy filtering for liquid & bubble reflective artifacts
4. Laplacian variance filtering for motion ghosting blur
5. Lab color space gradient detection for chromatic aberration fringe artifacts
6. Manual double-check to eliminate irrelevant content (instruments, black screen, external scene)

## Pre-training Setting
- Backbone: EfficientNet-B1 (7.8M params)
- Framework: SimMIM masked image modeling
- Mask strategy: 16x16 block mask, mask ratio 60%
- Optimizer: AdamW, 300 training epochs, batch size 64
- Input size: 384x384 RGB
- Hardware: NVIDIA RTX 4090, PyTorch 2.8.0

## Repository Structure
```
ColoTexture-Hybrid/
├── LICENSE                          # Apache License 2.0
├── README.md                        # This document
├── pretrained_weights/              # Pre-trained model checkpoints
│   ├── effb1_simmim_colo.pth        # SimMIM pre-trained EfficientNet-B1
│   ├── effb1_imagenet.pth           # ImageNet pre-trained (for comparison)
│   └── download_pretrained.py       # Script to download ImageNet weights
├── simmim_pretraining/              # Self-supervised pre-training code
│   ├── model.py                     # EfficientNet-B1 + SimMIM framework
│   ├── dataset.py                   # Dataset & augmentations
│   ├── train.py                     # Pre-training script
│   └── load_weights.py              # Weight loading utilities
└── data_cleaning/                   # Data cleaning pipeline tools
    ├── video_to_frames_nvidia_linux.py   # GPU-accelerated video frame extraction
    ├── image_deduplication_parallel_linux.py  # Large-scale parallel deduplication
    ├── image_sequence_deduplication.py    # Sequence-based deduplication
    ├── bad_frame_extractor_linux.py       # MAD-based bad frame filtering
    ├── image_manual_selector.py           # Interactive manual selection (Windows)
    └── image_manual_selector_linux.py     # Interactive manual selection (Linux)
```

## Download Guide
### Dataset
- **International (SourceForge):** [https://sourceforge.net/projects/colotexture-hybrid/files/](https://sourceforge.net/projects/colotexture-hybrid/files/)
- **China (Baidu Pan):** [https://pan.baidu.com/s/1S3BKb8geYhzixLbKJyySig?pwd=2026](https://pan.baidu.com/s/1S3BKb8geYhzixLbKJyySig?pwd=2026) (Password: 2026)
- Raw source data: Refer to official release of HyperKvasir & LDPolypVideo

### Pre-trained Weights
SimMIM pre-trained EfficientNet-B1 weights are available in the `pretrained_weights/` folder of this repository:
- `effb1_simmim_colo.pth` — Pre-trained on ColoTexture-Hybrid (300 epochs, SimMIM)

## Usage Instructions

### 1. Self-supervised Pre-training
```bash
cd train_simmim
python train.py   # Edit DATA_DIR in train.py to point to your dataset
```

### 2. Load Pre-trained Weights for Downstream Tasks
```python
from train_simmim.load_weights import load_encoder_only

encoder = load_encoder_only('pretrained_weights/effb1_simmim_colo.pth')
```

See [train_simmim/load_weights.py](train_simmim/load_weights.py) for more usage examples.

### 3. Run Data Cleaning Pipeline
```bash
# Step 1: Extract frames from videos
cd data_cleaning_code
python video_to_frames_nvidia_linux.py

# Step 2: Deduplicate and filter bad frames
python image_deduplication_parallel_linux.py

# Step 3 (optional): Manual review
python image_manual_selector_linux.py
```

## Ethics Statement
This study does not collect original patient clinical data. All image materials are derived from two publicly available datasets (LDPolypVideo, HyperKvasir), which have completed independent ethical review by their original releasing institutions. No additional ethical approval is required for this secondary data analysis research.

## License
This project is open-source under the Apache License 2.0.
See the [LICENSE](LICENSE) file for full license text.

## Contact
For dataset issues, code bugs or academic cooperation, please open an Issue or contact the corresponding author.
