# ColoTexture-Hybrid

**A Large-Scale Hybrid Dataset for Intestinal Texture Self-Supervised Pre-training**  
**面向肠道纹理自监督预训练的大规模混合数据集**

[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)](https://pytorch.org/)

<div align="center">

[**English**](#english) &nbsp;|&nbsp; [**简体中文**](#简体中文)

</div>

---

<div id="english"></div>

## English

### Introduction
Colorectal cancer mostly evolves from colon polyps, and deep learning-based analysis is limited by scarce pixel-level annotations and significant intestinal mucosal texture heterogeneity. Besides, ImageNet pre-trained backbones suffer severe domain shift with endoscopic images, failing to capture subtle lesion boundaries.

To address these bottlenecks, we fuse two public endoscopic datasets (LDPolypVideo & HyperKvasir) to build **ColoTexture-Hybrid**, a large-scale unlabeled intestinal endoscopic dataset containing 27,124 clinical static colonoscopy images, designed specifically for domain self-supervised pre-training of gastrointestinal vision models.

#### Core Contributions
1. The first large-scale hybrid intestinal texture dataset for self-supervised pre-training on gastrointestinal endoscopy, with 27,124 unlabeled colonoscopy images.
2. A 5-stage cascaded automatic data cleaning pipeline based on Median Absolute Deviation (MAD) to filter blurry frames, duplicate frames, reflective artifacts, ghosting and chromatic aberration.
3. Open-source full dataset resources and matched SimMIM pre-trained EfficientNet-B1 weights under Apache 2.0 license.

### Dataset Overview
#### Basic Information
- Total images: 27,124 unlabeled JPG colonoscopy frames
- Total storage size: ~5.1 GB
- Source datasets:
  - HyperKvasir: 19,261 images (71.0%)
  - LDPolypVideo: 7,863 images (29.0%)
- Resolution range: 443×393 ~ 3264×2448
- File naming rule: `colo_1.jpg`, `colo_2.jpg`, ..., `colo_27124.jpg`
- No pixel masks, bounding boxes or category labels — fully adapted to self-supervised representation learning.

#### Visual Coverage
1. **Polyp Morphology (Paris Classification)**
    0-IIa, 0-IIb, 0-IIc, 0-Ip, 0-Is, 0-Isp, LST (granular & non-granular subtypes)
2. **Colon Anatomy Segments**
    Transverse colon, descending colon, ascending colon, cecum, sigmoid colon, rectum
3. Common imaging interferences: specular reflection, bubbles, residual fluid, motion blur, color fringe artifacts

#### Data Cleaning Pipeline
All raw frames pass a 5-stage MAD-based automatic filtering + manual secondary review:
1. Perceptual hash deduplication for redundant consecutive video frames
2. High-frequency energy filtering to remove motion/out-of-focus blur
3. ROI low-frequency energy filtering for liquid & bubble reflective artifacts
4. Laplacian variance filtering for motion ghosting blur
5. Lab color space gradient detection for chromatic aberration fringe artifacts
6. Manual double-check to eliminate irrelevant content (instruments, black screen, external scene)

### Pre-training Setting
- Backbone: EfficientNet-B1 (7.8M params)
- Framework: SimMIM masked image modeling
- Mask strategy: 16×16 block mask, mask ratio 60%
- Optimizer: AdamW, 300 training epochs, batch size 64
- Input size: 384×384 RGB
- Hardware: NVIDIA RTX 4090, PyTorch 2.8.0

### Repository Structure
```
ColoTexture-Hybrid/
├── LICENSE                                 # Apache License 2.0
├── README.md                               # This document
├── pretrained_weights/                     # Pre-trained model checkpoints
│   ├── effb1_simmim_colo.pth               # SimMIM pre-trained EfficientNet-B1
│   ├── effb1_imagenet.pth                  # ImageNet pre-trained (for comparison)
│   └── download_pretrained.py              # Script to download ImageNet weights
├── train_simmim/                           # Self-supervised pre-training code
│   ├── model.py                            # EfficientNet-B1 + SimMIM framework
│   ├── dataset.py                          # Dataset & augmentations
│   ├── train.py                            # Pre-training script
│   └── load_weights.py                     # Weight loading utilities
└── data_cleaning_code/                     # Data cleaning pipeline tools
    ├── video_to_frames_nvidia_linux.py     # GPU-accelerated video frame extraction
    ├── image_deduplication_parallel_linux.py  # Large-scale parallel deduplication
    ├── image_sequence_deduplication.py     # Sequence-based deduplication
    ├── bad_frame_extractor_linux.py        # MAD-based bad frame filtering
    ├── image_manual_selector.py            # Interactive manual selection (Windows)
    └── image_manual_selector_linux.py      # Interactive manual selection (Linux)
```

### Download Guide
#### Dataset
- **International (SourceForge):** [https://sourceforge.net/projects/colotexture-hybrid/files/](https://sourceforge.net/projects/colotexture-hybrid/files/)
- **China (Baidu Pan):** [https://pan.baidu.com/s/1S3BKb8geYhzixLbKJyySig?pwd=2026](https://pan.baidu.com/s/1S3BKb8geYhzixLbKJyySig?pwd=2026) (Password: 2026)
- Raw source data: Refer to official release of HyperKvasir & LDPolypVideo

#### Pre-trained Weights
SimMIM pre-trained EfficientNet-B1 weights are available in the `pretrained_weights/` folder of this repository:
- `effb1_simmim_colo.pth` — Pre-trained on ColoTexture-Hybrid (300 epochs, SimMIM)

### Usage Instructions

#### 1. Self-supervised Pre-training
```bash
cd train_simmim
python train.py   # Edit DATA_DIR in train.py to point to your dataset
```

#### 2. Load Pre-trained Weights for Downstream Tasks
```python
from train_simmim.load_weights import load_encoder_only

encoder = load_encoder_only('pretrained_weights/effb1_simmim_colo.pth')
```

See [train_simmim/load_weights.py](train_simmim/load_weights.py) for more usage examples.

#### 3. Run Data Cleaning Pipeline
```bash
# Step 1: Extract frames from videos
cd data_cleaning_code
python video_to_frames_nvidia_linux.py

# Step 2: Deduplicate and filter bad frames
python image_deduplication_parallel_linux.py

# Step 3 (optional): Manual review
python image_manual_selector_linux.py
```

### Ethics Statement
This study does not collect original patient clinical data. All image materials are derived from two publicly available datasets (LDPolypVideo, HyperKvasir), which have completed independent ethical review by their original releasing institutions. No additional ethical approval is required for this secondary data analysis research.

### Citation
If you use this dataset or pre-trained weights in your research, please cite our paper:
```bibtex
@article{ColoTextureHybrid2026,
  title={ColoTexture-Hybrid: A Large-Scale Hybrid Dataset for Intestinal Texture Self-Supervised Pre-training},
  author={XXX, XXX, XXX},
  journal={XXX},
  year={2026}
}
```

### License
This project is open-source under the Apache License 2.0. See the [LICENSE](LICENSE) file for full license text.

### Contact
For dataset issues, code bugs or academic cooperation, please open an Issue or contact the corresponding author.

---

<div align="center">

[▲ Back to Top](#colotexture-hybrid)

</div>

---

<div id="简体中文"></div>

## 简体中文

### 简介
结直肠癌多由结肠息肉演化而来，基于深度学习的分析受限于稀缺的像素级标注与肠粘膜纹理异质性。此外，ImageNet 预训练主干网络与内镜图像存在严重的领域偏移，难以捕捉细微的病灶边界。

为此，我们融合两个公开内镜数据集（LDPolypVideo 与 HyperKvasir），构建了 **ColoTexture-Hybrid**——一个包含 27,124 张临床静态结肠镜图像的大规模无标注肠道内镜数据集，专为消化内镜视觉模型的领域自监督预训练设计。

#### 核心贡献
1. 首个面向消化内镜自监督预训练的大规模混合肠道纹理数据集，含 27,124 张无标注结肠镜图像。
2. 基于中位数绝对偏差（MAD）的 5 级级联自动化数据清洗流水线，可过滤模糊帧、重复帧、反光伪影、鬼影及色差。
3. 在 Apache 2.0 协议下开源完整数据集资源及配套 SimMIM 预训练 EfficientNet-B1 权重。

### 数据集概览
#### 基本信息
- 图像总数：27,124 张无标注 JPG 结肠镜帧
- 存储大小：约 5.1 GB
- 来源数据集：
  - HyperKvasir：19,261 张（71.0%）
  - LDPolypVideo：7,863 张（29.0%）
- 分辨率范围：443×393 ~ 3264×2448
- 命名规则：`colo_1.jpg`, `colo_2.jpg`, ..., `colo_27124.jpg`
- 无像素掩膜、边界框或类别标签，完全适配自监督表征学习。

#### 视觉覆盖
1. **息肉形态（巴黎分型）**
    0-IIa、0-IIb、0-IIc、0-Ip、0-Is、0-Isp、LST（颗粒型及非颗粒亚型）
2. **结肠解剖分段**
    横结肠、降结肠、升结肠、盲肠、乙状结肠、直肠
3. 常见成像干扰：镜面反光、气泡、残液、运动模糊、色差条纹伪影

#### 数据清洗流水线
所有原始帧经过 5 级 MAD 自动过滤 + 人工二次复核：
1. 感知哈希去重，剔除连续视频冗余帧
2. 高频能量过滤，去除运动/失焦模糊
3. ROI 低频能量过滤，剔除液体及气泡反光伪影
4. Laplacian 方差过滤，去除运动鬼影模糊
5. Lab 色彩空间梯度检测，剔除色差条纹伪影
6. 人工逐张复核，去除无关内容（器械、黑屏、院外场景）

### 预训练设置
- 主干网络：EfficientNet-B1（780 万参数）
- 框架：SimMIM 掩码图像建模
- 掩码策略：16×16 块掩码，掩码比例 60%
- 优化器：AdamW，300 训练轮次，批次大小 64
- 输入尺寸：384×384 RGB
- 硬件：NVIDIA RTX 4090，PyTorch 2.8.0

### 仓库结构
```
ColoTexture-Hybrid/
├── LICENSE                                 # Apache 2.0 许可证
├── README.md                               # 本文档
├── pretrained_weights/                     # 预训练模型权重
│   ├── effb1_simmim_colo.pth               # SimMIM 预训练 EfficientNet-B1
│   ├── effb1_imagenet.pth                  # ImageNet 预训练（用于对比）
│   └── download_pretrained.py              # 下载 ImageNet 权重的脚本
├── train_simmim/                           # 自监督预训练代码
│   ├── model.py                            # EfficientNet-B1 + SimMIM 框架
│   ├── dataset.py                          # 数据集与数据增强
│   ├── train.py                            # 预训练脚本
│   └── load_weights.py                     # 权重加载工具
└── data_cleaning_code/                     # 数据清洗流水线工具
    ├── video_to_frames_nvidia_linux.py     # GPU 加速视频抽帧
    ├── image_deduplication_parallel_linux.py  # 大规模并行去重
    ├── image_sequence_deduplication.py     # 序列去重
    ├── bad_frame_extractor_linux.py        # 基于 MAD 的坏帧过滤
    ├── image_manual_selector.py            # 交互式手动筛选 (Windows)
    └── image_manual_selector_linux.py      # 交互式手动筛选 (Linux)
```

### 下载指南
#### 数据集
- **国际（SourceForge）：** [https://sourceforge.net/projects/colotexture-hybrid/files/](https://sourceforge.net/projects/colotexture-hybrid/files/)
- **国内（百度网盘）：** [https://pan.baidu.com/s/1S3BKb8geYhzixLbKJyySig?pwd=2026](https://pan.baidu.com/s/1S3BKb8geYhzixLbKJyySig?pwd=2026) （密码：2026）
- 原始数据源：请参阅 HyperKvasir 与 LDPolypVideo 的官方发布

#### 预训练权重
SimMIM 预训练的 EfficientNet-B1 权重位于本仓库的 `pretrained_weights/` 文件夹中：
- `effb1_simmim_colo.pth` — 在 ColoTexture-Hybrid 上预训练（300 轮，SimMIM）

### 使用说明

#### 1. 自监督预训练
```bash
cd train_simmim
python train.py   # 修改 train.py 中的 DATA_DIR 指向你的数据集路径
```

#### 2. 加载预训练权重用于下游任务
```python
from train_simmim.load_weights import load_encoder_only

encoder = load_encoder_only('pretrained_weights/effb1_simmim_colo.pth')
```

更多用法示例请参阅 [train_simmim/load_weights.py](train_simmim/load_weights.py)。

#### 3. 运行数据清洗流水线
```bash
# 步骤 1：视频抽帧
cd data_cleaning_code
python video_to_frames_nvidia_linux.py

# 步骤 2：去重与坏帧过滤
python image_deduplication_parallel_linux.py

# 步骤 3（可选）：人工复核
python image_manual_selector_linux.py
```

### 伦理声明
本研究未采集原始患者临床数据，所有图像素材来源于两个公开数据集（LDPolypVideo、HyperKvasir），其原始发布机构已完成独立伦理审查。本二次数据分析研究无需额外伦理审批。

### 引用
如果您在研究中使用了本数据集或预训练权重，请引用我们的论文：
```bibtex
@article{ColoTextureHybrid2026,
  title={ColoTexture-Hybrid: A Large-Scale Hybrid Dataset for Intestinal Texture Self-Supervised Pre-training},
  author={XXX, XXX, XXX},
  journal={XXX},
  year={2026}
}
```

### 许可证
本项目基于 Apache License 2.0 开源。完整许可证文本请参阅 [LICENSE](LICENSE) 文件。

### 联系方式
如有数据集问题、代码缺陷或学术合作，请提交 Issue 或联系通讯作者。

---

<div align="center">

[▲ 回到顶部](#colotexture-hybrid)

</div>
