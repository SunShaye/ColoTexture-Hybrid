<div align="center">

# ColoTexture-Hybrid

一个包含多种黏膜纹理的无标签结肠镜图像混合数据集

[![lang: English](https://img.shields.io/badge/lang-English-8b949e?style=flat)](README.md)
[![lang: 简体中文](https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-2f81f7?style=flat)](README.zh-CN.md)

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey?style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FJKBW5-1f6feb?style=flat-square)](https://doi.org/10.17605/OSF.IO/JKBW5)
[![SourceForge](https://img.shields.io/badge/download-SourceForge-FC6D26?style=flat-square&logo=sourceforge&logoColor=white)](https://sourceforge.net/projects/colotexture-hybrid/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat-square)](https://www.python.org)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-EE4C2C?style=flat-square)](https://pytorch.org)

</div>

## 目录

[1 运行环境](#1-运行环境) · [2 仓库结构](#2-仓库结构) · [3 数据](#3-数据) · [4 预训练权重](#4-预训练权重) · [5 使用方法](#5-使用方法) · [6 脚本说明](#6-脚本说明) · [7 输出与指标](#7-输出与指标) · [8 结果摘要](#8-结果摘要) · [9 复现顺序](#9-复现顺序) · [10 引用](#10-引用) · [11 许可证](#11-许可证) · [参考文献](#参考文献)

## 1. 运行环境

全部实验在名为 `Unet` 的 conda 环境中完成：

```bash
conda activate Unet
```

**表 1** 运行环境。

| 项目 | 要求 |
| :-- | :-- |
| Python | 3.9 及以上 |
| PyTorch | 2.x，CUDA 版本（使用 `torch.amp` 进行混合精度训练） |
| torchvision | 与所安装的 PyTorch 版本匹配 |
| 其他依赖 | `opencv-python`、`numpy`、`pandas`、`scikit-learn`、`seaborn`、`umap-learn`、`natsort`、`tqdm`、`matplotlib`、`psutil`、`numba`、`scipy` |
| 硬件 | 单张 NVIDIA GPU；所记录的实验使用 24 个 CPU worker |

仓库未提供固定版本的依赖文件，上述环境由人工解析得到。请先安装与 CUDA 匹配的 PyTorch，
再安装其余依赖。

基准脚本的 `--workers` 默认为 16，已发表结果使用 24。`PolypDataset` 会将全部图像预加载到
内存，因此 worker 数量与内存占用相互关联。

## 2. 仓库结构

```text
ColoTexture-Source/
├── README.md  README.zh-CN.md  LICENSE
├── backbones/ladder/
│   ├── ColoTexture-Hybrid-SIMMIM.pth      # 在精选混合语料上预训练
│   ├── ColoTexture-Stage5-SIMMIM.pth      # 在五阶段清洗语料上预训练
│   └── ColoTexture-Raw-SIMMIM.pth         # 在未清洗语料上预训练
├── efficientnet_b1_imagenet.pth           # ImageNet 基线骨干
├── ColoTexture-Hybrid/                    # 26,193 帧精选预训练图像（jpg）
├── compare_downstream/                    # 逐 run 产物：ImageNet 与 Hybrid-SIMMIM
├── compare_stage/                         # 逐 run 产物：三骨干阶梯
├── compare_fewshot/                       # 逐 run 产物：训练比例扫描
├── eval_results/                          # 特征空间分析图
├── figure/                                # 箱线图与箱体统计
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

仓库根目录下的 `*.txt` 文件是基准脚本写出的五次运行汇总报告；`compare_*/` 目录保存用于
计算这些报告的逐 run 产物。

## 3. 数据

### 3.1 下游分割基准

```text
/home/linux/Dataset/MixDB/ProcessedDataset/
├── train/{images,masks}       # 1,160 张
├── val/{images,masks}         # 290 张
├── test/{images,masks}        # 797 张：CVC-300、CVC-ClinicDB、CVC-ColonDB、ETIS-LaribPolypDB、Kvasir
├── train_list.txt
├── val_list.txt
└── dataset_info.md
```

图像缩放至 384 × 384，二值化阈值为 0.5；`test/` 按数据集分为子目录。三个基准脚本对这些目录
的使用方式并不相同。

**表 2** 各基准脚本使用的数据划分。

| 脚本 | 训练集 | 模型选择 | 报告评估集 |
| :-- | :-- | :-- | :-- |
| `compare_downstream.py` | `train/`，1,160 张 | `val/`，290 张 | `test/`，797 张；五个子集分别报告并给出汇总 |
| `compare_stage.py` | `train/`，1,160 张 | `val/`，290 张 | `test/`，797 张；五个子集分别报告并给出汇总 |
| `compare_fewshot.py` | `train/`，按 100 / 50 / 20 / 10 % 采样（1,160 / 580 / 232 / 116 张） | `val/`，290 张 | `val/`，290 张；不使用测试集 |

### 3.2 预训练语料

**表 3** 预训练所用纹理语料。

| 语料 | 默认路径 | 图像数 |
| :-- | :-- | :-- |
| ColoTexture-Hybrid | `/mnt/sda/Dataset/ColoTexture-Hybrid` | 全部（26,193） |
| ColoTexture-Stage5 | `/mnt/sda/Dataset/stage5` | 26,193（固定种子采样） |
| ColoTexture-Raw | `/mnt/sda/Dataset/stage1` | 26,193（固定种子采样） |

采样使用 `SAMPLE_SEED = 42`，因此子集可复现。所有帧由结肠镜视频提取，并与公开息肉基准做
去重处理；相应流程由第 6 节列出的脚本实现。该语料与预训练权重一同按 CC BY 4.0 分发，
获取方式见第 4 节。

## 4. 预训练权重

**分发渠道。** ColoTexture-Hybrid 语料与已发布权重通过 OSF 提供，DOI 为
[10.17605/OSF.IO/JKBW5](https://doi.org/10.17605/OSF.IO/JKBW5)；另在百度网盘
（<https://pan.baidu.com/s/1_iIi95t7aU_Zjk4VfnhOgA?pwd=2026>，提取码 `2026`）与
SourceForge（<https://sourceforge.net/projects/colotexture-hybrid/>）提供镜像。

三个阶梯权重均为普通的 `torch.save` 字典：

```python
{
    'encoder': OrderedDict(...),   # 506 个张量，标准 EfficientNet-B1 编码器
    'epoch':   24,                 # 快照序号
    'arch':    'EfficientNetB1Encoder',
}
```

`unet.EfficientNetB1UNetStd.load_backbone()` 接受以下任一形式：

- 上述字典（自动解包 `'encoder'` 条目）；
- 带 `encoder.` 前缀的 state dict；
- 官方 torchvision `efficientnet_b1` 权重（`features.*` 键会被重映射）。

重新生成 ImageNet 基线权重：

```bash
python download_pretrained.py    # torchvision IMAGENET1K_V1 -> efficientnet_b1_imagenet.pth
```

## 5. 使用方法

### 5.1 获取代码

```bash
git clone https://github.com/<user>/ColoTexture-Source.git
cd ColoTexture-Source
```

仅使用已发布权重时，无需预训练语料，也无需执行预训练：下游脚本只读取
`backbones/ladder/*.pth` 与 `efficientnet_b1_imagenet.pth`。

### 5.2 预训练

```bash
python SIMMIM-Hybrid.py
python SIMMIM-Stage5.py
python SIMMIM-Raw.py
```

**表 4** 预训练配置（三个脚本完全一致）。

| 设置 | 取值 |
| :-- | :-- |
| 输入分辨率 | 384 × 384 |
| 掩码 | 随机遮挡 16 × 16 图像块的 60 % |
| 损失 | L1，仅在掩码区域计算 |
| 优化器 | AdamW，学习率 1e-4，最小学习率 1e-6，权重衰减 0.05 |
| 学习率调度 | 15 个 epoch 预热 + 余弦退火，共 300 个 epoch |
| 批大小 | 64 |
| 初始化 | ImageNet EfficientNet-B1 |

每个脚本写出 `backbones/ladder/<name>.pth`，以及一个可视化目录
（`*_simmim_visualizations/`），其中每隔若干 epoch 保存一次重建对照图。输出路径、语料
路径与采样数量均为各脚本顶部的模块级常量。

### 5.3 下游基准

```bash
python compare_downstream.py     # ImageNet vs ColoTexture-Hybrid-SIMMIM，各 5 次运行
python compare_stage.py          # 三个阶梯骨干，各 5 次运行
python compare_fewshot.py        # ImageNet vs Hybrid-SIMMIM，100/50/20/10 % 训练数据
```

**表 5** 下游训练配置（三个脚本完全一致）。

| 设置 | 取值 |
| :-- | :-- |
| Epochs / 早停耐心 | 100 / 20 |
| 批大小 | 16 |
| 学习率 / 最小学习率 | 1e-3 / 1e-5 |
| 权重衰减 / 预热 | 0.01 / 5 个 epoch |
| 随机种子 | 42、43、44、45、46 |
| 图像尺寸 / 阈值 | 384 / 0.5 |

所有运行使用同一分割模型：`unet.EfficientNetB1UNetStd`，即标准 U-Net 解码器（双线性上采样、
跳连拼接、双卷积）接在 EfficientNet-B1 编码器上。模型不含 ASPP、注意力、边缘分支或域自适应
归一化，因此运行之间的差异可归因于骨干权重。

`compare_fewshot.py` 的划分与另外两个脚本不同（表 2）：验证集固定为 290 张，且始终不使用
测试集。仅训练子集发生变化（1,160 张训练图像的 100、50、20、10 %），每个比例都在同样的
290 张图像上评估，因此各比例之间可相互比较。

常用覆盖参数：

```bash
python compare_fewshot.py --epochs 60 --patience 15 --seeds 42 43 44
python compare_fewshot.py --ratios 0.5 0.1
python compare_downstream.py --workers 8
```

每个脚本训练一个全新初始化的 U-Net、按表 2 指定的划分完成评估、写出全部产物，随后自动进入
下一次运行或下一个骨干。切换骨干之前，五次运行的汇总会写入仓库根目录。在单张 GPU 上请顺序
运行三个脚本。

### 5.4 特征分析与绘图

```bash
python evaluate.py             # 写出 evaluate_result.txt 与 eval_results/
python generate_boxplot.py     # 读取 compare_fewshot/，写出 figure/
```

`evaluate.py` 在同一批图像上比较两个骨干，使用 Linear Probe、k-NN、Fisher 判别比、Silhouette、
Davies-Bouldin、Calinski-Harabasz 与 CKA，并输出 Grad-CAM、多层特征图、t-SNE / UMAP 嵌入、
校准曲线与特征分布。

`generate_boxplot.py` 以每个设置的五次运行为样本：箱体上下边为 Q1/Q3，中线为中位数，须线
覆盖观测到的最小值与最大值。

## 6. 脚本说明

**数据清洗。**

| 脚本 | 作用 |
| :-- | :-- |
| `extract_frames.py` | 将视频解码为帧；优先使用 NVIDIA 硬件解码，并提供 CPU 多进程回退。 |
| `extract_bad_frames.py` | 导出由自适应质量阈值判定为低质量的帧。 |
| `deduplicate_sequence.py` | 参考实现：串行 pHash 去重与低质量帧过滤。 |
| `deduplicate_sequence_parallel.py` | 同一流程的 24 核批处理版本，带内存监控。 |
| `check_dataset_overlap.py` | 跨数据集泄漏检查：CUDA pHash/dHash 召回候选，再进行 MD5、描述子 MSE/ZNCC 与翻转/旋转验证。 |
| `select_images_mouse.py` | 鼠标点击的人工复核。 |
| `select_images_keyboard.py` | 左右方向键的人工复核。 |

**模型与数据。**

| 脚本 | 作用 |
| :-- | :-- |
| `dataset.py` | `PolypDataset`（图像常驻内存、随机尺寸裁剪、垂直翻转、HSV 抖动、运动模糊）与各增强类。 |
| `model.py` | `EfficientNetB1Encoder`、`EfficientNetB1SIMMIM`、掩码 L1 损失、torchvision 键重映射。 |
| `unet.py` | `EfficientNetB1UNetStd`：在 EfficientNet-B1 编码器上使用标准 U-Net 解码器。 |
| `download_pretrained.py` | 下载 ImageNet EfficientNet-B1 权重。 |

**训练、评估与报告。**

| 脚本 | 作用 |
| :-- | :-- |
| `SIMMIM-Hybrid.py` | 在完整 ColoTexture-Hybrid 语料上做 SimMIM 预训练。 |
| `SIMMIM-Stage5.py` | 在五阶段清洗语料的 26,193 张采样上做 SimMIM 预训练。 |
| `SIMMIM-Raw.py` | 在未清洗语料的 26,193 张采样上做 SimMIM 预训练。 |
| `evaluate.py` | 骨干特征空间对比与可视化。 |
| `compare_downstream.py` | 两骨干下游基准，各五次运行。 |
| `compare_stage.py` | 三骨干下游基准，各五次运行。 |
| `compare_fewshot.py` | 固定验证集的四档训练比例基准。 |
| `generate_boxplot.py` | 小样本结果的箱线图。 |

## 7. 输出与指标

```text
<实验>/<骨干>/[ratio_XXX/]runYY_seedZZ/
├── train.log               # 逐 epoch 训练日志
├── epoch_metrics.csv       # 逐 epoch：train_loss、val dice/iou/precision/recall/f2、lr、epoch_seconds
├── training_curves.png     # 损失与验证指标曲线
├── best_unet.pth           # 验证集最优权重
├── run_config.json         # 本次运行的完整配置
├── test_per_image.csv      # 每张测试图一行，含全部六项指标   （compare_*）
├── test_per_dataset.csv    # 逐数据集统计                     （compare_*）
├── test_overall.csv        # 汇总统计                         （compare_*）
├── test_summary.txt        # 人类可读的运行报告               （compare_*）
├── val_per_image.csv       # 每张验证图一行                   （compare_fewshot）
├── val_per_group.csv       # 逐划分统计                       （compare_fewshot）
├── val_overall.csv         # 汇总统计                         （compare_fewshot）
└── run_summary.txt         # 人类可读的运行报告               （compare_fewshot）
```

五次运行的汇总写入仓库根目录（`compare_downstream_<backbone>.txt`、
`compare_downstream_comparison.txt`，以及 `compare_stage` 与 `compare_fewshot` 的对应文件）。
每份汇总包含逐 run 概览、逐 run × 逐数据集明细、五次运行的聚合
mean/std/min/max/median/Q1/Q3/IQR/95 % 置信区间、汇总的逐图统计以及产物清单。

**表 6** 报告指标。所有指标先逐图计算，再聚合。

| 指标 | 方向 |
| :-- | :-- |
| Dice | 越高越好 |
| IoU | 越高越好 |
| Precision | 越高越好 |
| Recall | 越高越好 |
| F2 | 越高越好 |
| HD95 | 越低越好 |

逐图数值保留在 `*_per_image.csv` 中，因此附加统计量可在不重新训练的情况下重算。

## 8. 结果摘要

**表 7** 五个测试集汇总，五次运行的均值 ± 标准差。来源：`compare_downstream_comparison.txt`。

| 骨干 | Dice | IoU | Precision | Recall | F2 | HD95 |
| :-- | --: | --: | --: | --: | --: | --: |
| ImageNet | 0.7980 ± 0.0211 | 0.7269 ± 0.0223 | 0.8142 ± 0.0077 | 0.8403 ± 0.0256 | 0.8139 ± 0.0223 | 45.186 ± 6.333 |
| ColoTexture-Hybrid-SIMMIM | 0.9011 ± 0.0090 | 0.8367 ± 0.0095 | 0.8929 ± 0.0085 | 0.9285 ± 0.0134 | 0.9147 ± 0.0121 | 19.437 ± 4.043 |

仓库另提供三骨干阶梯（`compare_stage_comparison.txt`）、训练比例扫描
（`compare_fewshot_comparison.txt`，其评估集为验证划分，见表 2）与特征空间分析
（`evaluate_result.txt`）的同类报告。

`compare_downstream.py` 与 `compare_stage.py` 执行独立的种子扫描，因此同一 Hybrid-SIMMIM
骨干在两份报告中的聚合值略有差异（Dice 0.9011 与 0.9026），二者使用相同训练配方。

## 9. 复现顺序

```text
extract_frames.py
  → deduplicate_sequence_parallel.py
  → extract_bad_frames.py
  → check_dataset_overlap.py
  → （可选）select_images_mouse.py / select_images_keyboard.py
download_pretrained.py
SIMMIM-Hybrid.py / SIMMIM-Stage5.py / SIMMIM-Raw.py
evaluate.py
compare_downstream.py → compare_stage.py → compare_fewshot.py
generate_boxplot.py
```

## 10. 引用

数据集已注册 DOI：[10.17605/OSF.IO/JKBW5](https://doi.org/10.17605/OSF.IO/JKBW5)。

```bibtex
@dataset{colotexture_hybrid_2026,
  title     = {ColoTexture-Hybrid},
  author    = {z z and Sun, Xiaoyan and Xing, Xiaoyu and Yi, Xiaowen},
  year      = {2026},
  publisher = {OSF},
  doi       = {10.17605/OSF.IO/JKBW5},
  url       = {https://osf.io/jkbw5/},
  note      = {数据集，采用 CC BY 4.0 许可}
}
```

引用本仓库代码时可使用：

```bibtex
@misc{colotexture_source_2026,
  title        = {ColoTexture-Source: Self-Supervised Colonoscopy-Texture Pretraining
                  for Efficient Polyp Segmentation},
  author       = {<作者>},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/<user>/ColoTexture-Source}},
  note         = {Licensed under CC BY 4.0}
}
```

## 11. 许可证

本仓库采用[知识共享 署名 4.0 国际许可协议（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)
发布，完整法律文本见 [`LICENSE`](LICENSE)。以整体或部分形式再分发本材料时，须署名作者、
注明许可协议，并说明是否做过修改，示例：

```text
本作品基于 <作者> 的 “ColoTexture-Source”
（https://github.com/<user>/ColoTexture-Source）构建，采用 CC BY 4.0 许可
（https://creativecommons.org/licenses/by/4.0/）。修改内容：<说明>。
```

许可范围覆盖本仓库分发的源代码、训练权重与文档。公开息肉基准（Kvasir-SEG、CVC-300、
CVC-ClinicDB、CVC-ColonDB、ETIS-LaribPolypDB）以及原始内窥镜视频未在此重新授权，仍适用
其原始条款；再分发图像或其派生帧之前，请先查阅相应条款。

## 参考文献

- Xie, Z., et al. SimMIM: a Simple Framework for Masked Image Modeling. CVPR, 2022.
- Tan, M., Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks. ICML, 2019.
- Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical
  Image Segmentation. MICCAI, 2015.
