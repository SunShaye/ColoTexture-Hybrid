"""
EfficientNet-B1 自监督预训练权重加载器
========================================
训练方法: SIMMIM (Simple Masked Image Modeling)
输入尺寸: 384×384 RGB
预训练数据: Polyp 内窥镜图像
训练轮数: 300 epochs

=== 保存的 checkpoint 结构 ===
best_model.pth 是一个 dict:
{
    'epoch': 最优 epoch 编号,
    'model_state_dict': EfficientNetB1SIMMIM 的完整 state_dict,
    'optimizer_state_dict': 优化器状态,
    'loss': 最优损失值,
}

=== 下游任务使用方法 ===
只需要 Encoder（EfficientNetB1Encoder），Decoder 和 MaskGenerator 是训练辅助模块。
Encoder 以 384×384 RGB 图像为输入，输出多尺度特征图。

=== Encoder 架构概览 ===
输入: [B, 3, 384, 384]
stem: Conv+BN+Swish, stride=2 → [B, 32, 192, 192]
Stage 1: MBConv x1, 不降采样 → [B, 16, 192, 192]
Stage 2: MBConv x2, stride=2   → [B, 24, 96, 96]
Stage 3: MBConv x2, stride=2   → [B, 40, 48, 48]
Stage 4: MBConv x3, stride=2   → [B, 80, 24, 24]
Stage 5: MBConv x3, stride=1   → [B, 112, 24, 24]
Stage 6: MBConv x4, stride=2   → [B, 192, 12, 12]
Stage 7: MBConv x1, stride=1   → [B, 320, 12, 12]
top_conv: Conv1x1+BN+Swish     → [B, 1280, 12, 12]
"""

import torch
import torch.nn as nn
from model import EfficientNetB1Encoder, EfficientNetB1SIMMIM


# ============================================================
# 方法 1: 从完整 checkpoint 只提取 Encoder 权重
# ============================================================
def load_encoder_only(checkpoint_path='best_model.pth'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    full_state = checkpoint['model_state_dict']

    encoder_state = {}
    for key, value in full_state.items():
        if key.startswith('encoder.'):
            encoder_state[key[len('encoder.'):]] = value

    encoder = EfficientNetB1Encoder()
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    print(f"Encoder loaded from epoch {checkpoint['epoch'] + 1}, loss {checkpoint['loss']:.4f}")
    return encoder


# ============================================================
# 方法 2: 直接加载 EfficientNetB1SIMMIM 再取 encoder
# ============================================================
def load_encoder_from_full(checkpoint_path='best_model.pth'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = EfficientNetB1SIMMIM()
    model.load_state_dict(checkpoint['model_state_dict'])

    encoder = model.encoder
    encoder.eval()

    print(f"Encoder loaded from epoch {checkpoint['epoch'] + 1}, loss {checkpoint['loss']:.4f}")
    return encoder


# ============================================================
# 方法 3: 作为特征提取器，获取多尺度特征
# ============================================================
class FeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path='best_model.pth'):
        super().__init__()
        self.encoder = load_encoder_only(checkpoint_path)

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder_only('best_model.pth').to(device)

    # ---------- 示例 1: 提取多尺度特征 ----------
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    with torch.no_grad():
        topconv_output, stage_outputs = encoder(dummy_input)

    print(f'\n--- 各层输出形状 ---')
    print(f'输入:                   [1, 3, 384, 384]')
    print(f'stem (stride=2):        {None}')
    for i, out in enumerate(stage_outputs):
        print(f'Stage {i+1} 输出:        {list(out.shape)}')

    # ---------- 示例 2: 用于分类下游任务 ----------
    class PolypClassifier(nn.Module):
        def __init__(self, checkpoint_path='best_model.pth', num_classes=2):
            super().__init__()
            self.encoder = load_encoder_only(checkpoint_path)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1280, num_classes)

        def forward(self, x):
            topconv_output, _ = self.encoder(x)
            pooled = self.pool(topconv_output).flatten(1)
            return self.fc(pooled)

    classifier = PolypClassifier('best_model.pth').to(device)
    with torch.no_grad():
        logits = classifier(dummy_input)
    print(f'\n--- 分类输出 ---')
    print(f'分类头 logits 形状:      {list(logits.shape)}')

    # ---------- 示例 3: 冻结 / 微调 ----------
    print(f'\n--- 微调策略 ---')
    print(f'Encoder 总参数量:        {sum(p.numel() for p in encoder.parameters()):,}')
    print(f'1. 冻结所有层: encoder.requires_grad_(False)')
    print(f'2. 只微调最后几层: 逐步解冻 Stage 6,7 和 top_conv')
    print(f'3. 全量微调: encoder.requires_grad_(True)')