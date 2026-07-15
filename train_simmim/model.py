"""
EfficientNet-B1 with SimMIM (Simple Masked Image Modeling) Framework
====================================================================
Implements EfficientNet-B1 encoder, lightweight SimMIM decoder, and
random block-wise mask generation for self-supervised pre-training.

Reference:
    SimMIM: A Simple Framework for Masked Image Modeling (CVPR 2022)
    EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, in_channels, se_ratio):
        super().__init__()
        self.se_channels = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv2d(in_channels, self.se_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(self.se_channels, in_channels, kernel_size=1)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out.expand_as(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with optional SE."""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size // 2,
                                groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(Swish())

        # Squeeze-and-Excitation
        layers.append(SqueezeExcitation(hidden_dim, se_ratio))

        # Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNetB1Encoder(nn.Module):
    """EfficientNet-B1 backbone: 7.8M params, output 1280-dim features at 12x12."""
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        # (in_ch, out_ch, kernel, stride, expand_ratio, num_repeat)
        block_args = [
            {'in_ch': 32, 'out_ch': 16, 'kernel': 3, 'stride': 1, 'expand': 1, 'repeat': 1},
            {'in_ch': 16, 'out_ch': 24, 'kernel': 3, 'stride': 2, 'expand': 6, 'repeat': 2},
            {'in_ch': 24, 'out_ch': 40, 'kernel': 5, 'stride': 2, 'expand': 6, 'repeat': 2},
            {'in_ch': 40, 'out_ch': 80, 'kernel': 3, 'stride': 2, 'expand': 6, 'repeat': 3},
            {'in_ch': 80, 'out_ch': 112, 'kernel': 5, 'stride': 1, 'expand': 6, 'repeat': 3},
            {'in_ch': 112, 'out_ch': 192, 'kernel': 5, 'stride': 2, 'expand': 6, 'repeat': 4},
            {'in_ch': 192, 'out_ch': 320, 'kernel': 3, 'stride': 1, 'expand': 6, 'repeat': 1},
        ]

        self.stages = nn.ModuleList()
        se_ratio = 0.25

        for args in block_args:
            stage_blocks = []
            for i in range(args['repeat']):
                in_ch = args['in_ch'] if i == 0 else args['out_ch']
                stride = args['stride'] if i == 0 else 1
                stage_blocks.append(MBConv(
                    in_ch, args['out_ch'], args['kernel'], stride,
                    args['expand'], se_ratio
                ))
            self.stages.append(nn.Sequential(*stage_blocks))

        self.top_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)
        x = self.top_conv(x)
        stage_outputs.append(x)
        return x, stage_outputs


class SIMMIMDecoder(nn.Module):
    """Lightweight convolutional decoder for pixel reconstruction in SimMIM."""
    def __init__(self, in_channels=1280, patch_size=16, num_patches=24):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.decoder(x)
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        return x


class MaskGenerator:
    """Generates random block-wise binary masks for SimMIM pretraining.

    Args:
        input_size: Input image size (square).
        mask_patch_size: Size of each mask patch.
        mask_ratio: Proportion of patches to mask (0.0 - 1.0).
    """
    def __init__(self, input_size=384, mask_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = input_size // mask_patch_size

    def generate_mask(self, batch_size, device):
        """Generate random binary masks for a batch."""
        n_patches = self.num_patches
        n_masked = int(n_patches * n_patches * self.mask_ratio)

        masks = torch.zeros(batch_size, n_patches, n_patches, device=device)

        for i in range(batch_size):
            perm = torch.randperm(n_patches * n_patches, device=device)
            mask_indices = perm[:n_masked]
            masks[i].view(-1)[mask_indices] = 1

        return masks

    def apply_mask(self, images, masks):
        """Apply generated masks to images (zero out masked regions)."""
        b, c, h, w = images.shape
        patch_size = self.mask_patch_size

        masks_expanded = masks.unsqueeze(1)
        masks_upsampled = F.interpolate(masks_expanded.float(),
                                        size=(h, w), mode='nearest')
        masks_upsampled = masks_upsampled.bool()

        masked_images = images.clone()
        masked_images[masks_upsampled.expand_as(images)] = 0

        return masked_images


class EfficientNetB1SIMMIM(nn.Module):
    """EfficientNet-B1 backbone with SimMIM masked image modeling head.

    Combines encoder, mask generator, and decoder for end-to-end
    self-supervised pre-training.
    """
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetB1Encoder()
        self.decoder = SIMMIMDecoder(in_channels=1280)
        self.mask_generator = MaskGenerator(input_size=384, mask_patch_size=16, mask_ratio=0.6)

    def forward(self, images):
        masks = self.mask_generator.generate_mask(images.size(0), images.device)
        masked_images = self.mask_generator.apply_mask(images, masks)

        encoded, stage_outputs = self.encoder(masked_images)

        # Use the final stage output for reconstruction
        stage7_output = stage_outputs[-1]
        reconstructed = self.decoder(stage7_output)

        return reconstructed, masks, images, masked_images


def compute_masked_l1_loss(reconstructed, original, masks):
    """L1 loss computed only over masked patches."""
    b, c, h, w = reconstructed.shape

    masks_expanded = masks.unsqueeze(1)
    masks_upsampled = F.interpolate(masks_expanded.float(), size=(h, w), mode='nearest')

    masked_reconstructed = reconstructed * masks_upsampled
    masked_original = original * masks_upsampled

    loss = F.l1_loss(masked_reconstructed, masked_original, reduction='sum')

    num_masked_pixels = masks_upsampled.sum() * c
    loss = loss / num_masked_pixels

    return loss
