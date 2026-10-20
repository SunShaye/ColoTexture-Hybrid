"""Vanilla U-Net with an EfficientNet-B1 encoder.

Deliberately a *standard* U-Net: plain double-conv decoder blocks with bilinear
upsampling and skip connections only. No ASPP / attention / edge modules /
domain-adaptive normalisation, so downstream differences between pretrained
backbones can be attributed to the backbone alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (EfficientNetB1Encoder, remap_torchvision_b1_keys,
                   _resolve_placeholders)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """bilinear upsample to skip resolution -> concat -> double conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EfficientNetB1UNetStd(nn.Module):
    """For a 384x384 input the encoder taps are:

    stem (32 @ H/2), stages[1] (24 @ H/4), stages[2] (40 @ H/8),
    stages[4] (112 @ H/16), top_conv (1280 @ H/32) == bottleneck.
    """

    def __init__(self, backbone_path=None, num_classes=1):
        super().__init__()
        self.encoder = EfficientNetB1Encoder()
        if backbone_path is not None:
            self.load_backbone(backbone_path)

        self.up4 = UpBlock(1280, 112, 256)  # H/32 -> H/16
        self.up3 = UpBlock(256, 40, 128)    # H/16 -> H/8
        self.up2 = UpBlock(128, 24, 64)     # H/8  -> H/4
        self.up1 = UpBlock(64, 32, 32)      # H/4  -> H/2
        self.head = nn.Conv2d(32, num_classes, 1)

    def load_backbone(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        if isinstance(ckpt, dict) and 'encoder' in ckpt:
            ckpt = ckpt['encoder']
        if any(k.startswith('features.') for k in ckpt):
            # official torchvision efficientnet_b1 state_dict -> our encoder keys
            state = remap_torchvision_b1_keys(ckpt)
            state = _resolve_placeholders(self.encoder, state)
        else:
            prefixed = {k[len('encoder.'):]: v for k, v in ckpt.items()
                        if k.startswith('encoder.')}
            state = prefixed if prefixed else ckpt
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f'  [backbone] {path} -> missing={len(missing)} '
              f'unexpected={len(unexpected)}')
        return missing, unexpected

    def forward(self, x):
        input_size = x.shape[2:]

        stem = self.encoder.stem(x)               # 32   @ H/2
        feat = stem
        stages = []
        for stage in self.encoder.stages:
            feat = stage(feat)
            stages.append(feat)
        top = self.encoder.top_conv(feat)         # 1280 @ H/32

        x = self.up4(top, stages[4])              # 112 @ H/16
        x = self.up3(x, stages[2])                # 40  @ H/8
        x = self.up2(x, stages[1])                # 24  @ H/4
        x = self.up1(x, stem)                     # 32  @ H/2

        x = F.interpolate(x, size=input_size, mode='bilinear',
                          align_corners=False)
        return self.head(x)
