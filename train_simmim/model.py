import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import MBConv as _TVMBConv
from torchvision.ops import StochasticDepth


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, sd_prob=0.0):
        super().__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        self.in_channels = in_channels
        hidden_dim = in_channels * expand_ratio
        squeeze_channels = max(1, in_channels // 4)

        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )

        self.se = SqueezeExcitation(hidden_dim, squeeze_channels)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.stochastic_depth = StochasticDepth(sd_prob, mode="row")

    def forward(self, x):
        out = x
        if self.expand is not None:
            out = self.expand(out)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = x + self.stochastic_depth(out)
        return out


class EfficientNetB1Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        block_args = [
            {'in_ch': 32, 'out_ch': 16, 'kernel': 3, 'stride': 1, 'expand': 1, 'repeat': 2},
            {'in_ch': 16, 'out_ch': 24, 'kernel': 3, 'stride': 2, 'expand': 6, 'repeat': 3},
            {'in_ch': 24, 'out_ch': 40, 'kernel': 5, 'stride': 2, 'expand': 6, 'repeat': 3},
            {'in_ch': 40, 'out_ch': 80, 'kernel': 3, 'stride': 2, 'expand': 6, 'repeat': 4},
            {'in_ch': 80, 'out_ch': 112, 'kernel': 5, 'stride': 1, 'expand': 6, 'repeat': 4},
            {'in_ch': 112, 'out_ch': 192, 'kernel': 5, 'stride': 2, 'expand': 6, 'repeat': 5},
            {'in_ch': 192, 'out_ch': 320, 'kernel': 3, 'stride': 1, 'expand': 6, 'repeat': 2},
        ]

        self.stages = nn.ModuleList()
        se_ratio = 0.25

        total_blocks = sum(args['repeat'] for args in block_args)
        block_idx = 0
        for stage_i, args in enumerate(block_args):
            stage_blocks = []
            for i in range(args['repeat']):
                in_ch = args['in_ch'] if i == 0 else args['out_ch']
                stride = args['stride'] if i == 0 else 1
                sd_prob = 0.2 * block_idx / max(1, total_blocks - 1)
                stage_blocks.append(MBConv(
                    in_ch, args['out_ch'], args['kernel'], stride,
                    args['expand'], se_ratio, sd_prob
                ))
                block_idx += 1
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
    def __init__(self, input_size=384, mask_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = input_size // mask_patch_size

    def generate_mask(self, batch_size, device):
        n_patches = self.num_patches
        n_masked = int(n_patches * n_patches * self.mask_ratio)

        masks = torch.zeros(batch_size, n_patches, n_patches, device=device)

        for i in range(batch_size):
            perm = torch.randperm(n_patches * n_patches, device=device)
            mask_indices = perm[:n_masked]
            masks[i].view(-1)[mask_indices] = 1

        return masks

    def apply_mask(self, images, masks):
        b, c, h, w = images.shape
        patch_size = self.mask_patch_size
        n_patches = self.num_patches

        masks_expanded = masks.unsqueeze(1)
        masks_upsampled = F.interpolate(masks_expanded.float(),
                                        size=(h, w), mode='nearest')
        masks_upsampled = masks_upsampled.bool()

        masked_images = images.clone()
        masked_images[masks_upsampled.expand_as(images)] = 0

        return masked_images


class EfficientNetB1SIMMIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetB1Encoder()
        self.decoder = SIMMIMDecoder(in_channels=1280)
        self.mask_generator = MaskGenerator(input_size=384, mask_patch_size=16, mask_ratio=0.6)

    def forward(self, images):
        masks = self.mask_generator.generate_mask(images.size(0), images.device)
        masked_images = self.mask_generator.apply_mask(images, masks)

        encoded, stage_outputs = self.encoder(masked_images)

        stage7_output = stage_outputs[-1]
        reconstructed = self.decoder(stage7_output)

        return reconstructed, masks, images, masked_images


def compute_masked_l1_loss(reconstructed, original, masks):
    b, c, h, w = reconstructed.shape

    masks_expanded = masks.unsqueeze(1)
    masks_upsampled = F.interpolate(masks_expanded.float(), size=(h, w), mode='nearest')

    masked_reconstructed = reconstructed * masks_upsampled
    masked_original = original * masks_upsampled

    loss = F.l1_loss(masked_reconstructed, masked_original, reduction='sum')

    num_masked_pixels = masks_upsampled.sum() * c
    loss = loss / num_masked_pixels

    return loss


def remap_torchvision_b1_keys(tv_state_dict):
    """Map torchvision efficientnet_b1 state_dict keys to EfficientNetB1Encoder keys.

    torchvision: features.{idx}.{block_idx}.block.{layer_idx}.{sub_idx}...
    my encoder:
        stages.{s}.{b}.expand.0.weight  (if expand_ratio != 1)
        stages.{s}.{b}.depthwise.0.weight
        stages.{s}.{b}.se.fc1.weight
        stages.{s}.{b}.project.0.weight
    """
    my_state = {}
    for key, val in tv_state_dict.items():
        if not key.startswith('features.'):
            continue
        parts = key.split('.')
        idx = int(parts[1])
        rest = parts[2:]
        if idx == 0:
            new_key = 'stem.' + '.'.join(rest)
            my_state[new_key] = val
        elif idx == 8:
            new_key = 'top_conv.' + '.'.join(rest)
            my_state[new_key] = val
        elif 1 <= idx <= 7:
            stage_idx = idx - 1
            # rest = [block_idx, 'block', layer_idx, sub_idx, ...]
            block_idx = rest[0]
            assert rest[1] == 'block'
            layer_idx = int(rest[2])
            sub_idx = rest[3]
            tail = rest[4:]
            # layer_idx in TV: 0=expand (or depthwise if expand=1), 1=depthwise (or SE if expand=1), 2=SE (or project if expand=1), 3=project
            # We determine the right sub-module by structure:
            # Since SE has fc1/fc2, project has 0/1 (Conv/BN), depthwise/expand have 0/1/2 (Conv/BN/Activation)
            if sub_idx == 'fc1' or sub_idx == 'fc2':
                sub_module = 'se'
                new_sub = sub_idx
                new_tail = '.'.join(tail)
                new_key = f'stages.{stage_idx}.{block_idx}.se.{new_sub}.{new_tail}'
            else:
                new_key = f'stages.{stage_idx}.{block_idx}.__PH__.{layer_idx}.{sub_idx}.' + '.'.join(tail)
            my_state[new_key] = val
        else:
            continue
    return my_state


def _resolve_placeholders(encoder, my_state):
    """Replace __PH__ placeholder with actual sub-module names (expand/depthwise/project)."""
    out = {}
    for key, val in my_state.items():
        if '__PH__' not in key:
            out[key] = val
            continue
        parts = key.split('.')
        s, b, block_idx, ph, layer_idx_s, sub_idx = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        assert ph == '__PH__'
        layer_idx = int(layer_idx_s)
        tail = parts[6:]
        block = encoder.stages[int(b)][int(block_idx)]
        if layer_idx == 0:
            sub = 'expand' if block.expand is not None else 'depthwise'
        elif layer_idx == 1:
            sub = 'depthwise' if block.expand is not None else 'expand'
        elif layer_idx == 2:
            sub = 'project'
        elif layer_idx == 3:
            sub = 'project'
        else:
            raise ValueError(f'Unexpected layer_idx {layer_idx}')
        new_key = f'{s}.{b}.{block_idx}.{sub}.{sub_idx}.' + '.'.join(tail)
        out[new_key] = val
    return out


def load_torchvision_b1_weights(encoder, weights_path='efficientnet_b1_imagenet.pth', strict=True):
    tv_state = torch.load(weights_path, map_location='cpu', weights_only=True)
    my_state = remap_torchvision_b1_keys(tv_state)
    my_state = _resolve_placeholders(encoder, my_state)
    missing, unexpected = encoder.load_state_dict(my_state, strict=strict)
    if not strict:
        print(f'Loaded B1 ImageNet weights (non-strict): missing={len(missing)}, unexpected={len(unexpected)}')
        if missing:
            print(f'  Missing keys (first 5): {missing[:5]}')
        if unexpected:
            print(f'  Unexpected keys (first 5): {unexpected[:5]}')
    return missing, unexpected
