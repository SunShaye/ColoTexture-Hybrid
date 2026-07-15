"""
EfficientNet-B1 SimMIM Pretrained Weight Loader
================================================
Training method: SimMIM (Simple Masked Image Modeling)
Input size: 384x384 RGB
Pretraining data: ColoTexture-Hybrid (colonoscopy images)
Training epochs: 300

=== Checkpoint structure ===
The checkpoint is a dict:
{
    'epoch': best epoch index,
    'model_state_dict': full state_dict of EfficientNetB1SIMMIM,
    'optimizer_state_dict': optimizer state,
    'loss': best loss value,
}

=== Downstream usage ===
Only the Encoder (EfficientNetB1Encoder) is needed; the Decoder and
MaskGenerator are training auxiliary modules.
The Encoder takes 384x384 RGB images as input and outputs multi-scale feature maps.

=== Encoder architecture overview ===
Input:                     [B, 3, 384, 384]
stem: Conv+BN+Swish, s=2   -> [B, 32, 192, 192]
Stage 1: MBConv x1, s=1    -> [B, 16, 192, 192]
Stage 2: MBConv x2, s=2    -> [B, 24, 96, 96]
Stage 3: MBConv x2, s=2    -> [B, 40, 48, 48]
Stage 4: MBConv x3, s=2    -> [B, 80, 24, 24]
Stage 5: MBConv x3, s=1    -> [B, 112, 24, 24]
Stage 6: MBConv x4, s=2    -> [B, 192, 12, 12]
Stage 7: MBConv x1, s=1    -> [B, 320, 12, 12]
top_conv: Conv1x1+BN+Swish -> [B, 1280, 12, 12]
"""

import torch
import torch.nn as nn
from model import EfficientNetB1Encoder, EfficientNetB1SIMMIM


# ============================================================
# Method 1: Extract encoder weights from full checkpoint
# ============================================================
def load_encoder_only(checkpoint_path='../pretrained_weights/effb1_simmim_colo.pth'):
    """Load only the encoder part from a SimMIM checkpoint."""
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
# Method 2: Load full EfficientNetB1SIMMIM then extract encoder
# ============================================================
def load_encoder_from_full(checkpoint_path='../pretrained_weights/effb1_simmim_colo.pth'):
    """Load the full SimMIM model and return its encoder."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = EfficientNetB1SIMMIM()
    model.load_state_dict(checkpoint['model_state_dict'])

    encoder = model.encoder
    encoder.eval()

    print(f"Encoder loaded from epoch {checkpoint['epoch'] + 1}, loss {checkpoint['loss']:.4f}")
    return encoder


# ============================================================
# Method 3: Feature extractor wrapper for multi-scale features
# ============================================================
class FeatureExtractor(nn.Module):
    """Wrapper that loads the encoder and exposes multi-scale feature extraction."""
    def __init__(self, checkpoint_path='../pretrained_weights/effb1_simmim_colo.pth'):
        super().__init__()
        self.encoder = load_encoder_only(checkpoint_path)

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# Usage examples
# ============================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder_only('../pretrained_weights/effb1_simmim_colo.pth').to(device)

    # ---- Example 1: Extract multi-scale features ----
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    with torch.no_grad():
        topconv_output, stage_outputs = encoder(dummy_input)

    print(f'\n--- Stage output shapes ---')
    print(f'Input:                   [1, 3, 384, 384]')
    for i, out in enumerate(stage_outputs):
        print(f'Stage {i+1} output:      {list(out.shape)}')

    # ---- Example 2: Downstream classification ----
    class PolypClassifier(nn.Module):
        """Example classifier head for polyp detection."""
        def __init__(self, checkpoint_path='../pretrained_weights/effb1_simmim_colo.pth', num_classes=2):
            super().__init__()
            self.encoder = load_encoder_only(checkpoint_path)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1280, num_classes)

        def forward(self, x):
            topconv_output, _ = self.encoder(x)
            pooled = self.pool(topconv_output).flatten(1)
            return self.fc(pooled)

    classifier = PolypClassifier('../pretrained_weights/effb1_simmim_colo.pth').to(device)
    with torch.no_grad():
        logits = classifier(dummy_input)
    print(f'\n--- Classification output ---')
    print(f'Logits shape:            {list(logits.shape)}')

    # ---- Example 3: Freezing / fine-tuning strategies ----
    print(f'\n--- Fine-tuning strategies ---')
    print(f'Encoder total params:    {sum(p.numel() for p in encoder.parameters()):,}')
    print(f'1. Freeze all:           encoder.requires_grad_(False)')
    print(f'2. Partial unfreeze:     progressively unfreeze Stage 6, 7, and top_conv')
    print(f'3. Full fine-tuning:     encoder.requires_grad_(True)')
