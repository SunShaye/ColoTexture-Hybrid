"""
Download official ImageNet pre-trained EfficientNet-B1 weights.

This script downloads the ImageNet1K_V1 weights from torchvision for
comparison with the SimMIM pre-trained weights.
"""

import torch
import torchvision.models as models

print("Downloading EfficientNet-B1 ImageNet pre-trained weights...")

weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
model = models.efficientnet_b1(weights=weights)

state_dict = model.state_dict()
torch.save(state_dict, 'effb1_imagenet.pth')

print(f"Download complete! Saved to: effb1_imagenet.pth")
print(f"Number of weight layers: {len(state_dict)}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
