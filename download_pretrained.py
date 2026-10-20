"""
Download the official EfficientNet-B1 ImageNet pretrained weights.
"""
import torch
import torchvision.models as models

print("Downloading EfficientNet-B1 ImageNet pretrained weights...")

weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
model = models.efficientnet_b1(weights=weights)

state_dict = model.state_dict()
torch.save(state_dict, 'efficientnet_b1_imagenet.pth')

print(f"Download complete! Saved to: efficientnet_b1_imagenet.pth")
print(f"Number of weight layers: {len(state_dict)}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
