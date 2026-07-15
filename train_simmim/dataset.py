"""
Dataset and augmentations for SimMIM self-supervised pre-training.

Provides custom augmentations tailored for colonoscopy images:
- RandomMotionBlur: Simulates motion blur during endoscopy
- RandomHSV: Random HSV color jittering for robustness
- PolypDataset: Unlabeled image dataset with composite augmentations
"""

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import cv2


class RandomMotionBlur:
    """Random directional motion blur augmentation.

    Simulates motion blur commonly seen in endoscopic video frames
    caused by camera movement.
    """
    def __init__(self, max_kernel_size=5):
        self.max_kernel_size = max_kernel_size

    def __call__(self, img):
        kernel_size = random.randint(2, self.max_kernel_size)
        angle = random.uniform(0, 180)
        angle_rad = np.deg2rad(angle)

        center = (kernel_size - 1) / 2.0
        x0 = center + center * np.cos(angle_rad)
        y0 = center + center * np.sin(angle_rad)
        x1 = center - center * np.cos(angle_rad)
        y1 = center - center * np.sin(angle_rad)

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.line(kernel, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1.0, 1)
        kernel = kernel / kernel.sum()

        img_array = np.array(img)
        img_blurred = np.zeros_like(img_array)
        for c in range(img_array.shape[2]):
            img_blurred[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)

        return Image.fromarray(img_blurred)

    def __repr__(self):
        return f'{self.__class__.__name__}(max_kernel_size={self.max_kernel_size})'


class RandomHSV:
    """Random HSV color space jittering for endoscopic images.

    Adjusts hue, saturation, and value within small ranges to improve
    robustness to varying lighting conditions in colonoscopy.
    """
    def __init__(self, h_range=0.02, s_range=0.1, v_range=0.1):
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, img):
        img_array = np.array(img)

        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.float32)

        h_shift = random.uniform(-self.h_range, self.h_range)
        s_shift = random.uniform(-self.s_range, self.s_range)
        v_shift = random.uniform(-self.v_range, self.v_range)

        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_shift * 180) % 180
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * (1 + s_shift), 0, 255)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * (1 + v_shift), 0, 255)

        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(img_rgb)

    def __repr__(self):
        return f'{self.__class__.__name__}(h_range={self.h_range}, s_range={self.s_range}, v_range={self.v_range})'


class PolypDataset(Dataset):
    """Unlabeled image dataset for SimMIM self-supervised pre-training.

    Recursively scans a directory for images and applies a composite
    augmentation pipeline:

    - RandomResizedCrop (384x384, scale 0.6-1.0)
    - RandomVerticalFlip
    - RandomHSV color jitter
    - RandomMotionBlur
    - ToTensor

    Args:
        data_dir: Root directory containing images (scanned recursively).
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomVerticalFlip(),
            RandomHSV(h_range=0.02, s_range=0.1, v_range=0.1),
            RandomMotionBlur(max_kernel_size=5),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image
