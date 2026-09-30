# -*- coding: utf-8 -*-
"""
图像序列自动剔除脚本
基于pHash的序列去重算法 + 自适应坏帧质量筛选
"""

import os
import shutil
import warnings
import numpy as np
import cv2
from natsort import natsorted
from tqdm import tqdm
from numba import cuda, jit

warnings.filterwarnings("ignore")

INPUT_DIR = r"E:\Images-RAW"
OUTPUT_DIR = r"E:\Datasets"

HAMMING_THRESHOLD = 4
REF_QUEUE_MAX_SIZE = 5

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(input_dir):
    """
    获取输入目录中的所有图像文件，按文件名自然排序
    """
    image_files = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(filename)
    image_files = natsorted(image_files)
    return image_files


# ---------- 纯CPU pHash计算 ----------
def compute_phash(image_path):
    """
    计算图像的感知哈希值(pHash)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img_resized))
    dct_low = dct[:8, :8]
    dct_low_flat = dct_low.flatten()[1:]
    median = np.median(dct_low_flat)
    hash_bits = (dct_low_flat > median).astype(np.uint8)

    phash = np.uint64(0)
    for bit in hash_bits:
        phash = (phash << 1) | bit

    return phash


def hamming_distance(hash1, hash2):
    """
    计算两个64位哈希值的汉明距离
    """
    return (hash1 ^ hash2).bit_count()


# ---------- 坏帧检测函数 ----------
def detect_blur_fft(image, high_freq_ratio=0.1):
    """
    模糊检测：频域高频能量占比，数值越高越清晰
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = int(min(rows, cols) * high_freq_ratio)
    cv2.circle(mask, (ccol, crow), r, 0, -1)

    total_energy = np.sum(magnitude_spectrum)
    high_freq_energy = np.sum(magnitude_spectrum * mask)
    return high_freq_energy / total_energy if total_energy > 0 else 0


def detect_ghost_phase_corr(image):
    """
    重影检测：边缘清晰度方差，数值越高边缘越锐利（重影越小）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def detect_artifact_band_energy(image):
    """
    伪影检测：选取画面水平中线最左侧的一小块区域，
    计算其低频能量占比。该值越高，压缩伪影越强。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    center_y = rows // 2
    half_h = max(1, int(rows * 0.05))
    region_w = max(1, int(cols * 0.1))
    y1, y2 = center_y - half_h, center_y + half_h
    x1, x2 = 0, region_w

    y1, y2 = max(0, y1), min(rows, y2)
    x1, x2 = max(0, x1), min(cols, x2)

    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    total_energy = np.sum(magnitude)
    if total_energy == 0:
        return 0.0

    ph, pw = patch.shape
    crow, ccol = ph // 2, pw // 2
    r = min(crow, ccol) // 2
    y_low1, y_low2 = crow - r, crow + r
    x_low1, x_low2 = ccol - r, ccol + r
    low_band = magnitude[y_low1:y_low2, x_low1:x_low2]
    low_energy = np.sum(low_band)

    return low_energy / total_energy


def detect_color_fringing(image):
    """
    彩色边缘检测：色度高梯度但亮度低梯度的像素占比
    数值越高彩色边缘越严重
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    L_grad = np.abs(cv2.Sobel(L, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(L, cv2.CV_64F, 0, 1))
    a_grad = np.abs(cv2.Sobel(a, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(a, cv2.CV_64F, 0, 1))
    b_grad = np.abs(cv2.Sobel(b, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(b, cv2.CV_64F, 0, 1))

    chroma_grad = np.sqrt(a_grad**2 + b_grad**2)

    L_grad_norm = L_grad / (L_grad.max() + 1e-8)
    chroma_grad_norm = chroma_grad / (chroma_grad.max() + 1e-8)

    fringe_mask = (chroma_grad_norm > 0.1) & (L_grad_norm < 0.3)
    return np.sum(fringe_mask) / image.size


def compute_quality_scores(image):
    """
    返回四个质量指标构成的字典
    """
    return {
        'blur': detect_blur_fft(image),
        'ghost': detect_ghost_phase_corr(image),
        'artifact': detect_artifact_band_energy(image),
        'fringe': detect_color_fringing(image)
    }


def adaptive_outlier_threshold(values, higher_is_better):
    """
    使用中位数绝对偏差(MAD)自动确定离群阈值
    """
    if len(values) == 0:
        return np.array([], dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.ones_like(values, dtype=bool)
    if higher_is_better:
        keep = values >= (median - 3 * mad)
    else:
        keep = values <= (median + 3 * mad)
    return keep


# ---------- 主流程 ----------
def main():
    print("=" * 60)
    print("图像序列自动剔除脚本")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"汉明距离阈值: {HAMMING_THRESHOLD}")
    print(f"参考队列最大长度: {REF_QUEUE_MAX_SIZE}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 步骤1: 扫描图像文件
    print("[步骤 1/4] 扫描图像文件...")
    image_files = get_image_files(INPUT_DIR)
    total_images = len(image_files)

    if total_images == 0:
        print("错误: 输入目录中没有找到图像文件")
        return
    print(f"  ✓ 找到 {total_images} 张图像\n")

    # 步骤2: pHash计算和序列去重
    print("[步骤 2/4] 计算pHash并执行序列去重...")
    ref_queue = []
    retained_indices = []

    for i, filename in enumerate(tqdm(image_files, desc="  处理", unit="帧")):
        image_path = os.path.join(INPUT_DIR, filename)
        phash = compute_phash(image_path)

        if phash is None:
            continue

        if not ref_queue:
            retained_indices.append(i)
            ref_queue.append(phash)
        else:
            should_retain = True
            for ref_hash in ref_queue:
                if hamming_distance(phash, ref_hash) < HAMMING_THRESHOLD:
                    should_retain = False
                    break

            if should_retain:
                retained_indices.append(i)
                ref_queue.append(phash)
                if len(ref_queue) > REF_QUEUE_MAX_SIZE:
                    ref_queue.pop(0)

    print(f"  ✓ pHash去重完成，共保留 {len(retained_indices)} 张图像\n")

    # 步骤3: 坏帧质量筛选
    print("[步骤 3/4] 执行自适应坏帧质量筛选...")
    quality_scores = {
        'blur': [],
        'ghost': [],
        'artifact': [],
        'fringe': []
    }
    valid_frames = []

    for idx in tqdm(retained_indices, desc="  评估质量", unit="帧"):
        image_path = os.path.join(INPUT_DIR, image_files[idx])
        img = cv2.imread(image_path)
        if img is None:
            continue
        scores = compute_quality_scores(img)
        for k in quality_scores:
            quality_scores[k].append(scores[k])
        valid_frames.append(idx)

    for k in quality_scores:
        quality_scores[k] = np.array(quality_scores[k])

    keep_blur = adaptive_outlier_threshold(quality_scores['blur'], higher_is_better=True)
    keep_ghost = adaptive_outlier_threshold(quality_scores['ghost'], higher_is_better=True)
    keep_artifact = adaptive_outlier_threshold(quality_scores['artifact'], higher_is_better=False)
    keep_fringe = adaptive_outlier_threshold(quality_scores['fringe'], higher_is_better=False)

    final_keep = keep_blur & keep_ghost & keep_artifact & keep_fringe
    final_indices = [valid_frames[i] for i in range(len(valid_frames)) if final_keep[i]]
    print(f"  ✓ 坏帧剔除后保留 {len(final_indices)} 张图像（剔除 {len(retained_indices) - len(final_indices)} 张）\n")

    # 步骤4: 复制图像到输出目录
    print("[步骤 4/4] 复制筛选后的图像到输出目录...")
    for idx in tqdm(final_indices, desc="  复制", unit="帧"):
        src_path = os.path.join(INPUT_DIR, image_files[idx])
        dst_path = os.path.join(OUTPUT_DIR, image_files[idx])
        shutil.copy2(src_path, dst_path)

    print()
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始图像数量: {total_images}")
    print(f"pHash去重后保留: {len(retained_indices)}")
    print(f"最终保留图像数量: {len(final_indices)}")
    print(f"总剔除率: {(total_images - len(final_indices)) / total_images * 100:.2f}%")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
