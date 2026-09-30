#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像序列自动剔除脚本 - 24核心并行处理版本
基于pHash的序列去重算法 + 自适应坏帧质量筛选
支持内存缓存（使用分批处理 + 内存监控）
"""

import os
import shutil
import warnings
import gc
import time
import numpy as np
import cv2
from natsort import natsorted
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import psutil
import ctypes

warnings.filterwarnings("ignore")

INPUT_DIR = "/mnt/sda/Dataset/Polyp-videoflame&pic_clean"
OUTPUT_DIR = "/mnt/sda/Dataset/Polyp-videoflame&pic_clean2"

HAMMING_THRESHOLD = 14
REF_QUEUE_MAX_SIZE = 10
NUM_WORKERS = 24  # 并行工作进程数

# 内存限制配置
MEMORY_LIMIT_GB = 400  # 内存使用上限400GB
BATCH_SIZE = 100000  # 每批处理10万张图像

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(input_dir):
    """获取输入目录中的所有图像文件，按文件名自然排序"""
    image_files = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(filename)
    return natsorted(image_files)


def create_batches_fixed_size(image_files, batch_size=BATCH_SIZE):
    """按固定数量分批"""
    batches = []
    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size

    print(f"  按固定数量分批（每批 {batch_size} 张图像）...")
    print(f"  总图像数: {total_images}, 预计分成 {num_batches} 批")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]

        batches.append({
            'files': batch_files,
            'batch_size': len(batch_files),
            'indices': list(range(start_idx, end_idx))
        })

    return batches


def load_images_to_memory(args):
    """加载图像到内存（用于并行）"""
    idx, image_path = args
    img = cv2.imread(image_path)
    if img is not None:
        return idx, img
    return idx, None


def compute_phash_from_image(img):
    """从内存中的图像计算pHash"""
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
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
    """计算两个64位哈希值的汉明距离"""
    return (hash1 ^ hash2).bit_count()


# ---------- 坏帧检测函数 ----------
def detect_blur_fft(image, high_freq_ratio=0.1):
    """模糊检测：频域高频能量占比"""
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
    """重影检测：边缘清晰度方差"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def detect_artifact_band_energy(image):
    """伪影检测：低频能量占比"""
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
    """彩色边缘检测"""
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
    """返回四个质量指标构成的字典"""
    return {
        'blur': detect_blur_fft(image),
        'ghost': detect_ghost_phase_corr(image),
        'artifact': detect_artifact_band_energy(image),
        'fringe': detect_color_fringing(image)
    }


def process_quality_batch(images_batch):
    """批量处理质量评估（用于并行）"""
    results = []
    for img in images_batch:
        if img is not None:
            results.append(compute_quality_scores(img))
        else:
            results.append(None)
    return results


def adaptive_outlier_threshold(values, higher_is_better):
    """使用中位数绝对偏差(MAD)自动确定离群阈值"""
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


def copy_file(args):
    """复制文件（用于多进程并行复制）"""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return None
    except Exception as e:
        return (src, str(e))


def process_single_batch(batch_info, input_dir, image_files, batch_num, total_batches):
    """处理单个批次的图像（在主进程中运行，使用tqdm显示进度）"""
    batch_files = batch_info['files']
    batch_indices = batch_info['indices']

    print(f"\n{'='*60}")
    print(f"处理第 {batch_num}/{total_batches} 批")
    print(f"  图像数量: {len(batch_files)} 张")
    print(f"{'='*60}\n")

    # 步骤1: 加载批次图像到内存
    print(f"[批次 {batch_num}] 加载图像到内存...")
    image_paths = [(idx, os.path.join(input_dir, filename))
                   for idx, filename in zip(batch_indices, batch_files)]

    with Pool(processes=NUM_WORKERS) as pool:
        loaded_images = list(tqdm(
            pool.imap(load_images_to_memory, image_paths),
            total=len(image_paths),
            desc="  加载",
            unit="帧"
        ))

    # 整理加载的图像
    images_in_memory = {}
    for idx, img in loaded_images:
        if img is not None:
            images_in_memory[idx] = img

    actual_memory = sum(img.nbytes for img in images_in_memory.values())
    print(f"  ✓ 成功加载 {len(images_in_memory)}/{len(batch_files)} 张图像")
    print(f"  ✓ 图像数据内存占用: {actual_memory / (1024**3):.2f} GB")

    # 报告加载后的系统内存状态
    mem_after_load = psutil.virtual_memory()
    print(f"  ✓ 加载后系统内存: {mem_after_load.used / (1024**3):.2f} GB / {mem_after_load.total / (1024**3):.2f} GB ({mem_after_load.percent}%)")
    print()

    # 步骤2: pHash计算和序列去重
    print(f"[批次 {batch_num}] 计算pHash并执行序列去重...")
    ref_queue = []
    retained_indices = []

    # 按原始顺序处理批次内的图像
    sorted_indices = sorted(images_in_memory.keys())

    for idx in tqdm(sorted_indices, desc="  去重", unit="帧"):
        img = images_in_memory[idx]
        phash = compute_phash_from_image(img)

        if phash is None:
            continue

        if not ref_queue:
            retained_indices.append(idx)
            ref_queue.append(phash)
        else:
            should_retain = True
            for ref_hash in ref_queue:
                if hamming_distance(phash, ref_hash) < HAMMING_THRESHOLD:
                    should_retain = False
                    break

            if should_retain:
                retained_indices.append(idx)
                ref_queue.append(phash)
                if len(ref_queue) > REF_QUEUE_MAX_SIZE:
                    ref_queue.pop(0)

    print(f"  ✓ pHash去重完成，保留 {len(retained_indices)}/{len(images_in_memory)} 张图像\n")

    # 步骤3: 坏帧质量筛选（并行处理）
    print(f"[批次 {batch_num}] 执行自适应坏帧质量筛选...")

    # 准备批量数据
    batch_size = max(1, len(retained_indices) // NUM_WORKERS)
    quality_batches = []
    current_batch = []

    for idx in retained_indices:
        if idx in images_in_memory:
            current_batch.append(images_in_memory[idx])
            if len(current_batch) >= batch_size:
                quality_batches.append(current_batch)
                current_batch = []
    if current_batch:
        quality_batches.append(current_batch)

    # 并行处理质量评估
    quality_scores = {
        'blur': [],
        'ghost': [],
        'artifact': [],
        'fringe': []
    }

    with Pool(processes=NUM_WORKERS) as pool:
        batch_results = list(tqdm(
            pool.imap(process_quality_batch, quality_batches),
            total=len(quality_batches),
            desc="  评估",
            unit="批"
        ))

    # 整理结果
    for batch_result in batch_results:
        for scores in batch_result:
            if scores is not None:
                for k in quality_scores:
                    quality_scores[k].append(scores[k])

    # 找到对应的帧索引
    valid_frames = []
    for idx in retained_indices:
        if idx in images_in_memory:
            valid_frames.append(idx)

    for k in quality_scores:
        quality_scores[k] = np.array(quality_scores[k])

    keep_blur = adaptive_outlier_threshold(quality_scores['blur'], higher_is_better=True)
    keep_ghost = adaptive_outlier_threshold(quality_scores['ghost'], higher_is_better=True)
    keep_artifact = adaptive_outlier_threshold(quality_scores['artifact'], higher_is_better=False)
    keep_fringe = adaptive_outlier_threshold(quality_scores['fringe'], higher_is_better=False)

    final_keep = keep_blur & keep_ghost & keep_artifact & keep_fringe
    final_indices = [valid_frames[i] for i in range(len(valid_frames)) if final_keep[i]]

    print(f"  ✓ 坏帧剔除后保留 {len(final_indices)} 张图像（剔除 {len(retained_indices) - len(final_indices)} 张）")

    # 彻底清理内存，防止批次间内存累积
    # 步骤1: 清理包含图像引用的批次列表（必须先做）
    for batch in quality_batches:
        batch.clear()
    quality_batches.clear()

    # 步骤2: 清理主图像字典
    images_in_memory.clear()

    # 步骤3: 清理加载的图像列表
    for i in range(len(loaded_images)):
        loaded_images[i] = (loaded_images[i][0], None)
    loaded_images.clear()

    # 步骤4: 清理质量评估相关的大型数组
    for k in list(quality_scores.keys()):
        quality_scores[k] = None
    quality_scores.clear()

    # 步骤5: 清理其他大型变量
    batch_results.clear() if 'batch_results' in locals() else None
    valid_frames.clear() if 'valid_frames' in locals() else None
    retained_indices.clear() if 'retained_indices' in locals() else None
    ref_queue.clear() if 'ref_queue' in locals() else None

    # 步骤6: 删除变量引用并强制垃圾回收
    del images_in_memory, loaded_images, quality_batches, quality_scores
    del batch_results, valid_frames, final_keep, keep_blur, keep_ghost, keep_artifact, keep_fringe
    del retained_indices, ref_queue, current_batch, batch_size

    # 步骤7: 强制垃圾回收并尝试释放内存给操作系统
    gc.collect()

    # 步骤8: 使用malloc_trim强制归还内存给操作系统（Linux only）
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except:
        pass

    # 报告释放内存后的系统内存状态
    mem_after_release = psutil.virtual_memory()
    print(f"  ✓ 内存释放后 - 系统内存占用: {mem_after_release.used / (1024**3):.2f} GB / {mem_after_release.total / (1024**3):.2f} GB ({mem_after_release.percent}%)")

    return final_indices


def wait_for_memory_stable(baseline_memory, max_wait=30, stable_threshold=0.5):
    """等待内存释放到稳定状态"""
    print("  等待内存释放到稳定状态...")
    stable_count = 0
    last_memory = psutil.virtual_memory().used / (1024**3)
    wait_iteration = 0

    while stable_count < 3 and wait_iteration < max_wait:
        time.sleep(1)
        current_memory = psutil.virtual_memory().used / (1024**3)
        memory_change = abs(current_memory - last_memory)

        print(f"    内存: {current_memory:.2f} GB (变化: {memory_change:+.3f} GB)")

        if memory_change < stable_threshold:
            stable_count += 1
        else:
            stable_count = 0

        last_memory = current_memory
        wait_iteration += 1

        gc.collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except:
            pass


def main():
    print("=" * 60)
    print("图像序列自动剔除脚本 - 24核心并行处理")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"汉明距离阈值: {HAMMING_THRESHOLD}")
    print(f"参考队列最大长度: {REF_QUEUE_MAX_SIZE}")
    print(f"并行工作进程: {NUM_WORKERS}")
    print(f"每批处理数量: {BATCH_SIZE}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 步骤1: 扫描图像文件
    print("[步骤 1/5] 扫描图像文件...")
    image_files = get_image_files(INPUT_DIR)
    total_images = len(image_files)

    if total_images == 0:
        print("错误: 输入目录中没有找到图像文件")
        return
    print(f"  ✓ 找到 {total_images} 张图像")

    # 检查系统内存
    total_mem = psutil.virtual_memory().total / (1024**3)
    available_mem = psutil.virtual_memory().available / (1024**3)
    print(f"  系统总内存: {total_mem:.1f} GB")
    print(f"  可用内存: {available_mem:.1f} GB")
    print()

    # 步骤2: 固定数量分批
    print("[步骤 2/5] 固定数量分批...")
    batches = create_batches_fixed_size(image_files, batch_size=BATCH_SIZE)
    print(f"  ✓ 分成 {len(batches)} 个批次处理")
    for i, batch in enumerate(batches):
        print(f"    批次 {i+1}: {len(batch['files'])} 张图像")
    print()

    # 步骤3: 逐批处理
    print("[步骤 3/5] 逐批处理图像（24核并行 + 内存监控）...")
    all_final_indices = []

    # 记录基线内存
    baseline_memory = psutil.virtual_memory().used / (1024**3)
    print(f"  系统基线内存: {baseline_memory:.2f} GB")

    for batch_num, batch_info in enumerate(batches, 1):
        print(f"\n{'='*60}")
        print(f"[批次 {batch_num}/{len(batches)}]")
        print(f"{'='*60}")

        # 非首轮：等待内存释放到稳定状态
        if batch_num > 1:
            wait_for_memory_stable(baseline_memory)

        mem_before = psutil.virtual_memory()
        print(f"  加载前系统内存: {mem_before.used / (1024**3):.2f} GB / {mem_before.total / (1024**3):.2f} GB ({mem_before.percent}%)")
        print(f"  图像数量: {len(batch_info['files'])} 张")

        # 在主进程中处理批次（使用tqdm显示进度）
        batch_indices = process_single_batch(
            batch_info, INPUT_DIR, image_files, batch_num, len(batches)
        )
        all_final_indices.extend(batch_indices)

        # 显示进度
        print(f"  累计保留: {len(all_final_indices)} 张图像")
        print()

    print(f"✓ 所有批次处理完成，共保留 {len(all_final_indices)}/{total_images} 张图像\n")

    # 步骤4: 复制图像到输出目录
    print("[步骤 4/5] 复制筛选后的图像到输出目录...")

    if len(all_final_indices) == 0:
        print("  ⚠ 没有图像需要复制")
    else:
        copy_tasks = [(os.path.join(INPUT_DIR, image_files[idx]),
                       os.path.join(OUTPUT_DIR, image_files[idx]))
                      for idx in all_final_indices]

        success_count = 0
        fail_count = 0
        with Pool(processes=NUM_WORKERS) as pool:
            for result in tqdm(
                pool.imap(copy_file, copy_tasks),
                total=len(copy_tasks),
                desc="  复制",
                unit="帧"
            ):
                if result is None:
                    success_count += 1
                else:
                    fail_count += 1

        if fail_count > 0:
            print(f"  ⚠ 复制完成: {success_count} 成功, {fail_count} 失败")
        else:
            print(f"  ✓ 成功复制 {success_count} 张图像")

    # 步骤5: 完成统计
    print("\n[步骤 5/5] 生成处理报告...")
    print()
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始图像数量: {total_images}")
    print(f"最终保留图像数量: {len(all_final_indices)}")
    if total_images > 0:
        print(f"总剔除率: {(total_images - len(all_final_indices)) / total_images * 100:.2f}%")
    else:
        print(f"总剔除率: 0.00%")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
