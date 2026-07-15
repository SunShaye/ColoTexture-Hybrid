#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Sequence Auto-Deduplication Script - 24-core parallel processing version
pHash-based sequence deduplication + adaptive bad-frame quality filtering
With memory caching (batch processing + memory monitoring)
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
NUM_WORKERS = 24  # Number of parallel worker processes

# Memory limit configuration
MEMORY_LIMIT_GB = 400  # Memory usage cap: 400GB
BATCH_SIZE = 100000  # Process 100K images per batch

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(input_dir):
    """Get all image files in the input directory, sorted naturally by filename."""
    image_files = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(filename)
    return natsorted(image_files)


def create_batches_fixed_size(image_files, batch_size=BATCH_SIZE):
    """Split into fixed-size batches."""
    batches = []
    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size

    print(f"  Batching by fixed size ({batch_size} images per batch)...")
    print(f"  Total images: {total_images}, expected {num_batches} batches")

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
    """Load image into memory (for parallel processing)."""
    idx, image_path = args
    img = cv2.imread(image_path)
    if img is not None:
        return idx, img
    return idx, None


def compute_phash_from_image(img):
    """Compute pHash from an in-memory image."""
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
    """Compute the Hamming distance between two 64-bit hash values."""
    return (hash1 ^ hash2).bit_count()


# ---------- Bad frame detection functions ----------
def detect_blur_fft(image, high_freq_ratio=0.1):
    """Blur detection: high-frequency energy ratio in frequency domain."""
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
    """Ghost detection: edge sharpness variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def detect_artifact_band_energy(image):
    """Artifact detection: low-frequency energy ratio."""
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
    """Color fringing detection."""
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
    """Return a dictionary of four quality metrics."""
    return {
        'blur': detect_blur_fft(image),
        'ghost': detect_ghost_phase_corr(image),
        'artifact': detect_artifact_band_energy(image),
        'fringe': detect_color_fringing(image)
    }


def process_quality_batch(images_batch):
    """Batch quality assessment (for parallel processing)."""
    results = []
    for img in images_batch:
        if img is not None:
            results.append(compute_quality_scores(img))
        else:
            results.append(None)
    return results


def adaptive_outlier_threshold(values, higher_is_better):
    """Automatically determine outlier thresholds using Median Absolute Deviation (MAD)."""
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
    """Copy file (for multi-process parallel copying)."""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return None
    except Exception as e:
        return (src, str(e))


def process_single_batch(batch_info, input_dir, image_files, batch_num, total_batches):
    """Process a single batch of images (runs in main process, uses tqdm for progress)."""
    batch_files = batch_info['files']
    batch_indices = batch_info['indices']

    print(f"\n{'='*60}")
    print(f"Processing batch {batch_num}/{total_batches}")
    print(f"  Image count: {len(batch_files)}")
    print(f"{'='*60}\n")

    # Step 1: Load batch images into memory
    print(f"[Batch {batch_num}] Loading images into memory...")
    image_paths = [(idx, os.path.join(input_dir, filename))
                   for idx, filename in zip(batch_indices, batch_files)]

    with Pool(processes=NUM_WORKERS) as pool:
        loaded_images = list(tqdm(
            pool.imap(load_images_to_memory, image_paths),
            total=len(image_paths),
            desc="  Loading",
            unit="frames"
        ))

    # Organize loaded images
    images_in_memory = {}
    for idx, img in loaded_images:
        if img is not None:
            images_in_memory[idx] = img

    actual_memory = sum(img.nbytes for img in images_in_memory.values())
    print(f"  ✓ Successfully loaded {len(images_in_memory)}/{len(batch_files)} images")
    print(f"  ✓ Image data memory usage: {actual_memory / (1024**3):.2f} GB")

    # Report system memory status after loading
    mem_after_load = psutil.virtual_memory()
    print(f"  ✓ System memory after loading: {mem_after_load.used / (1024**3):.2f} GB / {mem_after_load.total / (1024**3):.2f} GB ({mem_after_load.percent}%)")
    print()

    # Step 2: Compute pHash and perform sequence deduplication
    print(f"[Batch {batch_num}] Computing pHash and performing sequence deduplication...")
    ref_queue = []
    retained_indices = []

    # Process images in batch in original order
    sorted_indices = sorted(images_in_memory.keys())

    for idx in tqdm(sorted_indices, desc="  Dedup", unit="frames"):
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

    print(f"  ✓ pHash dedup complete, retained {len(retained_indices)}/{len(images_in_memory)} images\n")

    # Step 3: Bad frame quality filtering (parallel processing)
    print(f"[Batch {batch_num}] Performing adaptive bad-frame quality filtering...")

    # Prepare batch data
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

    # Parallel quality assessment
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
            desc="  Assessing",
            unit="batches"
        ))

    # Aggregate results
    for batch_result in batch_results:
        for scores in batch_result:
            if scores is not None:
                for k in quality_scores:
                    quality_scores[k].append(scores[k])

    # Find corresponding frame indices
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

    print(f"  ✓ After bad frame removal: retained {len(final_indices)} images (removed {len(retained_indices) - len(final_indices)})")

    # Thoroughly clean memory to prevent accumulation between batches
    # Step 1: Clear batch lists containing image references (must do first)
    for batch in quality_batches:
        batch.clear()
    quality_batches.clear()

    # Step 2: Clear main image dict
    images_in_memory.clear()

    # Step 3: Clear loaded image list
    for i in range(len(loaded_images)):
        loaded_images[i] = (loaded_images[i][0], None)
    loaded_images.clear()

    # Step 4: Clear large arrays from quality assessment
    for k in list(quality_scores.keys()):
        quality_scores[k] = None
    quality_scores.clear()

    # Step 5: Clear other large variables
    batch_results.clear() if 'batch_results' in locals() else None
    valid_frames.clear() if 'valid_frames' in locals() else None
    retained_indices.clear() if 'retained_indices' in locals() else None
    ref_queue.clear() if 'ref_queue' in locals() else None

    # Step 6: Delete variable references and force garbage collection
    del images_in_memory, loaded_images, quality_batches, quality_scores
    del batch_results, valid_frames, final_keep, keep_blur, keep_ghost, keep_artifact, keep_fringe
    del retained_indices, ref_queue, current_batch, batch_size

    # Step 7: Force garbage collection and try to return memory to OS
    gc.collect()

    # Step 8: Use malloc_trim to force memory return to OS (Linux only)
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except:
        pass

    # Report system memory status after release
    mem_after_release = psutil.virtual_memory()
    print(f"  ✓ After memory release - system memory: {mem_after_release.used / (1024**3):.2f} GB / {mem_after_release.total / (1024**3):.2f} GB ({mem_after_release.percent}%)")

    return final_indices


def wait_for_memory_stable(baseline_memory, max_wait=30, stable_threshold=0.5):
    """Wait for memory to stabilize after release."""
    print("  Waiting for memory to stabilize...")
    stable_count = 0
    last_memory = psutil.virtual_memory().used / (1024**3)
    wait_iteration = 0

    while stable_count < 3 and wait_iteration < max_wait:
        time.sleep(1)
        current_memory = psutil.virtual_memory().used / (1024**3)
        memory_change = abs(current_memory - last_memory)

        print(f"    Memory: {current_memory:.2f} GB (change: {memory_change:+.3f} GB)")

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
    print("Image Sequence Auto-Deduplication Script - 24-core Parallel Processing")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Hamming distance threshold: {HAMMING_THRESHOLD}")
    print(f"Reference queue max size: {REF_QUEUE_MAX_SIZE}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Scan image files
    print("[Step 1/5] Scanning image files...")
    image_files = get_image_files(INPUT_DIR)
    total_images = len(image_files)

    if total_images == 0:
        print("Error: No image files found in input directory")
        return
    print(f"  ✓ Found {total_images} images")

    # Check system memory
    total_mem = psutil.virtual_memory().total / (1024**3)
    available_mem = psutil.virtual_memory().available / (1024**3)
    print(f"  Total system memory: {total_mem:.1f} GB")
    print(f"  Available memory: {available_mem:.1f} GB")
    print()

    # Step 2: Fixed-size batching
    print("[Step 2/5] Fixed-size batching...")
    batches = create_batches_fixed_size(image_files, batch_size=BATCH_SIZE)
    print(f"  ✓ Split into {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"    Batch {i+1}: {len(batch['files'])} images")
    print()

    # Step 3: Process batch by batch
    print("[Step 3/5] Processing images batch by batch (24-core parallel + memory monitoring)...")
    all_final_indices = []

    # Record baseline memory
    baseline_memory = psutil.virtual_memory().used / (1024**3)
    print(f"  System baseline memory: {baseline_memory:.2f} GB")

    for batch_num, batch_info in enumerate(batches, 1):
        print(f"\n{'='*60}")
        print(f"[Batch {batch_num}/{len(batches)}]")
        print(f"{'='*60}")

        # Non-first batch: wait for memory to stabilize
        if batch_num > 1:
            wait_for_memory_stable(baseline_memory)

        mem_before = psutil.virtual_memory()
        print(f"  System memory before loading: {mem_before.used / (1024**3):.2f} GB / {mem_before.total / (1024**3):.2f} GB ({mem_before.percent}%)")
        print(f"  Image count: {len(batch_info['files'])}")

        # Process batch in main process (uses tqdm for progress)
        batch_indices = process_single_batch(
            batch_info, INPUT_DIR, image_files, batch_num, len(batches)
        )
        all_final_indices.extend(batch_indices)

        # Show progress
        print(f"  Cumulative retained: {len(all_final_indices)} images")
        print()

    print(f"✓ All batches complete, retained {len(all_final_indices)}/{total_images} images total\n")

    # Step 4: Copy filtered images to output directory
    print("[Step 4/5] Copying filtered images to output directory...")

    if len(all_final_indices) == 0:
        print("  ⚠ No images to copy")
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
                desc="  Copying",
                unit="frames"
            ):
                if result is None:
                    success_count += 1
                else:
                    fail_count += 1

        if fail_count > 0:
            print(f"  ⚠ Copy complete: {success_count} succeeded, {fail_count} failed")
        else:
            print(f"  ✓ Successfully copied {success_count} images")

    # Step 5: Generate report
    print("\n[Step 5/5] Generating processing report...")
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Original image count: {total_images}")
    print(f"Final retained image count: {len(all_final_indices)}")
    if total_images > 0:
        print(f"Total removal rate: {(total_images - len(all_final_indices)) / total_images * 100:.2f}%")
    else:
        print(f"Total removal rate: 0.00%")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
