#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bad frame extraction script - 24-core parallel processing version
Adaptive threshold-based bad frame quality filtering.
Only retains poor-quality images (bad frames) for subsequent analysis.
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
OUTPUT_DIR = "/mnt/sda/Dataset/Polyp-bad-frames"

NUM_WORKERS = 24  # Number of parallel worker processes

# Memory limit configuration
MEMORY_LIMIT_GB = 400  # Memory usage cap 400 GB
BATCH_SIZE = 100000  # Process 100k images per batch

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(input_dir):
    """Retrieve all image files from input directory, sorted naturally."""
    image_files = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(filename)
    return natsorted(image_files)


def create_batches_fixed_size(image_files, batch_size=BATCH_SIZE):
    """Split image list into fixed-size batches."""
    batches = []
    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size

    print(f"  Batch size: {batch_size} images")
    print(f"  Total images: {total_images}, Batches: {num_batches}")

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
    """Load an image into memory (for parallel use)."""
    idx, image_path = args
    img = cv2.imread(image_path)
    if img is not None:
        return idx, img
    return idx, None


def detect_blur_fft(image, high_freq_ratio=0.1):
    """Blur detection: ratio of high-frequency energy in FFT domain."""
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
    """Ghost/blur detection: variance of Laplacian (edge sharpness)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def detect_artifact_band_energy(image):
    """Artifact detection: ratio of low-frequency energy."""
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
    """Return a dict of four quality metrics."""
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


def select_bad_frames(values, higher_is_better):
    """
    Auto-determine bad frame thresholds using Median Absolute Deviation (MAD).
    Opposite of adaptive_outlier_threshold: selects outliers (low-quality images).
    """
    if len(values) == 0:
        return np.array([], dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values, dtype=bool)  # No variance: select none
    if higher_is_better:
        # Higher is better, select those below threshold (poor quality)
        bad = values < (median - 3 * mad)
    else:
        # Lower is better, select those above threshold (poor quality)
        bad = values > (median + 3 * mad)
    return bad


def copy_file(args):
    """Copy a file (for multiprocessing parallel copy)."""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return None
    except Exception as e:
        return (src, str(e))


def process_single_batch(batch_info, input_dir, image_files, batch_num, total_batches):
    """Process a single batch (runs in main process, uses tqdm for progress)."""
    batch_files = batch_info['files']
    batch_indices = batch_info['indices']

    print(f"\n{'='*60}")
    print(f"Processing batch {batch_num}/{total_batches}")
    print(f"  Images: {len(batch_files)}")
    print(f"{'='*60}\n")

    # Step 1: Load batch images into memory
    print(f"[Batch {batch_num}] Loading images to memory...")
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
    print(f"  ✓ Loaded {len(images_in_memory)}/{len(batch_files)} images")
    print(f"  ✓ Image data memory: {actual_memory / (1024**3):.2f} GB")

    # Report system memory status after loading
    mem_after_load = psutil.virtual_memory()
    print(f"  ✓ System memory after loading: {mem_after_load.used / (1024**3):.2f} GB / {mem_after_load.total / (1024**3):.2f} GB ({mem_after_load.percent}%)")
    print()

    # Step 2: Quality assessment (parallel)
    print(f"[Batch {batch_num}] Performing quality assessment...")

    # Prepare batch data
    batch_size = max(1, len(images_in_memory) // NUM_WORKERS)
    quality_batches = []
    current_batch = []

    sorted_indices = sorted(images_in_memory.keys())

    for idx in sorted_indices:
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
            desc="  Evaluation",
            unit="batches"
        ))

    # Collect results
    for batch_result in batch_results:
        for scores in batch_result:
            if scores is not None:
                for k in quality_scores:
                    quality_scores[k].append(scores[k])

    # Map to corresponding frame indices
    valid_frames = []
    for idx in sorted_indices:
        if idx in images_in_memory:
            valid_frames.append(idx)

    for k in quality_scores:
        quality_scores[k] = np.array(quality_scores[k])

    # Use select_bad_frames to identify low-quality images
    bad_blur = select_bad_frames(quality_scores['blur'], higher_is_better=True)
    bad_ghost = select_bad_frames(quality_scores['ghost'], higher_is_better=True)
    bad_artifact = select_bad_frames(quality_scores['artifact'], higher_is_better=False)
    bad_fringe = select_bad_frames(quality_scores['fringe'], higher_is_better=False)

    # Keep frame if ANY quality metric indicates it is a bad frame
    final_bad = bad_blur | bad_ghost | bad_artifact | bad_fringe
    bad_indices = [valid_frames[i] for i in range(len(valid_frames)) if final_bad[i]]

    print(f"  ✓ Quality assessment completed")
    print(f"  ✓ Bad frames detected: {len(bad_indices)}/{len(valid_frames)}")
    
    # Print per-metric bad frame counts
    print(f"    - Blurry frames: {np.sum(bad_blur)}")
    print(f"    - Ghost frames: {np.sum(bad_ghost)}")
    print(f"    - Artifact frames: {np.sum(bad_artifact)}")
    print(f"    - Color fringing frames: {np.sum(bad_fringe)}")
    print(f"    - Unique bad frames (union): {len(bad_indices)}")
    print()

    # Thorough memory cleanup to prevent accumulation across batches
    # Step 1: Clear image reference list used in quality batches
    for batch in quality_batches:
        batch.clear()
    quality_batches.clear()

    # Step 2: Clear main image dictionary
    images_in_memory.clear()

    # Step 3: Clear loaded image list
    for i in range(len(loaded_images)):
        loaded_images[i] = (loaded_images[i][0], None)
    loaded_images.clear()

    # Step 4: Clear large quality score arrays
    for k in list(quality_scores.keys()):
        quality_scores[k] = None
    quality_scores.clear()

    # Step 5: Clear other large variables
    batch_results.clear() if 'batch_results' in locals() else None
    valid_frames.clear() if 'valid_frames' in locals() else None

    # Step 6: Delete variable references and force garbage collection
    del images_in_memory, loaded_images, quality_batches, quality_scores
    del batch_results, valid_frames, final_bad, bad_blur, bad_ghost, bad_artifact, bad_fringe
    del current_batch, batch_size

    # Step 7: Force garbage collection and release memory to OS
    gc.collect()

    # Step 8: Use malloc_trim to return memory to OS (Linux only)
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except:
        pass

    # Report system memory status after release
    mem_after_release = psutil.virtual_memory()
    print(f"  ✓ Memory after release: {mem_after_release.used / (1024**3):.2f} GB / {mem_after_release.total / (1024**3):.2f} GB ({mem_after_release.percent}%)")

    return bad_indices


def wait_for_memory_stable(baseline_memory, max_wait=30, stable_threshold=0.5):
    """Wait until memory release stabilizes."""
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
    print("Bad Frame Extraction Script - 24-core Parallel Processing")
    print("=" * 60)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Parallel Workers: {NUM_WORKERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Scan image files
    print("[Step 1/5] Scanning image files...")
    image_files = get_image_files(INPUT_DIR)
    total_images = len(image_files)

    if total_images == 0:
        print("Error: No image files found")
        return
    print(f"  ✓ Found {total_images} images")

    # Check system memory
    total_mem = psutil.virtual_memory().total / (1024**3)
    available_mem = psutil.virtual_memory().available / (1024**3)
    print(f"  Total memory: {total_mem:.1f} GB")
    print(f"  Available memory: {available_mem:.1f} GB")
    print()

    # Step 2: Create fixed-size batches
    print("[Step 2/5] Creating batches...")
    batches = create_batches_fixed_size(image_files, batch_size=BATCH_SIZE)
    print(f"  ✓ Created {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"    Batch {i+1}: {len(batch['files'])} images")
    print()

    # Step 3: Process batch by batch
    print("[Step 3/5] Processing batches (24-core parallel + memory monitoring)...")
    all_bad_indices = []

    # Record baseline memory
    baseline_memory = psutil.virtual_memory().used / (1024**3)
    print(f"  Baseline memory: {baseline_memory:.2f} GB")

    for batch_num, batch_info in enumerate(batches, 1):
        print(f"\n{'='*60}")
        print(f"[Batch {batch_num}/{len(batches)}]")
        print(f"{'='*60}")

        # From batch 2 onward: wait for memory to stabilize
        if batch_num > 1:
            wait_for_memory_stable(baseline_memory)

        mem_before = psutil.virtual_memory()
        print(f"  Memory before loading: {mem_before.used / (1024**3):.2f} GB / {mem_before.total / (1024**3):.2f} GB ({mem_before.percent}%)")
        print(f"  Images: {len(batch_info['files'])}")

        # Process batch in main process (uses tqdm for progress)
        batch_indices = process_single_batch(
            batch_info, INPUT_DIR, image_files, batch_num, len(batches)
        )
        all_bad_indices.extend(batch_indices)

        # Show progress
        print(f"  Cumulative bad frames: {len(all_bad_indices)}")
        print()

    print(f"✓ Processing complete. Total bad frames: {len(all_bad_indices)}/{total_images}\n")

    # Step 4: Copy bad frames to output directory
    print("[Step 4/5] Copying bad frames to output directory...")

    if len(all_bad_indices) == 0:
        print("  No bad frames found")
    else:
        copy_tasks = [(os.path.join(INPUT_DIR, image_files[idx]),
                       os.path.join(OUTPUT_DIR, image_files[idx]))
                      for idx in all_bad_indices]

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
            print(f"  Copy complete: {success_count} success, {fail_count} failed")
        else:
            print(f"  ✓ Successfully copied {success_count} bad frames")

    # Step 5: Generate report
    print("\n[Step 5/5] Generating report...")
    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Bad frames extracted: {len(all_bad_indices)}")
    if total_images > 0:
        print(f"Bad frame ratio: {len(all_bad_indices) / total_images * 100:.2f}%")
    else:
        print(f"Bad frame ratio: 0.00%")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
