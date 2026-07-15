# -*- coding: utf-8 -*-
"""
Image Sequence Auto-Deduplication Script
pHash-based sequence deduplication + adaptive bad-frame quality filtering
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
    Get all image files in the input directory, sorted naturally by filename.
    """
    image_files = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(filename)
    image_files = natsorted(image_files)
    return image_files


# ---------- CPU-only pHash computation ----------
def compute_phash(image_path):
    """
    Compute the perceptual hash (pHash) of an image.
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
    Compute the Hamming distance between two 64-bit hash values.
    """
    return (hash1 ^ hash2).bit_count()


# ---------- Bad frame detection functions ----------
def detect_blur_fft(image, high_freq_ratio=0.1):
    """
    Blur detection: high-frequency energy ratio in frequency domain; higher is sharper.
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
    Ghost detection: Laplacian variance of edges; higher indicates sharper edges (less ghosting).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def detect_artifact_band_energy(image):
    """
    Artifact detection: samples a small left-aligned region on the horizontal midline
    and computes its low-frequency energy ratio. Higher values indicate stronger compression artifacts.
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
    Color fringing detection: proportion of pixels with high chroma gradient but low luminance gradient.
    Higher values indicate more severe color fringing.
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
    Return a dictionary of four quality metrics.
    """
    return {
        'blur': detect_blur_fft(image),
        'ghost': detect_ghost_phase_corr(image),
        'artifact': detect_artifact_band_energy(image),
        'fringe': detect_color_fringing(image)
    }


def adaptive_outlier_threshold(values, higher_is_better):
    """
    Automatically determine outlier thresholds using Median Absolute Deviation (MAD).
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


# ---------- Main pipeline ----------
def main():
    print("=" * 60)
    print("Image Sequence Auto-Deduplication Script")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Hamming distance threshold: {HAMMING_THRESHOLD}")
    print(f"Reference queue max size: {REF_QUEUE_MAX_SIZE}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Scan image files
    print("[Step 1/4] Scanning image files...")
    image_files = get_image_files(INPUT_DIR)
    total_images = len(image_files)

    if total_images == 0:
        print("Error: No image files found in input directory")
        return
    print(f"  ✓ Found {total_images} images\n")

    # Step 2: Compute pHash and perform sequence deduplication
    print("[Step 2/4] Computing pHash and performing sequence deduplication...")
    ref_queue = []
    retained_indices = []

    for i, filename in enumerate(tqdm(image_files, desc="  Processing", unit="frames")):
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

    print(f"  ✓ pHash dedup complete, retained {len(retained_indices)} images\n")

    # Step 3: Bad frame quality filtering
    print("[Step 3/4] Performing adaptive bad-frame quality filtering...")
    quality_scores = {
        'blur': [],
        'ghost': [],
        'artifact': [],
        'fringe': []
    }
    valid_frames = []

    for idx in tqdm(retained_indices, desc="  Assessing quality", unit="frames"):
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
    print(f"  ✓ After bad frame removal: retained {len(final_indices)} images (removed {len(retained_indices) - len(final_indices)})\n")

    # Step 4: Copy filtered images to output directory
    print("[Step 4/4] Copying filtered images to output directory...")
    for idx in tqdm(final_indices, desc="  Copying", unit="frames"):
        src_path = os.path.join(INPUT_DIR, image_files[idx])
        dst_path = os.path.join(OUTPUT_DIR, image_files[idx])
        shutil.copy2(src_path, dst_path)

    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Original image count: {total_images}")
    print(f"After pHash dedup: {len(retained_indices)}")
    print(f"Final retained image count: {len(final_indices)}")
    print(f"Total removal rate: {(total_images - len(final_indices)) / total_images * 100:.2f}%")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
