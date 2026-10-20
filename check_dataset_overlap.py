#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-dataset image overlap detection script (check_dataset_overlap.py, CUDA-accelerated pHash + multi-method cross-validation)

Compare:
  Query set: /mnt/sda/Dataset/ColoTexture-Hybrid                 (27124 jpg images)
  Reference set: /mnt/sda/Dataset/Kvasir-SEG/images              (1000 jpg images)

Pipeline:
  1. Load all images from both sides into memory (24 threads, cv2 decoding releases the GIL).
  2. Batch-compute pHash (DCT) and dHash perceptual hashes on GPU (CUDA),
     then compute all cross Hamming distances at once via matrix multiplication,
     producing the candidate pairs (union).
  3. Rigorous multi-stage verification:
       a. Exact match on file bytes MD5 / decoded pixel MD5;
       b. Pairwise vectorized comparison of 128x128 grayscale descriptors (MSE + zero-mean
          normalized cross-correlation ZNCC), while also testing all four orientations
          (horizontal flip / vertical flip / 180-degree rotation) to catch flip leakage;
       c. Re-verify confirmed pairs at 256x256, and at original resolution in grayscale
          when the sizes match.
  4. Write a txt report: final statistics + the specific overlapping image names.

Runtime environment: conda Unet (torch 2.8 + cuda 12.9)
"""

import os
import math
import time
import hashlib
import numpy as np
import cv2
import torch
from natsort import natsorted
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ======================================================================
# Configuration
# ======================================================================
QUERY_DIR = "/mnt/sda/Dataset/ColoTexture-Hybrid"
REF_DIRS = {
    "Kvasir-SEG/images": "/mnt/sda/Dataset/Kvasir-SEG/images",
}
REPORT_PATH = "/home/linux/PycharmProjects/Process_Datasets/overlap_report_colotexture_vs_kvasir_seg.txt"
OLD_REPORT_PATH = "/home/linux/PycharmProjects/Process_Datasets/overlap_report.txt"

NUM_WORKERS = 24              # Number of parallel worker threads (cv2 releases the GIL while decoding)
PHASH_THRESHOLD = 10          # pHash Hamming distance threshold (<= is treated as a candidate)
DHASH_THRESHOLD = 10          # dHash Hamming distance threshold (<= is treated as a candidate)
DESC_SIZE = 128               # Grayscale descriptor size used for pairwise verification
CONFIRM_MSE = 50.0            # Descriptor MSE threshold (<= is treated as a pixel-level confirmation)
CONFIRM_CORR = 0.995          # Descriptor ZNCC threshold (>= is treated as a pixel-level confirmation)
REVERIFY_SIZE = 256           # Re-verification size for confirmed pairs
HASH_BATCH = 1024             # Number of images processed by the GPU per batch
CMP_CHUNK = 4096              # Chunk size for cross comparison
VERIFY_CHUNK = 4096           # Chunk size for pairwise verification


SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ======================================================================
# Basic Utilities
# ======================================================================
def list_images(directory):
    files = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXT:
            files.append(name)
    return natsorted(files)


def make_dct_matrix(n, device):
    """Orthogonal n x n DCT-II matrix (row k corresponds to frequency k)"""
    idx = torch.arange(n, device=device, dtype=torch.float32)
    C = torch.cos(math.pi * (2.0 * idx.view(-1, 1) + 1.0) * idx.view(1, -1) / (2.0 * n))
    C[0] *= math.sqrt(1.0 / n)
    C[1:] *= math.sqrt(2.0 / n)
    return C


def load_one(path):
    """Read a single image and compute its CPU-side features"""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        file_md5 = hashlib.md5(raw).hexdigest()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {'path': path, 'img': None, 'file_md5': file_md5, 'pix_md5': None,
                    'gray32': None, 'gray98': None, 'gray128': None, 'shape': None}
        pix_md5 = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {
            'path': path, 'img': img, 'file_md5': file_md5, 'pix_md5': pix_md5,
            'gray32': cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA),
            'gray98': cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA),
            'gray128': cv2.resize(gray, (DESC_SIZE, DESC_SIZE), interpolation=cv2.INTER_AREA),
            'shape': img.shape,
        }
    except Exception as e:
        return {'path': path, 'img': None, 'file_md5': None, 'pix_md5': None,
                'gray32': None, 'gray98': None, 'gray128': None, 'shape': None, 'error': str(e)}


def load_dataset(directory, tag):
    names = list_images(directory)
    paths = [os.path.join(directory, n) for n in names]
    print(f"  [{tag}] Found {len(paths)} images, loading into memory ...")
    records = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for i, rec in enumerate(tqdm(ex.map(load_one, paths),
                                     total=len(paths), desc=f"  Loading {tag}", unit="images")):
            records[i] = rec
    ok = sum(1 for r in records if r['img'] is not None)
    mem = sum(r['img'].nbytes for r in records if r['img'] is not None)
    print(f"  [{tag}] Successfully loaded {ok}/{len(records)} images (failed {len(records) - ok}), decoded memory {mem / (1024 ** 3):.2f} GB")
    return records


def build_hashes(records, tag):
    """Batch-compute pHash / dHash on the GPU (returns 0/1 bit matrices)"""
    valid = [r for r in records if r['img'] is not None]
    n = len(valid)
    ph = torch.zeros((n, 64), dtype=torch.float32, device=DEVICE)
    dh = torch.zeros((n, 64), dtype=torch.float32, device=DEVICE)
    C = make_dct_matrix(32, DEVICE)
    print(f"  [{tag}] Computing pHash/dHash on {DEVICE.upper()} ({n} images) ...")
    for start in tqdm(range(0, n, HASH_BATCH), desc=f"  Hash {tag}", unit="batch"):
        end = min(start + HASH_BATCH, n)
        chunk = valid[start:end]
        g32 = np.stack([r['gray32'] for r in chunk]).astype(np.float32) / 255.0
        g98 = np.stack([r['gray98'] for r in chunk]).astype(np.float32)
        t32 = torch.from_numpy(g32).to(DEVICE)
        t98 = torch.from_numpy(g98).to(DEVICE)
        with torch.no_grad():
            dct = torch.einsum('kr,nrc,sc->nks', C, t32, C)
            low = dct[:, :8, :8].reshape(-1, 64)
            med = low.median(dim=1, keepdim=True).values
            ph[start:end] = (low > med).float()
            dh[start:end] = (t98[:, :, 1:] > t98[:, :, :-1]).reshape(-1, 64).float()
    return valid, ph, dh


def hamming_candidates(q_bits, r_bits, threshold):
    """Compute cross Hamming distances by matrix multiplication and return (qi, ri, dist) arrays"""
    popr = r_bits.sum(dim=1)
    out_q, out_r, out_d = [], [], []
    qn = q_bits.shape[0]
    for start in range(0, qn, CMP_CHUNK):
        end = min(start + CMP_CHUNK, qn)
        qc = q_bits[start:end]
        popq = qc.sum(dim=1)
        H = popq[:, None] + popr[None, :] - 2.0 * (qc @ r_bits.t())
        idx = (H <= threshold).nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        out_q.append(idx[:, 0].cpu().numpy() + start)
        out_r.append(idx[:, 1].cpu().numpy())
        out_d.append(H[idx[:, 0], idx[:, 1]].cpu().numpy())
    if not out_q:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(out_q), np.concatenate(out_r), np.concatenate(out_d)


def build_md5_index(records, key):
    idx = {}
    for i, r in enumerate(records):
        if r.get(key) is not None:
            idx.setdefault(r[key], []).append(i)
    return idx


def verify_candidates(q_desc, r_desc, qi, ri):
    """Pairwise verification of candidate pairs using 128x128 descriptors (including 4 orientations), vectorized + chunked

    q_desc / r_desc are the descriptor arrays of the query set and reference set respectively; qi / ri are the pairing indices.
    """
    results = np.empty(len(qi), dtype=[
        ('mse', np.float32), ('corr', np.float32), ('orient', 'U8')])
    for s in range(0, len(qi), VERIFY_CHUNK):
        e = min(s + VERIFY_CHUNK, len(qi))
        A = q_desc[qi[s:e]].astype(np.float32)
        B0 = r_desc[ri[s:e]].astype(np.float32)
        Am = A - A.mean(axis=(1, 2), keepdims=True)
        An = np.sqrt((Am * Am).sum(axis=(1, 2)))

        best_mse = np.full(e - s, np.inf, np.float32)
        best_corr = np.full(e - s, -2.0, np.float32)
        best_orient = np.array(['none'] * (e - s), dtype='U8')
        variants = [
            ('none',   B0),
            ('fliplr', B0[:, :, ::-1]),
            ('flipud', B0[:, ::-1, :]),
            ('rot180', B0[:, ::-1, ::-1]),
        ]
        for name, B in variants:
            mse = ((A - B) ** 2).mean(axis=(1, 2))
            Bm = B - B.mean(axis=(1, 2), keepdims=True)
            Bn = np.sqrt((Bm * Bm).sum(axis=(1, 2)))
            num = (Am * Bm).sum(axis=(1, 2))
            den = An * Bn
            corr = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
            upd = corr > best_corr
            best_corr[upd] = corr[upd]
            best_mse[upd] = mse[upd]
            best_orient[upd] = name
        results['mse'][s:e] = best_mse
        results['corr'][s:e] = best_corr
        results['orient'][s:e] = best_orient
    return results


def reverify(img_q, img_r, orient):
    """Re-verify a confirmed pair at 256x256 and return (mse, corr)"""
    gq = cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    if orient == 'fliplr':
        gr = gr[:, ::-1]
    elif orient == 'flipud':
        gr = gr[::-1, :]
    elif orient == 'rot180':
        gr = gr[::-1, ::-1]
    gq = cv2.resize(gq, (REVERIFY_SIZE, REVERIFY_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gr = cv2.resize(gr, (REVERIFY_SIZE, REVERIFY_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    mse = float(np.mean((gq - gr) ** 2))
    a = gq.ravel() - gq.mean()
    b = gr.ravel() - gr.mean()
    den = math.sqrt(float((a * a).sum())) * math.sqrt(float((b * b).sum()))
    corr = float((a * b).sum() / den) if den > 0 else 0.0
    return mse, corr


# ======================================================================
# Main Pipeline
# ======================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("Cross-dataset image overlap detection (CUDA pHash + rigorous multi-method verification)")
    print("=" * 72)
    print(f"Query directory     : {QUERY_DIR}")
    for k, v in REF_DIRS.items():
        print(f"Reference directory : {k} -> {v}")
    print(f"Device              : {DEVICE.upper()} | parallel threads: {NUM_WORKERS}")
    print(f"pHash/dHash threshold: {PHASH_THRESHOLD} | verification: MSE<={CONFIRM_MSE} or ZNCC>={CONFIRM_CORR}")
    print("=" * 72)

    # 1) Load into memory
    print("\n[Step 1/6] Loading all images into memory ...")
    q_records = load_dataset(QUERY_DIR, "ColoTexture-Hybrid")
    ref_pack = {}
    for tag, d in REF_DIRS.items():
        ref_pack[tag] = load_dataset(d, tag)

    # 2) GPU hashing
    print("\n[Step 2/6] Computing perceptual hashes (GPU) ...")
    q_valid, q_ph, q_dh = build_hashes(q_records, "ColoTexture-Hybrid")
    ref_data = {}
    for tag, recs in ref_pack.items():
        rv, rph, rdh = build_hashes(recs, tag)
        desc = np.stack([r['gray128'] for r in rv]) if rv else np.zeros((0, DESC_SIZE, DESC_SIZE), np.uint8)
        ref_data[tag] = {'valid': rv, 'ph': rph, 'dh': rdh, 'desc': desc}

    q_desc = np.stack([r['gray128'] for r in q_valid]) if q_valid else np.zeros((0, DESC_SIZE, DESC_SIZE), np.uint8)

    # 3) Exact matching (MD5)
    print("\n[Step 3/6] Exact matching (file MD5 / decoded pixel MD5) ...")
    q_file_idx = build_md5_index(q_valid, 'file_md5')
    q_pix_idx = build_md5_index(q_valid, 'pix_md5')
    for tag, data in ref_data.items():
        md5_file_pairs = []
        md5_pix_pairs = []
        r_file_idx = build_md5_index(data['valid'], 'file_md5')
        r_pix_idx = build_md5_index(data['valid'], 'pix_md5')
        for h, qis in q_file_idx.items():
            if h in r_file_idx:
                for qi in qis:
                    for ri in r_file_idx[h]:
                        md5_file_pairs.append((qi, ri))
        for h, qis in q_pix_idx.items():
            if h in r_pix_idx:
                for qi in qis:
                    for ri in r_pix_idx[h]:
                        md5_pix_pairs.append((qi, ri))
        data['md5_file'] = md5_file_pairs
        data['md5_pix'] = md5_pix_pairs
        print(f"  [{tag}] File MD5 matches: {len(md5_file_pairs)} pairs | pixel MD5 matches: {len(md5_pix_pairs)} pairs")

    # 4) Perceptual hash candidates
    print("\n[Step 4/6] Cross-comparison of perceptual hashes (pHash / dHash, union) ...")
    for tag, data in ref_data.items():
        ph_q, ph_r, ph_d = hamming_candidates(q_ph, data['ph'], PHASH_THRESHOLD)
        dh_q, dh_r, dh_d = hamming_candidates(q_dh, data['dh'], DHASH_THRESHOLD)
        cand = {}
        for qi, ri, d in zip(ph_q, ph_r, ph_d):
            cand[(int(qi), int(ri))] = {'phash': float(d), 'dhash': None}
        for qi, ri, d in zip(dh_q, dh_r, dh_d):
            cand.setdefault((int(qi), int(ri)), {'phash': None, 'dhash': None})['dhash'] = float(d)
        for (qi, ri) in data['md5_file']:
            cand.setdefault((qi, ri), {'phash': None, 'dhash': None})
        for (qi, ri) in data['md5_pix']:
            cand.setdefault((qi, ri), {'phash': None, 'dhash': None})
        data['cand'] = cand
        print(f"  [{tag}] pHash candidates: {len(ph_q)} pairs, dHash candidates: {len(dh_q)} pairs, {len(cand)} pairs after merging and deduplication")

    # 5) Pairwise verification
    print("\n[Step 5/6] Pairwise descriptor verification (128x128, including flips) + re-verification ...")
    detailed = []
    for tag, data in ref_data.items():
        keys = list(data['cand'].keys())
        if not keys:
            data['detailed'] = []
            continue
        qi = np.array([k[0] for k in keys], np.int64)
        ri = np.array([k[1] for k in keys], np.int64)
        vres = verify_candidates(q_desc, data['desc'], qi, ri)
        md5_file_set = set(data['md5_file'])
        md5_pix_set = set(data['md5_pix'])
        rows = []
        for k, (qi_, ri_) in enumerate(tqdm(keys, desc=f"  Collecting {tag}", unit="pairs")):
            mse = float(vres['mse'][k]); corr = float(vres['corr'][k]); orient = str(vres['orient'][k])
            rec = data['cand'][(qi_, ri_)]
            is_md5 = (qi_, ri_) in md5_file_set or (qi_, ri_) in md5_pix_set
            pixel_ok = (mse <= CONFIRM_MSE or corr >= CONFIRM_CORR)
            if is_md5 or (pixel_ok and orient == 'none'):
                kind = 'exact'
            elif pixel_ok and orient != 'none':
                kind = 'flip'
            else:
                continue
            methods = []
            if (qi_, ri_) in md5_file_set:
                methods.append("file_md5")
            if (qi_, ri_) in md5_pix_set:
                methods.append("pixel_md5")
            if rec['phash'] is not None and rec['phash'] <= PHASH_THRESHOLD:
                methods.append("phash")
            if rec['dhash'] is not None and rec['dhash'] <= DHASH_THRESHOLD:
                methods.append("dhash")
            methods.append("pixel_verify" if orient == 'none' else f"flip:{orient}")
            rows.append({
                'qi': qi_, 'ri': ri_, 'orient': orient, 'kind': kind,
                'phash': rec['phash'], 'dhash': rec['dhash'],
                'mse': mse, 'corr': corr, 'methods': methods,
            })
        # Re-verification
        for r in tqdm(rows, desc=f"  Re-verifying {tag}", unit="pairs"):
            img_q = q_valid[r['qi']]['img']
            img_r = data['valid'][r['ri']]['img']
            rmse, rcorr = reverify(img_q, img_r, r['orient'])
            r['rev_mse'] = rmse
            r['rev_corr'] = rcorr
            r['q_name'] = os.path.basename(q_valid[r['qi']]['path'])
            r['r_name'] = os.path.basename(data['valid'][r['ri']]['path'])
            r['q_shape'] = q_valid[r['qi']]['shape']
            r['r_shape'] = data['valid'][r['ri']]['shape']
        data['detailed'] = rows
        detailed.extend([dict(tag=tag, **r) for r in rows])

    # 6) Report
    print("\n[Step 6/6] Generating txt report ...")
    if os.path.exists(OLD_REPORT_PATH):
        os.remove(OLD_REPORT_PATH)

    q_ok = sum(1 for r in q_records if r['img'] is not None)
    exact = [d for d in detailed if d['kind'] == 'exact']
    flip_matches = [d for d in detailed if d['kind'] == 'flip']
    confirmed_q_names = sorted({d['q_name'] for d in exact})
    flip_q_names = sorted({d['q_name'] for d in flip_matches})

    L = []
    L.append("=" * 80)
    L.append("Cross-Dataset Image Overlap Detection Report")
    L.append("=" * 80)
    L.append(f"Generated at        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Runtime environment : conda Unet | torch {torch.__version__} | device {DEVICE.upper()}")
    L.append(f"Query directory     : {QUERY_DIR}")
    for k, v in REF_DIRS.items():
        L.append(f"Reference directory : {k}  ({v})")
    L.append(f"pHash/dHash threshold : {PHASH_THRESHOLD}")
    L.append(f"Pairwise verification : {DESC_SIZE}x{DESC_SIZE} grayscale MSE<={CONFIRM_MSE} or ZNCC>={CONFIRM_CORR}, including flip detection")
    L.append(f"Confirmed pair re-check : {REVERIFY_SIZE}x{REVERIFY_SIZE} grayscale")
    L.append("")

    L.append("-" * 80)
    L.append("[I] Dataset Sizes")
    L.append("-" * 80)
    L.append(f"ColoTexture-Hybrid (query set) : {q_ok} images (failed to read {len(q_records) - q_ok})")
    for tag, recs in ref_pack.items():
        ok = sum(1 for r in recs if r['img'] is not None)
        L.append(f"{tag:<27} : {ok} images (failed to read {len(recs) - ok})")
    L.append("")

    L.append("-" * 80)
    L.append("[II] Overall Conclusion")
    L.append("-" * 80)
    L.append(f"Overlap present    : {'yes' if exact else 'no'}")
    L.append(f"Confirmed overlapping (same orientation) image pairs : {len(exact)} pairs")
    L.append(f"Overlapping ColoTexture images (deduplicated): {len(confirmed_q_names)} images"
             f" ({len(confirmed_q_names) / max(q_ok, 1) * 100:.4f}% of the query set)")
    L.append(f"Additional flip overlaps (mirror/rotation) pairs : {len(flip_matches)} pairs"
             f" (involving {len(flip_q_names)} images) -- see [III-b]")
    L.append("")
    L.append("Statistics by method (same-orientation confirmed pairs):")
    L.append(f"  Identical file bytes (file MD5)  : {sum(1 for d in exact if 'file_md5' in d['methods'])} pairs")
    L.append(f"  Identical decoded pixels (pixel MD5) : {sum(1 for d in exact if 'pixel_md5' in d['methods'])} pairs")
    L.append(f"  pHash Hamming distance <= {PHASH_THRESHOLD}        : {sum(1 for d in exact if 'phash' in d['methods'])} pairs")
    L.append(f"  dHash Hamming distance <= {DHASH_THRESHOLD}        : {sum(1 for d in exact if 'dhash' in d['methods'])} pairs")
    L.append(f"  Confirmed by descriptor pixel verification : {sum(1 for d in exact if 'pixel_verify' in d['methods'])} pairs")
    L.append("")

    L.append("-" * 80)
    L.append("[III] Overlap Details (same-orientation confirmed pairs)")
    L.append("-" * 80)
    for tag in REF_DIRS.keys():
        rows = [d for d in exact if d['tag'] == tag]
        rows.sort(key=lambda x: (x['phash'] if x['phash'] is not None else 99,
                                 x['dhash'] if x['dhash'] is not None else 99))
        L.append(f"\n>>> Overlap with {tag}: {len(rows)} pairs, "
                 f"involving {len({r['q_name'] for r in rows})} ColoTexture images")
        if not rows:
            L.append("    (none)")
            continue
        for d in rows:
            ph = "-" if d['phash'] is None else str(int(d['phash']))
            dh = "-" if d['dhash'] is None else str(int(d['dhash']))
            L.append(f"    {d['q_name']}  <->  {d['r_name']}  "
                     f"| methods[{'/'.join(d['methods'])}] pHash={ph} dHash={dh} "
                     f"| descriptor MSE={d['mse']:.2f} ZNCC={d['corr']:.5f} "
                     f"| re-check MSE={d['rev_mse']:.2f} ZNCC={d['rev_corr']:.5f} "
                     f"| size {d['q_shape']} vs {d['r_shape']}")

    L.append("")
    L.append("-" * 80)
    L.append(f"[III-b] Flip Overlap Details (mirror/rotation, {len(flip_matches)} pairs)")
    L.append("-" * 80)
    if flip_matches:
        for d in sorted(flip_matches, key=lambda x: (x['tag'], x['q_name'])):
            ph = "-" if d['phash'] is None else str(int(d['phash']))
            dh = "-" if d['dhash'] is None else str(int(d['dhash']))
            L.append(f"    [{d['tag']}] {d['q_name']}  <->  {d['r_name']}  "
                     f"| orientation={d['orient']} pHash={ph} dHash={dh} "
                     f"| descriptor MSE={d['mse']:.2f} ZNCC={d['corr']:.5f} "
                     f"| re-check MSE={d['rev_mse']:.2f} ZNCC={d['rev_corr']:.5f} "
                     f"| size {d['q_shape']} vs {d['r_shape']}")
    else:
        L.append("    (none)")

    suspect_counts = {tag: max(0, len(data['cand']) - len(data['detailed'])) for tag, data in ref_data.items()}
    L.append("")
    L.append("-" * 80)
    L.append("[IV] Number of Candidates That Failed Verification (for reference)")
    L.append("-" * 80)
    for tag, c in suspect_counts.items():
        L.append(f"    {tag}: {c} pairs (pHash/dHash hit but pixel verification failed, judged as non-overlapping)")
    L.append("")

    L.append("-" * 80)
    L.append("[V] List of Overlapping ColoTexture-Hybrid Image Names (deduplicated)")
    L.append("-" * 80)
    if confirmed_q_names:
        for n in confirmed_q_names:
            L.append(f"    {n}")
    else:
        L.append("    (none)")
    L.append("")
    L.append("=" * 80)
    L.append(f"Detection finished, total time {time.time() - t0:.1f} seconds")
    L.append("=" * 80)

    report = "\n".join(L)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
