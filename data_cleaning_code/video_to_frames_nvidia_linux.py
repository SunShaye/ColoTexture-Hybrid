#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Frame Extraction Script - NVIDIA GPU Acceleration + CPU Multi-core Parallel
Extract video frames using ffmpeg NVIDIA hardware decoder; CPU decoding uses multi-process parallelism.
"""

import os
import subprocess
import re
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import threading

INPUT_DIR = "/mnt/sda/Dataset/Polyp-video"
OUTPUT_DIR = "/mnt/sda/Dataset/Polyp-video-flame"

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

FFMPEG_PATH = "/usr/bin/ffmpeg"  # Use system ffmpeg (supports NVIDIA hardware acceleration)

# Number of CPU parallel worker processes
CPU_WORKERS = min(24, cpu_count())  # Use 24 cores or max available cores


def get_video_files(input_dir):
    """Get all video files."""
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, file))
    return sorted(video_files)


def get_video_codec(video_path):
    """Detect video codec format."""
    try:
        cmd = [
            FFMPEG_PATH, "-i", video_path,
            "-hide_banner"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr

        # Extract video stream info
        video_stream_match = re.search(r'Stream #.*Video:\s*(\w+)', output)
        if video_stream_match:
            codec = video_stream_match.group(1).lower()
            return codec
        return None
    except:
        return None


def get_video_info(video_path):
    """Get video info (frame count, fps, etc.)."""
    try:
        cmd = [
            FFMPEG_PATH, "-i", video_path,
            "-hide_banner"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr  # ffmpeg outputs to stderr

        # Extract frame count
        frame_match = re.search(r'frame=\s*(\d+)', output)
        fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', output)
        duration_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', output)

        frames = None
        fps = None
        duration = None

        if fps_match:
            fps = float(fps_match.group(1))

        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            seconds = float(duration_match.group(3))
            duration = hours * 3600 + minutes * 60 + seconds

        if fps and duration:
            frames = int(fps * duration)

        return {'frames': frames, 'fps': fps, 'duration': duration}
    except Exception as e:
        return {'frames': None, 'fps': None, 'duration': None}


def extract_frames_cpu_worker(args):
    """
    CPU decode worker process (for multi-process parallel).
    args: (video_path, output_dir, start_frame)
    """
    video_path, output_dir, start_frame = args
    video_name = Path(video_path).stem

    cmd = [
        FFMPEG_PATH,
        "-threads", "1",  # Each ffmpeg instance uses single thread; parallelism controlled by multi-process
        "-i", video_path,
        "-pix_fmt", "bgr24",
        "-start_number", str(start_frame),
        os.path.join(output_dir, "%011d.jpg"),
        "-y"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Count extracted frames
            frame_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
            new_frames = len([f for f in frame_files
                            if int(Path(f).stem) >= start_frame])
            return (video_path, new_frames, True, "")
        else:
            return (video_path, 0, False, result.stderr[:200])
    except Exception as e:
        return (video_path, 0, False, str(e))


def extract_frames_nvidia(video_path, output_dir, global_frame_counter, progress_queue=None):
    """
    Extract video frames using NVIDIA GPU acceleration.
    """
    video_name = Path(video_path).stem

    # Check if NVIDIA decoder is available
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-decoders"],
            capture_output=True, text=True
        )
        has_nvidia = "h264_cuvid" in result.stdout or "hevc_cuvid" in result.stdout
    except:
        has_nvidia = False

    # Get video info and codec
    info = get_video_info(video_path)
    codec = get_video_codec(video_path)

    # Determine if codec supports NVIDIA hardware decoding
    nvidia_supported_codecs = {'h264', 'h264_vdpau', 'hevc', 'h265', 'hevc_vdpau'}
    can_use_nvidia = has_nvidia and codec in nvidia_supported_codecs

    start_frame = global_frame_counter[0]

    # Build ffmpeg command
    if can_use_nvidia:
        # Use NVIDIA hardware decoding
        if codec in {'hevc', 'h265', 'hevc_vdpau'}:
            decoder = "hevc_cuvid"
        else:
            decoder = "h264_cuvid"

        cmd = [
            FFMPEG_PATH,
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-c:v", decoder,
            "-i", video_path,
            "-vf", "format=nv12,hwdownload,format=nv12",
            "-pix_fmt", "bgr24",
            "-start_number", str(start_frame),
            os.path.join(output_dir, "%011d.jpg"),
            "-y"
        ]
        mode = f"NVIDIA hardware acceleration ({decoder})"
    else:
        # Use CPU decoding - return None indicates multi-process processing needed
        if not has_nvidia:
            mode = "CPU (NVIDIA decoder unavailable)"
        else:
            mode = f"CPU (codec {codec} not supported by NVIDIA hardware)"
        return None, mode, info, codec, start_frame

    # Execute ffmpeg
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Read output to get progress
        frame_pattern = re.compile(r'frame=\s*(\d+)')
        last_frame = 0

        while True:
            line = process.stderr.readline()
            if not line:
                break

            match = frame_pattern.search(line)
            if match:
                current_frame = int(match.group(1))
                if current_frame > last_frame:
                    last_frame = current_frame

        process.wait()

        if process.returncode == 0:
            extracted_frames = last_frame if last_frame > 0 else info.get('frames', 0)
            global_frame_counter[0] += extracted_frames
            if progress_queue:
                progress_queue.put((video_path, extracted_frames, True, ""))
            return extracted_frames, mode, info, codec, start_frame
        else:
            if progress_queue:
                progress_queue.put((video_path, 0, False, f"ffmpeg return code {process.returncode}"))
            return 0, mode, info, codec, start_frame

    except Exception as e:
        if progress_queue:
            progress_queue.put((video_path, 0, False, str(e)))
        return 0, mode, info, codec, start_frame


def classify_videos(video_files):
    """Classify videos into NVIDIA-accelerated and CPU-processing groups."""
    nvidia_videos = []
    cpu_videos = []

    # Check if NVIDIA decoder is available
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-decoders"],
            capture_output=True, text=True
        )
        has_nvidia = "h264_cuvid" in result.stdout or "hevc_cuvid" in result.stdout
    except:
        has_nvidia = False

    nvidia_supported_codecs = {'h264', 'h264_vdpau', 'hevc', 'h265', 'hevc_vdpau'}

    print("Analyzing video codec formats...")
    for video_path in tqdm(video_files, desc="Detecting codec"):
        codec = get_video_codec(video_path)
        info = get_video_info(video_path)

        if has_nvidia and codec in nvidia_supported_codecs:
            nvidia_videos.append((video_path, info, codec))
        else:
            cpu_videos.append((video_path, info, codec))

    return nvidia_videos, cpu_videos


def main():
    print("=" * 60)
    print("Video Frame Extraction Tool - NVIDIA GPU Acceleration + CPU Multi-core Parallel")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CPU parallel workers: {CPU_WORKERS}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return

    # Check ffmpeg
    try:
        result = subprocess.run([FFMPEG_PATH, "-version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"Detected: {version_line}")
        else:
            print("Error: ffmpeg not properly installed")
            return
    except FileNotFoundError:
        print(f"Error: ffmpeg not found. Please ensure it is installed and added to PATH")
        return

    # Check NVIDIA support
    try:
        result = subprocess.run([FFMPEG_PATH, "-encoders"],
                              capture_output=True, text=True)
        if "h264_nvenc" in result.stdout:
            print("✓ NVIDIA encoder support detected")
        else:
            print("⚠ NVIDIA encoder not detected, will use CPU")
    except:
        pass

    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get video files
    video_files = get_video_files(INPUT_DIR)

    if not video_files:
        print("Error: No video files found")
        return

    print(f"Found {len(video_files)} video files")
    print()

    # Classify videos
    nvidia_videos, cpu_videos = classify_videos(video_files)

    print(f"\nClassification results:")
    print(f"  - NVIDIA GPU accelerated: {len(nvidia_videos)} videos")
    print(f"  - CPU multi-core parallel: {len(cpu_videos)} videos")
    print()

    # Global frame counter
    manager = Manager()
    global_frame_counter = manager.list([1])
    total_extracted = manager.list([0])

    # ========== Process NVIDIA videos (serial, GPU handles one at a time) ==========
    if nvidia_videos:
        print(f"Processing {len(nvidia_videos)} NVIDIA-accelerated videos...")
        for i, (video_path, info, codec) in enumerate(nvidia_videos, 1):
            print(f"[GPU {i}/{len(nvidia_videos)}] {Path(video_path).name}")
            print(f"  Codec: {codec}, {info.get('frames', '?')} frames, {info.get('fps', '?')} fps")

            extracted, mode, _, _, _ = extract_frames_nvidia(
                video_path, OUTPUT_DIR, global_frame_counter
            )
            print(f"  Mode: {mode}")

            if extracted and extracted > 0:
                total_extracted[0] += extracted
                print(f"  ✓ Extracted {extracted} frames")
            else:
                # NVIDIA failed, add to CPU queue
                print(f"  ⚠ NVIDIA failed, falling back to CPU")
                cpu_videos.append((video_path, info, codec))
            print()

    # ========== Process CPU videos (parallel, multi-process) ==========
    if cpu_videos:
        print(f"Processing {len(cpu_videos)} CPU-parallel videos (using {CPU_WORKERS} cores)...")
        print()

        # Prepare task list
        cpu_tasks = []
        for video_path, info, codec in cpu_videos:
            start_frame = global_frame_counter[0]
            # Estimate frame count to assign start frame number
            estimated_frames = info.get('frames', 0) or int(info.get('fps', 30) * info.get('duration', 0))
            global_frame_counter[0] += estimated_frames + 100  # Reserve space
            cpu_tasks.append((video_path, OUTPUT_DIR, start_frame, info, codec))

        # Use multi-process parallel processing
        with Pool(processes=CPU_WORKERS) as pool:
            results = []
            with tqdm(total=len(cpu_tasks), desc="CPU parallel processing") as pbar:
                for result in pool.imap_unordered(process_cpu_video, cpu_tasks):
                    results.append(result)
                    pbar.update(1)

        # Summarize results
        success_count = 0
        for video_path, extracted, success, error in results:
            if success:
                total_extracted[0] += extracted
                success_count += 1
            else:
                print(f"  ✗ Failed: {Path(video_path).name} - {error}")

        print(f"\nCPU processing complete: {success_count}/{len(cpu_videos)} succeeded")
        print()

    # Verify output
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    output_files.sort()

    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Videos processed: {len(video_files)}")
    print(f"Total frames extracted: {len(output_files)}")
    print(f"Output directory: {OUTPUT_DIR}")

    if output_files:
        print(f"First frame: {output_files[0]}")
        print(f"Last frame: {output_files[-1]}")
    print("=" * 60)


def process_cpu_video(task):
    """CPU video processing function (for multi-process)."""
    video_path, output_dir, start_frame, info, codec = task
    video_name = Path(video_path).name

    cmd = [
        FFMPEG_PATH,
        "-threads", "1",  # Each ffmpeg instance uses single thread
        "-i", video_path,
        "-pix_fmt", "bgr24",
        "-start_number", str(start_frame),
        os.path.join(output_dir, "%011d.jpg"),
        "-y"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Count extracted frames
            frame_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
            new_frames = len([f for f in frame_files
                            if int(Path(f).stem) >= start_frame])
            return (video_path, new_frames, True, "")
        else:
            return (video_path, 0, False, result.stderr[:200])
    except Exception as e:
        return (video_path, 0, False, str(e))


if __name__ == "__main__":
    main()
