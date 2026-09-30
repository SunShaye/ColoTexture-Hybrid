#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频抽帧脚本 - NVIDIA GPU加速 + CPU多核并行版本
使用ffmpeg的NVIDIA硬件解码器提取视频帧，CPU解码使用多进程并行
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

FFMPEG_PATH = "/usr/bin/ffmpeg"  # 使用系统ffmpeg（支持NVIDIA硬件加速）

# CPU并行工作进程数
CPU_WORKERS = min(24, cpu_count())  # 使用24核心或最大可用核心数


def get_video_files(input_dir):
    """获取所有视频文件"""
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, file))
    return sorted(video_files)


def get_video_codec(video_path):
    """检测视频编码格式"""
    try:
        cmd = [
            FFMPEG_PATH, "-i", video_path,
            "-hide_banner"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr

        # 提取视频流信息
        video_stream_match = re.search(r'Stream #.*Video:\s*(\w+)', output)
        if video_stream_match:
            codec = video_stream_match.group(1).lower()
            return codec
        return None
    except:
        return None


def get_video_info(video_path):
    """获取视频信息（帧数、fps等）"""
    try:
        cmd = [
            FFMPEG_PATH, "-i", video_path,
            "-hide_banner"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr  # ffmpeg输出到stderr

        # 提取帧数
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
    CPU解码工作进程（用于多进程并行）
    args: (video_path, output_dir, start_frame)
    """
    video_path, output_dir, start_frame = args
    video_name = Path(video_path).stem

    cmd = [
        FFMPEG_PATH,
        "-threads", "1",  # 每个ffmpeg实例使用单线程，由多进程控制并行度
        "-i", video_path,
        "-pix_fmt", "bgr24",
        "-start_number", str(start_frame),
        os.path.join(output_dir, "%011d.jpg"),
        "-y"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # 统计提取的帧数
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
    使用NVIDIA GPU加速提取视频帧
    """
    video_name = Path(video_path).stem

    # 检查NVIDIA解码器是否可用
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-decoders"],
            capture_output=True, text=True
        )
        has_nvidia = "h264_cuvid" in result.stdout or "hevc_cuvid" in result.stdout
    except:
        has_nvidia = False

    # 获取视频信息和编码格式
    info = get_video_info(video_path)
    codec = get_video_codec(video_path)

    # 判断编码格式是否支持NVIDIA硬件解码
    nvidia_supported_codecs = {'h264', 'h264_vdpau', 'hevc', 'h265', 'hevc_vdpau'}
    can_use_nvidia = has_nvidia and codec in nvidia_supported_codecs

    start_frame = global_frame_counter[0]

    # 构建ffmpeg命令
    if can_use_nvidia:
        # 使用NVIDIA硬件解码
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
        mode = f"NVIDIA硬件加速 ({decoder})"
    else:
        # 使用CPU解码 - 返回None表示需要用多进程处理
        if not has_nvidia:
            mode = "CPU (NVIDIA解码器不可用)"
        else:
            mode = f"CPU (编码 {codec} 不支持NVIDIA硬件)"
        return None, mode, info, codec, start_frame

    # 执行ffmpeg
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # 读取输出以获取进度
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
                progress_queue.put((video_path, 0, False, f"ffmpeg返回码 {process.returncode}"))
            return 0, mode, info, codec, start_frame

    except Exception as e:
        if progress_queue:
            progress_queue.put((video_path, 0, False, str(e)))
        return 0, mode, info, codec, start_frame


def classify_videos(video_files):
    """将视频分类为NVIDIA可加速和CPU处理两类"""
    nvidia_videos = []
    cpu_videos = []

    # 检查NVIDIA解码器是否可用
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-decoders"],
            capture_output=True, text=True
        )
        has_nvidia = "h264_cuvid" in result.stdout or "hevc_cuvid" in result.stdout
    except:
        has_nvidia = False

    nvidia_supported_codecs = {'h264', 'h264_vdpau', 'hevc', 'h265', 'hevc_vdpau'}

    print("正在分析视频编码格式...")
    for video_path in tqdm(video_files, desc="检测编码"):
        codec = get_video_codec(video_path)
        info = get_video_info(video_path)

        if has_nvidia and codec in nvidia_supported_codecs:
            nvidia_videos.append((video_path, info, codec))
        else:
            cpu_videos.append((video_path, info, codec))

    return nvidia_videos, cpu_videos


def main():
    print("=" * 60)
    print("视频抽帧工具 - NVIDIA GPU加速 + CPU多核并行")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"CPU并行工作进程: {CPU_WORKERS}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        return

    # 检查ffmpeg
    try:
        result = subprocess.run([FFMPEG_PATH, "-version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"检测到: {version_line}")
        else:
            print("错误: ffmpeg未正确安装")
            return
    except FileNotFoundError:
        print(f"错误: 未找到ffmpeg，请确保已安装并添加到PATH")
        return

    # 检查NVIDIA支持
    try:
        result = subprocess.run([FFMPEG_PATH, "-encoders"],
                              capture_output=True, text=True)
        if "h264_nvenc" in result.stdout:
            print("✓ 检测到NVIDIA编码器支持")
        else:
            print("⚠ 未检测到NVIDIA编码器，将使用CPU")
    except:
        pass

    print()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取视频文件
    video_files = get_video_files(INPUT_DIR)

    if not video_files:
        print("错误: 未找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")
    print()

    # 分类视频
    nvidia_videos, cpu_videos = classify_videos(video_files)

    print(f"\n分类结果:")
    print(f"  - NVIDIA GPU加速: {len(nvidia_videos)} 个视频")
    print(f"  - CPU多核并行: {len(cpu_videos)} 个视频")
    print()

    # 全局帧计数器
    manager = Manager()
    global_frame_counter = manager.list([1])
    total_extracted = manager.list([0])

    # ========== 处理NVIDIA视频（串行，GPU一次处理一个）==========
    if nvidia_videos:
        print(f"开始处理 {len(nvidia_videos)} 个NVIDIA加速视频...")
        for i, (video_path, info, codec) in enumerate(nvidia_videos, 1):
            print(f"[GPU {i}/{len(nvidia_videos)}] {Path(video_path).name}")
            print(f"  编码: {codec}, {info.get('frames', '?')}帧, {info.get('fps', '?')}fps")

            extracted, mode, _, _, _ = extract_frames_nvidia(
                video_path, OUTPUT_DIR, global_frame_counter
            )
            print(f"  模式: {mode}")

            if extracted and extracted > 0:
                total_extracted[0] += extracted
                print(f"  ✓ 提取 {extracted} 帧")
            else:
                # NVIDIA失败，加入CPU队列
                print(f"  ⚠ NVIDIA失败，转CPU处理")
                cpu_videos.append((video_path, info, codec))
            print()

    # ========== 处理CPU视频（并行，多进程）==========
    if cpu_videos:
        print(f"开始处理 {len(cpu_videos)} 个CPU并行视频（使用 {CPU_WORKERS} 核心）...")
        print()

        # 准备任务列表
        cpu_tasks = []
        for video_path, info, codec in cpu_videos:
            start_frame = global_frame_counter[0]
            # 预估帧数来分配起始帧号
            estimated_frames = info.get('frames', 0) or int(info.get('fps', 30) * info.get('duration', 0))
            global_frame_counter[0] += estimated_frames + 100  # 预留空间
            cpu_tasks.append((video_path, OUTPUT_DIR, start_frame, info, codec))

        # 使用多进程并行处理
        with Pool(processes=CPU_WORKERS) as pool:
            results = []
            with tqdm(total=len(cpu_tasks), desc="CPU并行处理") as pbar:
                for result in pool.imap_unordered(process_cpu_video, cpu_tasks):
                    results.append(result)
                    pbar.update(1)

        # 汇总结果
        success_count = 0
        for video_path, extracted, success, error in results:
            if success:
                total_extracted[0] += extracted
                success_count += 1
            else:
                print(f"  ✗ 失败: {Path(video_path).name} - {error}")

        print(f"\nCPU处理完成: {success_count}/{len(cpu_videos)} 成功")
        print()

    # 验证输出
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    output_files.sort()

    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"处理视频数: {len(video_files)}")
    print(f"提取总帧数: {len(output_files)}")
    print(f"输出目录: {OUTPUT_DIR}")

    if output_files:
        print(f"第一帧: {output_files[0]}")
        print(f"最后一帧: {output_files[-1]}")
    print("=" * 60)


def process_cpu_video(task):
    """CPU视频处理函数（用于多进程）"""
    video_path, output_dir, start_frame, info, codec = task
    video_name = Path(video_path).name

    cmd = [
        FFMPEG_PATH,
        "-threads", "1",  # 每个ffmpeg实例使用单线程
        "-i", video_path,
        "-pix_fmt", "bgr24",
        "-start_number", str(start_frame),
        os.path.join(output_dir, "%011d.jpg"),
        "-y"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # 统计提取的帧数
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
