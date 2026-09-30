# -*- coding: utf-8 -*-
"""
图像手动筛选脚本
通过鼠标点击选择保留或删除图像
"""

import os
import shutil
import cv2
import numpy as np
from natsort import natsorted


INPUT_DIR = r"E:\Pycharm\Process_Datasets\input_images"
OUTPUT_DIR = r"E:\Pycharm\Process_Datasets\output_images"

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

WINDOW_NAME = "图像筛选工具"


class ImageSelector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_files = []
        self.current_index = 0
        self.retained_files = []
        self.click_state = None  # None, 'left_pending', 'right_pending'
        self.finished = False

    def get_image_files(self):
        """获取输入目录中的所有图像文件"""
        image_files = []
        for filename in os.listdir(self.input_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_files.append(filename)
        return natsorted(image_files)

    def load_current_image(self):
        """加载当前图像"""
        if self.current_index >= len(self.image_files):
            return None

        image_path = os.path.join(self.input_dir, self.image_files[self.current_index])
        img = cv2.imread(image_path)
        return img

    def resize_to_window(self, img, max_width=1280, max_height=720):
        """将图像缩放到适合窗口大小"""
        if img is None:
            return None

        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    def draw_status(self, img):
        """在图像上绘制状态信息"""
        if img is None:
            return None

        display = img.copy()
        h, w = display.shape[:2]

        # 绘制底部信息栏背景
        bar_height = 80
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # 当前图像信息
        current_file = self.image_files[self.current_index] if self.current_index < len(self.image_files) else ""
        info_text = f"[{self.current_index + 1}/{len(self.image_files)}] {current_file}"
        cv2.putText(display, info_text, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 操作提示
        if self.click_state == 'left_pending':
            cv2.putText(display, "左键再次点击确认保留 | 右键取消", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.click_state == 'right_pending':
            cv2.putText(display, "右键再次点击确认删除 | 左键取消", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, "左键: 保留 | 右键: 删除 | ESC: 退出", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if self.finished:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_state == 'right_pending':
                # 撤销右键操作
                self.click_state = None
                self.update_display()
            elif self.click_state == 'left_pending':
                # 确认保留
                self.retained_files.append(self.image_files[self.current_index])
                self.next_image()
            else:
                # 第一次左键点击
                self.click_state = 'left_pending'
                self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.click_state == 'left_pending':
                # 撤销左键操作
                self.click_state = None
                self.update_display()
            elif self.click_state == 'right_pending':
                # 确认删除
                self.next_image()
            else:
                # 第一次右键点击
                self.click_state = 'right_pending'
                self.update_display()

    def update_display(self):
        """更新显示"""
        if self.finished:
            return

        img = self.load_current_image()
        if img is None:
            self.show_finished()
            return

        img = self.resize_to_window(img)
        display = self.draw_status(img)

        cv2.imshow(WINDOW_NAME, display)
        cv2.resizeWindow(WINDOW_NAME, display.shape[1], display.shape[0])

    def next_image(self):
        """跳转到下一张图像"""
        self.click_state = None
        self.current_index += 1

        if self.current_index >= len(self.image_files):
            self.show_finished()
        else:
            self.update_display()

    def show_finished(self):
        """显示处理完成"""
        self.finished = True

        # 创建完成提示图像
        display = np.zeros((400, 600, 3), dtype=np.uint8)

        cv2.putText(display, "All Images Processed!", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"Total: {len(self.image_files)}", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(display, f"Retained: {len(self.retained_files)}", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(display, f"Deleted: {len(self.image_files) - len(self.retained_files)}", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.putText(display, "Press any key to exit and copy files...", (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(0)

    def copy_retained_files(self):
        """复制保留的文件到输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)

        for filename in self.retained_files:
            src_path = os.path.join(self.input_dir, filename)
            dst_path = os.path.join(self.output_dir, filename)
            shutil.copy2(src_path, dst_path)

        print(f"\n已将 {len(self.retained_files)} 张图像复制到: {self.output_dir}")

    def run(self):
        """运行主循环"""
        print("=" * 60)
        print("图像手动筛选工具")
        print("=" * 60)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)

        if not os.path.exists(self.input_dir):
            print(f"错误: 输入目录不存在: {self.input_dir}")
            return

        self.image_files = self.get_image_files()

        if len(self.image_files) == 0:
            print("错误: 输入目录中没有找到图像文件")
            return

        print(f"找到 {len(self.image_files)} 张图像")
        print("\n操作说明:")
        print("  - 左键点击两次: 保留图像")
        print("  - 右键点击两次: 删除图像")
        print("  - ESC: 退出程序")
        print()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        self.update_display()

        while not self.finished:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()

        if self.finished and len(self.retained_files) > 0:
            print(f"\n处理完成!")
            print(f"保留图像: {len(self.retained_files)} 张")
            print(f"删除图像: {len(self.image_files) - len(self.retained_files)} 张")
            self.copy_retained_files()
        elif not self.finished:
            print("\n用户取消操作")
        else:
            print("\n没有保留任何图像")


def main():
    selector = ImageSelector(INPUT_DIR, OUTPUT_DIR)
    selector.run()


if __name__ == "__main__":
    main()
