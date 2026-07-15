# -*- coding: utf-8 -*-
"""
Manual image selection script.
Select to keep or delete images via mouse clicks.
"""

import os
import shutil
import cv2
import numpy as np
from natsort import natsorted


INPUT_DIR = r"E:\Pycharm\Process_Datasets\input_images"
OUTPUT_DIR = r"E:\Pycharm\Process_Datasets\output_images"

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

WINDOW_NAME = "Image Selector"


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
        """Retrieve all image files from the input directory."""
        image_files = []
        for filename in os.listdir(self.input_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_files.append(filename)
        return natsorted(image_files)

    def load_current_image(self):
        """Load the current image."""
        if self.current_index >= len(self.image_files):
            return None

        image_path = os.path.join(self.input_dir, self.image_files[self.current_index])
        img = cv2.imread(image_path)
        return img

    def resize_to_window(self, img, max_width=1280, max_height=720):
        """Resize image to fit the display window."""
        if img is None:
            return None

        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    def draw_status(self, img):
        """Draw status overlay on the image."""
        if img is None:
            return None

        display = img.copy()
        h, w = display.shape[:2]

        # Draw bottom info bar background
        bar_height = 80
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Current image info
        current_file = self.image_files[self.current_index] if self.current_index < len(self.image_files) else ""
        info_text = f"[{self.current_index + 1}/{len(self.image_files)}] {current_file}"
        cv2.putText(display, info_text, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Action hints
        if self.click_state == 'left_pending':
            cv2.putText(display, "Left-click again to KEEP | Right-click to cancel", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.click_state == 'right_pending':
            cv2.putText(display, "Right-click again to DELETE | Left-click to cancel", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Left: KEEP | Right: DELETE | ESC: Exit", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback handler."""
        if self.finished:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_state == 'right_pending':
                # Cancel right-click action
                self.click_state = None
                self.update_display()
            elif self.click_state == 'left_pending':
                # Confirm keep
                self.retained_files.append(self.image_files[self.current_index])
                self.next_image()
            else:
                # First left-click
                self.click_state = 'left_pending'
                self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.click_state == 'left_pending':
                # Cancel left-click action
                self.click_state = None
                self.update_display()
            elif self.click_state == 'right_pending':
                # Confirm delete
                self.next_image()
            else:
                # First right-click
                self.click_state = 'right_pending'
                self.update_display()

    def update_display(self):
        """Refresh the display."""
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
        """Advance to the next image."""
        self.click_state = None
        self.current_index += 1

        if self.current_index >= len(self.image_files):
            self.show_finished()
        else:
            self.update_display()

    def show_finished(self):
        """Display completion screen."""
        self.finished = True

        # Create completion info screen
        display = np.zeros((400, 600, 3), dtype=np.uint8)

        cv2.putText(display, "All Images Processed!", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"Total: {len(self.image_files)}", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(display, f"Retained: {len(self.retained_files)}", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(display, f"Deleted: {len(self.image_files) - len(self.retained_files)}", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.putText(display, "Press any key to exit and copy files...", (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(0)

    def copy_retained_files(self):
        """Copy retained files to the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        for filename in self.retained_files:
            src_path = os.path.join(self.input_dir, filename)
            dst_path = os.path.join(self.output_dir, filename)
            shutil.copy2(src_path, dst_path)

        print(f"\nCopied {len(self.retained_files)} images to: {self.output_dir}")

    def run(self):
        """Run the main loop."""
        print("=" * 60)
        print("Manual Image Selection Tool")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

        if not os.path.exists(self.input_dir):
            print(f"Error: Input directory not found: {self.input_dir}")
            return

        self.image_files = self.get_image_files()

        if len(self.image_files) == 0:
            print("Error: No image files found in input directory")
            return

        print(f"Found {len(self.image_files)} images")
        print("\nInstructions:")
        print("  - Left-click twice: keep image")
        print("  - Right-click twice: delete image")
        print("  - ESC: quit")
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
            print(f"\nProcessing complete!")
            print(f"Retained images: {len(self.retained_files)}")
            print(f"Deleted images: {len(self.image_files) - len(self.retained_files)}")
            self.copy_retained_files()
        elif not self.finished:
            print("\nOperation cancelled by user")
        else:
            print("\nNo images were retained")


def main():
    selector = ImageSelector(INPUT_DIR, OUTPUT_DIR)
    selector.run()


if __name__ == "__main__":
    main()
