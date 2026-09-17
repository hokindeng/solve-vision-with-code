#!/usr/bin/env python3
"""Subtractive (multiply) pigment mixing animation.

Fills the black-bordered mixing zone of first_frame.png with the per-channel
product of the two pigments, blending in gradually over the whole clip.
"""
import os
import subprocess
import numpy as np
from PIL import Image
import cv2

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 44

C1 = np.array([136, 158, 164], dtype=np.float64)  # left pigment
C2 = np.array([68, 180, 242], dtype=np.float64)   # right pigment
MIXED = np.round(C1 * C2 / 255.0).astype(np.uint8)  # -> (36, 112, 156)


def interior_mask(img):
    """White pixels enclosed by the black border of the central square."""
    h, w, _ = img.shape
    black = img.sum(axis=2) < 60
    ys, xs = np.where(black)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # flood fill from the centre of the square, stopping at black pixels
    region = np.zeros((h, w), dtype=np.uint8)
    region[y0:y1 + 1, x0:x1 + 1] = 1
    walls = (black | (region == 0)).astype(np.uint8)  # 1 = blocked
    free = (1 - walls) * 255
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    cv2.floodFill(free.copy(), ff_mask, (cx, cy), 128, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    return ff_mask[1:-1, 1:-1] == 255


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    mask = interior_mask(base)
    white = base[mask].astype(np.float64)

    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        a = ease(t)
        frame = base.copy()
        if i == N_FRAMES - 1:
            frame[mask] = MIXED
        elif i > 0:
            frame[mask] = np.round(white * (1 - a) + MIXED * a).astype(np.uint8)
        frames.append(frame)

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("mixed colour:", tuple(int(v) for v in MIXED), "interior px:", int(mask.sum()), "->", OUT)


if __name__ == "__main__":
    main()
