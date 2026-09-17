#!/usr/bin/env python3
"""Slide the dark rectangular mask straight down until it exits the frame."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 58, 16

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape

# Locate the mask: the solid gray rectangle.
mask_color = np.array([209, 209, 209], dtype=np.uint8)
is_mask = (first == mask_color).all(axis=2)
rows = np.where(is_mask.any(axis=1))[0]
cols = np.where(is_mask.any(axis=0))[0]
y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
mask_h = y1 - y0

# Background with the mask removed (it sits on a pure-white area).
background = first.copy()
background[y0:y1, x0:x1] = 255

# Move continuously from the start position until the top edge is past the bottom.
total_travel = H - y0


def make_frame(i):
    t = i / (N_FRAMES - 1)
    dy = int(round(t * total_travel))
    frame = background.copy()
    top, bottom = y0 + dy, min(y1 + dy, H)
    if top < H:
        frame[top:bottom, x0:x1] = mask_color
    return frame


os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i in range(N_FRAMES):
    Image.fromarray(make_frame(i)).save(os.path.join(frames_dir, f"{i:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(frames_dir, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT,
], check=True)
print("wrote", OUT)
