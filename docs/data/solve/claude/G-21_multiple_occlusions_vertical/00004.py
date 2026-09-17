#!/usr/bin/env python3
"""Move the gray rectangular mask straight down until it exits the frame."""
import os, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 58
MASK_COLOR = np.array([209, 209, 209], dtype=np.uint8)

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape

# Locate the mask (solid gray rectangle) in the first frame.
m = (first == MASK_COLOR).all(axis=2)
ys, xs = np.where(m)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
mask_patch = first[y0:y1, x0:x1].copy()

# Background = first frame with the mask removed (it sits over pure white).
bg = first.copy()
bg[y0:y1, x0:x1] = 255

# Mask travels from its start to fully below the frame over the full duration.
total_dy = H - y0
os.makedirs(OUT_DIR, exist_ok=True)
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    dy = int(round(t * total_dy))
    frame = bg.copy()
    top, bot = y0 + dy, min(y1 + dy, H)
    if top < H:
        frame[top:bot, x0:x1] = mask_patch[: bot - top]
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()
print("wrote", OUT)
