#!/usr/bin/env python3
"""Slide the dark rectangular mask straight down over the 5 objects until it exits the frame."""
import os, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 58
W = H = 1024

first = np.array(Image.open(FIRST).convert("RGB"))
bg_color = first[0, 0].copy()

# Locate the mask: a solid rectangle in the upper part of the frame (above the objects).
nonbg = np.abs(first.astype(int) - bg_color.astype(int)).sum(2) > 30
rows = nonbg.sum(1)
top = int(np.argmax(rows > 0))
bottom = top
while bottom + 1 < H and rows[bottom + 1] > 0:
    bottom += 1
xs = np.where(nonbg[top:bottom + 1].any(0))[0]
left, right = int(xs.min()), int(xs.max())
mask_h, mask_w = bottom - top + 1, right - left + 1
mask_patch = first[top:bottom + 1, left:right + 1].copy()

# Background: first frame with the mask region restored to the background colour.
background = first.copy()
background[top:bottom + 1, left:right + 1] = bg_color

# Vertical trajectory: from starting position to fully off the bottom edge, linear pace.
y_start, y_end = top, H
frames = []
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    y = int(round(y_start + (y_end - y_start) * t))
    fr = background.copy()
    y0, y1 = max(y, 0), min(y + mask_h, H)
    if y1 > y0:
        fr[y0:y1, left:right + 1] = mask_patch[y0 - y:y1 - y]
    frames.append(fr)

os.makedirs(OUT_DIR, exist_ok=True)
raw = os.path.join(OUT_DIR, "_frames.rgb")
with open(raw, "wb") as f:
    for fr in frames:
        f.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw,
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
], check=True)
os.remove(raw)
print(f"wrote {OUT}: {N_FRAMES} frames @ {FPS} fps, mask {mask_w}x{mask_h} from y={y_start} to y={y_end}")
