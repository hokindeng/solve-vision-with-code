#!/usr/bin/env python3
"""Animate a red perpendicular (vertical) line through the point at the vertical
midpoint between the two horizontal parallel lines in first_frame.png."""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 50
RED = (255, 0, 0)

base = Image.open(SRC).convert("RGB")
arr = np.array(base)

# Locate the two horizontal black parallel lines.
black = np.all(arr == 0, axis=2)
rows = np.where(black.sum(1) > arr.shape[1] // 2)[0]
gaps = np.where(np.diff(rows) > 1)[0]
upper = rows[: gaps[0] + 1]
lower = rows[gaps[0] + 1 :]
y_top, y_bot = upper.mean(), lower.mean()
thickness = int(round((len(upper) + len(lower)) / 2))
y_mid = (y_top + y_bot) / 2.0

# Find the point (dot) lying at the vertical midpoint: the non-white, non-black
# blob whose centre is closest to y_mid.
import cv2
colored = (~np.all(arr == 255, axis=2)) & (~black)
n, lab, stats, cents = cv2.connectedComponentsWithStats(colored.astype(np.uint8))
best = min(range(1, n), key=lambda i: abs(cents[i][1] - y_mid))
x_pt = float(cents[best][0])

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

def frame(k):
    t = k / (N_FRAMES - 1)
    p = ease(t)
    im = base.copy()
    if p > 0:
        y_end = y_top + (y_bot - y_top) * p
        d = ImageDraw.Draw(im)
        d.line([(x_pt, y_top), (x_pt, y_end)], fill=RED, width=thickness)
    return im

os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{base.width}x{base.height}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
       "-movflags", "+faststart", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for k in range(N_FRAMES):
    proc.stdin.write(np.asarray(frame(k)).tobytes())
proc.stdin.close()
proc.wait()
print(f"wrote {OUT}: point=({x_pt:.1f},{y_mid:.1f}) line y {y_top:.1f}->{y_bot:.1f} x={x_pt:.1f}")
