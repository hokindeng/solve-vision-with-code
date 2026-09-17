#!/usr/bin/env python3
"""Color the triple intersection of the three circles red, animated over 60 frames."""
import os, subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16
RED = np.array([255, 0, 0], dtype=np.uint8)

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg = np.array([255, 255, 255], dtype=np.uint8)

# --- find the stroke colors of the circles (exact, non-background colors) ---
flat = img.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
stroke_cols = [tuple(c) for c, n in zip(cols, counts) if n > 2000 and tuple(c) != tuple(bg)]

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(sol[2] + cx**2 + cy**2)
    return cx, cy, r

circles = []
for i in range(len(stroke_cols)):
    ys, xs = np.nonzero(np.all(img == np.array(stroke_cols[i], dtype=np.uint8), axis=2))
    cx, cy, r = fit_circle(xs.astype(float), ys.astype(float))
    rad = np.hypot(xs - cx, ys - cy)
    # exact-color pixels span the solid core; add ~1px for the antialiased fringe
    half_w = (np.percentile(rad, 99) - np.percentile(rad, 1)) / 2 + 1.0
    circles.append((cx, cy, r, half_w))

# --- triple intersection mask: inside the inner edge of all three strokes, on background ---
yy, xx = np.mgrid[0:H, 0:W]
inside = np.ones((H, W), bool)
for cx, cy, r, hw in circles[:3]:
    inside &= np.hypot(xx - cx, yy - cy) <= r - hw
mask = inside & np.all(img == bg, axis=2)

# --- animation: radial reveal from the region's centroid, eased ---
my, mx = np.nonzero(mask)
c_y, c_x = my.mean(), mx.mean()
dist = np.hypot(xx - c_x, yy - c_y)
dmax = dist[mask].max() + 1.0

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "frames")
os.makedirs(tmp, exist_ok=True)
for f in range(N_FRAMES):
    t = f / (N_FRAMES - 1)
    ease = t * t * (3 - 2 * t)  # smoothstep
    frame = img.copy()
    if f > 0:
        reveal = mask & (dist <= ease * dmax)
        if f == N_FRAMES - 1:
            reveal = mask
        frame[reveal] = RED
    Image.fromarray(frame).save(os.path.join(tmp, f"f{f:03d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "f%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    OUT,
], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("circles:", [(round(a, 1), round(b, 1), round(c, 1), round(d, 1)) for a, b, c, d in circles])
print("red pixels:", int(mask.sum()), "->", OUT)
