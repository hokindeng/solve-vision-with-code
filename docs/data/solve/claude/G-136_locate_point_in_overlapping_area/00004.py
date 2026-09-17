#!/usr/bin/env python3
"""Locate the points inside the overlap of two semi-transparent shapes and
circle them in red, step by step. Regenerates /app/output/video.mp4."""
import os, subprocess, tempfile, shutil
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
FPS, N_FRAMES = 16, 37
W = H = 1024

base = np.array(Image.open(FIRST).convert("RGB"))

# --- 1. Identify the overlap region -----------------------------------------
# The background is white; both shapes are semi-transparent, so the overlap is
# the region whose colour is neither white, nor either of the single-shape
# colours. Pick the shape colours as the 2nd/3rd most common non-white, non-
# dark colours; the overlap is the darkest remaining large flat colour.
flat = base.reshape(-1, 3)
dark = flat.sum(1) < 150
cols, counts = np.unique(flat[~dark], axis=0, return_counts=True)
order = np.argsort(-counts)
cols, counts = cols[order], counts[order]
big = [tuple(int(v) for v in c) for c, n in zip(cols, counts) if n > 2000 and tuple(c) != (255, 255, 255)]
# single-shape colours are lighter (higher luminance) than the overlap colour
big.sort(key=lambda c: sum(c))
overlap_col = big[0]
overlap = np.all(base == overlap_col, axis=2).astype(np.uint8)

# --- 2. Detect the dots -----------------------------------------------------
dots = dark.reshape(H, W).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(dots)
dot_r = max(stats[1:, cv2.CC_STAT_WIDTH].max(), stats[1:, cv2.CC_STAT_HEIGHT].max()) / 2.0

# Fill the dot holes punched into the overlap so the distance transform is
# measured against the real shape boundary.
filled = overlap.copy()
k = np.ones((3, 3), np.uint8)
for _ in range(30):
    grown = cv2.dilate(filled, k) & (overlap | dots)
    if (grown == filled).all():
        break
    filled = grown
dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)

inside = []
for i in range(1, n):
    cx, cy = cent[i]
    x, y = int(round(cx)), int(round(cy))
    # fully inside: whole dot (radius dot_r) is strictly within the overlap
    if filled[y, x] and dist[y, x] > dot_r + 1.0:
        inside.append((cx, cy))
inside.sort(key=lambda p: (p[0], p[1]))  # left-to-right order for the reveal
print("points inside overlap:", [(round(x), round(y)) for x, y in inside])

# --- 3. Animate: draw a red circle around each point, one after another ------
RED = (220, 30, 30)
R = int(round(dot_r * 3.2))      # circle radius
T = 3                             # line thickness

def draw_arc(img, c, frac):
    """Draw an arc sweeping `frac` of the full circle (anti-aliased)."""
    if frac <= 0:
        return
    cx, cy = c
    if frac >= 1.0:
        cv2.circle(img, (int(round(cx)), int(round(cy))), R, RED, T, cv2.LINE_AA)
        return
    ang = 360.0 * frac
    cv2.ellipse(img, (int(round(cx)), int(round(cy))), (R, R), 0, -90, -90 + ang,
                RED, T, cv2.LINE_AA)

frames = []
frames.append(base.copy())                      # frame 0: untouched first frame
n_anim = N_FRAMES - 1                           # 36 frames of action
per = n_anim / max(1, len(inside))              # frames per point
for f in range(1, N_FRAMES):
    img = base.copy()
    t = f  # 1..36
    for j, c in enumerate(inside):
        start = j * per
        # sweep takes ~70% of this point's slot, then holds
        sweep = max(1.0, per * 0.7)
        frac = (t - start) / sweep
        frac = min(1.0, max(0.0, frac))
        if f == N_FRAMES - 1:
            frac = 1.0
        draw_arc(img, c, frac)
    frames.append(img)

# --- 4. Encode -----------------------------------------------------------------
tmp = tempfile.mkdtemp()
try:
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"f{i:04d}.png"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%04d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "12",
        "-pix_fmt", "yuv420p", "-r", str(FPS), OUT,
    ], check=True)
finally:
    shutil.rmtree(tmp)
print("wrote", OUT, "frames:", len(frames))
