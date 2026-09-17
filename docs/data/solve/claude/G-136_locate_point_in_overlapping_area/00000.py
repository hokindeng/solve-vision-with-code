#!/usr/bin/env python3
"""Locate the points lying fully inside the overlap of two shapes and circle them in red."""
import os, subprocess, tempfile
import numpy as np, cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N_FRAMES, FPS = 37, 16

base = np.array(Image.open(FIRST).convert("RGB"))
H, W = base.shape[:2]

# --- 1. find the colours: white bg, black dots, shape A, shape B, overlap (darkest non-black blend)
cols, cnt = np.unique(base.reshape(-1, 3), axis=0, return_counts=True)
cols = [tuple(int(v) for v in c) for c, n in zip(cols, cnt) if n > 500]
black = (0, 0, 0)
shape_cols = [c for c in cols if c != black and c != (255, 255, 255)]
# overlap colour is the blend => darkest (lowest sum) of the three shape colours
overlap_col = min(shape_cols, key=sum)
ov = np.all(base == overlap_col, axis=2)
blk = np.all(base == black, axis=2).astype(np.uint8)

# --- 2. locate dot centres (split touching dots by eroding)
dist = cv2.distanceTransform(blk, cv2.DIST_L2, 5)
dot_r = float(dist.max())                        # ~6.5 px
# local maxima of the distance transform -> one peak blob per dot, even when two dots touch
dil = cv2.dilate(dist, np.ones((7, 7), np.uint8))
peaks = ((dist >= dil - 1e-3) & (dist > dot_r * 0.6)).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(peaks)
centers = [tuple(cen[i]) for i in range(1, n)]

# --- 3. a point is fully inside the overlap iff a thin ring just outside the dot is entirely overlap colour
def fully_inside(c):
    cx, cy = c
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    ring = (d > dot_r + 0.5) & (d <= dot_r + 3.0)
    vals = ring & ~(blk.astype(bool))            # ignore neighbouring black dots
    return vals.sum() > 0 and np.all(ov[vals])

inside = [c for c in centers if fully_inside(c)]
# order them for a tidy step-by-step reveal (top-to-bottom, left-to-right)
inside.sort(key=lambda c: (round(c[1] / 40), c[0]))
print(f"{len(centers)} points found, {len(inside)} fully inside the overlap")

# --- 4. animate: frame 0 untouched, then each circle is swept in, all complete by the last frame
R, TH = int(round(dot_r * 2.4)), 3
RED = (220, 30, 30)
avail = N_FRAMES - 1                             # frames 1..36
per = avail / max(len(inside), 1)
SWEEP = max(2, int(per * 0.8))                   # frames used to sweep one circle

def draw_arc(img, c, frac):
    if frac <= 0: return
    cx, cy = int(round(c[0])), int(round(c[1]))
    if frac >= 1:
        cv2.circle(img, (cx, cy), R, RED, TH, cv2.LINE_AA); return
    cv2.ellipse(img, (cx, cy), (R, R), -90, 0, 360 * frac, RED, TH, cv2.LINE_AA)

frames = []
for f in range(N_FRAMES):
    img = base.copy()
    for k, c in enumerate(inside):
        start = 1 + k * per
        frac = (f - start + 1) / SWEEP
        if f == N_FRAMES - 1: frac = 1.0
        draw_arc(img, c, min(1.0, frac))
    frames.append(img)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(td, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "12", "-preset", "slow", OUT], check=True)
print("wrote", OUT)
