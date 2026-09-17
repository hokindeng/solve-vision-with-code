#!/usr/bin/env python3
"""Locate the intersection of two line segments in first_frame.png and animate
a red circle being drawn around it. Output: /app/output/video.mp4."""
import os, subprocess, shutil, tempfile
import numpy as np, cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N_FRAMES, FPS = 30, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W = base.shape[:2]

# --- 1. Locate the two lines -------------------------------------------------
# foreground = anything that is not (near) white background
gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
mask = (gray < 235).astype(np.uint8) * 255
ys, xs = np.nonzero(mask)
pts = np.stack([xs, ys], 1).astype(np.float64)

def fit_line(p):
    """Total least-squares line: returns (point, unit direction)."""
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt[0]

def dist(p, c, d):
    n = np.array([-d[1], d[0]])
    return np.abs((p - c) @ n)

# seed with the two most distinct Hough segments, then refine by re-assignment
hl = cv2.HoughLinesP(mask, 1, np.pi / 360, 100, minLineLength=100, maxLineGap=20)
segs = [tuple(int(v) for v in l.reshape(-1)) for l in hl]
def ang(s): return np.arctan2(s[3] - s[1], s[2] - s[0]) % np.pi
a0 = segs[0]
b0 = max(segs, key=lambda s: min(abs(ang(s) - ang(a0)), np.pi - abs(ang(s) - ang(a0))))
lines = []
for s in (a0, b0):
    c = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2], float)
    d = np.array([s[2] - s[0], s[3] - s[1]], float); d /= np.linalg.norm(d)
    lines.append((c, d))
for _ in range(5):
    d0 = dist(pts, *lines[0]); d1 = dist(pts, *lines[1])
    lab = d1 < d0
    lines = [fit_line(pts[(~lab) & (d0 < 6)]), fit_line(pts[lab & (d1 < 6)])]

(c0, d0v), (c1, d1v) = lines
A = np.array([d0v, -d1v]).T
t = np.linalg.solve(A, c1 - c0)
ix, iy = c0 + t[0] * d0v
print(f"intersection at ({ix:.1f}, {iy:.1f})")

# --- 2. Animate the red circle ----------------------------------------------
RADIUS, THICK, RED = 30, 5, (230, 30, 30)
SHIFT = 4
center = (int(round(ix * 2 ** SHIFT)), int(round(iy * 2 ** SHIFT)))
axes = (RADIUS * 2 ** SHIFT, RADIUS * 2 ** SHIFT)

def ease(u): return 0.5 - 0.5 * np.cos(np.pi * u)

frames = []
for i in range(N_FRAMES):
    f = base.copy()
    if i > 0:
        # frames 1..N-2 progressively sweep the arc; final frames show it complete
        u = min(1.0, (i - 1) / (N_FRAMES - 4))
        sweep = 360.0 * ease(u)
        if sweep > 0:
            cv2.ellipse(f, center, axes, 0, -90, -90 + sweep, RED, THICK,
                        lineType=cv2.LINE_AA, shift=SHIFT)
    frames.append(f)

# --- 3. Encode ---------------------------------------------------------------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = tempfile.mkdtemp()
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
shutil.rmtree(tmp)
print("wrote", OUT)
