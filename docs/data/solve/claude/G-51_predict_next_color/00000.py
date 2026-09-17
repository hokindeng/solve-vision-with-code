"""Predict the next color in the sequence and animate filling the empty hexagon.

Sequence in first_frame.png: green, blue, purple, green, (empty) -> period 3 -> blue.
Only the interior of the empty outlined hexagon changes; every other pixel is kept.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from collections import deque

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 64

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape

# --- locate the outlined (empty) hexagon and its interior ------------------
outline_col = np.array([100, 100, 100], dtype=np.uint8)
outline = np.all(first == outline_col, axis=2)
ys, xs = np.nonzero(outline)
cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0

# flood-fill from the center over non-outline pixels to get the interior mask
interior = np.zeros((H, W), dtype=bool)
q = deque([(int(round(cy)), int(round(cx)))])
interior[q[0]] = True
while q:
    y, x = q.popleft()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and not interior[ny, nx] and not outline[ny, nx]:
            interior[ny, nx] = True
            q.append((ny, nx))

# --- infer the next color from the filled hexagons ---------------------------
# sample fill colors at the centers of the filled hexagons (same row as the empty one)
row = first[int(round(cy))]
seq, prev = [], None
for x in range(W):
    c = tuple(int(v) for v in row[x])
    if c != (255, 255, 255) and c != (0, 0, 0) and c != tuple(outline_col) and c != prev:
        seq.append(c)
    prev = c
# seq: colors in reading order (dedup consecutive). find smallest period.
def next_color(s):
    for p in range(1, len(s) + 1):
        if all(s[i] == s[i - p] for i in range(p, len(s))):
            return s[len(s) - p]
    return s[-1]
fill = np.array(next_color(seq), dtype=np.uint8)

# --- hexagon geometry (radius from outline bbox height) ----------------------
R = (ys.max() - ys.min()) / 2.0 - 3.0  # inner radius (inside the stroke)
def hex_points(scale):
    r = R * scale
    return [(cx + r * np.cos(np.deg2rad(90 + 60 * k)),
             cy + r * np.sin(np.deg2rad(90 + 60 * k))) for k in range(6)]

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

# --- render frames -----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "frames")
os.makedirs(tmp, exist_ok=True)
grow_frames = 52  # animation, then hold
for i in range(N_FRAMES):
    frame = first.copy()
    if i > 0:
        t = min(1.0, (i) / (grow_frames - 1))
        s = ease(t)
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(hex_points(s), fill=255)
        mask = (np.array(m) > 0) & interior
        if t >= 1.0:
            mask = interior
        frame[mask] = fill
    Image.fromarray(frame).save(os.path.join(tmp, f"{i:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT
], check=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print("wrote", OUT, "fill color", fill.tolist())
