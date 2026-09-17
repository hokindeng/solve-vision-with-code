#!/usr/bin/env python3
"""Majority-color task: fade out all objects whose color is not the majority."""
import os, subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 40
FADE_START, FADE_END = 4, 36   # frame indices over which the vanish happens

img = np.array(Image.open(SRC).convert("RGB")).astype(np.uint8)
H, W, _ = img.shape
bg = np.array([255, 255, 255], np.uint8)

# --- find fill colors (everything except white background and black outlines)
flat = img.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
is_white = (cols == 255).all(1)
is_black = (cols == 0).all(1)
fill_cols = [tuple(c) for c, w, b, n in zip(cols, is_white, is_black, counts)
             if not w and not b and n > 50]

# --- connected components per fill color -> object labels
label = np.zeros((H, W), np.int32)
obj_color = {}   # label -> color tuple
nxt = 1
for c in fill_cols:
    mask = (img == np.array(c, np.uint8)).all(2).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    for k in range(1, n):
        m = lab == k
        if m.sum() < 30:      # ignore anti-alias specks
            continue
        label[m] = nxt
        obj_color[nxt] = c
        nxt += 1

# --- assign black outline pixels to the nearest object (iterative dilation)
black = (img == 0).all(2)
unassigned = black & (label == 0)
kernel = np.ones((3, 3), np.uint8)
for _ in range(6):
    if not unassigned.any():
        break
    dil = cv2.dilate(label.astype(np.float32), kernel).astype(np.int32)
    # dilate with max; fill only currently unassigned black pixels
    fill = unassigned & (dil > 0)
    label[fill] = dil[fill]
    unassigned = black & (label == 0)

# --- count objects per color, pick majority
color_count = {}
for l, c in obj_color.items():
    color_count[c] = color_count.get(c, 0) + 1
majority = max(color_count, key=color_count.get)
print("object counts per color:", {str(k): v for k, v in color_count.items()})
print("majority color:", majority)

vanish = np.zeros((H, W), bool)
for l, c in obj_color.items():
    if c != majority:
        vanish |= label == l
vanish_f = vanish.astype(np.float32)[..., None]

def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

os.makedirs(OUT_DIR, exist_ok=True)
base = img.astype(np.float32)
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for i in range(N_FRAMES):
    a = smoothstep((i - FADE_START) / (FADE_END - FADE_START))
    if i >= FADE_END:
        a = 1.0
    frame = base * (1 - a * vanish_f) + bg.astype(np.float32) * (a * vanish_f)
    frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    if i == 0:
        frame = img.copy()
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print("wrote", OUT)
