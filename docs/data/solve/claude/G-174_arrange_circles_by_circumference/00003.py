#!/usr/bin/env python3
"""Rearrange the circles of first_frame.png into a centered horizontal row,
sorted left-to-right by circumference (largest to smallest), and render the
motion as an 80-frame, 16 fps, 1024x1024 H.264 video."""
import os, subprocess, shutil, tempfile
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80
HOLD_START, HOLD_END = 6, 8          # frames of stillness at both ends
GAP = 20                             # horizontal gap between neighbouring circles

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg_color = np.array([255, 255, 255], np.uint8)

# --- detect circles as connected non-background components ------------------
mask = (img != bg_color).any(axis=2).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
circles = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 50:
        continue
    m = (lab[y:y + h, x:x + w] == i)
    sprite = img[y:y + h, x:x + w].copy()
    circles.append(dict(
        sprite=sprite, mask=m, w=w, h=h,
        cx=x + (w - 1) / 2.0, cy=y + (h - 1) / 2.0,
        radius=(w - 1) / 2.0,
    ))

# Background = first frame with all circles removed
background = img.copy()
background[mask.astype(bool)] = bg_color

# --- target layout: one row, centered, sorted by circumference (∝ radius) ----
order = sorted(range(len(circles)), key=lambda k: -circles[k]["radius"])
total_w = sum(circles[k]["w"] for k in order) + GAP * (len(order) - 1)
x = (W - total_w) / 2.0
row_y = (H - 1) / 2.0
for k in order:
    c = circles[k]
    c["tx"] = x + (c["w"] - 1) / 2.0
    c["ty"] = row_y
    x += c["w"] + GAP

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def render(t):
    frame = background.copy()
    # draw larger circles first so smaller ones stay visible when paths cross
    for k in sorted(range(len(circles)), key=lambda k: -circles[k]["radius"]):
        c = circles[k]
        cx = c["cx"] + (c["tx"] - c["cx"]) * t
        cy = c["cy"] + (c["ty"] - c["cy"]) * t
        x0 = int(round(cx - (c["w"] - 1) / 2.0))
        y0 = int(round(cy - (c["h"] - 1) / 2.0))
        xs, ys = max(0, x0), max(0, y0)
        xe, ye = min(W, x0 + c["w"]), min(H, y0 + c["h"])
        if xe <= xs or ye <= ys:
            continue
        sp = c["sprite"][ys - y0:ye - y0, xs - x0:xe - x0]
        mk = c["mask"][ys - y0:ye - y0, xs - x0:xe - x0]
        region = frame[ys:ye, xs:xe]
        region[mk] = sp[mk]
    return frame

os.makedirs(OUT_DIR, exist_ok=True)
tmp = tempfile.mkdtemp()
move_frames = N_FRAMES - HOLD_START - HOLD_END
for f in range(N_FRAMES):
    if f < HOLD_START:
        t = 0.0
    elif f >= N_FRAMES - HOLD_END:
        t = 1.0
    else:
        t = ease((f - HOLD_START) / float(move_frames - 1))
    Image.fromarray(render(t)).save(os.path.join(tmp, f"{f:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT,
], check=True)
shutil.rmtree(tmp)
print("wrote", OUT)
