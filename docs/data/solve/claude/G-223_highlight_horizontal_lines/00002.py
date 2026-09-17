#!/usr/bin/env python3
"""Circle all horizontal lines in first_frame.png with black circles, animated over 3 s."""
import os, subprocess, math, tempfile
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48
SS = 4  # supersampling factor for antialiased strokes

base = np.array(Image.open(SRC).convert("RGB"))
H, W = base.shape[:2]

# --- detect line segments (non-background connected components) ---
bg = np.array([255, 255, 255])
mask = np.any(np.abs(base.astype(int) - bg) > 20, axis=2)
import cv2
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
horizontal = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 30:
        continue
    if w > 3 * h:  # horizontal line
        horizontal.append((x, y, w, h))
print("horizontal lines:", horizontal)

# --- ellipse ("circle") geometry per line ---
STROKE = 4
shapes = []
for (x, y, w, h) in horizontal:
    cx, cy = x + w / 2.0, y + h / 2.0
    rx = w / 2.0 + 14
    ry = h / 2.0 + 18
    shapes.append((cx, cy, rx, ry))

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

def render(frame_idx):
    """Return frame as np array; only the black stroke pixels differ from base."""
    t = frame_idx / (N_FRAMES - 1)
    layer = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(layer)
    k = len(shapes)
    for j, (cx, cy, rx, ry) in enumerate(shapes):
        # stagger: each circle drawn over an overlapping window
        start = 0.08 + j * 0.42 / max(k, 1)
        dur = 0.5
        p = min(max((t - start) / dur, 0.0), 1.0)
        p = ease(p)
        if p <= 0:
            continue
        bbox = [(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS]
        end_ang = -90 + 360 * p
        if p >= 1.0:
            d.ellipse(bbox, outline=255, width=STROKE * SS)
        else:
            d.arc(bbox, start=-90, end=end_ang, fill=255, width=STROKE * SS)
    alpha = np.array(layer.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0, 1)[..., None]
    out = base.astype(np.float32) * (1 - alpha)  # black stroke
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = tempfile.mkdtemp(prefix="frames_")
for i in range(N_FRAMES):
    f = base if i == 0 else render(i)
    Image.fromarray(f).save(os.path.join(frames_dir, f"{i:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(frames_dir, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT
], check=True)
import shutil
shutil.rmtree(frames_dir, ignore_errors=True)
print("wrote", OUT)
