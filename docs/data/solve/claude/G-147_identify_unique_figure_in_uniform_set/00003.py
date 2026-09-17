#!/usr/bin/env python3
"""Identify the unique shape in first_frame.png and circle it in red, step by step."""
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import subprocess

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 60

base = Image.open(FIRST).convert("RGB")
W, H = base.size
arr = np.array(base)

# --- Step 1: find shapes (connected components of non-background pixels) ---
bg = arr[0, 0].astype(int)
mask = (np.abs(arr.astype(int) - bg).sum(axis=2) > 60).astype(np.uint8)
n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
shapes = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 50:
        continue
    fill = area / float(w * h)          # square ~1.0, circle ~0.785, triangle ~0.5
    shapes.append(dict(cx=x + w / 2.0, cy=y + h / 2.0, w=w, h=h, area=area, fill=fill))

# --- Step 2: identify the unique one (largest deviation from median features) ---
feats = np.array([[s["fill"], s["area"], s["w"], s["h"]] for s in shapes], dtype=float)
med = np.median(feats, axis=0)
scale = np.maximum(np.abs(med), 1e-6)
dev = (np.abs(feats - med) / scale).sum(axis=1)
odd = shapes[int(np.argmax(dev))]
print("shapes:", [(round(s['fill'], 2), s['area']) for s in shapes])
print("unique shape at", odd["cx"], odd["cy"])

cx, cy = odd["cx"], odd["cy"]
radius = max(odd["w"], odd["h"]) / 2.0 * 1.55 + 8
RED = (255, 0, 0)
LINE_W = 6

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

# --- Step 3: animate: pause (identify), then sweep the red circle, then hold ---
HOLD_START = 14   # frames of looking at the scene
SWEEP = 34        # frames drawing the circle
frames = []
for f in range(N_FRAMES):
    im = base.copy()
    if f >= HOLD_START:
        t = min(1.0, (f - HOLD_START) / float(SWEEP - 1))
        sweep_deg = 360.0 * ease(t)
        if sweep_deg > 0:
            big = 4  # supersample for smooth anti-aliased arc
            layer = Image.new("RGBA", (W * big, H * big), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            bbox = [(cx - radius) * big, (cy - radius) * big, (cx + radius) * big, (cy + radius) * big]
            start = -90
            if sweep_deg >= 359.9:
                d.ellipse(bbox, outline=RED + (255,), width=LINE_W * big)
            else:
                d.arc(bbox, start=start, end=start + sweep_deg, fill=RED + (255,), width=LINE_W * big)
            layer = layer.resize((W, H), Image.LANCZOS)
            im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
    frames.append(np.array(im))

# Frame 0 must be exactly first_frame.png
frames[0] = arr.copy()

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames.raw")
with open(tmp, "wb") as fh:
    for fr in frames:
        fh.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", tmp,
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT
], check=True)
os.remove(tmp)
print("wrote", OUT, len(frames), "frames")
