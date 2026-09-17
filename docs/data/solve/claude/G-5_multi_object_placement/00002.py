#!/usr/bin/env python3
"""Move each colored object in a straight line onto the star marker of the same color."""
import subprocess, os
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT_DIR = "/app/output"
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48
W = H = 1024

img = np.array(Image.open(SRC).convert("RGB"))
bg = img[0, 0].copy()

# Segment all non-background components.
mask = np.any(img != bg, axis=2)
lab, n = ndimage.label(mask)
comps = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    color = tuple(int(c) for c in np.median(img[ys, xs], axis=0))
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    comps.append(dict(id=i, color=color, bbox=bbox, n=len(xs)))

# Stars are the small components; objects are the large ones. Pair them by color.
sizes = sorted(c["n"] for c in comps)
thresh = (sizes[len(sizes) // 2 - 1] + sizes[len(sizes) // 2]) / 2 if len(sizes) > 1 else 0
stars = [c for c in comps if c["n"] <= thresh]
objs = [c for c in comps if c["n"] > thresh]

def center(c):
    x0, y0, x1, y1 = c["bbox"]
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

def color_dist(a, b):
    return sum((p - q) ** 2 for p, q in zip(a, b))

moves = []
base = img.copy()  # background + stars (objects erased)
for o in objs:
    s = min(stars, key=lambda s: color_dist(s["color"], o["color"]))
    m = lab == o["id"]
    x0, y0, x1, y1 = o["bbox"]
    sprite = img[y0:y1 + 1, x0:x1 + 1].copy()
    smask = m[y0:y1 + 1, x0:x1 + 1]
    base[m] = bg
    start = center(o)
    end = center(s)
    moves.append(dict(sprite=sprite, mask=smask, x0=x0, y0=y0, delta=end - start))

def blit(frame, sprite, smask, x, y):
    h, w = smask.shape
    fx0, fy0 = max(x, 0), max(y, 0)
    fx1, fy1 = min(x + w, W), min(y + h, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sx0, sy0 = fx0 - x, fy0 - y
    sub = smask[sy0:sy0 + (fy1 - fy0), sx0:sx0 + (fx1 - fx0)]
    region = frame[fy0:fy1, fx0:fx1]
    region[sub] = sprite[sy0:sy0 + (fy1 - fy0), sx0:sx0 + (fx1 - fx0)][sub]

os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in range(N_FRAMES):
    t = f / (N_FRAMES - 1)  # linear pacing: 0 at first frame, 1 at last
    frame = base.copy()
    for mv in moves:
        dx, dy = np.round(mv["delta"] * t).astype(int)
        blit(frame, mv["sprite"], mv["mask"], mv["x0"] + dx, mv["y0"] + dy)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()
print("wrote", OUT)
