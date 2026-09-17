#!/usr/bin/env python3
"""Move the green-bordered hexagon horizontally until it sits directly below the
red-star object, carrying the green border along.  Everything else is untouched."""
import os, subprocess, tempfile
import numpy as np, cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 60, 16

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

def color_mask(rgb, tol=10):
    return np.abs(base.astype(int) - np.array(rgb)).sum(2) < tol

# Moving object = green border (0,200,0) plus everything enclosed by it.
green = color_mask((0, 200, 0)).astype(np.uint8)
cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
obj_mask = np.zeros((H, W), np.uint8)
cv2.drawContours(obj_mask, cnts, -1, 1, -1)
obj_mask = obj_mask.astype(bool)

# Target: x-center of red star; start: x-center of the moving object.
ys, xs = np.where(color_mask((255, 0, 0)))
star_cx = (xs.min() + xs.max()) / 2.0
ys, xs = np.where(obj_mask)
obj_cx = (xs.min() + xs.max()) / 2.0
total_dx = int(round(star_cx - obj_cx))

# Background revealed when the object leaves: the frame with the object removed.
# The object sits on plain white, so fill with the surrounding colour.
bg = base.copy()
bg[obj_mask] = 255

obj_pixels = base[obj_mask]
oy, ox = np.where(obj_mask)

def ease(t):  # smooth ease-in-out
    return 0.5 - 0.5 * np.cos(np.pi * t)

tmp = tempfile.mkdtemp()
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    dx = int(round(total_dx * ease(t)))
    frame = bg.copy()
    nx = ox + dx
    ok = (nx >= 0) & (nx < W)
    frame[oy[ok], nx[ok]] = obj_pixels[ok]
    if i == 0:
        assert np.array_equal(frame, base)
    Image.fromarray(frame).save(os.path.join(tmp, f"f_{i:04d}.png"))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "f_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                "-g", "1", OUT], check=True)
print(f"wrote {OUT}: dx={total_dx}, {N_FRAMES} frames @ {FPS} fps")
