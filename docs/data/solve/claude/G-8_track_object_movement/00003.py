#!/usr/bin/env python3
"""Move the green-bordered object horizontally until it is directly below the red-star object."""
import os, subprocess, shutil
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape

# --- locate the moving object: green border + everything enclosed by it ---
green = np.all(img == (0, 200, 0), axis=-1)
obj_mask = ndimage.binary_fill_holes(green)
# also close small gaps in the border, then fill again, to be safe
obj_mask = ndimage.binary_fill_holes(ndimage.binary_closing(obj_mask, iterations=2)) | obj_mask
ys, xs = np.nonzero(obj_mask)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
obj_cx = (x0 + x1) / 2.0

# --- locate the red star (target column) ---
red = np.all(img == (255, 0, 0), axis=-1) | np.all(img == (200, 0, 0), axis=-1)
rys, rxs = np.nonzero(red)
star_cx = (rxs.min() + rxs.max()) / 2.0

dx_total = int(round(star_cx - obj_cx))

# --- background with the object removed (surroundings are pure white) ---
bg = img.copy()
bg[obj_mask] = (255, 255, 255)
patch = img[y0:y1 + 1, x0:x1 + 1]
pmask = obj_mask[y0:y1 + 1, x0:x1 + 1]

def ease(t):  # smooth ease-in-out
    return 0.5 - 0.5 * np.cos(np.pi * t)

frames = []
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    dx = int(round(dx_total * ease(t)))
    f = bg.copy()
    sx0, sx1 = x0 + dx, x1 + dx + 1
    region = f[y0:y1 + 1, sx0:sx1]
    region[pmask] = patch[pmask]
    frames.append(f)

# first frame must be identical to the source
assert np.array_equal(frames[0], img)

os.makedirs(OUT_DIR, exist_ok=True)
tmp = os.path.join(OUT_DIR, "_frames")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-r", str(FPS), OUT,
], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print(f"wrote {OUT}: {N_FRAMES} frames, dx={dx_total}")
