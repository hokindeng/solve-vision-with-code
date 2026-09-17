#!/usr/bin/env python3
"""Move the 3 left-side objects horizontally into their dashed targets."""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 30, 16
DASH = np.array([120, 120, 120])

img = np.array(Image.open(SRC).convert("RGB")).astype(np.float64)
H, W, _ = img.shape
nonwhite = (img != 255).any(axis=2)
lab, n = ndimage.label(nonwhite, structure=np.ones((3, 3)))
sizes = ndimage.sum(nonwhite, lab, range(1, n + 1))

# Solid objects = large components; dashed outlines = many small components.
objects = []
for i, s in enumerate(sizes, start=1):
    if s < 1000:
        continue
    mask = lab == i
    mask = ndimage.binary_fill_holes(mask)
    ys, xs = np.where(mask)
    # reference (most saturated / darkest) colour for alpha estimation
    px = img[mask]
    ref = px[np.argmax((255 - px).max(axis=1))]
    a = np.clip((255 - img).max(axis=2) / (255 - ref).max(), 0, 1) * mask
    rgb = np.where(a[..., None] > 0,
                   (img - (1 - a[..., None]) * 255) / np.maximum(a[..., None], 1e-6), 0)
    rgb = np.clip(rgb, 0, 255)
    objects.append(dict(mask=mask, alpha=a, rgb=rgb,
                        x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max()))

# Dash pixels: everything non-white that is not part of an object.
obj_union = np.zeros((H, W), bool)
for o in objects:
    obj_union |= ndimage.binary_dilation(o["mask"], iterations=2)
dash = nonwhite & ~obj_union

# For each object find the integer horizontal shift that best lays its
# boundary over dash pixels (targets are to the right).
for o in objects:
    edge = o["mask"] & ~ndimage.binary_erosion(o["mask"], iterations=3)
    ey, ex = np.where(edge)
    best, best_dx = -1, 0
    for dx in range(1, W - o["x1"]):
        xs = ex + dx
        score = dash[ey, xs].sum()
        if score > best:
            best, best_dx = score, dx
    o["dx"] = best_dx
    print(f"object bbox x[{o['x0']},{o['x1']}] y[{o['y0']},{o['y1']}] -> dx={best_dx} (score {best})")

# Background = frame with objects removed (they sit on pure white).
bg = img.copy()
for o in objects:
    bg[o["mask"]] = 255

def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for f in range(N_FRAMES):
    t = ease(f / (N_FRAMES - 1))
    frame = bg.copy()
    for o in objects:  # draw order: circle (largest) first, hexagons on top
        dx = int(round(o["dx"] * t))
        a = np.roll(o["alpha"], dx, axis=1)[..., None]
        rgb = np.roll(o["rgb"], dx, axis=1)
        frame = a * rgb + (1 - a) * frame
    frames.append(np.clip(frame + 0.5, 0, 255).astype(np.uint8))

# Sanity: first frame must equal the source exactly.
assert np.array_equal(frames[0], img.astype(np.uint8)), "first frame mismatch"

Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
raw = b"".join(fr.tobytes() for fr in frames)
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", OUT,
], input=raw, check=True)
print("wrote", OUT)
