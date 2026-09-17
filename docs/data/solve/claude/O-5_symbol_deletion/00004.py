#!/usr/bin/env python3
"""Symbol deletion: fade out the red-bordered target, then slide the
following symbols left to close the gap. Every other pixel stays as in
first_frame.png."""
import os, subprocess, shutil
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 46

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape
bg = base[0, 0].copy()

# ---- locate the red border (target) --------------------------------------
r, g, b = base[..., 0].astype(int), base[..., 1].astype(int), base[..., 2].astype(int)
red = (r > 200) & (g < 80) & (b < 80)
ys, xs = np.where(red)
rx0, rx1, ry0, ry1 = xs.min(), xs.max(), ys.min(), ys.max()

# ---- locate all cells via column occupancy in the symbol row band --------
nonbg = ~(base == bg).all(2)
band = nonbg[ry0:ry1 + 1].any(0)
edges = np.where(np.diff(band.astype(int)) != 0)[0]
runs = [(edges[i] + 1, edges[i + 1]) for i in range(0, len(edges) - 1, 2)]  # inclusive
# merge runs that fall inside the red border into one cell
cells = []
for a, c in runs:
    if cells and a <= rx1 and cells[-1][0] >= rx0:
        cells[-1] = (cells[-1][0], c)
    else:
        cells.append((a, c))
target_idx = next(i for i, (a, c) in enumerate(cells) if a >= rx0 - 1 and c <= rx1 + 1)
pitch = cells[1][0] - cells[0][0]

# region to fade: red border bbox (contains the target cell entirely)
fy0, fy1 = ry0, ry1 + 1
fx0, fx1 = rx0, rx1 + 1
# patch of everything right of the target that must slide left
row_y0 = min(ry0, nonbg[:, cells[0][0]:cells[-1][1] + 1].any(1).nonzero()[0].min())
row_y1 = max(ry1, nonbg[:, cells[0][0]:cells[-1][1] + 1].any(1).nonzero()[0].max()) + 1
right_x0 = fx1
right_x1 = W
right_patch = base[row_y0:row_y1, right_x0:right_x1].copy()


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


FADE_END = 18            # frames 0..FADE_END: target fades out
SLIDE_START, SLIDE_END = 20, 45

frames = []
for i in range(N_FRAMES):
    f = base.copy()
    # phase 1: fade target + red border toward background
    a = ease(i / FADE_END)
    reg = base[fy0:fy1, fx0:fx1].astype(float)
    f[fy0:fy1, fx0:fx1] = np.round(reg * (1 - a) + bg * a).astype(np.uint8)
    # phase 2: slide following symbols left by one pitch
    if i >= SLIDE_START:
        s = ease((i - SLIDE_START) / (SLIDE_END - SLIDE_START))
        dx = s * pitch
        # clear everything from the target's left edge rightwards, then draw shifted patch
        f[row_y0:row_y1, fx0:W] = bg
        # subpixel shift via linear interpolation between integer offsets
        d0, fr = int(np.floor(dx)), dx - np.floor(dx)
        canvas = np.zeros((row_y1 - row_y0, W - fx0 + pitch + 2, 3), float)
        canvas[:] = bg
        for d, wgt in ((d0, 1 - fr), (d0 + 1, fr)):
            if wgt <= 0:
                continue
            tmp = np.tile(bg.astype(float), (row_y1 - row_y0, W - fx0 + pitch + 2, 1))
            x_start = right_x0 - d - fx0 + pitch  # offset in canvas coords (canvas starts at fx0 - pitch)
            tmp[:, x_start:x_start + right_patch.shape[1]] = right_patch
            canvas = canvas * (1 - wgt) + tmp * wgt if d == d0 else canvas + (tmp - bg) * wgt
        # canvas x=0 corresponds to image x = fx0 - pitch; copy the part from fx0 onward
        off = pitch
        f[row_y0:row_y1, fx0:W] = np.clip(np.round(canvas[:, off:off + (W - fx0)]), 0, 255).astype(np.uint8)
    frames.append(f)

# frame 0 must equal first_frame exactly
assert (frames[0] == base).all()

os.makedirs(OUT_DIR, exist_ok=True)
tmp_dir = os.path.join(OUT_DIR, "_frames")
shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp_dir, f"{i:04d}.png"))
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(tmp_dir, "%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
    "-vf", "format=yuv420p", OUT,
], check=True)
shutil.rmtree(tmp_dir)
Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))
print("wrote", OUT, "cells:", cells, "target:", target_idx)
