#!/usr/bin/env python3
"""Symbol deletion: fade out the red-bordered target symbol, leaving everything else untouched."""
import os, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 46, 16

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape

# Locate the red border marking the deletion target.
r, g, b = base[..., 0].astype(int), base[..., 1].astype(int), base[..., 2].astype(int)
red = (r > 200) & (g < 60) & (b < 60)
ys, xs = np.where(red)
x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

# Background colour (sampled at a corner) used to erase the target region.
bg = base[2, 2].astype(float)

# Everything inside the red border (border + symbol) is the target to delete.
mask = np.zeros((H, W), dtype=bool)
mask[y0:y1, x0:x1] = True

def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

frames = []
hold_start, hold_end = 8, 6
anim = N_FRAMES - hold_start - hold_end
for i in range(N_FRAMES):
    if i < hold_start:
        alpha = 0.0
    elif i >= N_FRAMES - hold_end:
        alpha = 1.0
    else:
        alpha = ease((i - hold_start + 1) / anim)
    f = base.astype(float).copy()
    region = f[mask]
    f[mask] = region * (1 - alpha) + bg * alpha
    frames.append(np.clip(np.round(f), 0, 255).astype(np.uint8))

# Frame 0 must equal first_frame exactly; last frame must have target fully removed.
assert np.array_equal(frames[0], base)
assert np.all(frames[-1][y0:y1, x0:x1] == bg.astype(np.uint8))

os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close()
p.wait()
if p.returncode != 0:
    raise SystemExit("ffmpeg failed")
print(f"wrote {OUT}: {N_FRAMES} frames @ {FPS} fps, target at x[{x0},{x1}) y[{y0},{y1})")
