#!/usr/bin/env python3
"""Move the green attention box from the left object to the right object.

Everything except the green box is left exactly as in first_frame.png.
The box glides (ease-in-out) from its original position/size to a box that
frames the right object with the same padding and stroke style.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 25, 16

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape
Y, X = np.mgrid[0:H, 0:W]

# --- locate the green box -------------------------------------------------
GREEN = np.array([70, 140, 70], dtype=np.uint8)
gmask = np.abs(first.astype(np.int64) - GREEN.astype(np.int64)).sum(2) < 40
ys, xs = np.where(gmask)
ox0, ox1, oy0, oy1 = xs.min(), xs.max(), ys.min(), ys.max()   # outer extent
# stroke geometry measured from the frame: 8 px wide stroke whose centreline
# corners are 4 px inside the outer extent, with round (disc) corner joins.
STROKE, R = 8, 4.3
src = np.array([ox0 + 4, oy0 + 4, ox1 - 4, oy1 - 4], dtype=float)  # centreline corners


def box_mask(x0, y0, x1, y1):
    """Rasterise the attention box with the same style as the original."""
    m = np.zeros((H, W), bool)
    m[y0:y1 + 1, x0 - 4:x0 + 4] = True   # left edge
    m[y0:y1 + 1, x1 - 3:x1 + 5] = True   # right edge
    m[y0 - 3:y0 + 5, x0:x1 + 1] = True   # top edge
    m[y1 - 4:y1 + 4, x0:x1 + 1] = True   # bottom edge
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        m |= (X - cx) ** 2 + (Y - cy) ** 2 <= R * R
    return m


assert (box_mask(*src.astype(int)) == gmask).all(), "box model does not reproduce the original"

# background with the box erased (the box sits on plain background)
bg_color = first[0, 0].copy()
clean = first.copy()
clean[gmask] = bg_color

# --- object bounding boxes ------------------------------------------------
nonbg = np.abs(clean.astype(np.int64) - bg_color.astype(np.int64)).sum(2) > 30
left = nonbg.copy(); left[:, W // 2:] = False
right = nonbg.copy(); right[:, :W // 2] = False
ly, lx = np.where(left)
ry, rx = np.where(right)
# gap between object bbox and the stroke centreline, as in the first frame
pad = int(round(np.mean([lx.min() - src[0], ly.min() - src[1], src[2] - lx.max(), src[3] - ly.max()])))
dst = np.array([rx.min() - pad, ry.min() - pad, rx.max() + pad, ry.max() + pad], dtype=float)


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


os.makedirs(OUT_DIR, exist_ok=True)
frames = [first.copy()]
for i in range(1, N_FRAMES):
    s = ease(i / (N_FRAMES - 1))
    rect = np.round(src + (dst - src) * s).astype(int)
    f = clean.copy()
    f[box_mask(*rect)] = GREEN
    frames.append(f)

# --- encode with ffmpeg (H.264, yuv420p) ----------------------------------
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
       "-movflags", "+faststart", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
proc.stdin.close()
proc.wait()
assert proc.returncode == 0, "ffmpeg failed"
print(f"wrote {OUT}: {N_FRAMES} frames @ {FPS} fps, pad {pad}, box {src.astype(int)} -> {dst.astype(int)}")
