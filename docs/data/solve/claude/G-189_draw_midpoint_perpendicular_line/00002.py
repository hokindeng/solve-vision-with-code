#!/usr/bin/env python3
"""Animate drawing a red perpendicular (vertical) line through the midpoint marker
between the two horizontal parallel lines of first_frame.png."""
import subprocess, numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 50

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# Locate the two horizontal black lines (full-width rows).
black_rows = np.where((base == 0).all(2).all(1))[0]
gaps = np.where(np.diff(black_rows) > 1)[0]
upper = black_rows[: gaps[0] + 1]
lower = black_rows[gaps[0] + 1 :]
mid_y = (upper.mean() + lower.mean()) / 2.0

# Locate the marker closest to the vertical midpoint (non-white, non-black blob).
other = ~((base == 255).all(2) | (base == 0).all(2))
cols = {}
for c in np.unique(base[other].reshape(-1, 3), axis=0):
    m = (base == c).all(2)
    ys, xs = np.where(m)
    cols[tuple(c)] = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0, m)
_, (cx, cy, marker_mask) = min(cols.items(), key=lambda kv: abs(kv[1][1] - mid_y))
x0 = int(round(cx))

y_start = upper.max() + 1     # first row below the upper line
y_end = lower.min() - 1       # last row above the lower line
thickness = len(upper)        # match the black lines' thickness (4 px)
xs = slice(x0 - thickness // 2, x0 - thickness // 2 + thickness)
RED = np.array([255, 0, 0], np.uint8)

def frame(i):
    t = i / (N - 1)
    t = t * t * (3 - 2 * t)  # smoothstep easing
    img = base.copy()
    y_cur = int(round(y_start + t * (y_end + 1 - y_start)))
    if y_cur > y_start:
        seg = img[y_start:y_cur, xs]
        keep = marker_mask[y_start:y_cur, xs]   # leave the midpoint marker pixels untouched
        seg[~keep] = RED
    return img

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for i in range(N):
    ff.stdin.write(frame(i).tobytes())
ff.stdin.close(); ff.wait()
print("wrote", OUT, "line x =", x0, "rows", y_start, "-", y_end)
