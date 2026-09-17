#!/usr/bin/env python3
"""Highlight the minimum point of the line chart with a red rectangular border.

The border is traced progressively (top edge -> right -> bottom -> left) over the
duration of the clip; every other pixel is left exactly as in first_frame.png.
"""
import os
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 48
RED = (255, 0, 0)
THICK = 4


def find_min_marker(img):
    """Locate the purple marker of the minimum value (28) and return its bbox."""
    a = np.asarray(img).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r > 120) & (r < 180) & (g < 80) & (b > 200)
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def find_label_top(img, mx0, mx1, my0):
    """Top row of the dark label text sitting just above the marker."""
    a = np.asarray(img).astype(int)
    band = a[my0 - 40:my0 - 2, mx0 - 12:mx1 + 12]
    dark = band.max(-1) < 120
    rows = np.nonzero(dark.any(1))[0]
    return my0 - 40 + rows.min()


def rect_path(x0, y0, x1, y1):
    """Ordered list of pixel coordinates along the rectangle's centreline."""
    pts = []
    pts += [(x, y0) for x in range(x0, x1 + 1)]
    pts += [(x1, y) for y in range(y0 + 1, y1 + 1)]
    pts += [(x, y1) for x in range(x1 - 1, x0 - 1, -1)]
    pts += [(x0, y) for y in range(y1 - 1, y0, -1)]
    return pts


def main():
    base = Image.open(SRC).convert("RGB")
    W, H = base.size
    base_arr = np.asarray(base).copy()

    mx0, my0, mx1, my1 = find_min_marker(base)
    # The "28" data label sits directly above the marker; include it inside the
    # border so the rectangle does not cut through the text.
    lbl_top = find_label_top(base, mx0, mx1, my0)
    pad = 10  # gap between marker and border
    x0, y0 = mx0 - pad, lbl_top - pad
    x1, y1 = mx1 + pad, my1 + pad
    path = rect_path(x0, y0, x1, y1)
    total = len(path)
    half = THICK // 2

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        # frame 0 shows the untouched chart; last frames show the full border
        t = i / (N_FRAMES - 3) if i < N_FRAMES - 2 else 1.0
        t = min(max(t, 0.0), 1.0)
        n = int(round(t * total))
        fr = base_arr.copy()
        for (px, py) in path[:n]:
            ya, yb = max(0, py - half), min(H, py - half + THICK)
            xa, xb = max(0, px - half), min(W, px - half + THICK)
            fr[ya:yb, xa:xb] = RED
        frames.append(fr)

    # Frame 0 must equal first_frame.png exactly.
    assert np.array_equal(frames[0], base_arr)

    raw = np.stack(frames).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    subprocess.run(cmd, input=raw, check=True)
    print(f"wrote {OUT} ({N_FRAMES} frames @ {FPS} fps, border at "
          f"({x0},{y0})-({x1},{y1}))")


if __name__ == "__main__":
    main()
