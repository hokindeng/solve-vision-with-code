#!/usr/bin/env python3
"""Color the triple intersection of the three Venn circles red, animated over 60 frames."""
import os, subprocess, numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 60, 16
RED = np.array([255, 0, 0], dtype=np.uint8)

def fit_circle(xs, ys):
    """Algebraic least-squares circle fit -> (cx, cy, r)."""
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return cx, cy, r

def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = base.shape
    white = np.all(base == 255, axis=2)

    # Distinct stroke colours (anything non-white), clustered by hue via exact palette.
    cols, counts = np.unique(base[~white].reshape(-1, 3), axis=0, return_counts=True)
    # The three dominant pure stroke colours are the circles; antialiased pixels are blends.
    order = np.argsort(-counts)
    stroke_cols = [tuple(cols[i]) for i in order[:3]]

    circles = []
    for col in stroke_cols:
        # Include antialiased pixels: nearest palette colour by distance, non-white.
        d = np.linalg.norm(base.astype(float) - np.array(col, float), axis=2)
        others = [np.linalg.norm(base.astype(float) - np.array(c, float), axis=2)
                  for c in stroke_cols if c != col]
        m = (~white) & (d < 120) & np.all([d < o for o in others], axis=0)
        ys, xs = np.nonzero(m)
        circles.append(fit_circle(xs.astype(float), ys.astype(float)))

    yy, xx = np.mgrid[0:H, 0:W]
    inside = np.ones((H, W), bool)
    for cx, cy, r in circles:
        inside &= (xx - cx) ** 2 + (yy - cy) ** 2 < (r - 1.0) ** 2
    region = inside & white  # never touch the outline pixels

    ys, xs = np.nonzero(region)
    # Radial wipe from the region centroid outward.
    ccx, ccy = xs.mean(), ys.mean()
    dist = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2)
    dmax = dist[region].max()

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        t = t * t * (3 - 2 * t)  # smoothstep easing
        f = base.copy()
        if i > 0:
            m = region & (dist <= t * (dmax + 1.0))
            f[m] = RED
        frames.append(f)
    frames[-1] = base.copy(); frames[-1][region] = RED  # guarantee completion

    raw = np.stack(frames).tobytes()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    subprocess.run(cmd, input=raw, check=True)
    print("circles:", [tuple(round(v, 1) for v in c) for c in circles])
    print("region px:", int(region.sum()), "->", OUT)

if __name__ == "__main__":
    main()
