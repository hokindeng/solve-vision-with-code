#!/usr/bin/env python3
"""Outline the innermost of several concentric squares with a blue outline, step by step."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
FPS, N_FRAMES = 16, 85
BLUE = (0, 0, 255)
THICK = 8  # outline thickness in px


def find_innermost_square(img):
    """Return (x0, y0, x1, y1) inclusive bbox of the innermost concentric square."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    cols, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(h, w)
    best = None
    for i, c in enumerate(cols):
        if counts[i] < 200:
            continue
        ys, xs = np.where(inv == i)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        # a solid filled square: pixel count ~ bbox area, roughly square
        if counts[i] < 0.95 * bw * bh or abs(bw - bh) > 3:
            continue
        if best is None or bw * bh < best[0]:
            best = (bw * bh, (int(x0), int(y0), int(x1), int(y1)))
    return best[1]


def draw_partial_outline(frame, box, progress):
    """Draw the outline path clockwise (top, right, bottom, left) up to `progress` in [0,1]."""
    x0, y0, x1, y1 = box
    t = THICK
    # outline centered on the square's edge
    ox0, oy0, ox1, oy1 = x0 - t // 2, y0 - t // 2, x1 + t // 2 + 1, y1 + t // 2 + 1
    W, H = ox1 - ox0, oy1 - oy0
    perim = 2 * (W + H)
    dist = progress * perim
    segs = [  # (length, painter)
        (W, lambda d: frame.__setitem__((slice(oy0, oy0 + t), slice(ox0, ox0 + int(round(d)))), BLUE)),
        (H, lambda d: frame.__setitem__((slice(oy0, oy0 + int(round(d))), slice(ox1 - t, ox1)), BLUE)),
        (W, lambda d: frame.__setitem__((slice(oy1 - t, oy1), slice(ox1 - int(round(d)), ox1)), BLUE)),
        (H, lambda d: frame.__setitem__((slice(oy1 - int(round(d)), oy1), slice(ox0, ox0 + t)), BLUE)),
    ]
    for length, paint in segs:
        if dist <= 0:
            break
        paint(min(dist, length))
        dist -= length


def ease(u):
    return u * u * (3 - 2 * u)  # smoothstep


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    box = find_innermost_square(base)
    print("innermost square bbox:", box)

    hold_start, hold_end = 8, 12
    anim = N_FRAMES - hold_start - hold_end
    tmpdir = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        frame = base.copy()
        if i >= hold_start:
            u = min(1.0, (i - hold_start) / (anim - 1))
            draw_partial_outline(frame, box, ease(u))
        Image.fromarray(frame).save(os.path.join(tmpdir, f"f{i:04d}.png"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmpdir, "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
