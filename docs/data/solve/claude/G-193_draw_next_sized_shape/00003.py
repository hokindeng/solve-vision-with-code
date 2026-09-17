#!/usr/bin/env python3
"""Draw the next shape of a repeating size cycle into the empty dashed box.

The row contains squares whose sizes follow a repeating cycle. We detect the
squares, infer the cycle period, predict the next size, and animate drawing
that square inside the empty box. Every pixel outside the box interior is left
exactly as in first_frame.png in every frame.
"""
import os
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 60


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def analyse(img):
    """Return (shape colour, list of (cx, cy, size) sorted by x, box bounds)."""
    white = np.all(img == 255, axis=2)
    black = np.all(img == 0, axis=2)
    coloured = ~white & ~black
    lab, n = ndimage.label(coloured)
    shapes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 50:
            continue
        shapes.append(((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0,
                       xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    shapes.sort()
    ys, xs = np.where(black)  # dashed box
    box = (xs.min(), xs.max(), ys.min(), ys.max())
    colour = img[coloured][0].tolist()
    return colour, shapes, box


def next_size(sizes):
    """Find the shortest period consistent with the sequence; return next value."""
    for p in range(1, len(sizes) + 1):
        if all(abs(sizes[i] - sizes[i - p]) <= 2 for i in range(p, len(sizes))):
            return sizes[len(sizes) - p]
    return sizes[-1]


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    colour, shapes, box = analyse(base)
    sizes = [s[2] for s in shapes]
    size = int(round(next_size(sizes)))
    print("sizes:", sizes, "-> next size:", size)

    bx0, bx1, by0, by1 = box
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    # Target square (integer pixel bounds), same vertical alignment as row.
    x0 = int(round(cx - size / 2.0)); x1 = x0 + size - 1
    y0 = int(round(cy - size / 2.0)); y1 = y0 + size - 1
    col = np.array(colour, dtype=np.uint8)

    # Timeline (frames): hold, trace outline, fill from centre outward, hold.
    T_HOLD0, T_OUTLINE, T_FILL, T_END = 6, 22, 50, N_FRAMES
    perim = [(x, y0) for x in range(x0, x1 + 1)] + \
            [(x1, y) for y in range(y0 + 1, y1 + 1)] + \
            [(x, y1) for x in range(x1 - 1, x0 - 1, -1)] + \
            [(x0, y) for y in range(y1 - 1, y0, -1)]

    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        if f >= T_HOLD0:
            # Step 1: trace the outline of the predicted size clockwise.
            t = ease((f - T_HOLD0) / float(T_OUTLINE - T_HOLD0))
            n = int(round(t * len(perim)))
            for (x, y) in perim[:n]:
                img[y, x] = col
        if f >= T_OUTLINE:
            # Step 2: fill grows from the centre until it meets the outline.
            t = ease((f - T_OUTLINE) / float(T_FILL - T_OUTLINE))
            half = int(round(t * (size / 2.0)))
            if half > 0:
                fx0 = max(x0, int(round(cx - half))); fx1 = min(x1, int(round(cx + half)) - 1)
                fy0 = max(y0, int(round(cy - half))); fy1 = min(y1, int(round(cy + half)) - 1)
                img[fy0:fy1 + 1, fx0:fx1 + 1] = col
        if f >= T_FILL:
            img[y0:y1 + 1, x0:x1 + 1] = col
        frames.append(img)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
