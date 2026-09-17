#!/usr/bin/env python3
"""Complete the checkerboard by mirroring the left half onto the right half.

Reads /app/first_frame.png, detects the 6x6 grid, finds which right-half cells
must be filled to mirror the left half, and animates each fill (square growing
from the cell centre) over ~35 frames at 16 fps.
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
N_FRAMES = 35

GREY = np.array([203, 213, 225])
WHITE = np.array([255, 255, 255])
FILL = np.array([5, 150, 105])


def detect_grid(im):
    """Return sorted list of (start, end) pixel ranges of cell interiors along
    x and y, derived from the grey grid lines."""
    def runs(mask):
        idx = np.where(mask)[0]
        groups, cur = [], [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                groups.append((cur[0], cur[-1]))
                cur = [i]
        groups.append((cur[0], cur[-1]))
        return groups

    # choose a row/col that crosses cells (not a grid line): use grey mask
    grey = (np.abs(im.astype(int) - GREY).sum(2) == 0)
    # rows: a column near the middle of a cell
    ys, xs = np.where(grey)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    col_probe = grey[:, (x0 + x1) // 2 + 50]  # probably inside a cell column
    row_probe = grey[(y0 + y1) // 2 + 50, :]
    vlines = runs(row_probe)
    hlines = runs(col_probe)
    xcells = [(vlines[i][1] + 1, vlines[i + 1][0] - 1) for i in range(len(vlines) - 1)]
    ycells = [(hlines[i][1] + 1, hlines[i + 1][0] - 1) for i in range(len(hlines) - 1)]
    return xcells, ycells


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    xcells, ycells = detect_grid(base)
    n = len(xcells)
    assert n == len(ycells) == 6, (xcells, ycells)

    def is_filled(r, c):
        (xa, xb), (ya, yb) = xcells[c], ycells[r]
        px = base[(ya + yb) // 2, (xa + xb) // 2]
        return tuple(px) == tuple(FILL)

    # cells on the right half that must be filled to mirror the left half
    targets = []
    for r in range(n):
        for c in range(n // 2, n):
            mc = n - 1 - c
            if is_filled(r, mc) and not is_filled(r, c):
                targets.append((r, c))

    # schedule: each cell grows over `dur` frames, staggered so the last one
    # finishes exactly on the final frame; first frame is untouched.
    k = len(targets)
    dur = 10
    span = N_FRAMES - 1 - dur          # frames available for start offsets
    starts = [1 + round(i * span / max(k - 1, 1)) for i in range(k)]

    white = (np.abs(base.astype(int) - WHITE).sum(2) == 0)
    frames = []
    for f in range(N_FRAMES):
        im = base.copy()
        for (r, c), s in zip(targets, starts):
            t = (f - s + 1) / dur
            if t <= 0:
                continue
            t = min(t, 1.0)
            t = t * t * (3 - 2 * t)  # smoothstep
            (xa, xb), (ya, yb) = xcells[c], ycells[r]
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            hw, hh = (xb - xa + 1) / 2 * t, (yb - ya + 1) / 2 * t
            if t >= 1.0:
                sx, ex, sy, ey = xa, xb + 1, ya, yb + 1
            else:
                sx, ex = int(round(cx - hw)), int(round(cx + hw))
                sy, ey = int(round(cy - hh)), int(round(cy + hh))
                sx, sy = max(sx, xa), max(sy, ya)
                ex, ey = min(ex, xb + 1), min(ey, yb + 1)
            if ex <= sx or ey <= sy:
                continue
            region = im[sy:ey, sx:ex]
            m = white[sy:ey, sx:ex]
            region[m] = FILL
        frames.append(im)

    assert np.array_equal(frames[0], base)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-preset", "slow", "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print(f"wrote {OUT}: {len(frames)} frames, targets={targets}")


if __name__ == "__main__":
    main()
