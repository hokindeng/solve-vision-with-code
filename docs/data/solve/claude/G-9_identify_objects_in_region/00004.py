#!/usr/bin/env python3
"""Outline the square inside the square (green-bordered) region.

Frame 0 is first_frame.png unchanged. Over the remaining frames a green
outline is traced clockwise around the only square inside the left region
(the dark blue square). Nothing else is modified.
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
N_FRAMES = 40
GREEN = np.array([0, 200, 0], dtype=np.uint8)
GAP = 3        # white gap between the square's edge and the outline
THICK = 4      # outline thickness in pixels


def find_square(img):
    """Return (x0, y0, x1, y1) inclusive bbox of the dark blue square."""
    a = img.astype(int)
    fill = (abs(a[..., 0] - 53) < 25) & (abs(a[..., 1] - 86) < 25) & (abs(a[..., 2] - 153) < 25)
    ys, xs = np.where(fill)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # grow to include the thin dark edge drawn around the fill
    nonwhite = a.sum(axis=2) < 700
    while x0 > 0 and nonwhite[y0:y1 + 1, x0 - 1].any():
        x0 -= 1
    while x1 < img.shape[1] - 1 and nonwhite[y0:y1 + 1, x1 + 1].any():
        x1 += 1
    while y0 > 0 and nonwhite[y0 - 1, x0:x1 + 1].any():
        y0 -= 1
    while y1 < img.shape[0] - 1 and nonwhite[y1 + 1, x0:x1 + 1].any():
        y1 += 1
    return x0, y0, x1, y1


def outline_pixels(x0, y0, x1, y1):
    """Ordered list of pixel sets tracing the outline ring clockwise.

    The ring is THICK pixels wide, GAP pixels outside the bbox. Returned as a
    list of (rows, cols) index arrays, one per step along the perimeter.
    """
    ox0, oy0 = x0 - GAP - THICK, y0 - GAP - THICK   # outer corner
    ox1, oy1 = x1 + GAP + THICK, y1 + GAP + THICK
    steps = []
    # top edge, left -> right
    for x in range(ox0, ox1 + 1):
        steps.append((np.arange(oy0, oy0 + THICK), np.full(THICK, x)))
    # right edge, top -> bottom
    for y in range(oy0 + THICK, oy1 + 1):
        steps.append((np.full(THICK, y), np.arange(ox1 - THICK + 1, ox1 + 1)))
    # bottom edge, right -> left
    for x in range(ox1 - THICK, ox0 - 1, -1):
        steps.append((np.arange(oy1 - THICK + 1, oy1 + 1), np.full(THICK, x)))
    # left edge, bottom -> top
    for y in range(oy1 - THICK, oy0 + THICK - 1, -1):
        steps.append((np.full(THICK, y), np.arange(ox0, ox0 + THICK)))
    return steps


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    h, w = base.shape[:2]
    bbox = find_square(base)
    steps = outline_pixels(*bbox)
    total = len(steps)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        frame = base.copy()
        if i > 0:
            # ease-in-out progress; final frame fully drawn
            t = i / (N_FRAMES - 1)
            t = 0.5 - 0.5 * np.cos(np.pi * t)
            n = int(round(t * total))
            for rows, cols in steps[:n]:
                frame[rows, cols] = GREEN
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: square bbox={bbox}, {N_FRAMES} frames @ {FPS} fps")


if __name__ == "__main__":
    main()
