#!/usr/bin/env python3
"""Draw a green rectangular highlight around the kitchen in the floorplan.

The kitchen is the bottom-middle room (gray floor, counter run with cooktop/sink,
and a second counter). The box is traced progressively along the room's walls
over 28 frames; frame 0 is the untouched first frame.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 28

# Kitchen room bounds (centre lines of the surrounding walls).
X0, Y0, X1, Y1 = 305, 570, 564, 963
THICK = 7
GREEN = (0, 200, 0)


def perimeter_points(x0, y0, x1, y1):
    """Clockwise corners starting top-left."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def draw_partial_box(img, frac):
    """Draw the first `frac` (0..1) of the rectangle perimeter, clockwise."""
    if frac <= 0:
        return img
    pts = perimeter_points(X0, Y0, X1, Y1)
    seg_len = [abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1])
               for i in range(4)]
    total = sum(seg_len)
    remaining = frac * total
    d = ImageDraw.Draw(img)
    half = THICK // 2
    for i in range(4):
        if remaining <= 0:
            break
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        L = seg_len[i]
        t = min(1.0, remaining / L)
        ex = ax + (bx - ax) * t
        ey = ay + (by - ay) * t
        # Draw as a filled rectangle so corners are square and joins are clean.
        xa, xb = sorted([ax, ex])
        ya, yb = sorted([ay, ey])
        d.rectangle([xa - half, ya - half, xb + half, yb + half], fill=GREEN)
        remaining -= L
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    for i in range(N_FRAMES):
        img = base.copy()
        if i > 0:
            # Ease-out so the trace starts quickly and settles; completes at last frame.
            u = i / (N_FRAMES - 1)
            frac = 1 - (1 - u) ** 2
            draw_partial_box(img, frac)
        img.save(os.path.join(frames_dir, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-vf", "scale=1024:1024", OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
