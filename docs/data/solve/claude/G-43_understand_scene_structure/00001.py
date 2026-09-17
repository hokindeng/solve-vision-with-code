#!/usr/bin/env python3
"""Draw a green rectangular highlight around the living room of the floorplan.

The living room is the middle-left room (sofa, coffee table, rug, TV, shelf).
The box is traced progressively along the room's walls over the 28 frames so the
first frame is untouched and the last frame shows the complete rectangle.
"""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 28
GREEN = (0, 200, 0)
THICK = 8

# Living-room wall centre lines (measured from first_frame.png).
X0, Y0, X1, Y1 = 58, 329, 514, 634


def perimeter_points(t):
    """Return polyline points covering fraction t (0..1) of the rectangle
    perimeter, starting top-left and going clockwise."""
    corners = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1), (X0, Y0)]
    lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1])
               for a, b in zip(corners, corners[1:])]
    total = sum(lengths)
    remain = t * total
    pts = [corners[0]]
    for a, b, L in zip(corners, corners[1:], lengths):
        if remain <= 0:
            break
        if remain >= L:
            pts.append(b)
            remain -= L
        else:
            f = remain / L
            pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            remain = 0
    return pts


def render_frame(base, t):
    img = base.copy()
    if t <= 0:
        return img
    d = ImageDraw.Draw(img)
    if t >= 1:
        d.rectangle([X0 - THICK // 2, Y0 - THICK // 2,
                     X1 + THICK // 2, Y1 + THICK // 2], outline=GREEN, width=THICK)
        return img
    pts = perimeter_points(t)
    d.line(pts, fill=GREEN, width=THICK, joint="curve")
    # square-ish caps at the start and the moving tip
    for (x, y) in (pts[0], pts[-1]):
        d.rectangle([x - THICK // 2, y - THICK // 2,
                     x + THICK // 2 - 1, y + THICK // 2 - 1], fill=GREEN)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(N_FRAMES):
        # ease-in-out over frames 1..N-1; frame 0 is the untouched original
        if i == 0:
            t = 0.0
        else:
            u = i / (N_FRAMES - 1)
            t = 0.5 - 0.5 * np.cos(np.pi * u)
        render_frame(base, t).save(os.path.join(frames_dir, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
