#!/usr/bin/env python3
"""Animate a green rectangular highlight around the bathroom in the floorplan.

The bathroom is the bottom-left room (toilet, sink, bathtub). Its walls sit at
x 55..64 / 371..380 and y 697..707 / 959..968, so the box is drawn along the
wall centre-lines and traced progressively around the perimeter. Frame 0 is the
untouched first frame; the final frame shows the complete box.
"""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 28
FPS = 16
GREEN = (0, 200, 0)
THICK = 7

# Bathroom bounds (centre of surrounding walls) — bottom-left room.
X0, Y0, X1, Y1 = 60, 702, 375, 964


def perimeter_points(x0, y0, x1, y1):
    """Corners in drawing order starting top-left, clockwise."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def draw_partial_box(img, frac):
    """Draw the first `frac` (0..1) of the perimeter onto img."""
    if frac <= 0:
        return img
    pts = perimeter_points(X0, Y0, X1, Y1)
    seg_len = [abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1])
               for i in range(4)]
    total = sum(seg_len)
    remaining = frac * total
    d = ImageDraw.Draw(img)
    for i in range(4):
        if remaining <= 0:
            break
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        L = seg_len[i]
        t = min(1.0, remaining / L)
        ex, ey = ax + (bx - ax) * t, ay + (by - ay) * t
        h = THICK // 2
        # Draw as filled rectangle so the stroke has square, connected ends.
        xa, xb = sorted([ax, ex]); ya, yb = sorted([ay, ey])
        d.rectangle([xa - h, ya - h, xb + h, yb + h], fill=GREEN)
        remaining -= L
    return img


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    frames = []
    for i in range(N_FRAMES):
        # frame 0 -> untouched; last frame -> complete box
        frac = ease(i / (N_FRAMES - 1))
        frames.append(np.array(draw_partial_box(base.copy(), frac)))

    tmp = os.path.join(OUT_DIR, "frames_raw.rgb")
    with open(tmp, "wb") as f:
        for fr in frames:
            f.write(fr.astype(np.uint8).tobytes())
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", tmp,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        OUT,
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp)
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
