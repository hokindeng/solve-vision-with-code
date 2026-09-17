#!/usr/bin/env python3
"""Draw a red perpendicular line through the midpoint dot between two parallel lines."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS = 50, 16
RED = np.array([255, 0, 0], dtype=np.uint8)


def find_geometry(im):
    black = np.all(im == 0, axis=2)
    rows = np.where(black.sum(axis=1) > im.shape[1] // 2)[0]
    # split rows into two groups (upper / lower line)
    gaps = np.where(np.diff(rows) > 1)[0]
    upper, lower = rows[: gaps[0] + 1], rows[gaps[0] + 1 :]
    y_top, y_bot = upper.mean(), lower.mean()
    y_mid = (y_top + y_bot) / 2.0
    thickness = len(upper)
    # find dot whose center is at the vertical midpoint
    white = np.all(im == 255, axis=2)
    other = ~white & ~black
    best = None
    for c in {tuple(p) for p in im[other]}:
        mask = np.all(im == np.array(c, dtype=np.uint8), axis=2)
        ys, xs = np.where(mask)
        # a dot is a compact blob; split by connected columns is overkill: use median
        cy, cx = np.median(ys), np.median(xs)
        d = abs(cy - y_mid)
        if best is None or d < best[0]:
            best = (d, cx, cy)
    return y_top, y_bot, best[1], thickness


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    y_top, y_bot, cx, thick = find_geometry(base)
    x0 = int(round(cx - thick / 2.0))
    x1 = x0 + thick  # exclusive
    y_start, y_end = int(round(y_top)), int(round(y_bot)) + 1  # center-to-center

    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in-out
        frame = base.copy()
        y_cur = y_start + int(round(s * (y_end - y_start)))
        if y_cur > y_start:
            frame[y_start:y_cur, x0:x1] = RED
        Image.fromarray(frame).save(os.path.join(tmp, f"f{i:04d}.png"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", os.path.join(tmp, "f%04d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
        check=True,
    )
    print("wrote", OUT, "line x", x0, x1 - 1, "y", y_start, y_end - 1)


if __name__ == "__main__":
    main()
