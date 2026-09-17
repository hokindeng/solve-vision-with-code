#!/usr/bin/env python3
"""Move the green-bordered decagon horizontally so it sits directly below the red star."""
import os, subprocess
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 60

def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    a = base.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]

    # Locate the green border -> moving object bounding box (with margin for antialiasing).
    green = (g > 150) & (r < 100) & (b < 100)
    ys, xs = np.where(green)
    m = 3
    x0, x1 = xs.min() - m, xs.max() + m + 1
    y0, y1 = ys.min() - m, ys.max() + m + 1
    obj_cx = (xs.min() + xs.max()) / 2.0

    # Locate the red star -> target x.
    red = (r > 200) & (g < 60) & (b < 60)
    sy, sx = np.where(red)
    star_cx = (sx.min() + sx.max()) / 2.0

    dx_total = int(round(star_cx - obj_cx))

    patch = base[y0:y1, x0:x1].copy()
    bg_color = base[0, 0]
    background = base.copy()
    background[y0:y1, x0:x1] = bg_color  # region around the object is plain background

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in/out
        dx = int(round(dx_total * s))
        frame = background.copy()
        frame[y0:y1, x0 + dx:x1 + dx] = patch
        Image.fromarray(frame).save(os.path.join(frames_dir, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    print(f"wrote {OUT}: shift dx={dx_total}, {N_FRAMES} frames @ {FPS} fps")

if __name__ == "__main__":
    main()
