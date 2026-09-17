#!/usr/bin/env python3
"""Animate the 4 teal blocks moving 2 grid cells to the left (one cell per step)."""
import os, subprocess, shutil
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 35
STEPS, CELL = 2, 128
TEAL, BLACK = (0, 128, 128), (0, 0, 0)

def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)

def main():
    first = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = first.shape

    # find blocks: teal fill + surrounding black outline
    teal = (first == TEAL).all(-1)
    black = (first == BLACK).all(-1)
    lab, n = ndimage.label(teal | black)
    blocks = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        blocks.append((x0, y0, first[y0:y1, x0:x1].copy()))

    # background: erase blocks (they sit strictly inside cells, so white fill is exact)
    bg = first.copy()
    for x0, y0, patch in blocks:
        bg[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = 255

    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        # piecewise: each step gets an equal share of the time, eased within the step
        s = t * STEPS
        k = min(int(s), STEPS - 1)
        prog = k + ease(s - k)
        dx = int(round(-prog * CELL))
        frame = bg.copy()
        for x0, y0, patch in blocks:
            ph, pw = patch.shape[:2]
            nx = x0 + dx
            assert 0 <= nx and nx + pw <= w
            frame[y0:y0 + ph, nx:nx + pw] = patch
        if f == 0:
            assert (frame == first).all()
        Image.fromarray(frame).save(os.path.join(tmp, f"{f:04d}.png"))

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT], check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
