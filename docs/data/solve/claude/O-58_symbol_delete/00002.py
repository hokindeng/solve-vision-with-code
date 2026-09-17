#!/usr/bin/env python3
"""Delete the red triangle at position 5.

Phase 1: the triangle fades smoothly to the background colour.
Phase 2: the remaining symbols close the gap. The deleted symbol is the last
one, so no symbol has to move; the emptied slot (box + index label) fades away
to yield the 4-position post-deletion arrangement.
All other pixels are taken verbatim from first_frame.png.
"""
import os, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 45

base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
bg = base[0, 0].copy()

# Geometry measured from first_frame.png (slot 5).
BOX = (674, 464, 770, 560)          # x0,y0,x1,y1 inclusive of the grey frame
SYM = (BOX[0] + 1, BOX[1] + 1, BOX[2] - 1, BOX[3] - 1)  # interior only
LABEL = (700, 575, 745, 610)         # digit "5" below the slot


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fade(img, rect, a):
    x0, y0, x1, y1 = rect
    region = img[y0:y1 + 1, x0:x1 + 1]
    img[y0:y1 + 1, x0:x1 + 1] = region * (1 - a) + bg * a


def make_frame(i):
    img = base.copy()
    # Phase 1: frames 0..24 -> triangle fades out.
    a1 = ease(i / 24.0)
    fade(img, SYM, a1)
    # Phase 2: frames 27..44 -> empty slot 5 (box + label) fades out.
    a2 = ease((i - 27) / 17.0)
    if a2 > 0:
        fade(img, BOX, a2)
        fade(img, LABEL, a2)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(N):
        Image.fromarray(make_frame(i)).save(os.path.join(frames_dir, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-x264-params", "keyint=1", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
