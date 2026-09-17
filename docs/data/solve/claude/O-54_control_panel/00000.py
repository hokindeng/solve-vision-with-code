#!/usr/bin/env python3
"""Animate control unit 1: slide its lever from middle to right and turn its light pink."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N = 24

BLACK = np.array([0, 0, 0], np.uint8)
GRAY = np.array([128, 128, 128], np.uint8)
PURPLE = np.array([128, 0, 128], np.uint8)
PINK = np.array([255, 192, 203], np.uint8)

# Unit 1 lever geometry (measured from first_frame.png)
KNOB_Y0, KNOB_Y1 = 648, 705          # knob rows [y0, y1)
KNOB_W = 57
KNOB_X_START = 176                   # middle position, left edge
KNOB_X_END = 253                     # right position, left edge (same panel offset as units 2/3)
DOT_Y0, DOT_Y1 = 674, 679
DOTS_X = [(120, 125), (202, 207), (284, 289)]  # left, middle, right dot columns [x0, x1)
PANEL_INNER = (91, 622, 319, 732)    # x0, y0, x1, y1 of black interior


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def make_frame(base, purple_mask, i):
    f = base.copy()
    # Lever: slides during frames 2..16
    s = ease((i - 2) / 14.0)
    kx = int(round(KNOB_X_START + (KNOB_X_END - KNOB_X_START) * s))
    x0, y0, x1, y1 = PANEL_INNER
    f[y0:y1, x0:x1] = BLACK
    stamp = base[DOT_Y0:DOT_Y1, DOTS_X[0][0]:DOTS_X[0][1]]  # exact dot sprite from source
    for dx0, dx1 in DOTS_X:
        f[DOT_Y0:DOT_Y1, dx0:dx1] = stamp
    f[KNOB_Y0:KNOB_Y1, kx:kx + KNOB_W] = GRAY
    # Light: fades purple -> pink during frames 10..20
    c = ease((i - 10) / 10.0)
    col = np.round(PURPLE * (1 - c) + PINK * c).astype(np.uint8)
    f[purple_mask] = col
    return f


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    purple_mask = np.all(base == PURPLE, axis=2)
    frames = [make_frame(base, purple_mask, i) for i in range(N)]
    assert np.array_equal(frames[0], base), "first frame must match source"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print("wrote", OUT)


if __name__ == "__main__":
    main()
