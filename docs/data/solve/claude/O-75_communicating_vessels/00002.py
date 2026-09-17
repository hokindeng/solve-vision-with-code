#!/usr/bin/env python3
"""Communicating vessels: 4 equal-diameter tubes settling to a common level.

Only the liquid columns inside the four tube interiors are redrawn; every other
pixel is copied from first_frame.png.
"""
import os
import subprocess
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 53
K = 2.29                       # viscous damping coefficient (1/s)

YELLOW = np.array([255, 227, 75], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)

# Tube interiors (inclusive x ranges) measured from the first frame.
TUBES = [(106, 195), (340, 429), (574, 663), (808, 897)]
Y_TOP_INTERIOR = 230           # first interior row below the open top
Y_ZERO = 879.0                 # y of 0 cm (top of the connecting channel)
PX_PER_CM = 10.0


def measure_levels(img):
    """Read the initial liquid heights (cm) directly from the frame."""
    yel = np.all(img == YELLOW, axis=2)
    heights = []
    for x0, x1 in TUBES:
        col = yel[:, (x0 + x1) // 2]
        top = np.argmax(col)       # first yellow row
        heights.append((Y_ZERO - top) / PX_PER_CM)
    return np.array(heights, dtype=float)


def simulate(h0, t):
    """Hydrostatic pressure equalization damped by viscous resistance.

    For equal-diameter tubes joined at the bottom, the flow into each tube is
    proportional to the pressure (level) difference with the common mean, so
    dh_i/dt = -k (h_i - h_mean).  Volume is conserved (the mean is constant) and
    every tube relaxes exponentially to the average of the initial heights.
    """
    mean = h0.mean()
    return mean + (h0 - mean) * np.exp(-K * t)


def draw(base, heights):
    frame = base.copy()
    for (x0, x1), h in zip(TUBES, heights):
        top = int(round(Y_ZERO - h * PX_PER_CM))
        top = max(Y_TOP_INTERIOR, min(top, int(Y_ZERO)))
        frame[Y_TOP_INTERIOR:top, x0:x1 + 1] = WHITE
        frame[top:int(Y_ZERO) + 1, x0:x1 + 1] = YELLOW
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    h0 = measure_levels(base)

    frames = []
    for i in range(N_FRAMES):
        t = i / FPS
        h = h0 if i == 0 else simulate(h0, t)
        if i == N_FRAMES - 1:
            h = np.full_like(h0, h0.mean())   # exact final equilibrium
        frames.append(draw(base, h))

    # First frame must be identical to the reference image.
    assert np.array_equal(frames[0], base), "first frame mismatch"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"initial levels (cm): {h0.tolist()}  final: {h0.mean():.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
