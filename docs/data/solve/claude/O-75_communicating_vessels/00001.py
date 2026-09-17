#!/usr/bin/env python3
"""Communicating vessels (5 tubes, honey) settling animation.

Only the liquid columns inside the five tube interiors are redrawn; every other
pixel is copied unchanged from first_frame.png.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

FPS = 16
N_FRAMES = 63
DURATION = (N_FRAMES - 1) / FPS

# Geometry measured from first_frame.png
TUBES = [(81, 170), (269, 358), (457, 546), (645, 734), (833, 922)]  # interior x ranges (inclusive)
Y_ZERO = 879          # pixel row of the 0 cm mark (bottom of tube interiors)
PX_PER_CM = 10.0
Y_TOP = 229           # top of tube interiors
ORANGE = np.array([255, 191, 75], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)

K_VISC = 1.66         # viscous resistance coefficient
G = 9.8


def measure_initial_levels(img):
    """Read the liquid heights (cm) directly from the first frame so frame 0 is exact."""
    levels = []
    for x0, x1 in TUBES:
        col = img[Y_TOP:Y_ZERO + 1, (x0 + x1) // 2]
        is_orange = (col == ORANGE).all(axis=1)
        top = Y_TOP + int(np.argmax(is_orange))
        levels.append((Y_ZERO - top) / PX_PER_CM)
    return np.array(levels)


def simulate(h0, times):
    """Overdamped hydrostatic equalization through the bottom channels.

    Adjacent tubes exchange liquid at a rate proportional to their pressure
    (height) difference and inversely proportional to the viscous resistance:
        dh_i/dt = (G / (K_VISC * L)) * sum_j (h_j - h_i)   over neighbours j.
    L is an effective channel length scale chosen so the slowest mode of the
    5-tube chain has decayed to <0.3 % over the clip duration.
    """
    n = len(h0)
    lap = np.zeros((n, n))
    for i in range(n - 1):
        lap[i, i] -= 1; lap[i + 1, i + 1] -= 1
        lap[i, i + 1] += 1; lap[i + 1, i] += 1
    rate = 1.8                       # = G / (K_VISC * L) with L ~ 3.3 m
    A = rate * lap
    w, V = np.linalg.eigh(A)         # symmetric -> exact solution via eigendecomposition
    mean = h0.mean()
    c = V.T @ (h0 - mean)
    T = times[-1]
    t_settle = 0.65 * T              # from here the last few percent settle smoothly to rest
    out = []
    for t in times:
        dev = V @ (c * np.exp(w * t))
        if t > t_settle:             # smoothstep taper so the final frame is exact equilibrium
            u = (t - t_settle) / (T - t_settle)
            dev = dev * (1.0 - (3 * u * u - 2 * u ** 3))
        out.append(mean + dev)
    return np.array(out)


def render(base, levels):
    frame = base.copy()
    for (x0, x1), h in zip(TUBES, levels):
        y = int(round(Y_ZERO - h * PX_PER_CM))
        y = max(Y_TOP, min(Y_ZERO, y))
        frame[Y_TOP:y, x0:x1 + 1] = WHITE
        frame[y:Y_ZERO, x0:x1 + 1] = ORANGE
    return frame


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    h0 = measure_initial_levels(base)
    times = np.arange(N_FRAMES) / FPS
    levels = simulate(h0, times)
    levels[-1] = np.full(len(h0), h0.mean())   # exact equilibrium on the last frame
    print("initial levels (cm):", h0, " final:", levels[-1])

    tmp = tempfile.mkdtemp()
    try:
        for i, lv in enumerate(levels):
            frame = base if i == 0 else render(base, lv)
            Image.fromarray(frame).save(os.path.join(tmp, f"f{i:04d}.png"))
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(tmp, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS), OUT,
        ], check=True)
    finally:
        shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
