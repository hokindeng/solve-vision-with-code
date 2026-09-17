#!/usr/bin/env python3
"""Communicating vessels (4 tubes, honey) settling animation.

Only the liquid columns inside the four tubes are redrawn; every other pixel
is taken verbatim from first_frame.png.
"""
import subprocess
import numpy as np
from PIL import Image

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 58

# Geometry measured from the first frame (inner tube x-ranges, inclusive)
TUBES = [(106, 195), (340, 429), (574, 663), (808, 897)]
Y_TOP = 230          # top of tube interior
Y_BOTTOM = 880       # first row of the always-full bottom reservoir
PX_PER_CM = 10.0
Y_ZERO = 878.0       # y pixel for level 0 cm
ORANGE = np.array([255, 191, 75], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)

K_VISC = 1.87        # viscous resistance coefficient
G = 9.8
RATE_SCALE = 0.3      # converts (g/k) to 1/s in animation time so the settling spans the clip


def measure_initial_levels(img):
    """Return initial surface y (float) for each tube from the orange pixels."""
    ys = []
    for x0, x1 in TUBES:
        col = img[Y_TOP:Y_BOTTOM, (x0 + x1) // 2]
        orange = (np.abs(col.astype(int) - ORANGE.astype(int)) < 4).all(1)
        ys.append(float(np.argmax(orange) + Y_TOP))
    return np.array(ys)


def simulate(h0, duration, n_frames):
    """Heights (cm) per frame. Adjacent tubes exchange fluid through the bottom
    channels; flow rate is driven by hydrostatic pressure difference and damped
    by viscosity: dh_i/dt = (g/k) * sum_j (h_j - h_i) over neighbours j."""
    rate = RATE_SCALE * G / K_VISC
    h = h0.astype(float).copy()
    dt_frame = duration / (n_frames - 1)
    sub = 50
    dt = dt_frame / sub
    frames = [h.copy()]
    for _ in range(n_frames - 1):
        for _ in range(sub):
            dh = np.zeros_like(h)
            for i in range(len(h) - 1):
                q = rate * (h[i] - h[i + 1])  # flow from i to i+1
                dh[i] -= q
                dh[i + 1] += q
            h += dh * dt
        frames.append(h.copy())
    return frames


def draw(base, heights_cm, snap_equal=False):
    img = base.copy()
    ys = Y_ZERO - np.array(heights_cm) * PX_PER_CM
    if snap_equal:
        ys[:] = round(float(ys.mean()))
    for (x0, x1), y in zip(TUBES, ys):
        yi = int(round(y))
        yi = max(Y_TOP, min(Y_BOTTOM, yi))
        img[Y_TOP:yi, x0:x1 + 1] = WHITE
        img[yi:Y_BOTTOM, x0:x1 + 1] = ORANGE
    return img


def main():
    base = np.array(Image.open(BASE).convert("RGB"))
    y0 = measure_initial_levels(base)
    h0 = (Y_ZERO - y0) / PX_PER_CM
    duration = (N_FRAMES - 1) / FPS
    hs = simulate(h0, duration, N_FRAMES)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, h in enumerate(hs):
        frame = base if i == 0 else draw(base, h, snap_equal=(i == N_FRAMES - 1))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print("initial levels (cm):", np.round(h0, 2), "final:", np.round(hs[-1], 2),
          "mean:", round(float(h0.mean()), 2))


if __name__ == "__main__":
    main()
