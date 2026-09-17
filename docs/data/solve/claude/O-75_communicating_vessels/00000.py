#!/usr/bin/env python3
"""Communicating vessels (3 tubes) settling simulation.

Redraws only the liquid columns inside the three tubes of first_frame.png;
every other pixel is copied unchanged from the first frame.
"""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 53
DURATION = N_FRAMES / FPS
K = 2.58  # viscous damping coefficient

# Geometry measured from the first frame
TUBE_COLS = [(223, 312), (457, 546), (691, 780)]  # inclusive x ranges of tube interiors
Y_BOTTOM = 969        # last liquid row (inclusive)
Y_TOP_INTERIOR = 229  # first interior row of the tubes
Y_ZERO = 878.5        # y pixel of 0 cm
PX_PER_CM = 10.0
YELLOW = np.array([255, 227, 75], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)


def surface_row(h_cm):
    """First yellow row for a liquid height in cm."""
    return int(np.floor(Y_ZERO - PX_PER_CM * h_cm + 0.5))


def measure_initial_heights(img):
    yel = (img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] < 120)
    hs = []
    for x0, x1 in TUBE_COLS:
        c = (x0 + x1) // 2
        top = np.where(yel[:, c])[0].min()
        hs.append((Y_ZERO + 0.5 - top) / PX_PER_CM)
    return np.array(hs)


def simulate(h0, times):
    """Chain of tubes joined at the bottom. Flow between neighbours is driven
    by hydrostatic head difference and damped by viscous resistance:
        dh_i/dt = K * sum_j (h_j - h_i)   for neighbouring j.
    Integrated with small explicit steps; volume is conserved exactly since
    the coupling is antisymmetric."""
    h = h0.astype(float).copy()
    out = []
    t = 0.0
    dt = 1e-3
    for tt in times:
        while t < tt - 1e-12:
            step = min(dt, tt - t)
            dh = np.zeros(3)
            for i, j in ((0, 1), (1, 2)):
                q = K * (h[j] - h[i])
                dh[i] += q
                dh[j] -= q
            h += step * dh
            t += step
        out.append(h.copy())
    return np.array(out)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    h0 = measure_initial_heights(base)
    h_final = h0.mean()

    times = np.arange(N_FRAMES) / FPS
    hs = simulate(h0, times)
    hs[0] = h0
    hs[-1] = h_final  # final stable equilibrium: all tubes at the average

    frames = []
    for h in hs:
        f = base.copy()
        for (x0, x1), hi in zip(TUBE_COLS, h):
            top = surface_row(hi)
            top = max(Y_TOP_INTERIOR, min(top, Y_BOTTOM + 1))
            f[Y_TOP_INTERIOR:top, x0:x1 + 1] = WHITE
            f[top:Y_BOTTOM + 1, x0:x1 + 1] = YELLOW
        frames.append(f)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print("initial heights (cm):", np.round(h0, 2), "final:", round(h_final, 3))
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
