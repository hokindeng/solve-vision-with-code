#!/usr/bin/env python3
"""Communicating vessels (3 tubes, viscous syrup) -> /app/output/video.mp4"""
import numpy as np
from PIL import Image
import subprocess, os

BASE = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 125
W = H = 1024

base = np.array(Image.open(BASE).convert("RGB"))
MAROON = np.array([165, 75, 75], np.uint8)
WHITE = np.array([255, 255, 255], np.uint8)

# tube interiors (x0, x1 inclusive), measured from first frame
TUBES = [(223, 312), (457, 546), (691, 780)]
Y_TOP = 232          # top of open tubes (below cap), region we may redraw
Y_CHAN = 879         # top of connecting channel (always liquid below)
PX_PER_CM = 10.0
Y_ZERO = 874.0       # y of 0 cm level (from tubes 1 & 3)
h0 = np.array([58.0, 32.0, 14.0])
y_init = np.array([294, 559, 734])            # measured initial surfaces
off = y_init - (Y_ZERO - PX_PER_CM * h0)      # per-tube render offset (0,5,0)
k = 0.82

# ---- physics: overdamped hydrostatic equalisation through bottom channels
# dh_i/dt = (c/k) * sum_j (h_j - h_i) over neighbours (chain 1-2-3)
T = (N - 1) / FPS
c = 4.8 / T * k                               # slowest mode ~0.8% residual at end
L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], float)
dt = 1.0 / FPS / 20
h = h0.copy()
levels = [h.copy()]
for f in range(1, N):
    for _ in range(20):
        h = h - dt * (c / k) * L @ h          # volume conserved (rows of L sum to 0)
    levels.append(h.copy())
levels = np.array(levels)
levels[-1] = h0.mean()                        # final: exact equilibrium

frames = []
for f in range(N):
    img = base.copy()
    if f > 0:
        s = f / (N - 1)
        w = 1 - (1 - s) ** 2                  # smooth fade of initial offset
        for i, (x0, x1) in enumerate(TUBES):
            y = Y_ZERO - PX_PER_CM * levels[f, i] + off[i] * (1 - w)
            y = int(round(y))
            img[Y_TOP:y, x0:x1 + 1] = WHITE
            img[y:Y_CHAN, x0:x1 + 1] = MAROON
    frames.append(img)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "6", "-preset", "slow", "-r", str(FPS), OUT], stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
print("levels frame0", levels[0], "mid", levels[N // 2].round(2), "final", levels[-1])
