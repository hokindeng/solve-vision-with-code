#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: move the levers of units 1 and 3 from the
middle to the left position, which (per unit 2) turns their lights red."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N = 16, 24

base = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = base.shape

# Geometry measured from first_frame.png
KNOB_Y0, KNOB_Y1 = 648, 705          # knob rows (exclusive end)
KNOB_W = 57
GRAY = np.array([128, 128, 128], np.uint8)
BLACK = np.array([0, 0, 0], np.uint8)
MAGENTA = np.array([255, 0, 255])
RED = np.array([255, 0, 0])
# Units needing change: (knob_left_x at middle, knob_left_x at left position,
# light bbox slice)
UNITS = [
    dict(x_mid=176, x_left=99, light=(np.s_[200:300, 155:255])),
    dict(x_mid=790, x_left=713, light=(np.s_[200:300, 770:870])),
]
# The middle dot (hidden under the knob) - copy its shape from unit 2.
DOT = base[672:681, 507:516].copy()  # 9x9 patch centered on dot at (676, 511)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def make_frame(i):
    f = base.copy()
    t = i / (N - 1)
    lever_t = ease(min(1.0, t / 0.75))           # lever moves in first 75%
    for u in UNITS:
        # 1. lever: clear old knob, restore hidden middle dot, draw knob
        f[KNOB_Y0:KNOB_Y1, u["x_mid"]:u["x_mid"] + KNOB_W] = BLACK
        cx = u["x_mid"] + KNOB_W // 2
        f[672:681, cx - 4:cx + 5] = DOT
        x = int(round(u["x_mid"] + (u["x_left"] - u["x_mid"]) * lever_t))
        f[KNOB_Y0:KNOB_Y1, x:x + KNOB_W] = GRAY
        # 2. light: fade magenta -> red as lever approaches position
        c_t = ease(np.clip((t - 0.35) / 0.4, 0, 1))
        col = np.round(MAGENTA + (RED - MAGENTA) * c_t).astype(np.uint8)
        reg = f[u["light"]]
        m = (reg[..., 0] == 255) & (reg[..., 1] == 0) & (reg[..., 2] == 255)
        reg[m] = col
    return f


frames = [make_frame(i) for i in range(N)]
assert np.array_equal(frames[0], base)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close()
p.wait()
Image.fromarray(frames[-1]).save("/app/output/last_frame.png")
print("wrote", OUT)
