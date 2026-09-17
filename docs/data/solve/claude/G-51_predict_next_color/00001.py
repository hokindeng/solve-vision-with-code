#!/usr/bin/env python3
"""Predict the next color in the sequence (blue, blue, red, red, ?) -> blue.

The empty placeholder triangle is filled in, bottom-to-top, with an exact copy of
the first (blue) triangle. Every pixel outside the placeholder stays untouched.
"""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 64

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape
WHITE = np.array([255, 255, 255], np.uint8)

# --- locate elements -------------------------------------------------------
nonwhite = (first != 255).any(2)
cols = np.where(nonwhite.any(0))[0]
# split into connected column runs -> one per triangle
runs, start = [], cols[0]
for a, b in zip(cols[:-1], cols[1:]):
    if b != a + 1:
        runs.append((start, a)); start = b
runs.append((start, cols[-1]))
assert len(runs) == 5, runs

def region(run):
    x0, x1 = run
    sub = nonwhite[:, x0:x1 + 1]
    ys = np.where(sub.any(1))[0]
    return x0, x1, ys.min(), ys.max()

regs = [region(r) for r in runs]
# sequence colours (most common non-outline colour of each filled triangle)
def main_color(reg):
    x0, x1, y0, y1 = reg
    px = first[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
    px = px[(px != 255).any(1)]
    vals, cnt = np.unique(px, axis=0, return_counts=True)
    return tuple(vals[cnt.argmax()])

colors = [main_color(r) for r in regs[:4]]
# pattern A A B B -> next is A
predicted = colors[0] if colors[0] == colors[1] and colors[2] == colors[3] else colors[-1]
src_idx = colors.index(predicted)

# --- build final frame ------------------------------------------------------
sx0, sx1, sy0, sy1 = regs[src_idx]
tx0, tx1, ty0, ty1 = regs[4]
# centre-align the source triangle on the placeholder
scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
dx, dy = int(round(tcx - scx)), int(round(tcy - scy))

final = first.copy()
# clear the gray placeholder outline
tmask = np.zeros((H, W), bool)
tmask[ty0:ty1 + 1, tx0:tx1 + 1] = nonwhite[ty0:ty1 + 1, tx0:tx1 + 1]
final[tmask] = WHITE
# paste the source triangle (non-white pixels only) at the shifted location
smask = np.zeros((H, W), bool)
smask[sy0:sy1 + 1, sx0:sx1 + 1] = nonwhite[sy0:sy1 + 1, sx0:sx1 + 1]
ys, xs = np.where(smask)
final[ys + dy, xs + dx] = first[ys, xs]

changed = tmask.copy()
changed[ys + dy, xs + dx] = True
cy0, cy1 = np.where(changed.any(1))[0][[0, -1]]

# --- animate: fill rises from the base to the apex ------------------------
def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

hold_start, hold_end = 4, 8
active = N_FRAMES - hold_start - hold_end
frames = []
for i in range(N_FRAMES):
    if i < hold_start:
        p = 0.0
    elif i >= N_FRAMES - hold_end:
        p = 1.0
    else:
        p = ease((i - hold_start + 1) / active)
    # fill line: rows >= line take the final image
    line = cy1 + 1 - int(round(p * (cy1 - cy0 + 1)))
    f = first.copy()
    rows = np.zeros(H, bool); rows[line:cy1 + 1] = True
    m = changed & rows[:, None]
    f[m] = final[m]
    frames.append(f)

assert np.array_equal(frames[0], first)
assert np.array_equal(frames[-1], final)

# --- encode -----------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
       "-tune", "stillimage", "-movflags", "+faststart", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    proc.stdin.write(np.ascontiguousarray(f).tobytes())
proc.stdin.close()
proc.wait()
assert proc.returncode == 0
print("wrote", OUT, "predicted colour", predicted, "shift", (dx, dy))
