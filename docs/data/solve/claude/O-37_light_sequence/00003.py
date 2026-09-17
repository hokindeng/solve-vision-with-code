#!/usr/bin/env python3
"""Generate the light-sequence video: leftmost 3 lights on, all others off."""
import subprocess, os
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
FPS, N_FRAMES, N_LIGHTS = 16, 35, 10
TARGET_ON = 3

base = np.array(Image.open(FIRST).convert("RGB")).astype(np.float32)
H, W, _ = base.shape

# --- locate lights: non-white column runs -----------------------------------
mask = (base < 250).any(axis=2)
cols = mask.any(axis=0)
runs, s = [], None
for i, v in enumerate(cols):
    if v and s is None:
        s = i
    if not v and s is not None:
        runs.append((s, i - 1)); s = None
assert len(runs) == N_LIGHTS, runs

lights = []
for a, b in runs:
    ys = np.where(mask[:, a:b + 1].any(axis=1))[0]
    cx, cy = (a + b) / 2.0, (ys.min() + ys.max()) / 2.0
    c = base[int(round(cy)), int(round(cx))]
    on = c[0] > 180 and c[1] < 120  # red core
    lights.append(dict(x0=a, x1=b, y0=int(ys.min()), y1=int(ys.max()), cx=cx, cy=cy, on=on))

# --- pixel-exact templates for an "on" and an "off" light ---------------------
def patch(l):
    return base[l["y0"]:l["y1"] + 1, l["x0"]:l["x1"] + 1].copy()

on_tpl = patch(next(l for l in lights if l["on"]))
off_tpl = patch(next(l for l in lights if not l["on"]))

def render(tpl, cx, cy):
    """Return (y0, x0, image) of template placed with centre at (cx, cy) on white."""
    h, w = tpl.shape[:2]
    x0 = int(np.floor(cx - (w - 1) / 2.0 + 0.5))
    y0 = int(np.floor(cy - (h - 1) / 2.0 + 0.5))
    return y0, x0, tpl

# --- schedule ----------------------------------------------------------------
changes = [i for i, l in enumerate(lights) if l["on"] != (i < TARGET_ON)]
DUR = 8  # frames per transition
if changes:
    span = (N_FRAMES - 2 - DUR) - 1
    starts = {i: 1 + round(k * span / max(1, len(changes) - 1)) for k, i in enumerate(changes)}

def smooth(t):
    return t * t * (3 - 2 * t)

frames = []
for f in range(N_FRAMES):
    img = base.copy()
    for i in changes:
        l = lights[i]
        a = smooth(float(np.clip((f - starts[i]) / DUR, 0.0, 1.0)))
        if a <= 0:
            continue
        cur_tpl, new_tpl = (on_tpl, off_tpl) if l["on"] else (off_tpl, on_tpl)
        # union region covering both renderings
        ry0, rx0, _ = render(cur_tpl, l["cx"], l["cy"])
        ny0, nx0, _ = render(new_tpl, l["cx"], l["cy"])
        Y0 = min(ry0, ny0); X0 = min(rx0, nx0)
        Y1 = max(ry0 + cur_tpl.shape[0], ny0 + new_tpl.shape[0])
        X1 = max(rx0 + cur_tpl.shape[1], nx0 + new_tpl.shape[1])
        cur = np.full((Y1 - Y0, X1 - X0, 3), 255.0, np.float32)
        new = cur.copy()
        cur[ry0 - Y0:ry0 - Y0 + cur_tpl.shape[0], rx0 - X0:rx0 - X0 + cur_tpl.shape[1]] = cur_tpl
        new[ny0 - Y0:ny0 - Y0 + new_tpl.shape[0], nx0 - X0:nx0 - X0 + new_tpl.shape[1]] = new_tpl
        img[Y0:Y1, X0:X1] = cur * (1 - a) + new * a
    frames.append(np.clip(img + 0.5, 0, 255).astype(np.uint8))

# --- encode --------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
assert p.returncode == 0
print("wrote", OUT, len(frames), "frames")
