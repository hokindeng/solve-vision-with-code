#!/usr/bin/env python3
"""Move each colored object along a straight line to the star marker of the same color."""
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 48

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
bg_color = np.array([255, 255, 255], dtype=np.uint8)

# Detect colored components; classify stars (low bbox fill ratio) vs solid objects.
colors = [tuple(c) for c in np.unique(img.reshape(-1, 3), axis=0) if tuple(c) != tuple(bg_color)]
moves = []  # (mask, ys, xs, color, start_center, end_center)
background = img.copy()
for c in colors:
    m = np.all(img == np.array(c, dtype=np.uint8), axis=2)
    lab, n = ndimage.label(m)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        fill = len(xs) / (bw * bh)
        center = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
        comps.append(dict(ys=ys, xs=xs, fill=fill, center=center))
    stars = [k for k in comps if k["fill"] < 0.3]
    objs = [k for k in comps if k["fill"] >= 0.3]
    for o in objs:
        star = min(stars, key=lambda s: (s["center"][0] - o["center"][0]) ** 2 + (s["center"][1] - o["center"][1]) ** 2)
        moves.append((o["ys"], o["xs"], c, o["center"], star["center"]))
        background[o["ys"], o["xs"]] = bg_color  # remove object from static background

def render(t):
    frame = background.copy()
    for ys, xs, c, (sx, sy), (ex, ey) in moves:
        dx = int(round((ex - sx) * t))
        dy = int(round((ey - sy) * t))
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        frame[ny[ok], nx[ok]] = np.array(c, dtype=np.uint8)
    return frame

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)  # linear pacing, arrives exactly on the last frame
    p.stdin.write(render(t).tobytes())
p.stdin.close()
p.wait()
print("wrote", OUT)
