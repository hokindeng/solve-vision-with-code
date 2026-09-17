#!/usr/bin/env python3
"""Solve the 15x15 maze in first_frame.png and render the walk as a video."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N = 15
FPS = 16
FRAMES = 71
SIZE = 1024
CELL = SIZE / N

base = Image.open(FIRST).convert("RGB")
arr = np.array(base).astype(int)


def center(r, c):
    return ((c + 0.5) * CELL, (r + 0.5) * CELL)


def is_open(r, c):
    # A cell is a pathway if its centre neighbourhood is not wall-dark.
    x, y = center(r, c)
    x, y = int(x), int(y)
    patch = arr[y - 3:y + 4, x - 3:x + 4]
    dark = (patch.max(axis=2) < 80).mean()
    return dark < 0.5


def find_cell(mask):
    ys, xs = np.where(mask)
    return int(ys.mean() // CELL), int(xs.mean() // CELL)


green = (arr[:, :, 1] > 150) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100)
red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
start = find_cell(green)
goal = find_cell(red)

open_cells = {(r, c) for r in range(N) for c in range(N) if is_open(r, c)}
open_cells |= {start, goal}

# BFS
prev = {start: None}
q = deque([start])
while q:
    cur = q.popleft()
    if cur == goal:
        break
    r, c = cur
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (r + dr, c + dc)
        if nb in open_cells and nb not in prev:
            prev[nb] = cur
            q.append(nb)
assert goal in prev, "no path found"
path = []
cur = goal
while cur is not None:
    path.append(cur)
    cur = prev[cur]
path.reverse()
steps = len(path) - 1
print(f"start={start} goal={goal} path length={steps} steps")

# Pixels belonging to the start marker and the flag; restored on top of the
# trail so those elements stay exactly as in the first frame.
marker_mask = green | red | ((arr.max(axis=2) < 80) & (arr.min(axis=2) > 40))
marker_rgb = np.array(base)

TRAIL = (66, 133, 244)
HEAD = (30, 90, 220)
LINE_W = 14


def render(progress):
    """progress in [0, steps]: draw trail up to that fractional point."""
    im = base.copy()
    if progress <= 0:
        return im
    d = ImageDraw.Draw(im)
    k = int(progress)
    frac = progress - k
    pts = [center(*p) for p in path[: k + 1]]
    if k < steps and frac > 0:
        (x0, y0), (x1, y1) = center(*path[k]), center(*path[k + 1])
        pts.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
    if len(pts) >= 2:
        d.line(pts, fill=TRAIL, width=LINE_W, joint="curve")
    hx, hy = pts[-1]
    rad = 16
    d.ellipse([hx - rad, hy - rad, hx + rad, hy + rad], fill=HEAD)
    out = np.array(im)
    out[marker_mask] = marker_rgb[marker_mask]
    return Image.fromarray(out)


os.makedirs(OUT_DIR, exist_ok=True)
cmd = [
    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), OUT,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(FRAMES):
    progress = steps * i / (FRAMES - 1)
    frame = render(progress)
    proc.stdin.write(np.array(frame).tobytes())
proc.stdin.close()
proc.wait()
print("wrote", OUT)
