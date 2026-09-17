#!/usr/bin/env python3
"""Solve the 15x15 maze in first_frame.png and render the walk as a video."""
import subprocess, os
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N = 15
W = H = 1024
CS = W / N
FPS = 16
FRAMES = 63

base = Image.open(SRC).convert("RGB")
arr = np.array(base).astype(int)

# --- parse grid ---------------------------------------------------------
def cell_center(r, c):
    return ((c + 0.5) * CS, (r + 0.5) * CS)

def classify(r, c):
    x0, x1 = int(round(c * CS)), int(round((c + 1) * CS))
    y0, y1 = int(round(r * CS)), int(round((r + 1) * CS))
    patch = arr[y0:y1, x0:x1]
    g = ((patch[:, :, 1] > 150) & (patch[:, :, 0] < 120)).sum()
    rd = ((patch[:, :, 0] > 150) & (patch[:, :, 1] < 100)).sum()
    white = (patch.sum(axis=2) > 600).mean()
    if g > 200:
        return "S"
    if rd > 200:
        return "E"
    return "." if white > 0.5 else "#"

grid = [[classify(r, c) for c in range(N)] for r in range(N)]
start = end = None
for r in range(N):
    for c in range(N):
        if grid[r][c] == "S":
            start = (r, c)
        elif grid[r][c] == "E":
            end = (r, c)
assert start and end, (start, end)

# --- BFS ----------------------------------------------------------------
prev = {start: None}
q = deque([start])
while q:
    cur = q.popleft()
    if cur == end:
        break
    r, c = cur
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != "#" and (nr, nc) not in prev:
            prev[(nr, nc)] = cur
            q.append((nr, nc))
assert end in prev, "no path found"
path = []
cur = end
while cur is not None:
    path.append(cur)
    cur = prev[cur]
path.reverse()
L = len(path) - 1
print("start", start, "end", end, "steps", L)

# --- render -------------------------------------------------------------
# Mask of pixels we are allowed to touch: only white pathway pixels.
white_mask = (arr.sum(axis=2) > 600)
protect = ~white_mask  # walls, start marker, flag stay untouched

TRAIL = (70, 130, 230)
MARK = (30, 90, 200)
TRAIL_W = int(CS * 0.30)
MARK_R = CS * 0.28

def render(progress):
    """progress in cells travelled along path (0..L)."""
    img = base.copy()
    if progress <= 0:
        return img
    d = ImageDraw.Draw(img)
    k = int(np.floor(progress))
    frac = progress - k
    pts = [cell_center(*p) for p in path[: k + 1]]
    if k < L:
        (x0, y0), (x1, y1) = cell_center(*path[k]), cell_center(*path[k + 1])
        pos = (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)
    else:
        pos = pts[-1]
    pts.append(pos)
    if len(pts) >= 2:
        d.line(pts, fill=TRAIL, width=TRAIL_W, joint="curve")
    for (x, y) in pts[:-1]:
        d.ellipse([x - TRAIL_W / 2, y - TRAIL_W / 2, x + TRAIL_W / 2, y + TRAIL_W / 2], fill=TRAIL)
    x, y = pos
    d.ellipse([x - MARK_R, y - MARK_R, x + MARK_R, y + MARK_R], fill=MARK)
    out = np.array(img)
    out[protect] = arr[protect]
    return Image.fromarray(out.astype(np.uint8))

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for i in range(FRAMES):
    t = i / (FRAMES - 1)
    frames.append(render(t * L))

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", OUT],
    stdin=subprocess.PIPE)
for f in frames:
    ff.stdin.write(np.array(f).tobytes())
ff.stdin.close()
ff.wait()
print("wrote", OUT)
