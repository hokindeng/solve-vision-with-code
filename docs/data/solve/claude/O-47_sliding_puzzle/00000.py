#!/usr/bin/env python3
"""Animate the 11-move solution of the 3x3 sliding puzzle in first_frame.png."""
import subprocess, os
from collections import deque
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, N_FRAMES, N = 16, 76, 3

first = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = first.shape

# --- grid geometry: dark grid lines -------------------------------------
dark = first.sum(2) < 200
lines = np.where(dark.mean(1) > 0.9)[0]
# group consecutive rows into line bands
bands = []
for r in lines:
    if bands and r == bands[-1][-1] + 1:
        bands[-1].append(r)
    else:
        bands.append([r])
assert len(bands) == N + 1, bands
# cell interiors: (start, end) exclusive, same for rows and cols (square grid)
cells = [(bands[i][-1] + 1, bands[i + 1][0]) for i in range(N)]

def cell_box(r, c):
    y0, y1 = cells[r]; x0, x1 = cells[c]
    return x0, y0, x1, y1

# --- read board state (tile = non-white content in cell) ----------------
white = np.array([255, 255, 255])
def cell_has_tile(r, c):
    x0, y0, x1, y1 = cell_box(r, c)
    return np.any(first[y0:y1, x0:x1] != white)

# Tile sprites: crop tight non-white bbox inside each occupied cell, record offset
sprites = {}
board = [[0] * N for _ in range(N)]
# numbers of tiles, read from layout of first frame (row-major)
LAYOUT = [[5, 1, 3], [0, 7, 6], [2, 4, 8]]
for r in range(N):
    for c in range(N):
        has = cell_has_tile(r, c)
        assert has == (LAYOUT[r][c] != 0), (r, c)
        board[r][c] = LAYOUT[r][c]
        if has:
            x0, y0, x1, y1 = cell_box(r, c)
            sub = first[y0:y1, x0:x1]
            nz = np.any(sub != white, axis=2)
            ys, xs = np.where(nz)
            ty0, ty1, tx0, tx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sprites[LAYOUT[r][c]] = (first[y0 + ty0:y0 + ty1, x0 + tx0:x0 + tx1].copy(), tx0, ty0)

# background: first frame with every cell interior blanked to white
bg = first.copy()
for r in range(N):
    for c in range(N):
        x0, y0, x1, y1 = cell_box(r, c)
        bg[y0:y1, x0:x1] = 255

# --- BFS solver ------------------------------------------------------------
start = tuple(sum(board, []))
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
def neighbors(s):
    b = s.index(0); br, bc = divmod(b, N)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = br + dr, bc + dc
        if 0 <= r < N and 0 <= c < N:
            i = r * N + c
            l = list(s); l[b], l[i] = l[i], l[b]
            yield tuple(l), s[i]
prev = {start: None}
q = deque([start])
while q:
    s = q.popleft()
    if s == goal: break
    for nx, tile in neighbors(s):
        if nx not in prev:
            prev[nx] = (s, tile); q.append(nx)
path = []
s = goal
while prev[s] is not None:
    s, tile = prev[s]; path.append(tile)
path.reverse()
print("solution (tiles moved):", path, "moves:", len(path))
assert len(path) == 11

# --- animation schedule -----------------------------------------------------
HOLD_START, PER_MOVE = 3, 6
HOLD_END = N_FRAMES - HOLD_START - PER_MOVE * len(path)
assert HOLD_END >= 1

def ease(t):
    return t * t * (3 - 2 * t)

pos = {t: (r, c) for r in range(N) for c in range(N) for t in [board[r][c]] if t}
blank = [(r, c) for r in range(N) for c in range(N) if board[r][c] == 0][0]

def render(moving=None, frm=None, to=None, t=0.0):
    img = bg.copy()
    for tile, (r, c) in pos.items():
        spr, ox, oy = sprites[tile]
        if tile == moving:
            ry = frm[0] + (to[0] - frm[0]) * t
            rx = frm[1] + (to[1] - frm[1]) * t
            # interpolate pixel origin between the two cells
            fy, fx = cell_box(*frm)[1], cell_box(*frm)[0]
            ty, tx = cell_box(*to)[1], cell_box(*to)[0]
            y = int(round(fy + (ty - fy) * t)) + oy
            x = int(round(fx + (tx - fx) * t)) + ox
        else:
            x0, y0, _, _ = cell_box(r, c)
            y, x = y0 + oy, x0 + ox
        h, w = spr.shape[:2]
        img[y:y + h, x:x + w] = spr
    return img

frames = [first.copy() for _ in range(HOLD_START)]
for tile in path:
    frm = pos[tile]; to = blank
    for k in range(1, PER_MOVE + 1):
        frames.append(render(tile, frm, to, ease(k / PER_MOVE)))
    pos[tile] = to; blank = frm
frames += [render() for _ in range(HOLD_END)]
assert len(frames) == N_FRAMES
assert np.array_equal(frames[0], first)

# --- encode ------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "15", "-preset", "slow", OUT], stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(np.ascontiguousarray(f).tobytes())
p.stdin.close(); p.wait()
assert p.returncode == 0
Image.fromarray(frames[-1]).save(os.path.join(HERE, "output", "last_frame.png"))
print("wrote", OUT)
