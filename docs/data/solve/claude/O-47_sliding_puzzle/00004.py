#!/usr/bin/env python3
"""Generate the sliding-puzzle solution video from first_frame.png.

Geometry (grid lines, tile rectangles, tile sprites) is measured directly from
the first frame, so every pixel outside the moving tiles stays untouched.
"""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 76
N = 3  # 3x3 puzzle

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape

# ---------------------------------------------------------------- geometry
GRID = np.array([51, 51, 51])
WHITE = np.array([255, 255, 255])

def is_color(px, c):
    return np.all(np.abs(px.astype(int) - c) <= 8, axis=-1)

# Grid lines: dark rows/cols that are dark across most of the board.
dark_rows = np.where(is_color(base, GRID).mean(axis=1) > 0.9)[0]
dark_cols = np.where(is_color(base, GRID).mean(axis=0) > 0.9)[0]

def group(idx):
    runs, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            runs.append((s, a)); s = b
    runs.append((s, idx[-1]))
    return runs

row_lines = group(dark_rows)   # 4 horizontal lines
col_lines = group(dark_cols)   # 4 vertical lines
assert len(row_lines) == N + 1 and len(col_lines) == N + 1, (row_lines, col_lines)

# Cell interiors (exclusive of grid lines).
cell_y = [(row_lines[i][1] + 1, row_lines[i + 1][0]) for i in range(N)]
cell_x = [(col_lines[i][1] + 1, col_lines[i + 1][0]) for i in range(N)]

# Tile rectangles: bounding box of non-white pixels inside each cell.
tile_rect = {}
for r in range(N):
    for c in range(N):
        y0, y1 = cell_y[r]; x0, x1 = cell_x[c]
        sub = base[y0:y1, x0:x1]
        mask = ~is_color(sub, WHITE)
        if mask.mean() < 0.05:
            continue  # blank cell
        ys, xs = np.where(mask)
        tile_rect[(r, c)] = (y0 + ys.min(), y0 + ys.max() + 1, x0 + xs.min(), x0 + xs.max() + 1)

# Per-row tile y-range and per-column x-range (so the blank cell gets a slot too).
row_y = {}
col_x = {}
for (r, c), (ty0, ty1, tx0, tx1) in tile_rect.items():
    row_y.setdefault(r, (ty0, ty1)); col_x.setdefault(c, (tx0, tx1))
assert len(row_y) == N and len(col_x) == N

def slot(r, c):
    ty0, ty1 = row_y[r]; tx0, tx1 = col_x[c]
    return ty0, ty1, tx0, tx1

# ---------------------------------------------------------------- read digits
# Identify which number sits in each cell by matching against rendered glyphs
# is unnecessary: we only need a consistent labelling. But to solve we need
# actual numbers, so recognise them by template matching against PIL's font
# is fragile; instead read them from the known layout via simple OCR-free
# approach: compare white-pixel glyph masks with digits rendered by DejaVu Sans
# Bold at a fitted size.
from PIL import ImageDraw, ImageFont

def glyph_mask(r, c):
    ty0, ty1, tx0, tx1 = tile_rect[(r, c)]
    sub = base[ty0 + 6:ty1 - 6, tx0 + 6:tx1 - 6]
    m = sub.mean(axis=-1) > 160  # white-ish glyph on purple
    ys, xs = np.where(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def render_digit(d, size):
    font = None
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size); break
    if font is None:
        font = ImageFont.load_default()
    im = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(im).text((size // 2, size // 4), str(d), fill=255, font=font)
    a = np.array(im) > 128
    ys, xs = np.where(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def classify(mask):
    best, best_score = None, -1
    h, w = mask.shape
    for d in range(1, N * N):
        g = render_digit(d, 64)
        g = np.array(Image.fromarray(g.astype(np.uint8) * 255).resize((w, h))) > 128
        score = (g == mask).mean()
        if score > best_score:
            best, best_score = d, score
    return best

state = [0] * (N * N)
for (r, c) in tile_rect:
    state[r * N + c] = classify(glyph_mask(r, c))
assert sorted(state) == list(range(N * N)), state
print("initial state:", state)

# ---------------------------------------------------------------- solve (BFS)
goal = tuple(list(range(1, N * N)) + [0])
start = tuple(state)

def neighbours(s):
    b = s.index(0); br, bc = divmod(b, N)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = br + dr, bc + dc
        if 0 <= r < N and 0 <= c < N:
            t = list(s); i = r * N + c
            t[b], t[i] = t[i], t[b]
            yield tuple(t), i  # moved tile came from index i

prev = {start: None}
q = deque([start])
while q:
    s = q.popleft()
    if s == goal:
        break
    for t, i in neighbours(s):
        if t not in prev:
            prev[t] = (s, i); q.append(t)

moves = []  # list of (from_index, to_index) for the moving tile
s = goal
while prev[s] is not None:
    p, i = prev[s]
    moves.append((i, p.index(0)))
    s = p
moves.reverse()
print("moves:", len(moves))
assert len(moves) == 11, len(moves)

# ---------------------------------------------------------------- animate
sprites = {}  # tile number -> pixel sprite
for (r, c), (ty0, ty1, tx0, tx1) in tile_rect.items():
    sprites[state[r * N + c]] = base[ty0:ty1, tx0:tx1].copy()

def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep

# Frame budget: frame 0 = first frame, then moves, then a short hold.
hold_start, hold_end = 1, 4
budget = N_FRAMES - 1 - hold_start - hold_end
per = [budget // len(moves)] * len(moves)
for k in range(budget - sum(per)):
    per[k] += 1

frames = [base.copy() for _ in range(1 + hold_start)]
cur = list(start)
for (src, dst), n in zip(moves, per):
    tile = cur[src]
    sr, sc = divmod(src, N); dr, dc = divmod(dst, N)
    sy0, sy1, sx0, sx1 = slot(sr, sc)
    dy0, dy1, dx0, dx1 = slot(dr, dc)
    prev_frame = frames[-1]
    for k in range(1, n + 1):
        t = ease(k / n)
        f = prev_frame.copy()
        # clear both cell interiors to white
        for (r, c) in ((sr, sc), (dr, dc)):
            y0, y1 = cell_y[r]; x0, x1 = cell_x[c]
            f[y0:y1, x0:x1] = 255
        y = int(round(sy0 + (dy0 - sy0) * t))
        x = int(round(sx0 + (dx0 - sx0) * t))
        sp = sprites[tile]
        f[y:y + sp.shape[0], x:x + sp.shape[1]] = sp
        frames.append(f)
    cur[dst], cur[src] = tile, 0

while len(frames) < N_FRAMES:
    frames.append(frames[-1].copy())
frames = frames[:N_FRAMES]
assert tuple(cur) == goal

# ---------------------------------------------------------------- encode
os.makedirs(OUT_DIR, exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(np.ascontiguousarray(f).tobytes())
p.stdin.close()
p.wait()
assert p.returncode == 0
print("wrote", OUT, len(frames), "frames")
