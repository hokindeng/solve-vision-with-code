#!/usr/bin/env python3
"""Animate the agent collecting all keys (optimal order) and reaching the door."""
import itertools
from collections import deque

import numpy as np
from PIL import Image
import imageio.v2 as imageio

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES = 59
FPS = 16

img = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = img.shape
GRID = 13                      # 11x11 maze + 1-cell black border
CELL = W / GRID

def cell_box(r, c):
    return (int(round(c * CELL)), int(round(r * CELL)),
            int(round((c + 1) * CELL)), int(round((r + 1) * CELL)))

def cell_center(r, c):
    return (int(round((c + 0.5) * CELL)), int(round((r + 0.5) * CELL)))

# --- locate coloured objects -------------------------------------------------
black = np.all(img == 0, axis=2)
white = np.all(img == 255, axis=2)
coloured = ~(black | white)

def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()

cols = {tuple(c) for c in np.unique(img[coloured].reshape(-1, 3), axis=0)}
objects = {}
for col in cols:
    m = np.all(img == col, axis=2)
    x0, y0, x1, y1 = bbox(m)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r, c = int(cy // CELL), int(cx // CELL)
    # classify: hollow square (door) has a white/empty centre; the agent is green.
    inner = m[int(cy) - 5:int(cy) + 5, int(cx) - 5:int(cx) + 5].mean()
    if col == (0, 255, 0):
        kind = "agent"
    elif inner < 0.5:
        kind = "door"
    else:
        kind = "key"
    objects[col] = dict(kind=kind, cell=(r, c), mask=m)

agent_col = next(k for k, v in objects.items() if v["kind"] == "agent")
door_col = next(k for k, v in objects.items() if v["kind"] == "door")
key_cols = [k for k, v in objects.items() if v["kind"] == "key"]
start = objects[agent_col]["cell"]
door = objects[door_col]["cell"]
keys = {k: objects[k]["cell"] for k in key_cols}

# --- grid parsing -------------------------------------------------------------
# a cell is a path if its corner pixel (away from any centred object) is white
passable = np.zeros((GRID, GRID), bool)
for r in range(GRID):
    for c in range(GRID):
        x0, y0, x1, y1 = cell_box(r, c)
        passable[r, c] = white[y0 + 4, x0 + 4]
for cell in [start, door, *keys.values()]:
    passable[cell] = True

def bfs(src):
    dist = {src: 0}
    prev = {}
    q = deque([src])
    while q:
        u = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + d[0], u[1] + d[1])
            if 0 <= v[0] < GRID and 0 <= v[1] < GRID and passable[v] and v not in dist:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    return dist, prev

def path(a, b):
    dist, prev = bfs(a)
    p = [b]
    while p[-1] != a:
        p.append(prev[p[-1]])
    return p[::-1]

nodes = [start, *keys.values(), door]
D = {n: bfs(n)[0] for n in nodes}
best = None
for perm in itertools.permutations(key_cols):
    seq = [start, *(keys[k] for k in perm), door]
    tot = sum(D[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
    if best is None or tot < best[0]:
        best = (tot, perm)
total, order = best
print("optimal order:", [objects[k]["cell"] for k in order], "total steps:", total)

# full cell path and the step index at which each key is collected
full = [start]
pickup_step = {}
cur = start
for k in order:
    seg = path(cur, keys[k])
    full += seg[1:]
    cur = keys[k]
    pickup_step[k] = len(full) - 1
full += path(cur, door)[1:]
n_steps = len(full) - 1

# --- rendering -----------------------------------------------------------------
agent_mask = objects[agent_col]["mask"]
ax0, ay0, ax1, ay1 = bbox(agent_mask)
sprite = agent_mask[ay0:ay1 + 1, ax0:ax1 + 1]
sh, sw = sprite.shape

# background: first frame with agent and keys erased (their cells are white paths)
bg = img.copy()
for col in [agent_col, *key_cols]:
    bg[objects[col]["mask"]] = 255

def render(t_step, collected):
    frame = bg.copy()
    for k in key_cols:
        if k not in collected:
            frame[objects[k]["mask"]] = k
    i = int(np.floor(t_step))
    f = t_step - i
    if i >= n_steps:
        i, f = n_steps, 0.0
    (r0, c0) = full[i]
    (r1, c1) = full[min(i + 1, n_steps)]
    x0, y0 = cell_center(r0, c0)
    x1, y1 = cell_center(r1, c1)
    cx = int(round(x0 + (x1 - x0) * f))
    cy = int(round(y0 + (y1 - y0) * f))
    tx, ty = cx - sw // 2, cy - sh // 2
    frame[ty:ty + sh, tx:tx + sw][sprite] = agent_col
    return frame

frames = []
collected = set()
for n in range(N_FRAMES):
    t = n_steps * n / (N_FRAMES - 1)
    for k in key_cols:
        if t >= pickup_step[k] - 1e-9:
            collected.add(k)
    frames.append(render(t, collected))

assert np.array_equal(frames[0], img), "first frame must match input"

writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1, ffmpeg_params=["-crf", "12"])
for fr in frames:
    writer.append_data(fr)
writer.close()
print("wrote", OUT, len(frames), "frames")
