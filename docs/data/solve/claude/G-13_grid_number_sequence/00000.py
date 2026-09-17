#!/usr/bin/env python3
"""Animate the orange agent from the green cell through yellow waypoints 1,2,3,... to the red cell.

Grid geometry, colours and the agent sprite are all read from first_frame.png so every
pixel that is not the moving agent stays untouched in every frame.
"""
import subprocess
from collections import deque

import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 107
N, PITCH = 10, 102          # 10x10 grid, 102 px per cell (2 px grid lines)

first = np.array(Image.open(SRC).convert("RGB"))
H, W, _ = first.shape


def cell_center(c, r):
    return c * PITCH + 51, r * PITCH + 51


def cell_rgb(c, r):
    """Dominant colour of a cell interior."""
    x, y = cell_center(c, r)
    patch = first[y - 45:y + 45, x - 45:x + 45].reshape(-1, 3)
    vals, cnt = np.unique(patch, axis=0, return_counts=True)
    return tuple(int(v) for v in vals[cnt.argmax()])


def is_orange(rgb): return rgb[0] > 200 and 100 < rgb[1] < 200 and rgb[2] < 60
def is_green(rgb): return rgb[0] < 60 and rgb[1] > 200 and rgb[2] < 60
def is_red(rgb): return rgb[0] > 200 and rgb[1] < 60 and rgb[2] < 60
def is_yellow(rgb): return rgb[0] > 200 and rgb[1] > 200 and rgb[2] < 60


# ---- locate special cells ---------------------------------------------------
start = goal = None
yellows = []
for r in range(N):
    for c in range(N):
        x, y = cell_center(c, r)
        centre = tuple(int(v) for v in first[y, x])
        # the agent sits on the green start cell, so the centre pixel is orange there
        if is_orange(centre) or is_green(cell_rgb(c, r)):
            start = (c, r)
        elif is_red(cell_rgb(c, r)):
            goal = (c, r)
        elif is_yellow(cell_rgb(c, r)):
            yellows.append((c, r))

# order yellow waypoints by their digit: count dark "ink" pixels differs per digit, but a
# more robust approach is template-free ordering by reading the digit glyph. The generator
# uses a plain font, so we rank digits by matching against rendered templates.
from PIL import ImageDraw, ImageFont


def digit_glyph(c, r):
    x, y = cell_center(c, r)
    patch = first[y - 45:y + 45, x - 45:x + 45]
    ink = (patch.sum(axis=2) < 200)
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return None
    g = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray((g * 255).astype(np.uint8)).resize((20, 30))


def templates():
    out = {}
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    for d in range(1, 10):
        im = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(im).text((20, 15), str(d), fill=255, font=font)
        a = np.array(im) > 128
        ys, xs = np.nonzero(a)
        g = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        out[d] = np.array(Image.fromarray((g * 255).astype(np.uint8)).resize((20, 30))) > 128
    return out


T = templates()


def read_digit(c, r):
    g = digit_glyph(c, r)
    if g is None:
        return 99
    g = np.array(g) > 128
    return min(T, key=lambda d: np.count_nonzero(T[d] ^ g))


yellows.sort(key=lambda cr: read_digit(*cr))
waypoints = [start] + yellows + [goal]


# ---- shortest 4-neighbour paths between consecutive waypoints --------------
def bfs(a, b):
    prev = {a: None}
    q = deque([a])
    while q:
        u = q.popleft()
        if u == b:
            break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dc, u[1] + dr)
            if 0 <= v[0] < N and 0 <= v[1] < N and v not in prev:
                prev[v] = u
                q.append(v)
    path = []
    while b is not None:
        path.append(b)
        b = prev[b]
    return path[::-1]


path = [start]
for a, b in zip(waypoints, waypoints[1:]):
    path += bfs(a, b)[1:]

# ---- agent sprite (exact pixels) and clean background ----------------------
mask = np.zeros((H, W), bool)
for rgb in np.unique(first.reshape(-1, 3), axis=0):
    if is_orange(tuple(int(v) for v in rgb)):
        mask |= np.all(first == rgb, axis=2)
ys, xs = np.nonzero(mask)
sx, sy = cell_center(*start)
sprite_off = np.stack([ys - sy, xs - sx], axis=1)
sprite_rgb = first[ys, xs]

background = first.copy()
background[mask] = np.array(cell_rgb(*start), np.uint8)  # green under the agent

# ---- timing: short hold at start and end, constant speed in between --------
HOLD_START, HOLD_END = 4, 4
move_frames = N_FRAMES - HOLD_START - HOLD_END
n_steps = len(path) - 1
centers = np.array([cell_center(*p) for p in path], float)


def agent_pos(f):
    if f < HOLD_START:
        return centers[0]
    if f >= N_FRAMES - HOLD_END:
        return centers[-1]
    s = (f - HOLD_START) / (move_frames - 1) * n_steps
    i = min(int(np.floor(s)), n_steps - 1)
    t = s - i
    return centers[i] * (1 - t) + centers[i + 1] * t


def render(f):
    frame = background.copy()
    x, y = np.rint(agent_pos(f)).astype(int)
    yy = np.clip(sprite_off[:, 0] + y, 0, H - 1)
    xx = np.clip(sprite_off[:, 1] + x, 0, W - 1)
    frame[yy, xx] = sprite_rgb
    return frame


ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
    stdin=subprocess.PIPE)
for f in range(N_FRAMES):
    ff.stdin.write(render(f).tobytes())
ff.stdin.close()
ff.wait()
print("waypoints:", waypoints)
print("path length:", n_steps, "moves ->", OUT)
