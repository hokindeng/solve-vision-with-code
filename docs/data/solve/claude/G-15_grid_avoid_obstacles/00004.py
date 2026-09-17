#!/usr/bin/env python3
"""Animate the agent along the shortest obstacle-free path from start to end."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 22
N = 10

BLUE = np.array([0, 100, 255])
RED = np.array([255, 50, 50])
GRAY = np.array([200, 200, 200])


def is_color(img, c, tol=30):
    return (np.abs(img.astype(int) - c).sum(axis=2) < tol)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = first.shape

    # --- detect grid lines (gray pixels along a row/column crossing the grid) ---
    gray_cols = np.where(is_color(first[H // 2 + 20:H // 2 + 21], GRAY)[0])[0]
    line_starts = [x for x in gray_cols if x - 1 not in set(gray_cols)]
    origin = line_starts[0]
    cell = line_starts[1] - line_starts[0]

    def cell_of(x, y):
        return (int((x - origin) // cell), int((y - origin) // cell))

    def cell_center(c, r):
        return (origin + c * cell + cell // 2, origin + r * cell + cell // 2)

    # --- start / end cells ---
    ys, xs = np.where(is_color(first, BLUE))
    start = cell_of(xs.mean(), ys.mean())
    ys, xs = np.where(is_color(first, RED))
    end = cell_of(xs.mean(), ys.mean())

    # --- obstacles: cells containing black pixels ---
    black = (first.sum(axis=2) < 60)
    obstacles = set()
    for r in range(N):
        for c in range(N):
            x0, y0 = origin + c * cell + 3, origin + r * cell + 3
            if black[y0:y0 + cell - 6, x0:x0 + cell - 6].any():
                obstacles.add((c, r))

    # --- BFS shortest path ---
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dc, cur[1] + dr)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in obstacles and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    assert path[0] == start and path[-1] == end, "no path found"

    # --- extract agent sprite from the start cell (everything that's not blue) ---
    sx0, sy0 = origin + start[0] * cell, origin + start[1] * cell
    region = first[sy0:sy0 + cell, sx0:sx0 + cell]
    mask = ~is_color(region, BLUE, tol=1)
    ys, xs = np.where(mask)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sprite = region[by0:by1, bx0:bx1].copy()
    smask = mask[by0:by1, bx0:bx1]
    # sprite offset relative to the cell's centre
    cx, cy = cell_center(*start)
    off_x = (sx0 + bx0) - cx
    off_y = (sy0 + by0) - cy

    # background with the agent removed (fill with blue)
    bg = first.copy()
    bg[sy0:sy0 + cell, sx0:sx0 + cell][mask] = BLUE

    def render(t):
        """t in [0, 1] along the path."""
        seg = t * (len(path) - 1)
        i = min(int(np.floor(seg)), len(path) - 2)
        f = seg - i
        (x0, y0), (x1, y1) = cell_center(*path[i]), cell_center(*path[i + 1])
        x = int(round(x0 + (x1 - x0) * f))
        y = int(round(y0 + (y1 - y0) * f))
        frame = bg.copy()
        px, py = x + off_x, y + off_y
        h, w = smask.shape
        frame[py:py + h, px:px + w][smask] = sprite[smask]
        return frame

    frames = [first.copy()]
    for i in range(1, N_FRAMES):
        frames.append(render(i / (N_FRAMES - 1)))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr.astype(np.uint8)).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"start={start} end={end} obstacles={sorted(obstacles)}")
    print(f"path={path}")
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
