#!/usr/bin/env python3
"""Animate the yellow agent along the shortest obstacle-free path from the
blue start cell to the red end cell of the 10x10 grid in first_frame.png."""
import subprocess
from collections import deque

import numpy as np
from PIL import Image

FIRST = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 62, 16
GRID, CELL, ORIGIN = 10, 93, 49  # cell interior top-left = ORIGIN + CELL*i

BLUE, RED, BLACK, WHITE = (0, 100, 255), (255, 50, 50), (0, 0, 0), (255, 255, 255)


def cell_of(mask):
    ys, xs = np.where(mask)
    return int((ys.mean() - ORIGIN) // CELL), int((xs.mean() - ORIGIN) // CELL)


def is_color(a, col, tol=30):
    return (np.abs(a.astype(int) - np.array(col)).sum(axis=2) < tol)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    blue, red, black = is_color(base, BLUE), is_color(base, RED), is_color(base, BLACK)
    start, end = cell_of(blue), cell_of(red)

    # obstacles: cells containing black pixels
    obstacles = set()
    for r in range(GRID):
        for c in range(GRID):
            y0, x0 = ORIGIN + CELL * r, ORIGIN + CELL * c
            if black[y0:y0 + CELL - 2, x0:x0 + CELL - 2].sum() > 50:
                obstacles.add((r, c))

    # extract agent sprite: non-blue pixels inside the start cell
    y0, x0 = ORIGIN + CELL * start[0], ORIGIN + CELL * start[1]
    cell_img = base[y0:y0 + CELL - 2, x0:x0 + CELL - 2]
    cell_blue = blue[y0:y0 + CELL - 2, x0:x0 + CELL - 2]
    ag = ~cell_blue
    ys, xs = np.where(ag)
    sy0, sy1, sx0, sx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprite = cell_img[sy0:sy1, sx0:sx1].copy()
    smask = ag[sy0:sy1, sx0:sx1]
    off = (sy0, sx0)  # sprite offset relative to cell top-left

    # background with agent removed (restore blue under it)
    bg = base.copy()
    sub = bg[y0:y0 + CELL - 2, x0:x0 + CELL - 2]
    sub[ag] = BLUE

    # BFS shortest path
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = (cur[0] + dr, cur[1] + dc)
            if 0 <= nx[0] < GRID and 0 <= nx[1] < GRID and nx not in obstacles and nx not in prev:
                prev[nx] = cur
                q.append(nx)
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    print("start", start, "end", end, "obstacles", sorted(obstacles))
    print("path", path, "moves", len(path) - 1)

    moves = len(path) - 1
    hold = 3  # frames held at the end
    move_frames = N_FRAMES - 1 - hold

    def ease(t):
        return t * t * (3 - 2 * t)

    frames = []
    for f in range(N_FRAMES):
        t = min(f / move_frames, 1.0) * moves
        i = min(int(t), moves - 1)
        u = ease(t - i) if t < moves else 1.0
        (r0, c0), (r1, c1) = path[i], path[i + 1]
        r, c = r0 + (r1 - r0) * u, c0 + (c1 - c0) * u
        py = int(round(ORIGIN + CELL * r + off[0]))
        px = int(round(ORIGIN + CELL * c + off[1]))
        frame = bg.copy()
        h, w = smask.shape
        region = frame[py:py + h, px:px + w]
        region[smask] = sprite[smask]
        if f == 0:
            frame = base.copy()
        frames.append(frame)

    assert np.array_equal(frames[0], base)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
