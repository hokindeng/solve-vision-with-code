#!/usr/bin/env python3
"""Animate the agent along the shortest obstacle-free path from start to end."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 34
FPS = 16
N = 10

BLUE = np.array([0, 100, 255])
RED = np.array([255, 50, 50])


def find_grid(a):
    """Return grid origin/pitch by locating the (200,200,200) grid lines."""
    gray = (a == 200).all(-1)
    # pick a row that crosses the vertical lines but is not itself a horizontal line
    counts = gray.sum(1)
    y = int(np.argmin(np.where(counts > 0, counts, np.inf)))
    xs = [x for x in range(a.shape[1]) if gray[y, x]]
    # collapse 2-px-wide lines to their first pixel
    lines = [x for i, x in enumerate(xs) if i == 0 or x != xs[i - 1] + 1]
    origin = lines[0]
    pitch = round((lines[-1] - lines[0]) / N)
    return origin, pitch


def cell_slice(origin, pitch, c, r):
    x0 = origin + pitch * c
    y0 = origin + pitch * r
    return slice(y0, y0 + pitch + 2), slice(x0, x0 + pitch + 2)


def classify(a, origin, pitch):
    start = end = None
    obstacles = set()
    for r in range(N):
        for c in range(N):
            ys, xs = cell_slice(origin, pitch, c, r)
            cell = a[ys, xs]
            if (cell == BLUE).all(-1).sum() > 100:
                start = (c, r)
            elif (cell == RED).all(-1).sum() > 100:
                end = (c, r)
            elif (cell.sum(-1) < 150).sum() > 50:
                obstacles.add((c, r))
    return start, end, obstacles


def bfs(start, end, obstacles):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        c, r = cur
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (c + dc, r + dr)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in obstacles and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def main():
    a = np.array(Image.open(FIRST).convert("RGB"))
    origin, pitch = find_grid(a)
    start, end, obstacles = classify(a, origin, pitch)
    path = bfs(start, end, obstacles)
    print("start", start, "end", end, "obstacles", sorted(obstacles))
    print("path", path)

    # Lift the agent sprite: every non-blue pixel inside the start cell.
    ys, xs = cell_slice(origin, pitch, *start)
    blue_px = np.where((a[ys, xs] == BLUE).all(-1))
    # tighten to the blue square's own bounding box so grid lines are excluded
    ys = slice(ys.start + blue_px[0].min(), ys.start + blue_px[0].max() + 1)
    xs = slice(xs.start + blue_px[1].min(), xs.start + blue_px[1].max() + 1)
    cell = a[ys, xs]
    mask = ~(cell == BLUE).all(-1)
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    sy, sx = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    sprite = cell[sy, sx].copy()
    smask = mask[sy, sx]
    sprite_off = (xs.start + cols[0], ys.start + rows[0])  # top-left in frame coords
    start_px = (origin + pitch * start[0], origin + pitch * start[1])

    # Background with the agent removed (start cell restored to solid blue).
    base = a.copy()
    base[ys, xs][mask] = BLUE

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    segs = len(path) - 1
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1) * segs
        k = min(int(t), segs - 1)
        f = t - k
        (c0, r0), (c1, r1) = path[k], path[k + 1]
        cx = c0 + (c1 - c0) * f
        cy = r0 + (r1 - r0) * f
        dx = int(round(sprite_off[0] + (origin + pitch * cx - start_px[0])))
        dy = int(round(sprite_off[1] + (origin + pitch * cy - start_px[1])))
        frame = a.copy() if i == 0 else base.copy()
        if i > 0:
            region = frame[dy:dy + sprite.shape[0], dx:dx + sprite.shape[1]]
            region[smask] = sprite[smask]
        frames.append(frame)

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
