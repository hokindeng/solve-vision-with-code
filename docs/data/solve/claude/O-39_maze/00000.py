#!/usr/bin/env python3
"""Solve the 15x15 maze in first_frame.png and animate the path as a video."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N = 15
SIZE = 1024
CELL = SIZE / N
FPS = 16
NFRAMES = 60
PATH_COLOR = (40, 110, 220)
LINE_W = 16


def cell_center(r, c):
    return ((c + 0.5) * CELL, (r + 0.5) * CELL)


def build_grid(img):
    """Classify each cell as wall (False) or open (True), find start and end."""
    arr = np.asarray(img)
    grid = np.zeros((N, N), dtype=bool)
    start = end = None
    for r in range(N):
        for c in range(N):
            y0, y1 = int(r * CELL) + 8, int((r + 1) * CELL) - 8
            x0, x1 = int(c * CELL) + 8, int((c + 1) * CELL) - 8
            patch = arr[y0:y1, x0:x1].reshape(-1, 3).astype(int)
            g = np.all(patch == [50, 200, 50], axis=1).sum()
            rd = np.all(patch == [200, 50, 50], axis=1).sum()
            wh = np.all(patch == 255, axis=1).sum()
            dk = np.all(patch == 30, axis=1).sum()
            if g > 50:
                start = (r, c)
            if rd > 50:
                end = (r, c)
            grid[r, c] = (wh + g + rd) > dk
    return grid, start, end


def bfs(grid, start, end):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def render_frame(base, path, progress):
    """Draw the path from start up to `progress` (in cell steps, float)."""
    img = base.copy()
    if progress <= 0:
        return img
    draw = ImageDraw.Draw(img)
    pts = [cell_center(*p) for p in path]
    n_full = int(progress)
    frac = progress - n_full
    seg = pts[: n_full + 1]
    if n_full < len(pts) - 1 and frac > 0:
        (x0, y0), (x1, y1) = pts[n_full], pts[n_full + 1]
        seg.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
    if len(seg) >= 2:
        draw.line(seg, fill=PATH_COLOR, width=LINE_W, joint="curve")
    # round caps at the current head so the leading edge looks smooth
    hx, hy = seg[-1]
    rr = LINE_W / 2
    draw.ellipse((hx - rr, hy - rr, hx + rr, hy + rr), fill=PATH_COLOR)
    return img


def main():
    base = Image.open(FIRST).convert("RGB")
    grid, start, end = build_grid(base)
    path = bfs(grid, start, end)
    assert path and path[0] == start and path[-1] == end, "no path found"
    steps = len(path) - 1
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))
    # frame 0 is untouched; the path grows linearly and is complete on the last frame
    for i in range(NFRAMES):
        progress = steps * i / (NFRAMES - 1)
        frame = render_frame(base, path, progress)
        frame.save(os.path.join(frames_dir, f"{i:04d}.png"))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", os.path.join(frames_dir, "%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
        check=True,
    )
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))
    os.rmdir(frames_dir)
    print(f"start={start} end={end} path_len={len(path)} -> {OUT}")


if __name__ == "__main__":
    main()
