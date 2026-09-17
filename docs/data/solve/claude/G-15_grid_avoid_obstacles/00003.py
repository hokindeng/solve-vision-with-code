#!/usr/bin/env python3
"""Animate the agent along the shortest obstacle-free path on the 10x10 grid."""
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
N_FRAMES = 62
N = 10

BLUE = np.array([0, 100, 255])
RED = np.array([255, 50, 50])


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # --- detect grid geometry from the gray lines (200,200,200) -------------
    gray = np.all(base == 200, axis=2)
    cols = np.where(gray.sum(axis=0) > H // 2)[0]
    rows = np.where(gray.sum(axis=1) > W // 2)[0]

    def line_starts(idx):
        starts = [int(idx[0])]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                starts.append(int(b))
        return starts

    xs = line_starts(cols)  # 11 vertical line starts
    ys = line_starts(rows)  # 11 horizontal line starts
    assert len(xs) == N + 1 and len(ys) == N + 1, (len(xs), len(ys))

    def cell_box(r, c):
        # interior of a cell (exclusive of the 2px grid line)
        return ys[r] + 2, ys[r + 1] - 1, xs[c] + 2, xs[c + 1] - 1  # y0,y1,x0,x1

    def cell_center(r, c):
        y0, y1, x0, x1 = cell_box(r, c)
        return (y0 + y1) / 2.0, (x0 + x1) / 2.0

    # --- classify cells -----------------------------------------------------
    start = end = None
    obstacles = set()
    for r in range(N):
        for c in range(N):
            y0, y1, x0, x1 = cell_box(r, c)
            patch = base[y0:y1, x0:x1]
            if np.any(np.all(patch == BLUE, axis=2)):
                start = (r, c)
            elif np.any(np.all(patch == RED, axis=2)):
                end = (r, c)
            elif np.any(np.all(patch < 60, axis=2)):
                obstacles.add((r, c))
    assert start is not None and end is not None

    # --- extract the agent sprite (everything in the start cell that is not blue)
    y0, y1, x0, x1 = cell_box(*start)
    cell = base[y0:y1, x0:x1]
    mask = ~np.all(cell == BLUE, axis=2)
    yy, xx = np.where(mask)
    sy0, sy1, sx0, sx1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    sprite = cell[sy0:sy1, sx0:sx1].copy()
    sprite_mask = mask[sy0:sy1, sx0:sx1]
    sh, sw = sprite.shape[:2]
    cy, cx = cell_center(*start)
    # offset of sprite top-left relative to the agent's centre
    off_y = (y0 + sy0) - cy
    off_x = (x0 + sx0) - cx

    # background with the agent removed (start square is solid blue)
    bg = base.copy()
    bg[y0:y1, x0:x1][mask] = BLUE

    # --- BFS shortest path --------------------------------------------------
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in obstacles and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    assert end in prev, "no path"
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    n_moves = len(path) - 1
    print("start", start, "end", end, "obstacles", sorted(obstacles))
    print("path", path)

    # --- render frames ------------------------------------------------------
    centers = [cell_center(*p) for p in path]
    move_frames = N_FRAMES - 5  # last few frames hold the finished state

    def draw(frame_idx):
        if frame_idx == 0:
            return base.copy()
        t = min(1.0, frame_idx / float(move_frames)) * n_moves
        i = min(int(np.floor(t)), n_moves - 1)
        f = t - i
        py = centers[i][0] * (1 - f) + centers[i + 1][0] * f
        px = centers[i][1] * (1 - f) + centers[i + 1][1] * f
        ty = int(round(py + off_y))
        tx = int(round(px + off_x))
        img = bg.copy()
        region = img[ty:ty + sh, tx:tx + sw]
        region[sprite_mask] = sprite[sprite_mask]
        return img

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(N_FRAMES):
        proc.stdin.write(np.ascontiguousarray(draw(k)).tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0
    print("wrote", OUT)


if __name__ == "__main__":
    main()
