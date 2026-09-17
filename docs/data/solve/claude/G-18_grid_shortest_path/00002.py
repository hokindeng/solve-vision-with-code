#!/usr/bin/env python3
"""Animate the purple agent along the shortest path from the yellow start
square to the orange end square on the 10x10 grid in first_frame.png."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 55
FPS = 16
CELL = 102          # grid pitch in pixels (2px line + 100px interior)
GRID = 10

YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)


def find_cell(mask):
    ys, xs = np.where(mask)
    return int(ys.mean() // CELL), int(xs.mean() // CELL)   # (row, col)


def cell_center(rc):
    r, c = rc
    return CELL * c + CELL // 2, CELL * r + CELL // 2      # (x, y)


def shortest_path(start, goal):
    """BFS over 4-neighbour moves on the open grid."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < GRID and 0 <= nc < GRID and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def ease(t):
    return t * t * (3 - 2 * t)   # smoothstep


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    yel = np.all(base == YELLOW, axis=2)
    org = np.all(base == ORANGE, axis=2)
    pur = np.all(base == PURPLE, axis=2)

    start = find_cell(yel | pur)
    goal = find_cell(org)

    # Agent geometry: center and radius measured from the first frame.
    ys, xs = np.where(pur)
    cy0, cx0 = ys.mean(), xs.mean()
    half = (xs.max() - xs.min()) // 2      # PIL ellipse bbox half-extent

    # Background = first frame with the agent removed (start square is yellow).
    bg = base.copy()
    bg[pur] = YELLOW

    def draw_agent(img, cx, cy):
        cx, cy = int(round(cx)), int(round(cy))
        pil = Image.fromarray(img)
        ImageDraw.Draw(pil).ellipse((cx - half, cy - half, cx + half, cy + half), fill=PURPLE)
        return np.array(pil)

    # Sanity check: regenerated first frame must match exactly.
    f0 = draw_agent(bg.copy(), cx0, cy0)
    assert np.array_equal(f0, base), "agent re-render does not match first frame"

    path = shortest_path(start, goal)
    n_moves = len(path) - 1
    centers = [cell_center(rc) for rc in path]
    # Use the measured start centre for exact first-frame reproduction.
    centers[0] = (cx0, cy0)

    hold_start, hold_end = 4, 6
    move_frames = N_FRAMES - hold_start - hold_end
    per_move = move_frames / n_moves

    frames = []
    for i in range(N_FRAMES):
        if i < hold_start:
            cx, cy = centers[0]
        elif i >= N_FRAMES - hold_end:
            cx, cy = centers[-1]
        else:
            s = (i - hold_start) / per_move          # progress in moves
            k = min(int(s), n_moves - 1)
            t = ease(min(s - k, 1.0))
            (x0, y0), (x1, y1) = centers[k], centers[k + 1]
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        frames.append(draw_agent(bg.copy(), cx, cy))

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"path {path[0]} -> {path[-1]} in {n_moves} moves; wrote {OUT}")


if __name__ == "__main__":
    main()
