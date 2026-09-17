#!/usr/bin/env python3
"""Animate the yellow agent from the green start cell to the pink end cell
along a shortest (Manhattan) path on the 10x10 grid in first_frame.png."""
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
N_FRAMES = 80
CELL = 102          # grid pitch in pixels (100 px interior + 2 px line)
N = 10

GREEN = np.array([0, 255, 0], np.uint8)
PINK = np.array([255, 20, 147], np.uint8)
YELLOW = np.array([255, 255, 0], np.uint8)


def cell_of(mask):
    ys, xs = np.where(mask)
    return int(xs.mean() // CELL), int(ys.mean() // CELL)   # (col, row)


def shortest_path(start, goal):
    """BFS over the open grid (no walls) -> list of (col,row) from start to goal."""
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        x, y = c
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):   # prefer down/right first
            n = (x + dx, y + dy)
            if 0 <= n[0] < N and 0 <= n[1] < N and n not in prev:
                prev[n] = c
                q.append(n)
    path = []
    c = goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frame0 = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = frame0.shape

    agent_mask = (frame0 == YELLOW).all(2)
    start = cell_of((frame0 == GREEN).all(2) | agent_mask)
    goal = cell_of((frame0 == PINK).all(2))

    # Background without the agent: agent pixels sit inside the green start cell.
    background = frame0.copy()
    background[agent_mask] = GREEN

    # Agent sprite (exact pixels, no anti-aliasing) and its top-left origin.
    ys, xs = np.where(agent_mask)
    y0, x0 = ys.min(), xs.min()
    sprite = agent_mask[y0:ys.max() + 1, x0:xs.max() + 1]
    sh, sw = sprite.shape

    path = shortest_path(start, goal)
    n_moves = len(path) - 1                       # 12 for this instance

    # Pacing: short hold at start, uniform moves, short hold at end.
    hold_start = 4
    hold_end = 4
    move_frames = N_FRAMES - hold_start - hold_end   # 72 frames -> 6 per move
    per_move = move_frames / n_moves

    def agent_origin(t):
        """Top-left of sprite at frame t (integer pixels)."""
        if t < hold_start:
            k = 0.0
        elif t >= hold_start + move_frames:
            k = float(n_moves)
        else:
            k = (t - hold_start) / per_move
        i = min(int(k), n_moves - 1)
        f = k - i
        (cx0, cy0), (cx1, cy1) = path[i], path[i + 1]
        dx = ((1 - f) * cx0 + f * cx1 - start[0]) * CELL
        dy = ((1 - f) * cy0 + f * cy1 - start[1]) * CELL
        return x0 + int(round(dx)), y0 + int(round(dy))

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
         "-r", str(FPS), OUT],
        stdin=subprocess.PIPE,
    )
    for t in range(N_FRAMES):
        frame = background.copy()
        ax, ay = agent_origin(t)
        region = frame[ay:ay + sh, ax:ax + sw]
        region[sprite] = YELLOW
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {N_FRAMES} frames, path {start} -> {goal} in {n_moves} moves")


if __name__ == "__main__":
    main()
