#!/usr/bin/env python3
"""Animate the orange agent along a shortest path from the green start to the red goal.

Everything is derived from first_frame.png: grid geometry, cell positions, the exact
agent sprite. Only the agent's pixels change between frames.
"""
import os
import subprocess
import tempfile
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
N_FRAMES, FPS, GRID = 55, 16, 10

RED, GREEN, ORANGE, BLACK = (255, 0, 0), (0, 255, 0), (255, 165, 0), (0, 0, 0)


def color_mask(img, rgb):
    return (img == np.array(rgb, dtype=np.uint8)).all(axis=2)


def bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), xs.max(), ys.min(), ys.max()


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = first.shape
    pitch = W / GRID

    def cell_of(mask):
        x0, x1, y0, y1 = bbox(mask)
        return int((y0 + y1) / 2 // pitch), int((x0 + x1) / 2 // pitch)

    start = cell_of(color_mask(first, GREEN))
    goal = cell_of(color_mask(first, RED))

    # Obstacles: any cell whose interior is mostly black (none expected, but be safe).
    blocked = set()
    for r in range(GRID):
        for c in range(GRID):
            y0, y1 = int(r * pitch) + 6, int((r + 1) * pitch) - 6
            x0, x1 = int(c * pitch) + 6, int((c + 1) * pitch) - 6
            if color_mask(first[y0:y1, x0:x1], BLACK).mean() > 0.5:
                blocked.add((r, c))

    # BFS shortest path (4-neighbour).
    prev = {start: None}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        if cur == goal:
            break
        r, c = cur
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < GRID and 0 <= nc < GRID and (nr, nc) not in prev and (nr, nc) not in blocked:
                prev[(nr, nc)] = cur
                dq.append((nr, nc))
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    # Agent sprite: exact orange pixels and their offset from the start cell centre.
    agent_mask = color_mask(first, ORANGE)
    ax0, ax1, ay0, ay1 = bbox(agent_mask)
    sprite = agent_mask[ay0:ay1 + 1, ax0:ax1 + 1]
    sx, sy = start[1] * pitch + pitch / 2, start[0] * pitch + pitch / 2
    off_x, off_y = ax0 - sx, ay0 - sy

    # Background = first frame with the agent removed (it sits inside the green cell).
    background = first.copy()
    background[agent_mask] = GREEN

    def cell_center(rc):
        return rc[1] * pitch + pitch / 2, rc[0] * pitch + pitch / 2

    def render(cx, cy):
        img = background.copy()
        x, y = int(round(cx + off_x)), int(round(cy + off_y))
        h, w = sprite.shape
        region = img[y:y + h, x:x + w]
        region[sprite] = ORANGE
        return img

    # Timing: frame 0 static, motion over frames 1..N-3, short hold at the goal.
    n_moves = len(path) - 1
    move_start, move_end = 1, N_FRAMES - 3
    frames = []
    for f in range(N_FRAMES):
        if f <= move_start - 1 or n_moves == 0:
            cx, cy = cell_center(path[0])
        elif f >= move_end:
            cx, cy = cell_center(path[-1])
        else:
            t = (f - move_start) / (move_end - move_start) * n_moves
            i = min(int(t), n_moves - 1)
            u = t - i
            u = u * u * (3 - 2 * u)  # smoothstep per move
            (x0, y0), (x1, y1) = cell_center(path[i]), cell_center(path[i + 1])
            cx, cy = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        frames.append(render(cx, cy))
    frames[0] = first.copy()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(td, f"{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "8", "-pix_fmt", "yuv420p",
            "-r", str(FPS), OUT,
        ], check=True)
    print(f"path {path[0]} -> {path[-1]} in {n_moves} moves; wrote {OUT}")


if __name__ == "__main__":
    main()
