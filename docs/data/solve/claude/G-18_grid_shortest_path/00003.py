#!/usr/bin/env python3
"""Animate the red agent moving along the shortest path from the orange
start square to the blue end square on the 10x10 grid in first_frame.png."""
import os, subprocess, tempfile
from collections import deque
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N_FRAMES, FPS = 45, 16
ORANGE, RED, BLUE = (255, 165, 0), (255, 0, 0), (0, 0, 255)
CELL, N = 102, 10  # grid pitch in px, grid size


def cell_of(mask):
    ys, xs = np.where(mask)
    return int(ys.mean() // CELL), int(xs.mean() // CELL)


def shortest_path(start, goal, blocked):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in prev and nxt not in blocked:
                prev[nxt] = cur
                q.append(nxt)
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def smoothstep(x):
    return x * x * (3 - 2 * x)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    red = (base == RED).all(2)
    start = cell_of((base == ORANGE).all(2) | red)
    goal = cell_of((base == BLUE).all(2))
    path = shortest_path(start, goal, blocked=set())

    # Agent sprite: red pixels relative to start cell origin.
    ys, xs = np.where(red)
    sr, sc = start
    rel_y, rel_x = ys - sr * CELL, xs - sc * CELL
    # Background with the agent lifted off (restore the orange start square).
    bg = base.copy()
    bg[red] = ORANGE

    steps = len(path) - 1
    hold_start, hold_end = 3, 4
    move_frames = N_FRAMES - hold_start - hold_end

    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(base.copy())
            continue
        t = np.clip((f - hold_start + 1) / move_frames * steps, 0, steps)
        i = min(int(np.floor(t)), steps - 1)
        u = smoothstep(t - i)
        (r0, c0), (r1, c1) = path[i], path[i + 1]
        oy = int(round(((1 - u) * r0 + u * r1) * CELL))
        ox = int(round(((1 - u) * c0 + u * c1) * CELL))
        img = bg.copy()
        img[rel_y + oy, rel_x + ox] = RED
        frames.append(img)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for k, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(td, f"{k:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
        ], check=True)
    print(f"path={path}\nwrote {OUT}")


if __name__ == "__main__":
    main()
