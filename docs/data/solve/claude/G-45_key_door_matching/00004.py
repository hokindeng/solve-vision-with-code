#!/usr/bin/env python3
"""Animate the agent collecting the Yellow key and walking to the Yellow door."""
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
N_FRAMES = 52
CELL = 93
ORIGIN = 93
N = 9

GREEN = np.array([0, 255, 0])
YELLOW = np.array([255, 255, 0])
WHITE = np.array([255, 255, 255])


def parse_grid(a):
    grid = np.zeros((N, N), dtype=bool)  # True = walkable
    for r in range(N):
        for c in range(N):
            p = a[ORIGIN + r * CELL + 4, ORIGIN + c * CELL + 4]
            grid[r, c] = p.sum() > 100
    return grid


def cell_of(x, y):
    return int((y - ORIGIN) // CELL), int((x - ORIGIN) // CELL)


def components(mask):
    """Simple 4-connected component labelling (BFS) returning list of pixel index arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append(np.array(pts))
    return comps


def bfs_path(grid, start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def center(cell):
    r, c = cell
    return (ORIGIN + c * CELL + CELL / 2.0, ORIGIN + r * CELL + CELL / 2.0)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    grid = parse_grid(first)

    # Agent sprite: the green circle, stored as offsets relative to its centre.
    gmask = (first == GREEN).all(2)
    gy, gx = np.where(gmask)
    agent_cx, agent_cy = int(round(gx.mean())), int(round(gy.mean()))
    sprite_dy, sprite_dx = gy - agent_cy, gx - agent_cx
    start = cell_of(agent_cx, agent_cy)

    # Yellow objects: the diamond (key, larger filled) and the hollow square (door).
    ycomps = components((first == YELLOW).all(2))
    ycomps.sort(key=len, reverse=True)
    key_pts, door_pts = ycomps[0], ycomps[1]
    key_cell = cell_of(key_pts[:, 1].mean(), key_pts[:, 0].mean())
    door_cell = cell_of(door_pts[:, 1].mean(), door_pts[:, 0].mean())

    # Background without the agent (its cell is white underneath).
    bg = first.copy()
    bg[gy, gx] = WHITE
    bg_nokey = bg.copy()
    bg_nokey[key_pts[:, 0], key_pts[:, 1]] = WHITE

    # Route: start -> key -> door.
    p1 = bfs_path(grid, start, key_cell)
    p2 = bfs_path(grid, key_cell, door_cell)
    route = p1 + p2[1:]
    key_idx = len(p1) - 1  # index in route where the key is collected

    # Timing: hold at the start for 2 frames, at the end for 4 frames, move in between.
    hold_start, hold_end = 2, 4
    move_frames = N_FRAMES - hold_start - hold_end
    steps = len(route) - 1

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        t = min(max((f - hold_start) / float(move_frames), 0.0), 1.0)
        s = t * steps  # continuous position along the route
        i = min(int(np.floor(s)), steps - 1) if steps > 0 else 0
        frac = s - i if steps > 0 else 0.0
        (x0, y0), (x1, y1) = center(route[i]), center(route[min(i + 1, steps)])
        x, y = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
        cx, cy = int(round(x)), int(round(y))

        # Key collected once the agent has reached (or passed) the key cell.
        collected = s >= key_idx - 1e-9
        frame = (bg_nokey if collected else bg).copy()
        frame[sprite_dy + cy, sprite_dx + cx] = GREEN
        frames.append(frame)

    # Ensure first frame is identical to the source image.
    frames[0] = first.copy()

    # Encode with ffmpeg via raw RGB pipe.
    h, w = first.shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "12",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"route ({len(route)} cells): {route}")
    print(f"key at {key_cell}, door at {door_cell}; wrote {OUT}")


if __name__ == "__main__":
    main()
