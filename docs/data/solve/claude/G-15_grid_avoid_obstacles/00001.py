#!/usr/bin/env python3
"""Animate the yellow agent along the shortest obstacle-free path to the red goal."""
import subprocess
from collections import deque

import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 58

# Grid geometry measured from first_frame.png: lines at 47+93*k (2 px wide),
# cell interiors are 90x90 starting at 49+93*k.
ORIGIN, PITCH, CELL = 49, 93, 90
N = 10

BLUE = np.array([0, 100, 255])
RED = np.array([255, 50, 50])
BLACK = np.array([0, 0, 0])


def cell_box(c, r):
    x0, y0 = ORIGIN + PITCH * c, ORIGIN + PITCH * r
    return x0, y0, x0 + CELL, y0 + CELL


def classify(img):
    start = end = None
    obstacles = set()
    for r in range(N):
        for c in range(N):
            x0, y0, x1, y1 = cell_box(c, r)
            cell = img[y0:y1, x0:x1]
            if (cell == BLUE).all(axis=2).sum() > 500:
                start = (c, r)
            elif (cell == RED).all(axis=2).sum() > 500:
                end = (c, r)
            elif (cell == BLACK).all(axis=2).sum() > 50:
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
        for dc, dr in ((1, 0), (0, -1), (-1, 0), (0, 1)):
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
    first = np.array(Image.open(SRC).convert("RGB"))
    start, end, obstacles = classify(first)
    path = bfs(start, end, obstacles)

    # Extract agent sprite (everything non-blue inside the start cell).
    x0, y0, x1, y1 = cell_box(*start)
    cell = first[y0:y1, x0:x1]
    mask = ~(cell == BLUE).all(axis=2)
    ys, xs = np.where(mask)
    sy0, sy1, sx0, sx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprite = cell[sy0:sy1, sx0:sx1].copy()
    smask = mask[sy0:sy1, sx0:sx1]
    # Sprite offset relative to the cell's top-left corner.
    off_x, off_y = sx0, sy0

    # Background: first frame with the agent removed (start cell fully blue).
    bg = first.copy()
    bg[y0:y1, x0:x1][mask] = BLUE

    def cell_origin(c, r):
        return ORIGIN + PITCH * c, ORIGIN + PITCH * r

    # Timing: short hold, then steps evenly spaced, then hold on the goal.
    steps = len(path) - 1
    hold_start = 4
    hold_end = 6
    move_frames = N_FRAMES - hold_start - hold_end
    frames = []
    for f in range(N_FRAMES):
        if f == 0:
            frames.append(first)
            continue
        t = (f - hold_start) / move_frames * steps
        t = min(max(t, 0.0), float(steps))
        i = min(int(np.floor(t)), steps - 1)
        u = t - i
        u = u * u * (3 - 2 * u)  # smoothstep within each step
        ax, ay = cell_origin(*path[i])
        bx, by = cell_origin(*path[i + 1])
        px = int(round(ax + (bx - ax) * u)) + off_x
        py = int(round(ay + (by - ay) * u)) + off_y
        frame = bg.copy()
        region = frame[py:py + sprite.shape[0], px:px + sprite.shape[1]]
        region[smask] = sprite[smask]
        frames.append(frame)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("path:", path)
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
