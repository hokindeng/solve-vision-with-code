#!/usr/bin/env python3
"""Render a sliding-puzzle solution video from first_frame.png."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 76
N = 3
TILE_ORIGIN = 36      # pixel origin of the tile in cell (0, 0)
CELL_PITCH = 328      # distance between tile origins
TILE_SIZE = 296

# Initial board as read from first_frame.png (0 = blank).
START = (4, 0, 2,
         8, 1, 3,
         5, 7, 6)
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)
REQUIRED_MOVES = 11


def neighbors(state):
    b = state.index(0)
    r, c = divmod(b, N)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            i = nr * N + nc
            s = list(state)
            s[b], s[i] = s[i], s[b]
            yield tuple(s), state[i]


def solve(start, goal):
    """BFS -> list of tile values to slide, in order (shortest solution)."""
    prev = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            break
        for nxt, tile in neighbors(s):
            if nxt not in prev:
                prev[nxt] = (s, tile)
                q.append(nxt)
    moves = []
    s = goal
    while prev[s] is not None:
        s, tile = prev[s]
        moves.append(tile)
    return moves[::-1]


def tile_xy(r, c):
    return TILE_ORIGIN + CELL_PITCH * c, TILE_ORIGIN + CELL_PITCH * r


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    frame0 = np.array(Image.open(FIRST).convert("RGB"))
    moves = solve(START, GOAL)
    assert len(moves) == REQUIRED_MOVES, f"solution has {len(moves)} moves"

    # Extract each tile's sprite and build the static background (tiles removed).
    sprites = {}
    pos = {}  # tile value -> (row, col)
    bg = frame0.copy()
    for i, v in enumerate(START):
        r, c = divmod(i, N)
        if v == 0:
            continue
        x, y = tile_xy(r, c)
        sprites[v] = frame0[y:y + TILE_SIZE, x:x + TILE_SIZE].copy()
        bg[y:y + TILE_SIZE, x:x + TILE_SIZE] = 255
        pos[v] = (r, c)

    # Precompute per-move source/destination cells.
    blank = divmod(START.index(0), N)
    plan = []
    cur = dict(pos)
    for v in moves:
        src = cur[v]
        dst = blank
        plan.append((v, src, dst))
        cur[v] = dst
        blank = src
    # Sanity: final arrangement equals GOAL.
    final = [0] * (N * N)
    for v, (r, c) in cur.items():
        final[r * N + c] = v
    assert tuple(final) == GOAL

    # Time slots: frame 0 is the untouched first frame; moves fill frames 1..N_FRAMES-1.
    span = N_FRAMES - 1
    bounds = [1 + span * k / REQUIRED_MOVES for k in range(REQUIRED_MOVES + 1)]
    bounds[-1] = N_FRAMES - 1

    frames = []
    for f in range(N_FRAMES):
        img = bg.copy()
        cur = dict(pos)
        moving = None
        for k, (v, src, dst) in enumerate(plan):
            t0, t1 = bounds[k], bounds[k + 1]
            if f >= t1:
                cur[v] = dst
            elif f > t0:
                p = ease((f - t0) / (t1 - t0))
                moving = (v, src, dst, p)
                break
            else:
                break
        for v, (r, c) in cur.items():
            if moving and v == moving[0]:
                continue
            x, y = tile_xy(r, c)
            img[y:y + TILE_SIZE, x:x + TILE_SIZE] = sprites[v]
        if moving:
            v, (r0, c0), (r1, c1), p = moving
            x0, y0 = tile_xy(r0, c0)
            x1, y1 = tile_xy(r1, c1)
            x = int(round(x0 + (x1 - x0) * p))
            y = int(round(y0 + (y1 - y0) * p))
            img[y:y + TILE_SIZE, x:x + TILE_SIZE] = sprites[v]
        frames.append(img)

    assert np.array_equal(frames[0], frame0)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0
    print(f"moves ({len(moves)}): {moves}")
    print(f"wrote {OUT} ({len(frames)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
