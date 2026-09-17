#!/usr/bin/env python3
"""Render a sliding-puzzle solution video from /app/first_frame.png."""
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

# Geometry measured from first_frame.png
GRID_LINES = [20, 348, 675, 1003]      # 2px dark lines at these coordinates
CELL_INT = [(22, 347), (350, 674), (677, 1002)]  # inclusive cell interiors
TILE_ORIGIN = [36, 364, 692]           # tile top-left (incl. border) per row/col
TILE_SIZE = 296

START = (2, 3, 6,
         1, 5, 0,
         7, 8, 4)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
REQUIRED_MOVES = 11


def neighbors(s):
    b = s.index(0)
    r, c = divmod(b, N)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            j = nr * N + nc
            l = list(s)
            l[b], l[j] = l[j], l[b]
            yield tuple(l), j  # new state, index of tile that moved


def solve(start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            break
        for n, j in neighbors(s):
            if n not in prev:
                prev[n] = (s, j)
                q.append(n)
    moves = []
    s = goal
    while prev[s] is not None:
        s, j = prev[s]
        moves.append(j)
    return moves[::-1]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # Extract tile sprites from the first frame.
    sprites = {}
    for idx, v in enumerate(START):
        if v == 0:
            continue
        r, c = divmod(idx, N)
        y0, x0 = TILE_ORIGIN[r], TILE_ORIGIN[c]
        sprites[v] = base[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE].copy()

    # Background: first frame with every cell interior blanked to white.
    bg = base.copy()
    for (y0, y1) in CELL_INT:
        for (x0, x1) in CELL_INT:
            bg[y0:y1 + 1, x0:x1 + 1] = 255

    # Grid-line mask so lines stay on top of a sliding tile.
    grid_mask = np.zeros((H, W), dtype=bool)
    gy0, gy1 = GRID_LINES[0], GRID_LINES[-1] + 1
    for g in GRID_LINES:
        grid_mask[g:g + 2, gy0:gy1] = True
        grid_mask[gy0:gy1, g:g + 2] = True

    moves = solve(START, GOAL)
    assert len(moves) == REQUIRED_MOVES, len(moves)

    # Timing: hold, then 11 moves of (5 motion frames + 1 rest), then final hold.
    PRE_HOLD = 4
    MOTION = 5
    REST = 1
    per_move = MOTION + REST
    post_hold = N_FRAMES - PRE_HOLD - per_move * len(moves)
    assert post_hold >= 1

    def render(state, moving=None):
        """state: tuple; moving: (tile_value, from_idx, to_idx, t)."""
        img = bg.copy()
        mv_val = moving[0] if moving else None
        for idx, v in enumerate(state):
            if v == 0 or v == mv_val:
                continue
            r, c = divmod(idx, N)
            y0, x0 = TILE_ORIGIN[r], TILE_ORIGIN[c]
            img[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = sprites[v]
        if moving:
            v, fi, ti, t = moving
            fr, fc = divmod(fi, N)
            tr, tc = divmod(ti, N)
            y = TILE_ORIGIN[fr] + (TILE_ORIGIN[tr] - TILE_ORIGIN[fr]) * t
            x = TILE_ORIGIN[fc] + (TILE_ORIGIN[tc] - TILE_ORIGIN[fc]) * t
            y, x = int(round(y)), int(round(x))
            img[y:y + TILE_SIZE, x:x + TILE_SIZE] = sprites[v]
        img[grid_mask] = base[grid_mask]
        return img

    frames = []
    state = START
    for _ in range(PRE_HOLD):
        frames.append(render(state))
    for j in moves:
        b = state.index(0)
        v = state[j]
        l = list(state)
        l[b], l[j] = l[j], l[b]
        new_state = tuple(l)
        for k in range(1, MOTION + 1):
            t = ease(k / MOTION)
            frames.append(render(state, (v, j, b, t)))
        state = new_state
        for _ in range(REST):
            frames.append(render(state))
    assert state == GOAL
    while len(frames) < N_FRAMES:
        frames.append(render(state))
    frames = frames[:N_FRAMES]

    # Frame 0 must be identical to the input.
    assert np.array_equal(frames[0], base)

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
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames, moves={moves}")


if __name__ == "__main__":
    main()
