#!/usr/bin/env python3
"""Render the sliding-puzzle solution video from first_frame.png."""
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
# Tile geometry measured from first_frame.png: tiles (incl. dark border) are
# 296x296 px and sit at 36 + 328*k on both axes.
TILE0 = 36
PITCH = 328
TILE = 296

START = [[8, 0, 2],
         [1, 4, 3],
         [7, 6, 5]]
GOAL = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
MOVES_REQUIRED = 11


def tile_xy(r, c):
    return TILE0 + PITCH * c, TILE0 + PITCH * r


def find_blank(state):
    for r in range(N):
        for c in range(N):
            if state[r][c] == 0:
                return r, c


def neighbors(state):
    r, c = find_blank(state)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            s = [list(row) for row in state]
            s[r][c], s[nr][nc] = s[nr][nc], 0
            yield (nr, nc), tuple(tuple(row) for row in s)


def solve(start, goal, exact_len):
    """BFS for a shortest path; extend with a detour if it is shorter than exact_len."""
    start = tuple(tuple(row) for row in start)
    prev = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            break
        for (tr, tc), ns in neighbors(s):
            if ns not in prev:
                prev[ns] = (s, (tr, tc))
                q.append(ns)
    path = []
    s = goal
    while prev[s] is not None:
        s, mv = prev[s]
        path.append(mv)
    path.reverse()
    if len(path) > exact_len or (exact_len - len(path)) % 2:
        raise RuntimeError(f"cannot solve in exactly {exact_len} moves (shortest {len(path)})")
    # Pad with a back-and-forth on the first move if a longer path is required.
    while len(path) < exact_len:
        br, bc = find_blank(start)
        first = path[0]
        path = [first, (br, bc)] + path
    return path


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # Extract tile sprites and build a tile-free background.
    sprites = {}
    bg = base.copy()
    for r in range(N):
        for c in range(N):
            v = START[r][c]
            x, y = tile_xy(r, c)
            if v:
                sprites[v] = base[y:y + TILE, x:x + TILE].copy()
                bg[y:y + TILE, x:x + TILE] = 255

    moves = solve(START, GOAL, MOVES_REQUIRED)
    assert len(moves) == MOVES_REQUIRED

    # Frame schedule: hold, then 11 moves of 6 frames each, then hold.
    move_frames = 6
    lead = 3
    total_anim = MOVES_REQUIRED * move_frames
    assert lead + total_anim < N_FRAMES

    def ease(t):
        return t * t * (3 - 2 * t)

    state = [row[:] for row in START]
    frames = []
    for f in range(N_FRAMES):
        img = bg.copy()
        moving = None
        k = f - lead
        if 0 <= k < total_anim:
            mi, sub = divmod(k, move_frames)
            tr, tc = moves[mi]
            br, bc = find_blank(state)
            t = ease((sub + 1) / move_frames)
            x0, y0 = tile_xy(tr, tc)
            x1, y1 = tile_xy(br, bc)
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            moving = (state[tr][tc], x, y)
            if sub == move_frames - 1:
                state[br][bc], state[tr][tc] = state[tr][tc], 0
            else:
                pass
        # Draw resting tiles (skip the moving one).
        for r in range(N):
            for c in range(N):
                v = state[r][c]
                if v and not (moving and moving[0] == v):
                    x, y = tile_xy(r, c)
                    img[y:y + TILE, x:x + TILE] = sprites[v]
        if moving:
            v, x, y = moving
            img[y:y + TILE, x:x + TILE] = sprites[v]
        frames.append(img)

    assert tuple(tuple(r) for r in state) == GOAL
    assert np.array_equal(frames[0], base)

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "6", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames, moves={moves}")


if __name__ == "__main__":
    main()
