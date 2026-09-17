#!/usr/bin/env python3
"""Animate the blue triangular agent along the shortest directed path.

Graph (read from first_frame.png):
    TOP(554,318) -> RED(204,270)
    TOP          -> BL(304,744)
    TOP          -> GREEN(708,750)
    GREEN        -> BL
    BL           -> RED
Shortest path from GREEN to RED: GREEN -> BL -> RED (2 steps).

The triangle sprite is lifted pixel-exactly from the first frame and
translated by integer offsets, so every other pixel stays unchanged.
"""
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
N_FRAMES = 30
GREEN = np.array([0, 128, 0], dtype=np.uint8)

# Node centres measured from the first frame (x, y).
NODES = {
    "GREEN": (708, 750),
    "BL": (304, 744),
    "TOP": (554, 318),
    "RED": (204, 270),
}
EDGES = [("TOP", "RED"), ("TOP", "BL"), ("TOP", "GREEN"),
         ("GREEN", "BL"), ("BL", "RED")]

# Triangle sprite bounding box in the first frame (inclusive).
SPR_X0, SPR_X1, SPR_Y0, SPR_Y1 = 678, 738, 720, 780


def shortest_path(start, goal):
    adj = {}
    for u, v in EDGES:
        adj.setdefault(u, []).append(v)
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                q.append(v)
    path, n = [], goal
    while n is not None:
        path.append(n)
        n = prev[n]
    return path[::-1]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # Lift the sprite (everything inside the bbox that is not node green).
    patch = base[SPR_Y0:SPR_Y1 + 1, SPR_X0:SPR_X1 + 1].copy()
    mask = np.any(patch != GREEN, axis=-1)
    sx0 = SPR_X0 - NODES["GREEN"][0]  # sprite offset relative to node centre
    sy0 = SPR_Y0 - NODES["GREEN"][1]

    # Background = first frame with the agent removed (node fill restored).
    bg = base.copy()
    region = bg[SPR_Y0:SPR_Y1 + 1, SPR_X0:SPR_X1 + 1]
    region[mask] = GREEN

    path = shortest_path("GREEN", "RED")
    pts = [NODES[n] for n in path]
    n_seg = len(pts) - 1

    # Frame 0 is the start; the last frame rests at the goal.
    moving = N_FRAMES - 1
    per_seg = moving // n_seg
    positions = []
    for i in range(N_FRAMES):
        if i == 0:
            positions.append(pts[0])
            continue
        k = min((i - 1) // per_seg, n_seg - 1)
        local = i - 1 - k * per_seg
        t = ease((local + 1) / per_seg) if k < n_seg - 1 else ease(
            min(1.0, (local + 1) / (moving - k * per_seg)))
        x = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * t
        y = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t
        positions.append((int(round(x)), int(round(y))))
    positions[-1] = pts[-1]

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
           "-preset", "slow", "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, (cx, cy) in enumerate(positions):
        frame = base.copy() if i == 0 else bg.copy()
        if i > 0:
            x0, y0 = cx + sx0, cy + sy0
            dst = frame[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
            dst[mask] = patch[mask]
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {N_FRAMES} frames, path {' -> '.join(path)}")


if __name__ == "__main__":
    main()
