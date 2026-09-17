#!/usr/bin/env python3
"""Move the blue triangular agent along the shortest directed path G -> W -> R.

The scene is taken from first_frame.png; the only pixels that change are the
agent triangle (erased from the start node and redrawn along the path).
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
FPS, N_FRAMES = 16, 30

# Node centres measured from first_frame.png (circle radius 74 px).
NODES = {"A": (376, 256), "G": (724, 347), "W": (586, 642), "R": (276, 792)}
# Directed edges, read from the arrowhead positions in the frame.
EDGES = [("A", "G"), ("A", "R"), ("R", "G"), ("G", "W"), ("W", "R")]
START, END = "G", "R"

# Agent triangle colours (fill + dark outline); the sprite itself is lifted
# pixel-exactly from first_frame.png so it is reproduced faithfully.
TRI_FILL, TRI_OUTLINE = (0, 0, 255), (0, 0, 139)
NODE_FILL_START = (0, 128, 0)


def shortest_path(start, end):
    adj = {}
    for u, v in EDGES:
        adj.setdefault(u, []).append(v)
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == end:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                q.append(v)
    path, n = [], end
    while n is not None:
        path.append(n)
        n = prev[n]
    return path[::-1]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def extract_sprite(arr):
    """Return (sprite_rgb, mask, (ox, oy)) where (ox, oy) is the sprite's
    top-left offset relative to the start-node centre."""
    mask = np.all(arr == TRI_FILL, axis=-1) | np.all(arr == TRI_OUTLINE, axis=-1)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sx, sy = NODES[START]
    return arr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), (x0 - sx, y0 - sy)


def draw_agent(frame, sprite, mask, off, cx, cy):
    ox, oy = off
    x0, y0 = cx + ox, cy + oy
    h, w = mask.shape
    region = frame[y0:y0 + h, x0:x0 + w]
    region[mask] = sprite[mask]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = Image.open(FIRST).convert("RGB")
    arr = np.array(first)

    # Background = first frame with the agent erased (restore the node fill).
    base = arr.copy()
    tri_mask = np.all(arr == TRI_FILL, axis=-1) | np.all(arr == TRI_OUTLINE, axis=-1)
    base[tri_mask] = NODE_FILL_START
    sprite, mask, off = extract_sprite(arr)

    path = shortest_path(START, END)
    hops = len(path) - 1

    # Schedule: hold at start, animate each hop with a short pause between,
    # hold at the end.
    hold_start, hold_end, pause = 2, 2, 2
    move_frames = N_FRAMES - hold_start - hold_end - pause * (hops - 1)
    per_hop = [move_frames // hops + (1 if i < move_frames % hops else 0) for i in range(hops)]

    positions = []
    positions += [NODES[START]] * hold_start
    for i in range(hops):
        (x0, y0), (x1, y1) = NODES[path[i]], NODES[path[i + 1]]
        n = per_hop[i]
        for k in range(1, n + 1):
            t = ease(k / n)
            positions.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        if i < hops - 1:
            positions += [NODES[path[i + 1]]] * pause
    positions += [NODES[END]] * hold_end
    positions = positions[:N_FRAMES]
    while len(positions) < N_FRAMES:
        positions.append(NODES[END])

    frames = []
    for i, (x, y) in enumerate(positions):
        if i == 0:
            frames.append(first.copy())
            continue
        img = base.copy()
        draw_agent(img, sprite, mask, off, int(round(x)), int(round(y)))
        frames.append(Image.fromarray(img))

    # Sanity: frame 0 reproduced by drawing must equal the original.
    chk = base.copy()
    draw_agent(chk, sprite, mask, off, *NODES[START])
    assert np.array_equal(chk, arr), "agent drawing does not match frame 0"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1024x1024", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.array(f).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"path {' -> '.join(path)}; wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
