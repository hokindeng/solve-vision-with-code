#!/usr/bin/env python3
"""Move the blue triangular agent along the shortest directed path.

Graph read from first_frame.png (node centres in pixels):
  G  green start  (459, 268)      A white (310, 256)
  B  white        (770, 426)      C white (220, 706)
  D  white        (554, 699)      R red end (566, 558)
Directed edges: G->B, G->C, G->D (arrowhead hidden under R), C->A, C->R,
R->B, B->D.  BFS shortest path G -> R is G -> C -> R (2 steps).
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

NODES = {
    "G": (459, 268), "A": (310, 256), "B": (770, 426),
    "C": (220, 706), "D": (554, 699), "R": (566, 558),
}
EDGES = [("G", "B"), ("G", "C"), ("G", "D"), ("C", "A"),
         ("C", "R"), ("R", "B"), ("B", "D")]


def shortest_path(src, dst):
    adj = {n: [] for n in NODES}
    for u, v in EDGES:
        adj[u].append(v)
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    path, n = [], dst
    while n is not None:
        path.append(n)
        n = prev[n]
    return path[::-1]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    r, g, b = first[..., 0].astype(int), first[..., 1].astype(int), first[..., 2].astype(int)
    agent_mask = (b > 100) & (r < 60) & (g < 60)          # blue fill + dark-blue outline
    ys, xs = np.where(agent_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprite = first[y0:y1, x0:x1].copy()
    smask = agent_mask[y0:y1, x0:x1]
    gx, gy = NODES["G"]
    off = (x0 - gx, y0 - gy)                                # sprite top-left relative to node centre

    background = first.copy()
    background[agent_mask] = (0, 128, 0)                    # agent sits inside the green node

    path = shortest_path("G", "R")
    segs = len(path) - 1
    per = (N_FRAMES - 1) / segs

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(first.copy())
            continue
        s = min(int(i / per), segs - 1)
        t = ease(min(1.0, (i - s * per) / per))
        (ax, ay), (bx, by) = NODES[path[s]], NODES[path[s + 1]]
        cx = ax + (bx - ax) * t
        cy = ay + (by - ay) * t
        px, py = int(round(cx + off[0])), int(round(cy + off[1]))
        fr = background.copy()
        region = fr[py:py + sprite.shape[0], px:px + sprite.shape[1]]
        region[smask] = sprite[smask]
        frames.append(fr)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"f{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-x264-params", "keyint=1", OUT], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("path:", " -> ".join(path), "| wrote", OUT)


if __name__ == "__main__":
    main()
