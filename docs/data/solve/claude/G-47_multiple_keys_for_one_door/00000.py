#!/usr/bin/env python3
"""Animate the agent collecting all keys (optimal order) and reaching the door."""
import itertools
import subprocess
from collections import deque

import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 51

GREEN = (0, 255, 0)
WHITE = np.array([255, 255, 255], np.uint8)


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = img.shape

    # --- grid geometry: detect maze bounds and cell size from black/white pattern
    white = (img == 255).all(2)
    black = (img == 0).all(2)
    non_black = ~black
    ys, xs = np.nonzero(non_black)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # cell size: smallest run of column transitions in white/black mask along a mostly clean row
    trans = [x for x in range(x0 + 1, x1) if (white[:, x] != white[:, x - 1]).sum() > H // 4]
    steps = np.diff([x0] + trans + [x1])
    cell = int(np.min(steps[steps > 10]))
    ncols = round((x1 - x0) / cell)
    nrows = round((y1 - y0) / cell)

    def center(r, c):
        return (x0 + c * cell + cell // 2, y0 + r * cell + cell // 2)

    # --- classify cells & find objects
    free = np.zeros((nrows, ncols), bool)
    objs = {}  # colour -> (r, c)
    for r in range(nrows):
        for c in range(ncols):
            cx, cy = center(r, c)
            patch = img[cy - cell // 2 + 2: cy + cell // 2 - 2, cx - cell // 2 + 2: cx + cell // 2 - 2]
            if black[cy - cell // 2 + 2: cy + cell // 2 - 2, cx - cell // 2 + 2: cx + cell // 2 - 2].mean() > 0.9:
                continue
            free[r, c] = True
            cols, cnt = np.unique(patch.reshape(-1, 3), axis=0, return_counts=True)
            for col, n in zip(map(tuple, cols), cnt):
                if col not in ((0, 0, 0), (255, 255, 255)) and n > 100:
                    objs[col] = (r, c)

    start = objs.pop(GREEN)
    # door = hollow square (interior white); keys = filled diamonds
    door_col = None
    for col, (r, c) in objs.items():
        cx, cy = center(r, c)
        if white[cy, cx]:
            door_col = col
    door = objs.pop(door_col)
    keys = dict(objs)

    # --- BFS distances / paths
    def bfs(src):
        dist = {src: 0}
        prev = {}
        q = deque([src])
        while q:
            u = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < nrows and 0 <= v[1] < ncols and free[v] and v not in dist:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
        return dist, prev

    def path(a, b):
        dist, prev = bfs(a)
        p = [b]
        while p[-1] != a:
            p.append(prev[p[-1]])
        return p[::-1]

    nodes = [start] + list(keys.values()) + [door]
    D = {n: bfs(n)[0] for n in nodes}
    best = None
    for perm in itertools.permutations(keys.keys()):
        seq = [start] + [keys[k] for k in perm] + [door]
        total = sum(D[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
        if best is None or total < best[0]:
            best = (total, perm)
    total, order = best
    print("cells", nrows, ncols, "start", start, "keys", keys, "door", door)
    print("optimal order", order, "distance", total)

    full = [start]
    for k in list(order) + ["door"]:
        tgt = door if k == "door" else keys[k]
        full += path(full[-1], tgt)[1:]

    # --- sprites
    agent_mask = (img == GREEN).all(2)
    ay, ax = np.nonzero(agent_mask)
    sprite = agent_mask[ay.min(): ay.max() + 1, ax.min(): ax.max() + 1]
    scx, scy = center(*start)
    off_y, off_x = ay.min() - scy, ax.min() - scx  # sprite top-left relative to cell centre
    key_masks = {k: (img == k).all(2) for k in keys}

    base = img.copy()
    base[agent_mask] = WHITE
    for m in key_masks.values():
        base[m] = WHITE

    # --- frames
    frames = []
    L = len(full) - 1
    collected = set()
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1) * L
        i = min(int(np.floor(t)), L - 1)
        u = t - i
        (r0, c0), (r1, c1) = full[i], full[i + 1]
        p0, p1 = center(r0, c0), center(r1, c1)
        px = int(round(p0[0] + (p1[0] - p0[0]) * u))
        py = int(round(p0[1] + (p1[1] - p0[1]) * u))
        # key disappears the moment the agent reaches its cell
        reached = i + (1 if u > 1 - 1e-9 else 0)
        for k, cell_rc in keys.items():
            if k not in collected and cell_rc in full[: reached + 1]:
                collected.add(k)
        fr = base.copy()
        for k, m in key_masks.items():
            if k not in collected:
                fr[m] = np.array(k, np.uint8)
        sy, sx = py + off_y, px + off_x
        fr[sy: sy + sprite.shape[0], sx: sx + sprite.shape[1]][sprite] = np.array(GREEN, np.uint8)
        frames.append(fr)

    # sanity: first frame equals source
    assert (frames[0] == img).all(), "first frame mismatch"

    # --- encode
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT, len(frames), "frames")


if __name__ == "__main__":
    main()
