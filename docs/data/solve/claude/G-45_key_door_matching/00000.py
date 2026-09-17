#!/usr/bin/env python3
"""Animate the agent collecting the Red key and walking to the Red door."""
import subprocess, os
from collections import deque
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 198
OFF, CELL, N = 68, 68, 13          # maze geometry: 13x13 cells of 68 px, 68 px border

AGENT = (0, 255, 0)
TARGET = (255, 0, 0)               # Red key / Red door

def cell_center(rc):
    r, c = rc
    return OFF + c * CELL + CELL // 2, OFF + r * CELL + CELL // 2   # (x, y)

def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # --- grid: a cell is a wall only if its centre is black ---------------
    grid = np.zeros((N, N), bool)  # True = walkable
    for r in range(N):
        for c in range(N):
            x, y = cell_center((r, c))
            grid[r, c] = tuple(base[y, x]) != (0, 0, 0)

    def cell_of(mask):
        ys, xs = np.nonzero(mask)
        return (int((ys.mean() - OFF) // CELL), int((xs.mean() - OFF) // CELL))

    def cell_mask(rc):
        m = np.zeros((H, W), bool)
        r, c = rc
        m[OFF + r * CELL:OFF + (r + 1) * CELL, OFF + c * CELL:OFF + (c + 1) * CELL] = True
        return m

    # --- locate agent, key (solid diamond) and door (hollow square) ---------
    agent_mask = np.all(base == AGENT, axis=2)
    start = cell_of(agent_mask)
    red = np.all(base == TARGET, axis=2)
    # split red pixels into connected cells; the filled one is the key
    red_cells = {}
    ys, xs = np.nonzero(red)
    for y, x in zip(ys, xs):
        red_cells.setdefault(((y - OFF) // CELL, (x - OFF) // CELL), 0)
        red_cells[((y - OFF) // CELL, (x - OFF) // CELL)] += 1
    key_cell = max(red_cells, key=red_cells.get)      # diamond has more pixels
    door_cell = min(red_cells, key=red_cells.get)
    key_mask = red & cell_mask(key_cell)

    # agent sprite (offsets relative to its cell centre)
    ay, ax = np.nonzero(agent_mask)
    sx, sy = cell_center(start)
    sprite_dx, sprite_dy = ax - sx, ay - sy

    # --- BFS ------------------------------------------------------------------
    def bfs(a, b):
        prev = {a: None}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            r, c = cur
            for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] and (nr, nc) not in prev:
                    prev[(nr, nc)] = cur
                    q.append((nr, nc))
        path, cur = [], b
        while cur is not None:
            path.append(cur); cur = prev[cur]
        return path[::-1]

    path1 = bfs(start, key_cell)
    path2 = bfs(key_cell, door_cell)
    n1, n2 = len(path1) - 1, len(path2) - 1

    # --- timing ---------------------------------------------------------------
    hold_start, hold_key, hold_end = 8, 6, 12
    move_frames = N_FRAMES - hold_start - hold_key - hold_end
    f1 = round(move_frames * n1 / (n1 + n2))
    f2 = move_frames - f1

    def positions(path, nf):
        """nf frames going from path[0] to path[-1] (excluding the start pose)."""
        pts = [cell_center(p) for p in path]
        out = []
        for i in range(1, nf + 1):
            t = i / nf * (len(pts) - 1)
            k = min(int(t), len(pts) - 2)
            u = t - k
            x = pts[k][0] + (pts[k+1][0] - pts[k][0]) * u
            y = pts[k][1] + (pts[k+1][1] - pts[k][1]) * u
            out.append((x, y))
        return out

    timeline = []   # (pos, key_collected)
    p0 = cell_center(start)
    timeline += [(p0, False)] * hold_start
    timeline += [(p, False) for p in positions(path1, f1)]
    timeline += [(cell_center(key_cell), True)] * hold_key
    timeline += [(p, True) for p in positions(path2, f2)]
    timeline += [(cell_center(door_cell), True)] * hold_end
    assert len(timeline) == N_FRAMES

    # --- render ---------------------------------------------------------------
    background = base.copy()
    background[agent_mask] = (255, 255, 255)    # agent sits on a white path cell
    bg_nokey = background.copy()
    bg_nokey[key_mask] = (255, 255, 255)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUT],
        stdin=subprocess.PIPE)
    for i, ((x, y), collected) in enumerate(timeline):
        frame = (bg_nokey if collected else background).copy()
        cx, cy = int(round(x)), int(round(y))
        frame[sprite_dy + cy, sprite_dx + cx] = AGENT
        if i == 0:
            assert np.array_equal(frame, base)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"path to key: {n1} steps, key->door: {n2} steps; wrote {OUT}")

if __name__ == "__main__":
    main()
