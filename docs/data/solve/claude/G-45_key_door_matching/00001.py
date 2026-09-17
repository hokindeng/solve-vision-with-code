#!/usr/bin/env python3
"""Render the maze key/door video from first_frame.png."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 154

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)


def components(mask):
    """Simple 4-connected component labelling (no scipy dependency)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y0, x0 in zip(*np.where(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(np.array(pts))
    return comps


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = base.shape

    # ---- grid geometry from the white/black maze region -------------------
    nonblack = base.max(axis=2) > 0
    ys, xs = np.where(nonblack)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # Find the number of cells: cell size should be ~ (x1-x0)/n; use the
    # start circle diameter (~42px) as a hint that cells are ~68px.
    green = (base == GREEN).all(axis=2)
    gy, gx = np.where(green)
    circ_d = gx.max() - gx.min() + 1
    n_cells = int(round((x1 - x0) / (circ_d * 68 / 42)))
    cw = (x1 - x0) / n_cells
    ch = (y1 - y0) / n_cells

    def cell_center(r, c):
        return (x0 + (c + 0.5) * cw, y0 + (r + 0.5) * ch)

    def cell_of(px, py):
        return (int((py - y0) // ch), int((px - x0) // cw))

    # Passable = cell whose centre neighbourhood is not black.
    passable = np.zeros((n_cells, n_cells), dtype=bool)
    for r in range(n_cells):
        for c in range(n_cells):
            cx, cy = cell_center(r, c)
            patch = base[int(cy) - 3:int(cy) + 4, int(cx) - 3:int(cx) + 4]
            passable[r, c] = patch.max() > 0  # anything non-black (white / marker)

    # ---- locate agent, key and door ----------------------------------------
    start = cell_of(gx.mean(), gy.mean())
    yellow = (base == YELLOW).all(axis=2)
    ycomps = components(yellow)
    # The filled diamond has many more pixels than the hollow square.
    ycomps.sort(key=len)
    door_pts, key_pts = ycomps[0], ycomps[-1]
    key_cell = cell_of(key_pts[:, 1].mean(), key_pts[:, 0].mean())
    door_cell = cell_of(door_pts[:, 1].mean(), door_pts[:, 0].mean())

    # ---- BFS path planning -------------------------------------------------
    def bfs(a, b):
        prev = {a: None}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            r, c = cur
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < n_cells and 0 <= nb[1] < n_cells \
                        and passable[nb] and nb not in prev:
                    prev[nb] = cur
                    q.append(nb)
        path = []
        cur = b
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    leg1 = bfs(start, key_cell)
    leg2 = bfs(key_cell, door_cell)
    print(f"grid {n_cells}x{n_cells}, start {start}, key {key_cell}, door {door_cell}")
    print(f"leg1 {len(leg1) - 1} steps, leg2 {len(leg2) - 1} steps")

    # ---- timing --------------------------------------------------------------
    hold_start, hold_key, hold_end = 8, 10, 12
    move_frames = N_FRAMES - hold_start - hold_key - hold_end
    s1, s2 = len(leg1) - 1, len(leg2) - 1
    f1 = int(round(move_frames * s1 / (s1 + s2)))
    f2 = move_frames - f1

    def positions(path, nf):
        """nf frames moving along path at constant speed (excl. start pose)."""
        pts = np.array([cell_center(r, c) for r, c in path], dtype=float)
        seg = len(pts) - 1
        out = []
        for i in range(1, nf + 1):
            t = i / nf * seg
            k = min(int(t), seg - 1)
            u = t - k
            out.append(tuple(pts[k] * (1 - u) + pts[k + 1] * u))
        return out

    start_xy = cell_center(*start)
    frames_spec = []  # (x, y, key_visible)
    frames_spec += [(start_xy, True)] * hold_start
    frames_spec += [(p, True) for p in positions(leg1, f1)]
    frames_spec += [(cell_center(*key_cell), False)] * hold_key
    frames_spec += [(p, False) for p in positions(leg2, f2)]
    frames_spec += [(cell_center(*door_cell), False)] * hold_end
    assert len(frames_spec) == N_FRAMES

    # ---- rendering -------------------------------------------------------------
    # Background without the agent; and without the key once collected.
    bg = base.copy()
    bg[green] = WHITE
    bg_nokey = bg.copy()
    bg_nokey[key_pts[:, 0], key_pts[:, 1]] = WHITE
    radius = circ_d / 2.0

    def render(xy, key_visible):
        img = Image.fromarray(bg if key_visible else bg_nokey)
        d = ImageDraw.Draw(img)
        x, y = xy
        d.ellipse([x - radius, y - radius, x + radius - 1, y + radius - 1], fill=GREEN)
        return np.array(img)

    frames = [base]  # frame 0 is exactly the given first frame
    for spec in frames_spec[1:]:
        frames.append(render(*spec))

    # Sanity: our render of the start pose should match the original closely.
    diff = (render(start_xy, True) != base).any(axis=2).sum()
    print(f"start-pose render differs from first frame in {diff} px")

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
