#!/usr/bin/env python3
"""Animate the green agent collecting the red key and reaching the red door."""
import os, subprocess
from collections import deque
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")
FPS, NFRAMES = 16, 282
N, O, S = 17, 53, 54           # grid size, pixel origin, cell size

GREEN, RED, BLACK = (0, 255, 0), (255, 0, 0), (0, 0, 0)


def cell_of(xy):
    return (int((xy[1] - O) // S), int((xy[0] - O) // S))


def center(rc):
    return (O + S * rc[1] + S / 2.0, O + S * rc[0] + S / 2.0)


def bfs(grid, a, b):
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


def main():
    im = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = im.shape
    black = (im == BLACK).all(2)
    grid = np.zeros((N, N), bool)
    for r in range(N):
        for c in range(N):
            grid[r, c] = not black[O + S*r + S//2, O + S*c + S//2]

    # Agent: green circle -> find its cell and radius.
    gmask = (im == GREEN).all(2)
    ys, xs = np.nonzero(gmask)
    start = cell_of((xs.mean(), ys.mean()))
    radius = (xs.max() - xs.min() + 1) / 2.0

    # Red objects: the key is the solid diamond (more pixels), the door the hollow square.
    from scipy import ndimage
    rmask = (im == RED).all(2)
    lab, n = ndimage.label(rmask)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        comps.append((len(xs), cell_of((xs.mean(), ys.mean())), lab == i))
    comps.sort(key=lambda t: -t[0])
    key_cell, key_mask = comps[0][1], comps[0][2]
    door_cell = comps[1][1]

    p1 = bfs(grid, start, key_cell)
    p2 = bfs(grid, key_cell, door_cell)
    steps1, steps2 = len(p1) - 1, len(p2) - 1
    total = steps1 + steps2

    # Backgrounds: agent removed; after key pickup key removed too.
    bg = im.copy(); bg[gmask] = 255
    bg_after = bg.copy(); bg_after[key_mask] = 255

    # Timing: short holds at start, at pickup, and at the end; motion fills the rest.
    hold_start, hold_key, hold_end = 8, 10, 12
    move_frames = NFRAMES - hold_start - hold_key - hold_end
    fpm = move_frames / total
    b1 = round(steps1 * fpm)
    b2 = move_frames - b1

    yy, xx = np.mgrid[0:H, 0:W]

    def draw(bgimg, pos):
        cx, cy = pos
        f = bgimg.copy()
        m = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= radius ** 2
        f[m] = GREEN
        return f

    def pos_on(path, t):  # t in [0,1] along path
        t = min(max(t, 0.0), 1.0) * (len(path) - 1)
        i = int(np.floor(t)); i = min(i, len(path) - 2)
        a, b = center(path[i]), center(path[i + 1])
        u = t - i
        return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)

    frames = []
    for _ in range(hold_start):
        frames.append(im)
    for k in range(1, b1 + 1):
        frames.append(draw(bg, pos_on(p1, k / b1)))
    for _ in range(hold_key):
        frames.append(draw(bg_after, center(key_cell)))
    for k in range(1, b2 + 1):
        frames.append(draw(bg_after, pos_on(p2, k / b2)))
    for _ in range(hold_end):
        frames.append(draw(bg_after, center(door_cell)))
    frames[0] = im
    assert len(frames) == NFRAMES, len(frames)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close(); p.wait()
    print(f"start={start} key={key_cell} door={door_cell} steps={steps1}+{steps2} -> {OUT}")


if __name__ == "__main__":
    main()
