#!/usr/bin/env python3
"""Animate the green agent along a shortest path from the orange start to the red end square."""
import os, subprocess
import numpy as np
from PIL import Image
from collections import deque

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 60
N, CELL = 10, 102  # 10x10 grid, 102 px pitch (2 px lines)

ORANGE, RED, GREEN = (255, 165, 0), (255, 0, 0), (0, 255, 0)

def cell_of(mask):
    ys, xs = np.where(mask)
    return int(xs.mean() // CELL), int(ys.mean() // CELL)  # (col, row)

def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    green = (base == GREEN).all(2)
    orange = (base == ORANGE).all(2)
    red = (base == RED).all(2)

    start = cell_of(orange)
    end = cell_of(red)

    # Extract agent sprite (exact pixels) and its bounding box offset.
    ys, xs = np.where(green)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprite_mask = green[y0:y1, x0:x1]
    sprite_rgb = base[y0:y1, x0:x1].copy()

    # Background: first frame with the agent removed (start square is solid orange underneath).
    bg = base.copy()
    bg[green] = ORANGE

    # BFS shortest path on the 4-connected grid (no obstacles in this scene).
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == end:
            break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dc, c[1] + dr)
            if 0 <= n[0] < N and 0 <= n[1] < N and n not in prev:
                prev[n] = c
                q.append(n)
    path = []
    c = end
    while c is not None:
        path.append(c)
        c = prev[c]
    path.reverse()
    n_moves = len(path) - 1

    # Timing: short hold at start and end, smooth motion in between.
    hold_start, hold_end = 4, 6
    move_frames = N_FRAMES - hold_start - hold_end

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        t = (f - hold_start) / move_frames
        t = min(max(t, 0.0), 1.0)
        s = t * n_moves
        i = min(int(s), n_moves - 1)
        u = s - i
        u = u * u * (3 - 2 * u)  # ease each step slightly
        (ca, ra), (cb, rb) = path[i], path[i + 1]
        col = ca + (cb - ca) * u
        row = ra + (rb - ra) * u
        dx = int(round((col - start[0]) * CELL))
        dy = int(round((row - start[1]) * CELL))
        img = bg.copy()
        ty, tx = y0 + dy, x0 + dx
        region = img[ty:ty + sprite_mask.shape[0], tx:tx + sprite_mask.shape[1]]
        region[sprite_mask] = sprite_rgb[sprite_mask]
        frames.append(img)

    # Frame 0 must equal the first frame exactly.
    assert np.array_equal(frames[0], base)

    for k, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(OUT_DIR, f"frame_{k:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(OUT_DIR, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-vf", "scale=1024:1024", OUT], check=True)
    for k in range(N_FRAMES):
        os.remove(os.path.join(OUT_DIR, f"frame_{k:03d}.png"))
    print(f"path {path}\nwrote {OUT}")

if __name__ == "__main__":
    main()
