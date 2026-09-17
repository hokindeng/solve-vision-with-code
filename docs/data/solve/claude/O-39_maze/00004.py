#!/usr/bin/env python3
"""Solve the 15x15 maze in first_frame.png and render the walk as an mp4."""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N = 15
FPS = 16
FRAMES = 58
PATH_COLOR = (66, 133, 244)  # blue trail


def cell_bounds(i, size):
    step = size / N
    return int(i * step), int((i + 1) * step)


def read_grid(img):
    """Return open[r][c], start, end from the image colours."""
    h = img.shape[0]
    open_ = np.zeros((N, N), bool)
    start = end = None
    for r in range(N):
        y0, y1 = cell_bounds(r, h)
        for c in range(N):
            x0, x1 = cell_bounds(c, h)
            patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
            dark = (patch.max(1) < 60).mean()
            open_[r, c] = dark < 0.5
            green = ((patch[:, 1] > 150) & (patch[:, 0] < 100) & (patch[:, 2] < 100)).mean()
            red = ((patch[:, 0] > 150) & (patch[:, 1] < 100) & (patch[:, 2] < 100)).mean()
            if green > 0.05:
                start = (r, c)
            if red > 0.05:
                end = (r, c)
    return open_, start, end


def bfs(open_, start, end):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < N and 0 <= nb[1] < N and open_[nb] and nb not in prev:
                prev[nb] = cur
                q.append(nb)
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def center(cell, size):
    r, c = cell
    y0, y1 = cell_bounds(r, size)
    x0, x1 = cell_bounds(c, size)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def render(base, path, progress, markers):
    """Draw the trail up to `progress` (float, in path-step units) over base."""
    size = base.shape[0]
    im = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(im)
    pts = [center(p, size) for p in path]
    width = int(size / N * 0.3)
    full = int(progress)
    frac = progress - full
    seg = pts[: full + 1]
    if full < len(pts) - 1 and frac > 0:
        a, b = pts[full], pts[full + 1]
        seg.append((a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac))
    if len(seg) >= 2:
        draw.line(seg, fill=PATH_COLOR, width=width, joint="curve")
        rad = width / 2
        for (x, y) in (seg[0], seg[-1]):
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=PATH_COLOR)
    out = np.array(im)
    # keep the start marker and end flag untouched on top of the trail
    out[markers] = base[markers]
    return out


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    open_, start, end = read_grid(base)
    path = bfs(open_, start, end)
    print("start", start, "end", end, "path length", len(path) - 1)

    # non-white, non-dark pixels (marker, flag, pole) are preserved everywhere
    is_white = (base == 255).all(2)
    is_dark = (base.max(2) < 60) & (np.abs(base.astype(int) - 30).max(2) < 5)
    markers = ~(is_white | is_dark)

    steps = len(path) - 1
    frames = [base]
    for k in range(1, FRAMES):
        progress = steps * k / (FRAMES - 1)
        frames.append(render(base, path, progress, markers))

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{base.shape[1]}x{base.shape[0]}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
