#!/usr/bin/env python3
"""Solve the 15x15 maze in first_frame.png and render the path as a video."""
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N = 15
W = H = 1024
CS = W / N
FPS = 16
NFRAMES = 87


def detect_grid(a):
    grid = np.zeros((N, N), bool)  # True = open
    start = end = None
    for r in range(N):
        for c in range(N):
            y, x = int((r + 0.5) * CS), int((c + 0.5) * CS)
            w = a[y - 5:y + 6, x - 5:x + 6].reshape(-1, 3).astype(int)
            dark = (w.max(axis=1) < 60).mean()
            grid[r, c] = dark < 0.5
            if ((w[:, 1] > 150) & (w[:, 0] < 100) & (w[:, 2] < 100)).any():
                start = (r, c)
            if ((w[:, 0] > 150) & (w[:, 1] < 80) & (w[:, 2] < 80)).any():
                end = (r, c)
    return grid, start, end


def bfs(grid, start, end):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def center(cell):
    r, c = cell
    return ((c + 0.5) * CS, (r + 0.5) * CS)


def main():
    base = Image.open(SRC).convert("RGB")
    a = np.array(base)
    grid, start, end = detect_grid(a)
    path = bfs(grid, start, end)
    assert path and path[0] == start and path[-1] == end, "no path found"

    # Mask of the start/end markers so they stay pixel-identical on top of the path.
    green = (a[:, :, 1] > 150) & (a[:, :, 0] < 100) & (a[:, :, 2] < 100)
    red = (a[:, :, 0] > 150) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
    keep = green | red
    # also keep dark pixels (flag pole) inside the marker cells
    for cell in (start, end):
        r, c = cell
        y0, y1 = int(r * CS), int((r + 1) * CS)
        x0, x1 = int(c * CS), int((c + 1) * CS)
        sub = a[y0:y1, x0:x1]
        keep[y0:y1, x0:x1] |= sub.max(axis=2) < 120

    pts = [center(p) for p in path]
    total = len(pts) - 1
    line_w = int(CS * 0.32)
    color = (66, 133, 244)
    dot_r = int(CS * 0.28)

    frames = []
    for f in range(NFRAMES):
        t = f / (NFRAMES - 1) * total  # progress in path steps
        i = int(np.floor(t))
        frac = t - i
        if i >= total:
            i, frac = total, 0.0
        pos = pts[i]
        if frac > 0:
            nx = pts[i + 1]
            pos = (pos[0] + (nx[0] - pos[0]) * frac, pos[1] + (nx[1] - pos[1]) * frac)

        img = base.copy()
        if f > 0:
            d = ImageDraw.Draw(img)
            poly = pts[: i + 1] + [pos]
            if len(poly) >= 2:
                d.line(poly, fill=color, width=line_w, joint="curve")
            for p in poly[:-1]:
                d.ellipse([p[0] - line_w / 2, p[1] - line_w / 2,
                           p[0] + line_w / 2, p[1] + line_w / 2], fill=color)
            # moving marker
            d.ellipse([pos[0] - dot_r, pos[1] - dot_r, pos[0] + dot_r, pos[1] + dot_r],
                      fill=(30, 90, 200))
            arr = np.array(img)
            arr[keep] = a[keep]
            img = Image.fromarray(arr)
        frames.append(np.array(img))

    frames[0] = a.copy()
    raw = np.stack(frames).tobytes()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", OUT]
    subprocess.run(cmd, input=raw, check=True)
    print(f"path length {len(path)} cells, wrote {OUT} with {NFRAMES} frames")


if __name__ == "__main__":
    main()
