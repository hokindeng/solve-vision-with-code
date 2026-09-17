#!/usr/bin/env python3
"""Animate the orange marker from the green start cell through the numbered
yellow cells (in ascending order) to the red end cell, using shortest
4-neighbour paths between consecutive targets. Everything else in the frame
stays exactly as in first_frame.png."""
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 89
CELL = 102          # grid pitch in pixels (2 px line + 100 px interior)
N = 10              # 10x10 grid
DWELL = 3           # frames to pause on each numbered target

ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")


def cell_center(c):
    return (c[0] * CELL + CELL / 2, c[1] * CELL + CELL / 2)


def interior(base, c):
    x0, y0 = c[0] * CELL, c[1] * CELL
    return base[y0 + 4:y0 + CELL - 2, x0 + 4:x0 + CELL - 2]


def classify_cells(base):
    """Return start, end and list of (cell, glyph_mask) for yellow cells."""
    start = end = None
    yellow = []
    for cx in range(N):
        for cy in range(N):
            sub = interior(base, (cx, cy))
            corner = tuple(int(v) for v in sub[0, 0])
            if corner == GREEN:
                start = (cx, cy)
            elif corner == RED:
                end = (cx, cy)
            elif corner == YELLOW:
                dark = sub.sum(axis=2) < 200
                ys, xs = np.where(dark)
                yellow.append(((cx, cy), dark[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return start, end, yellow


def digit_templates(n):
    font = None
    for p in FONT_PATHS:
        try:
            font = ImageFont.truetype(p, 48)
            break
        except OSError:
            pass
    temps = []
    for d in range(1, n + 1):
        im = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(im).text((16, 16), str(d), fill=255, font=font)
        arr = np.array(im) > 128
        ys, xs = np.where(arr)
        temps.append(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return temps


def order_targets(yellow):
    """Order yellow cells by the numeral drawn in them via template matching."""
    n = len(yellow)
    temps = digit_templates(n)
    scores = np.zeros((n, n))
    for i, (_, mask) in enumerate(yellow):
        h, w = mask.shape
        for j, t in enumerate(temps):
            ref = np.array(Image.fromarray(t.astype(np.uint8) * 255)
                           .resize((w, h), Image.BILINEAR)) > 128
            scores[i, j] = (ref == mask).mean()
    order = [None] * n
    used_c, used_d = set(), set()
    for _ in range(n):
        cands = [(scores[i, j], i, j) for i in range(n) if i not in used_c
                 for j in range(n) if j not in used_d]
        _, i, j = max(cands)
        order[j] = yellow[i][0]
        used_c.add(i)
        used_d.add(j)
    return order


def bfs_path(a, b):
    """Shortest 4-neighbour path on the open grid (horizontal moves first)."""
    prev = {a: None}
    q = deque([a])
    while q and b not in prev:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in prev:
                prev[nxt] = (x, y)
                q.append(nxt)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def make_schedule(n_steps, stop_idx):
    """Per-frame position parameter s in [0, n_steps], pausing DWELL frames at
    each intermediate stop, ending exactly at n_steps on the last frame."""
    inner = stop_idx[1:-1]
    move_frames = (N_FRAMES - 1) - DWELL * len(inner)
    ds = n_steps / move_frames
    sched, s, k = [0.0], 0.0, 0
    while len(sched) < N_FRAMES:
        s_new = s + ds
        if k < len(inner) and s < inner[k] <= s_new + 1e-9:
            s_new = float(inner[k])
            sched.extend([s_new] * (DWELL + 1))
            k += 1
        else:
            sched.append(s_new)
        s = s_new
    sched = sched[:N_FRAMES]
    sched[-1] = float(n_steps)
    return sched


def draw_disc(img, x, y, radius, ss=4):
    pad = int(radius) + 3
    px0, py0 = int(np.floor(x)) - pad, int(np.floor(y)) - pad
    size = 2 * pad + 1
    patch = Image.new("L", (size * ss, size * ss), 0)
    lx, ly, r = (x - px0) * ss, (y - py0) * ss, radius * ss
    ImageDraw.Draw(patch).ellipse([lx - r, ly - r, lx + r, ly + r], fill=255)
    alpha = patch.resize((size, size), Image.LANCZOS)
    img.paste(Image.new("RGB", (size, size), ORANGE), (px0, py0), alpha)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W, _ = base.shape

    start, end, yellow = classify_cells(base)
    targets = order_targets(yellow)

    # marker geometry
    ys, xs = np.where(np.all(base == ORANGE, axis=2))
    radius = (xs.max() - xs.min() + 1) / 2

    # background with the marker removed (restore the green start cell)
    bg = base.copy()
    gy, gx = np.where(np.all(base == GREEN, axis=2))
    bg[gy.min():gy.max() + 1, gx.min():gx.max() + 1] = GREEN

    # full route through the waypoints
    waypoints = [start] + targets + [end]
    route, stop_idx = [start], [0]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        route.extend(bfs_path(a, b)[1:])
        stop_idx.append(len(route) - 1)
    n_steps = len(route) - 1

    schedule = make_schedule(n_steps, stop_idx)

    frames = [base.copy()]  # first frame identical to the source image
    for s in schedule[1:]:
        i = min(int(np.floor(s)), n_steps - 1)
        t = s - i
        p0, p1 = cell_center(route[i]), cell_center(route[i + 1])
        x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
        img = Image.fromarray(bg)
        draw_disc(img, x, y, radius)
        frames.append(np.array(img))

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"waypoints {waypoints}: {n_steps} steps, {len(frames)} frames -> {OUT}")


if __name__ == "__main__":
    main()
