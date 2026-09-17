#!/usr/bin/env python3
"""Animate the agent from green to red through the yellow numbered waypoints.

Grid, waypoints and the agent sprite are detected from first_frame.png so
every pixel other than the moving agent stays untouched in every frame.
"""
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = Path(__file__).resolve().parent
FIRST = APP / "first_frame.png"
OUT = APP / "output" / "video.mp4"
FPS = 16
N_FRAMES = 113

GRAY = (150, 150, 150)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


def color_mask(img, rgb):
    return (img == np.array(rgb, dtype=np.uint8)).all(-1)


def detect_grid(img):
    """Return gridline start coordinates (x list, y list)."""
    g = color_mask(img, GRAY)
    h, w = g.shape
    rows = np.where(g.sum(1) > w * 0.5)[0]
    cols = np.where(g.sum(0) > h * 0.5)[0]

    def group_starts(a):
        out = []
        for v in a:
            if not out or v - out[-1][-1] > 1:
                out.append([v])
            else:
                out[-1].append(v)
        return [grp[0] for grp in out]

    return group_starts(cols), group_starts(rows)


def interior(m, c, r, xs0, ys0):
    return m[ys0[r] + 2:ys0[r + 1], xs0[c] + 2:xs0[c + 1]]


def find_cells(img, rgb, xs0, ys0):
    """Cells whose interior is mostly the given colour."""
    m = color_mask(img, rgb)
    return [(c, r) for r in range(len(ys0) - 1) for c in range(len(xs0) - 1)
            if interior(m, c, r, xs0, ys0).mean() > 0.5]


def waypoint_order(img, yellow_cells, xs0, ys0):
    """Order yellow cells by the digit drawn on them (template matching)."""
    dark = img.astype(int).sum(-1) < 200

    def crop(a):
        ys, xs = np.where(a)
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def norm(a, size=(24, 32)):
        im = Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        return np.asarray(im).astype(float) / 255.0

    font = None
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            font = ImageFont.truetype(path, 60)
            break
    templates = {}
    for d in range(1, 10):
        canvas = Image.new("L", (80, 100), 0)
        ImageDraw.Draw(canvas).text((10, 10), str(d), fill=255, font=font)
        templates[d] = norm(crop(np.asarray(canvas) > 128))

    scored = []
    for (c, r) in yellow_cells:
        g = norm(crop(interior(dark, c, r, xs0, ys0)))
        best = min(templates, key=lambda d: np.abs(templates[d] - g).mean())
        scored.append((best, (c, r)))
    scored.sort()
    return [cell for _, cell in scored]


def bfs_path(start, goal, ncols, nrows):
    """Shortest 4-connected path from start to goal (cells incl. both ends)."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        c, r = cur
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (c + dc, r + dr)
            if 0 <= nxt[0] < ncols and 0 <= nxt[1] < nrows and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def frame_times(n_moves, seg_lengths, n_frames, hold_end=6, pause=2):
    """Per-frame position along the path, in units of moves."""
    n_pauses = len(seg_lengths) - 1
    motion = n_frames - 1 - hold_end - pause * n_pauses
    step = n_moves / motion
    bounds = np.cumsum(seg_lengths)[:-1]
    times, t, reached = [0.0], 0.0, set()
    while len(times) < 1 + motion + pause * n_pauses:
        t = min(t + step, n_moves)
        times.append(t)
        for k, b in enumerate(bounds):
            if k not in reached and t >= b - 1e-9:
                reached.add(k)
                t = float(b)
                times.extend([t] * pause)
    times = times[:1 + motion + pause * n_pauses]
    times.extend([float(n_moves)] * (n_frames - len(times)))
    return times


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    H, W, _ = img.shape
    xs0, ys0 = detect_grid(img)
    ncols, nrows = len(xs0) - 1, len(ys0) - 1

    green = find_cells(img, GREEN, xs0, ys0)[0]
    red = find_cells(img, RED, xs0, ys0)[0]
    order = waypoint_order(img, find_cells(img, YELLOW, xs0, ys0), xs0, ys0)

    # Agent sprite: everything in the green cell box that is not green/grey.
    gc, gr = green
    cx0, cx1 = xs0[gc], min(xs0[gc + 1] + 2, W)
    cy0, cy1 = ys0[gr], min(ys0[gr + 1] + 2, H)
    cell = img[cy0:cy1, cx0:cx1]
    sprite_mask = ~(color_mask(cell, GREEN) | color_mask(cell, GRAY))
    sprite_rgb = cell.copy()

    # Background: first frame with the agent removed (green restored).
    bg = img.copy()
    bg[cy0:cy1, cx0:cx1][sprite_mask] = GREEN

    # Shortest path between each consecutive pair of stops.
    stops = [green] + order + [red]
    cells, seg_lengths = [green], []
    for a, b in zip(stops, stops[1:]):
        seg = bfs_path(a, b, ncols, nrows)
        cells.extend(seg[1:])
        seg_lengths.append(len(seg) - 1)
    n_moves = len(cells) - 1

    def ease(u):
        return u * u * (3 - 2 * u)

    frames = []
    for t in frame_times(n_moves, seg_lengths, N_FRAMES):
        i = min(int(np.floor(t)), n_moves - 1)
        u = min(t - i, 1.0)
        (c0, r0), (c1, r1) = cells[i], cells[i + 1]
        ue = ease(u)
        ox = int(round(xs0[c0] + (xs0[c1] - xs0[c0]) * ue))
        oy = int(round(ys0[r0] + (ys0[r1] - ys0[r0]) * ue))
        f = bg.copy()
        h, w = sprite_mask.shape
        region = f[oy:oy + h, ox:ox + w]
        region[sprite_mask[:region.shape[0], :region.shape[1]]] = \
            sprite_rgb[:region.shape[0], :region.shape[1]][sprite_mask[:region.shape[0], :region.shape[1]]]
        frames.append(f)
    frames[0] = img.copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           "-movflags", "+faststart", str(OUT)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    print(f"stops={stops} moves={n_moves} frames={len(frames)} -> {OUT}")


if __name__ == "__main__":
    main()
