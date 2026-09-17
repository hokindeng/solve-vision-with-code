#!/usr/bin/env python3
"""Animate Pac-Man along the max-cost orthogonal path on the 4x4 cost grid."""
import subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024
FPS = 16
N_FRAMES = 91
CELL = 256
GREEN = (76, 175, 80)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
R = 64            # Pac-Man radius (outer edge of outline)
OUTLINE = 2
PAC_OFF = (0, 20)  # Pac-Man center offset from cell center (from first frame)

# Costs read from first_frame.png (row-major). Start cell cost is irrelevant.
COSTS = [
    [0, 50, 20, 40],
    [30, 30, 10, 40],
    [10, 30, 50, 10],
    [10, 20, 10, 20],
]
START, GOAL = (0, 0), (3, 3)


def best_path():
    """Simple path (no revisits) from START to GOAL maximizing collected cost."""
    best = {"sum": -1, "path": None}

    def dfs(r, c, visited, s, path):
        if (r, c) == GOAL:
            if s > best["sum"]:
                best["sum"], best["path"] = s, list(path)
            return
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in visited:
                visited.add((nr, nc)); path.append((nr, nc))
                dfs(nr, nc, visited, s + COSTS[nr][nc], path)
                path.pop(); visited.remove((nr, nc))

    dfs(*START, {START}, 0, [START])
    return best["path"], best["sum"]


def cell_center(rc):
    r, c = rc
    return (c * CELL + CELL // 2 + PAC_OFF[0], r * CELL + CELL // 2 + PAC_OFF[1])


def draw_pacman(img, cx, cy, d):
    """d = (dx, dy) unit facing direction; mouth is a white wedge toward d."""
    dr = ImageDraw.Draw(img)
    dr.ellipse([cx - R, cy - R, cx + R, cy + R], fill=YELLOW, outline=BLACK, width=OUTLINE)
    dx, dy = d
    px, py = -dy, dx  # perpendicular
    L, Hh = 44, 20
    tri = [(cx, cy),
           (cx + dx * L + px * Hh, cy + dy * L + py * Hh),
           (cx + dx * L - px * Hh, cy + dy * L - py * Hh)]
    dr.polygon(tri, fill=WHITE)


def main():
    first = Image.open("/app/first_frame.png").convert("RGB")
    bg = first.copy()
    # Erase Pac-Man from start cell: its region is solid green underneath.
    sx, sy = cell_center(START)
    ImageDraw.Draw(bg).rectangle([sx - R - 1, sy - R - 1, sx + R + 1, sy + R + 1], fill=GREEN)

    path, total = best_path()
    print("path:", path, "sum:", total)
    n_moves = len(path) - 1

    # Timing: short hold at start, moves spread across the middle, hold at end.
    hold_start, hold_end = 6, 8
    move_frames = N_FRAMES - hold_start - hold_end
    per = move_frames / n_moves

    frames = []
    facing = (1, 0)
    for f in range(N_FRAMES):
        img = bg.copy()
        t = f - hold_start
        if t <= 0:
            cx, cy = cell_center(path[0])
        elif t >= move_frames:
            cx, cy = cell_center(path[-1])
        else:
            i = min(int(t // per), n_moves - 1)
            u = (t - i * per) / per
            a, b = cell_center(path[i]), cell_center(path[i + 1])
            facing = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
            cx = round(a[0] + (b[0] - a[0]) * u)
            cy = round(a[1] + (b[1] - a[1]) * u)
        draw_pacman(img, cx, cy, facing)
        frames.append(np.array(img))

    frames[0] = np.array(first)  # exact first frame
    arr = np.stack(frames)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
           "/app/output/video.mp4"]
    p = subprocess.run(cmd, input=arr.tobytes())
    if p.returncode:
        sys.exit("ffmpeg failed")
    # Report how closely the redrawn start frame matches the original.
    diff = (frames[1].astype(int) - np.array(first).astype(int)) != 0
    print("redrawn-vs-original differing pixels:", int(diff.any(axis=2).sum()))


if __name__ == "__main__":
    main()
