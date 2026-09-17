#!/usr/bin/env python3
"""Animate Pac-Man along the max-cost simple path on the 4x4 grid."""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 91
CELL = 256
N = 4

# Costs read from first_frame.png (start cell shows no number; never re-entered).
COSTS = [
    [0, 10, 40, 30],
    [40, 30, 30, 20],
    [20, 10, 40, 30],
    [50, 30, 10, 40],
]
START, GOAL = (0, 0), (3, 3)

# Pac-Man geometry measured from first_frame.png
PAC_OFF = (128, 148)          # center offset inside a cell
R_OUT = 64                    # outer (black outline) radius
OUTLINE = 2
MOUTH_LEN, MOUTH_HALF = 44, 20
YELLOW, BLACK, WHITE = (255, 255, 0), (0, 0, 0), (255, 255, 255)
GREEN = (76, 175, 80)


def best_path():
    """DFS over simple orthogonal paths, maximizing sum of entered-cell costs."""
    best = {"sum": -1, "path": None}
    seen = [[False] * N for _ in range(N)]

    def dfs(r, c, total, path):
        if (r, c) == GOAL:
            if total > best["sum"]:
                best["sum"], best["path"] = total, list(path)
            return
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and not seen[nr][nc]:
                seen[nr][nc] = True
                path.append((nr, nc))
                dfs(nr, nc, total + COSTS[nr][nc], path)
                path.pop()
                seen[nr][nc] = False

    seen[START[0]][START[1]] = True
    dfs(START[0], START[1], 0, [START])
    return best["path"], best["sum"]


def cell_center(rc):
    r, c = rc
    return (c * CELL + PAC_OFF[0], r * CELL + PAC_OFF[1])


def draw_pacman(img, cx, cy, direction):
    d = ImageDraw.Draw(img)
    d.ellipse([cx - R_OUT, cy - R_OUT, cx + R_OUT, cy + R_OUT],
              fill=YELLOW, outline=BLACK, width=OUTLINE)
    dx, dy = direction
    # mouth: white wedge from the center, pointing in the move direction
    px, py = -dy, dx  # perpendicular
    tip = (cx + dx * MOUTH_LEN, cy + dy * MOUTH_LEN)
    p1 = (tip[0] + px * MOUTH_HALF, tip[1] + py * MOUTH_HALF)
    p2 = (tip[0] - px * MOUTH_HALF, tip[1] - py * MOUTH_HALF)
    d.polygon([(cx, cy), p1, p2], fill=WHITE)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = Image.open(FIRST).convert("RGB")

    # Background: first frame with Pac-Man removed from the (plain green) start cell.
    bg = first.copy()
    sx, sy = cell_center(START)
    ImageDraw.Draw(bg).rectangle([sx - R_OUT - 1, sy - R_OUT - 1, sx + R_OUT + 1, sy + R_OUT + 1], fill=GREEN)

    path, total = best_path()
    steps = len(path) - 1
    print("path:", path, "sum:", total, file=sys.stderr)

    hold_start, hold_end = 4, 6
    move_frames = N_FRAMES - hold_start - hold_end

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
         "-r", str(FPS), OUT],
        stdin=subprocess.PIPE)

    direction = (1, 0)
    for f in range(N_FRAMES):
        if f == 0:
            frame = first.copy()
        else:
            t = (f - hold_start) / move_frames * steps
            t = min(max(t, 0.0), float(steps))
            i = min(int(t), steps - 1)
            frac = t - i
            a, b = cell_center(path[i]), cell_center(path[i + 1])
            dr, dc = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
            if t < steps:
                direction = (dc, dr)
            cx = round(a[0] + (b[0] - a[0]) * frac)
            cy = round(a[1] + (b[1] - a[1]) * frac)
            frame = bg.copy()
            draw_pacman(frame, cx, cy, direction)
        ffmpeg.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        sys.exit("ffmpeg failed")
    print("wrote", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
