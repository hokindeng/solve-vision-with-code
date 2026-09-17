#!/usr/bin/env python3
"""Animate Pac-Man along the max-cost simple path on the 4x4 grid."""
import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS = 16
N_FRAMES = 91
CELL = 256
N = 4
GREEN = (76, 175, 80)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
R = 64                    # outer radius incl. outline
OUTLINE_W = 2
Y_OFF = 20                # Pac-Man sits 20 px below the cell centre in the source frame
MOUTH_LEN = 44
MOUTH_HALF = 20

# Costs read from first_frame.png (start cell has no visible cost)
COSTS = [[0, 40, 50, 50],
         [10, 10, 30, 40],
         [40, 50, 30, 10],
         [10, 10, 40, 20]]


def best_path():
    best = [None, None]

    def dfs(r, c, vis, s, path):
        if (r, c) == (N - 1, N - 1):
            if best[0] is None or s > best[0]:
                best[0], best[1] = s, list(path)
            return
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in vis:
                vis.add((nr, nc)); path.append((nr, nc))
                dfs(nr, nc, vis, s + COSTS[nr][nc], path)
                path.pop(); vis.remove((nr, nc))

    dfs(0, 0, {(0, 0)}, 0, [(0, 0)])
    return best[0], best[1]


def cell_center(rc):
    r, c = rc
    return (c * CELL + CELL // 2, r * CELL + CELL // 2 + Y_OFF)


def draw_pacman(img, cx, cy, angle):
    d = ImageDraw.Draw(img)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=YELLOW, outline=BLACK, width=OUTLINE_W)
    ca, sa = math.cos(angle), math.sin(angle)

    def rot(x, y):
        return (cx + x * ca - y * sa, cy + x * sa + y * ca)

    tri = [rot(0, 0), rot(MOUTH_LEN, -MOUTH_HALF), rot(MOUTH_LEN, MOUTH_HALF)]
    d.polygon(tri, fill=WHITE)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = Image.open(FIRST).convert("RGB")
    bg = first.copy()
    # remove Pac-Man from the start cell (plain green underneath)
    sx, sy = cell_center((0, 0))
    ImageDraw.Draw(bg).rectangle([sx - R - 1, sy - R - 1, sx + R + 1, sy + R + 1], fill=GREEN)

    total, path = best_path()
    moves = len(path) - 1
    per_move = 6
    frames = []
    frames.append(np.array(first))          # frame 0 exactly as given
    angle = 0.0
    for f in range(1, N_FRAMES):
        t = (f) / per_move
        i = min(int(math.floor(t)), moves - 1)
        frac = min(t - i, 1.0)
        a = cell_center(path[i]); b = cell_center(path[i + 1])
        if f <= moves * per_move:
            angle = math.atan2(b[1] - a[1], b[0] - a[0])
            x = a[0] + (b[0] - a[0]) * frac
            y = a[1] + (b[1] - a[1]) * frac
        else:
            x, y = cell_center(path[-1])
        img = bg.copy()
        draw_pacman(img, round(x), round(y), angle)
        frames.append(np.array(img))

    tmp = os.path.join(OUT_DIR, "frames.raw")
    with open(tmp, "wb") as fh:
        for fr in frames:
            fh.write(fr.astype(np.uint8).tobytes())
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1024x1024", "-r", str(FPS), "-i", tmp,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT
    ], check=True)
    os.remove(tmp)
    print(f"path={path} total={total} frames={len(frames)} -> {OUT}")


if __name__ == "__main__":
    main()
