#!/usr/bin/env python3
"""Animate Pac-Man along the maximum-cost right/down path on the 4x4 grid."""
import math, os, subprocess, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

FPS = 16
N_FRAMES = 91
CELL = 256
# Costs read from the first frame (start cell cost is hidden under Pac-Man and never collected).
COST = [[0, 10, 50, 30],
        [10, 50, 50, 10],
        [50, 40, 30, 40],
        [30, 30, 10, 30]]
GREEN = (76, 175, 80)

# Pac-Man geometry measured from first_frame.png
PAC_R = 62          # yellow radius
OUTLINE_W = 2       # black ring width
Y_OFF = 20          # Pac-Man is drawn 20px below the cell centre
MOUTH_LEN = 44      # mouth wedge depth (apex at centre)
MOUTH_HALF = 20     # half-height of mouth opening at its rim


def best_path():
    """DP over right/down moves maximising the sum of entered-cell costs."""
    n = 4
    best = [[-1] * n for _ in range(n)]
    prev = [[None] * n for _ in range(n)]
    best[0][0] = 0
    for r in range(n):
        for c in range(n):
            if r == 0 and c == 0:
                continue
            cands = []
            if r > 0:
                cands.append((best[r - 1][c], (r - 1, c)))  # came from above
            if c > 0:
                cands.append((best[r][c - 1], (r, c - 1)))  # came from the left
            v, p = max(cands, key=lambda t: t[0])           # ties: prefer the right-first route
            best[r][c] = v + COST[r][c]
            prev[r][c] = p
    path = [(n - 1, n - 1)]
    while path[-1] != (0, 0):
        path.append(prev[path[-1][0]][path[-1][1]])
    return path[::-1], best[n - 1][n - 1]


def cell_centre(rc):
    r, c = rc
    return (c * CELL + CELL // 2, r * CELL + CELL // 2 + Y_OFF)


def draw_pacman(img, cx, cy, angle):
    """Draw Pac-Man at float centre (cx, cy), mouth facing `angle` (radians), supersampled."""
    S = 4
    R = PAC_R + OUTLINE_W
    pad = R + 3
    x0, y0 = int(math.floor(cx - pad)), int(math.floor(cy - pad))
    w = h = 2 * pad + 2
    layer = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lx, ly = (cx - x0) * S, (cy - y0) * S
    d.ellipse([lx - R * S, ly - R * S, lx + R * S, ly + R * S], fill=(0, 0, 0, 255))
    d.ellipse([lx - PAC_R * S, ly - PAC_R * S, lx + PAC_R * S, ly + PAC_R * S], fill=(255, 255, 0, 255))
    ca, sa = math.cos(angle), math.sin(angle)
    tip = (lx, ly)
    p1 = (lx + (MOUTH_LEN * ca - MOUTH_HALF * sa) * S, ly + (MOUTH_LEN * sa + MOUTH_HALF * ca) * S)
    p2 = (lx + (MOUTH_LEN * ca + MOUTH_HALF * sa) * S, ly + (MOUTH_LEN * sa - MOUTH_HALF * ca) * S)
    d.polygon([tip, p1, p2], fill=(255, 255, 255, 255))
    layer = layer.resize((w, h), Image.LANCZOS)
    img.alpha_composite(layer, (x0, y0))


def main():
    first = Image.open(FIRST).convert("RGB")
    arr = np.array(first)

    # Background: first frame with Pac-Man removed from the start cell (fill with green).
    sx, sy = cell_centre((0, 0))
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    mask = (xx - sx) ** 2 + (yy - sy) ** 2 <= (PAC_R + OUTLINE_W + 2) ** 2
    bg = arr.copy()
    bg[mask] = GREEN
    bg_img = Image.fromarray(bg).convert("RGBA")

    path, total = best_path()
    print("path:", path, "total collected:", total)
    moves = len(path) - 1
    per_move = (N_FRAMES - 1) // moves  # 15 frames per step

    frames = [first.copy()]
    facing = 0.0
    for m in range(moves):
        a, b = cell_centre(path[m]), cell_centre(path[m + 1])
        facing = math.atan2(b[1] - a[1], b[0] - a[0])
        for k in range(1, per_move + 1):
            t = k / per_move
            cx = a[0] + (b[0] - a[0]) * t
            cy = a[1] + (b[1] - a[1]) * t
            img = bg_img.copy()
            draw_pacman(img, cx, cy, facing)
            frames.append(img.convert("RGB"))
    while len(frames) < N_FRAMES:  # hold on the goal if any frames remain
        frames.append(frames[-1].copy())
    frames = frames[:N_FRAMES]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp()
    try:
        for i, f in enumerate(frames):
            f.save(os.path.join(tmp, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(tmp, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
            "-r", str(FPS), OUT,
        ], check=True)
    finally:
        shutil.rmtree(tmp)
    print("wrote", OUT, "frames:", len(frames))


if __name__ == "__main__":
    main()
