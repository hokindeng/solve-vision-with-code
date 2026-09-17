#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: orange agent visits pink, purple, yellow,
brown, blue blocks in order (shortest Manhattan paths) then the red end cell."""
import os, subprocess
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS, N_FRAMES, SIZE = 16, 82, 1024
ORIGIN, PITCH, N = 47, 93, 10          # grid lines at 47 + 93*k (2 px wide)
RADIUS, OUTLINE_W = 23, 2
AGENT_FILL, AGENT_OUTLINE = (255, 165, 0), (200, 120, 0)

COLORS = {
    "green": (50, 200, 50), "red": (255, 50, 50), "pink": (255, 192, 203),
    "purple": (128, 0, 128), "yellow": (255, 255, 0), "brown": (139, 69, 19),
    "blue": (0, 0, 255),
}
ORDER = ["pink", "purple", "yellow", "brown", "blue", "red"]


def cell_center(r, c):
    return ORIGIN + PITCH * c + 46, ORIGIN + PITCH * r + 46


def locate(arr, color):
    m = np.all(arr == color, axis=2)
    ys, xs = np.where(m)
    return int((ys.mean() - ORIGIN) // PITCH), int((xs.mean() - ORIGIN) // PITCH)


def bfs(start, goal):
    """Shortest 4-neighbour path on the open 10x10 grid (blocks are passable)."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def main():
    orig = np.array(Image.open(FIRST).convert("RGB"))
    cells = {k: locate(orig, v) for k, v in COLORS.items()}
    start = cells["green"]

    # Background with the agent removed (restore the green start cell).
    base = orig.copy()
    r0, c0 = start
    y0, x0 = ORIGIN + PITCH * r0 + 2, ORIGIN + PITCH * c0 + 2
    base[y0:y0 + 90, x0:x0 + 90] = COLORS["green"]

    # Full route: shortest path between consecutive targets.
    route = [start]
    for name in ORDER:
        route += bfs(route[-1], cells[name])[1:]
    steps = len(route) - 1

    # Pace the motion over the whole clip, holding briefly at the end.
    hold = 4
    move_frames = N_FRAMES - hold
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    for i in range(N_FRAMES):
        t = min(i / (move_frames - 1), 1.0) * steps
        k = min(int(t), steps - 1)
        frac = t - k
        (ra, ca), (rb, cb) = route[k], route[k + 1]
        xa, ya = cell_center(ra, ca)
        xb, yb = cell_center(rb, cb)
        cx = int(round(xa + (xb - xa) * frac))
        cy = int(round(ya + (yb - ya) * frac))
        if i == 0:
            frame = Image.fromarray(orig)
        else:
            frame = Image.fromarray(base.copy())
            d = ImageDraw.Draw(frame)
            d.ellipse([cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS],
                      fill=AGENT_FILL, outline=AGENT_OUTLINE, width=OUTLINE_W)
        frame.save(os.path.join(tmp, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-vf", f"scale={SIZE}:{SIZE}", OUT,
    ], check=True)
    print(f"wrote {OUT}: {N_FRAMES} frames, route of {steps} steps")


if __name__ == "__main__":
    main()
