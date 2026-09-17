#!/usr/bin/env python3
"""Animate the orange agent through brown -> blue -> purple -> red on the 10x10 grid."""
import os, subprocess, shutil
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N = 10
ORIGIN = 47        # first grid line
PITCH = 93         # cell pitch (91 px interior + 2 px line)
FPS = 16
FRAMES = 102

AGENT_FILL = (255, 165, 0)
AGENT_OUTLINE = (200, 120, 0)
AGENT_R = 23
COLORS = {
    "green": (50, 200, 50),
    "red": (255, 50, 50),
    "brown": (139, 69, 19),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
}


def cell_center(c, r):
    return ORIGIN + PITCH * c + 46, ORIGIN + PITCH * r + 46


def find_cells(arr):
    """Locate each colored cell by sampling cell centers (skipping the agent pixels)."""
    found = {}
    for r in range(N):
        for c in range(N):
            x, y = cell_center(c, r)
            # sample a corner region of the cell interior, away from the agent
            px = tuple(int(v) for v in arr[y - 40, x - 40])
            for name, col in COLORS.items():
                if px == col:
                    found[name] = (c, r)
    return found


def bfs(start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        c, r = cur
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):  # up, down, left, right
            nxt = (c + dc, r + dr)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def draw_agent(img, cx, cy):
    d = ImageDraw.Draw(img)
    d.ellipse((cx - AGENT_R, cy - AGENT_R, cx + AGENT_R, cy + AGENT_R),
              fill=AGENT_FILL, outline=AGENT_OUTLINE, width=2)


def main():
    first = Image.open(FIRST).convert("RGB")
    arr = np.array(first)
    cells = find_cells(arr)
    start, end = cells["green"], cells["red"]
    targets = [cells["brown"], cells["blue"], cells["purple"], end]

    # Background: first frame with the agent erased (it sits fully inside the green cell).
    bg = first.copy()
    sx, sy = cell_center(*start)
    ImageDraw.Draw(bg).rectangle((sx - AGENT_R - 1, sy - AGENT_R - 1, sx + AGENT_R + 1, sy + AGENT_R + 1),
                                 fill=COLORS["green"])

    # Full route: concatenation of shortest paths between consecutive targets.
    route = [start]
    for t in targets:
        route += bfs(route[-1], t)[1:]
    n_moves = len(route) - 1

    # Timing: short hold at start and end, constant speed in between.
    hold_start, hold_end = 3, 4
    move_frames = FRAMES - 1 - hold_start - hold_end

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    for f in range(FRAMES):
        if f == 0:
            frame = first.copy()
        else:
            t = min(max((f - hold_start) / move_frames, 0.0), 1.0) * n_moves
            i = min(int(t), n_moves - 1)
            frac = t - i
            (c0, r0), (c1, r1) = route[i], route[i + 1]
            x0, y0 = cell_center(c0, r0)
            x1, y1 = cell_center(c1, r1)
            cx = int(round(x0 + (x1 - x0) * frac))
            cy = int(round(y0 + (y1 - y0) * frac))
            frame = bg.copy()
            draw_agent(frame, cx, cy)
        frame.save(os.path.join(tmp, f"{f:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-r", str(FPS), OUT,
    ], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"route ({n_moves} moves): {route}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
