"""Animate an orange agent visiting colored blocks in order on a 10x10 grid.

Regenerates /app/output/video.mp4 from /app/first_frame.png.
"""
import os
import subprocess
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N = 10
ORIGIN = 47          # first grid line
CELL = 93            # cell pitch; interior of cell 0 spans 49..138
FPS = 16
TOTAL_FRAMES = 134
FRAMES_PER_MOVE = 4

AGENT_FILL = (255, 165, 0)
AGENT_OUTLINE = (200, 120, 0)
AGENT_R = 23
AGENT_W = 2

COLORS = {
    "green": (50, 200, 50),
    "red": (255, 50, 50),
    "purple": (128, 0, 128),
    "brown": (139, 69, 19),
    "pink": (255, 192, 203),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}
ORDER = ["purple", "brown", "pink", "blue", "yellow"]


def find_cells(arr):
    """Locate each colored cell by its interior fill color."""
    found = {}
    for name, col in COLORS.items():
        m = np.all(arr == col, axis=2)
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        found[name] = (int((cy - ORIGIN) // CELL), int((cx - ORIGIN) // CELL))
    return found


def find_agent(arr):
    m = np.all(arr == AGENT_FILL, axis=2)
    ys, xs = np.where(m)
    return (int(round(ys.mean())), int(round(xs.mean())))


def bfs(start, goal):
    """Shortest 4-neighbour path on the open grid (blocks are targets, not walls)."""
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


def draw_agent(img, cx, cy):
    d = ImageDraw.Draw(img)
    d.ellipse([cx - AGENT_R, cy - AGENT_R, cx + AGENT_R, cy + AGENT_R],
              fill=AGENT_FILL, outline=AGENT_OUTLINE, width=AGENT_W)


def main():
    first = Image.open(FIRST).convert("RGB")
    arr = np.array(first)
    cells = find_cells(arr)
    agent_y, agent_x = find_agent(arr)
    start = cells["green"]
    end = cells["red"]

    # Background: first frame with the agent erased (restore the green cell fill).
    bg = arr.copy()
    r0, c0 = start
    y0, x0 = ORIGIN + 2 + CELL * r0, ORIGIN + 2 + CELL * c0
    bg[y0:y0 + 90, x0:x0 + 90] = COLORS["green"]
    bg_img = Image.fromarray(bg)

    # Agent pixel center relative to its cell (keeps the exact original placement).
    off_y = agent_y - y0
    off_x = agent_x - x0

    def center_of(cell):
        r, c = cell
        return ORIGIN + 2 + CELL * c + off_x, ORIGIN + 2 + CELL * r + off_y

    # Full route: start -> targets in order -> end, shortest path each leg.
    route = [start]
    for name in ORDER + ["red"]:
        goal = cells[name] if name != "red" else end
        leg = bfs(route[-1], goal)
        route.extend(leg[1:])

    n_moves = len(route) - 1
    move_frames = n_moves * FRAMES_PER_MOVE
    hold_start = 4
    hold_end = max(1, TOTAL_FRAMES - hold_start - move_frames)

    frames = []
    frames += [(route[0], route[0], 0.0)] * hold_start
    for i in range(n_moves):
        for k in range(FRAMES_PER_MOVE):
            frames.append((route[i], route[i + 1], (k + 1) / FRAMES_PER_MOVE))
    frames += [(route[-1], route[-1], 0.0)] * hold_end
    frames = frames[:TOTAL_FRAMES]
    while len(frames) < TOTAL_FRAMES:
        frames.append(frames[-1])

    os.makedirs(OUT_DIR, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "1024x1024", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT],
        stdin=subprocess.PIPE)
    for idx, (a, b, t) in enumerate(frames):
        ax, ay = center_of(a)
        bx, by = center_of(b)
        cx = int(round(ax + (bx - ax) * t))
        cy = int(round(ay + (by - ay) * t))
        img = bg_img.copy()
        draw_agent(img, cx, cy)
        if idx == 0:
            assert np.array_equal(np.array(img), arr), "first frame mismatch"
        ffmpeg.stdin.write(np.array(img).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {OUT}: {len(frames)} frames, {n_moves} moves, route {route}")


if __name__ == "__main__":
    main()
