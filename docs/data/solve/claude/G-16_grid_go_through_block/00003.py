#!/usr/bin/env python3
"""Animate an orange agent through a 10x10 grid: start -> brown -> blue -> purple -> yellow -> end,
taking shortest 4-neighbour paths between consecutive targets."""
import subprocess, collections
import numpy as np
from PIL import Image, ImageDraw

FIRST = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 94
N = 10
ORIGIN, PITCH, LINE = 47, 93, 2           # grid geometry measured from first frame

COLORS = {
    "green": (50, 200, 50), "red": (255, 50, 50), "brown": (139, 69, 19),
    "blue": (0, 0, 255), "purple": (128, 0, 128), "yellow": (255, 255, 0),
}
AGENT_FILL, AGENT_OUTLINE = (255, 165, 0), (200, 120, 0)
AGENT_R, AGENT_W = 23, 2

def cell_center(c, r):
    return ORIGIN + PITCH * c + LINE / 2 + (PITCH - LINE) / 2, ORIGIN + PITCH * r + LINE / 2 + (PITCH - LINE) / 2

def find_cells(img):
    a = np.array(img).astype(int)
    found = {}
    for name, col in COLORS.items():
        ys, xs = np.where((a == col).all(2))
        found[name] = (int(round((xs.mean() - ORIGIN) // PITCH)), int(round((ys.mean() - ORIGIN) // PITCH)))
    return found

def bfs(start, goal, blocked):
    prev = {start: None}
    q = collections.deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dc, cur[1] + dr)
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in prev and (nxt == goal or nxt not in blocked):
                prev[nxt] = cur
                q.append(nxt)
    path, cur = [], goal
    while cur is not None:
        path.append(cur); cur = prev[cur]
    return path[::-1]

def draw_agent(img, x, y):
    d = ImageDraw.Draw(img)
    d.ellipse([x - AGENT_R, y - AGENT_R, x + AGENT_R, y + AGENT_R],
              fill=AGENT_FILL, outline=AGENT_OUTLINE, width=AGENT_W)

def main():
    first = Image.open(FIRST).convert("RGB")
    cells = find_cells(first)
    start, end = cells["green"], cells["red"]
    order = [cells[k] for k in ("brown", "blue", "purple", "yellow")] + [end]

    # background: first frame with the agent erased (green cell repainted)
    bg = first.copy()
    d = ImageDraw.Draw(bg)
    x0, y0 = ORIGIN + PITCH * start[0] + LINE, ORIGIN + PITCH * start[1] + LINE
    d.rectangle([x0, y0, x0 + PITCH - LINE - 1, y0 + PITCH - LINE - 1], fill=COLORS["green"])

    # full route through targets in order (blocks other than the current target are avoided when possible)
    route, cur = [start], start
    for tgt in order:
        blocked = set(order) - {tgt}
        seg = bfs(cur, tgt, blocked)
        route += seg[1:]; cur = tgt
    steps = len(route) - 1

    hold_start, hold_end = 3, 7
    move_frames = N_FRAMES - hold_start - hold_end
    frames = []
    for i in range(N_FRAMES):
        t = min(max((i - hold_start) / move_frames, 0.0), 1.0) * steps
        k = min(int(t), steps - 1); f = t - k
        (c0, r0), (c1, r1) = route[k], route[k + 1]
        x0, y0 = cell_center(c0, r0); x1, y1 = cell_center(c1, r1)
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        fr = bg.copy()
        draw_agent(fr, round(x), round(y))
        frames.append(fr)

    frames[0] = first  # exact first frame
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x1024", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "16", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.array(fr, dtype=np.uint8).tobytes())
    p.stdin.close(); p.wait()
    print("route", route, "steps", steps, "frames", len(frames))

if __name__ == "__main__":
    main()
