#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: the orange agent walks from the green cell
through yellow waypoints 1,2,3 in order to the red cell, using shortest
4-connected paths between consecutive waypoints. Everything except the agent
disc (and the green cell it uncovers) is left pixel-identical to first_frame.png."""
import subprocess, numpy as np
from collections import deque
from PIL import Image

W = H = 1024; N = 10; CELL = 102          # grid lines at 102*k, cells 101px wide
FPS = 16; FRAMES = 122
first = np.array(Image.open('/app/first_frame.png').convert('RGB'))

def cell_of(mask):
    ys, xs = np.where(mask); return (int(ys.mean()) // CELL, int(xs.mean()) // CELL)

def center(rc): r, c = rc; return (CELL * c + 51, CELL * r + 51)   # (x, y)

R, G, B = first[..., 0].astype(int), first[..., 1].astype(int), first[..., 2].astype(int)
green  = cell_of((R < 50) & (G > 200) & (B < 50))
red    = cell_of((R > 200) & (G < 50) & (B < 50))
orange = (R == 255) & (G == 165) & (B == 0)
yellow = (R > 200) & (G > 200) & (B < 50)
ycells = sorted((r, c) for r in range(N) for c in range(N)
                if yellow[CELL * r + 10, CELL * c + 10])   # sample a corner point of each cell
# Digit labels read from the frame: "1" at row4/col3, "2" at row8/col6, "3" at row0/col0.
labels = {(4, 3): 1, (8, 6): 2, (0, 0): 3}
assert set(ycells) == set(labels), ycells
waypoints = [green] + sorted(ycells, key=labels.get) + [red]

def bfs(s, t):
    prev = {s: None}; q = deque([s])
    while q:
        u = q.popleft()
        if u == t: break
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + d[0], u[1] + d[1])
            if 0 <= v[0] < N and 0 <= v[1] < N and v not in prev:
                prev[v] = u; q.append(v)
    path = []; u = t
    while u is not None: path.append(u); u = prev[u]
    return path[::-1]

route = [waypoints[0]]
for a, b in zip(waypoints, waypoints[1:]):
    route += bfs(a, b)[1:]
moves = len(route) - 1
print('route', route, 'moves', moves)

# Agent sprite: exact pixel mask of the disc from the first frame, relative to its cell centre.
ys, xs = np.where(orange); gx, gy = center(green)
disc_dx, disc_dy = xs - gx, ys - gy
background = first.copy()
background[orange] = (0, 255, 0)             # green cell restored under the agent

def frame(i):
    t = i / (FRAMES - 1) * moves             # progress in "moves"
    k = min(int(t), moves - 1); f = t - k
    (x0, y0), (x1, y1) = center(route[k]), center(route[k + 1])
    cx, cy = round(x0 + (x1 - x0) * f), round(y0 + (y1 - y0) * f)
    img = background.copy()
    px, py = disc_dx + cx, disc_dy + cy
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    img[py[ok], px[ok]] = (255, 165, 0)
    return img

frames = [frame(i) for i in range(FRAMES)]
assert (frames[0] == first).all(), 'first frame must match first_frame.png'
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '15', '-preset', 'slow', '/app/output/video.mp4'], stdin=subprocess.PIPE)
for fr in frames: p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frames[-1]).save('/app/output/last_frame.png')
print('wrote /app/output/video.mp4', p.returncode)
