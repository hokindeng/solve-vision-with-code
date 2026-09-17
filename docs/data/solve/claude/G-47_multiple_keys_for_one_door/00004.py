import itertools, subprocess, os
from collections import deque
import numpy as np
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 73, 16
ORIGIN, N = 78, 11
CELL = (944 - 78 + 1) / N  # 78.8 approx

base = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = base.shape

def center(r, c):
    return ORIGIN + CELL * (c + 0.5), ORIGIN + CELL * (r + 0.5)

def find(color):
    m = np.all(base == color, axis=2)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    return m, (int((cy - ORIGIN) // CELL), int((cx - ORIGIN) // CELL))

agent_mask, start = find((0, 255, 0))
door_mask, door = find((0, 0, 255))
keys = {}
for col in [(0, 255, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (255, 165, 0)]:
    if np.any(np.all(base == col, axis=2)):
        keys[col] = find(col)

# Grid: cell is a wall if its centre is black
grid = np.zeros((N, N), bool)
for r in range(N):
    for c in range(N):
        x, y = center(r, c)
        px = base[int(y), int(x)]
        grid[r, c] = not (px == 0).all()  # True = passable (white or object)

def bfs(src):
    dist = {src: 0}; prev = {src: None}; q = deque([src])
    while q:
        u = q.popleft()
        for d in ((1,0),(-1,0),(0,1),(0,-1)):
            v = (u[0]+d[0], u[1]+d[1])
            if 0 <= v[0] < N and 0 <= v[1] < N and grid[v] and v not in dist:
                dist[v] = dist[u] + 1; prev[v] = u; q.append(v)
    return dist, prev

def path(a, b):
    _, prev = bfs(a)
    p = [b]
    while p[-1] != a:
        p.append(prev[p[-1]])
    return p[::-1]

key_cells = [v[1] for v in keys.values()]
pts = [start] + key_cells + [door]
D = {p: bfs(p)[0] for p in pts}
best = min(itertools.permutations(key_cells),
           key=lambda perm: sum(D[a][b] for a, b in zip((start,) + perm, perm + (door,))))
order = [start] + list(best) + [door]
full = [start]
for a, b in zip(order, order[1:]):
    full += path(a, b)[1:]
print('order', order, 'length', len(full) - 1)

# Agent sprite
ys, xs = np.nonzero(agent_mask)
acx, acy = center(*start)
sprite_dy, sprite_dx = ys - int(round(acy)), xs - int(round(acx))
bg = base.copy()
bg[agent_mask] = 255  # path under agent is white
key_masks = {cell: keys[col][0] for col, (m, cell) in [(k, v) for k, v in keys.items()]}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
frames = []
L = len(full) - 1
for f in range(N_FRAMES):
    t = f / (N_FRAMES - 1) * L
    i = min(int(t), L - 1); s = t - i
    (r0, c0), (r1, c1) = full[i], full[i + 1]
    r, c = r0 + (r1 - r0) * s, c0 + (c1 - c0) * s
    x, y = center(r, c)
    img = bg.copy()
    reached = set(full[:i + 1]) | ({full[i + 1]} if s >= 0.999 else set())
    for cell, m in key_masks.items():
        if cell in reached:
            img[m] = 255
    yy = np.clip(sprite_dy + int(round(y)), 0, H - 1); xx = np.clip(sprite_dx + int(round(x)), 0, W - 1)
    img[yy, xx] = (0, 255, 0)
    frames.append(img)

assert np.array_equal(frames[0], base)
tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(f'{tmp}/{i:03d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%03d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', OUT], check=True)
import shutil; shutil.rmtree(tmp)
print('wrote', OUT)
