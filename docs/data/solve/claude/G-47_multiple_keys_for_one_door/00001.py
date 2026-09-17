import itertools, subprocess, os
from collections import deque
import numpy as np
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 49, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
G = 13                      # 13x13 grid incl. outer wall
cs = W / G

def cell_center(r, c):
    return (int(round((c + 0.5) * cs)), int(round((r + 0.5) * cs)))

# ---- parse grid: a cell is path if its centre region is not black ----
def cell_pixels(r, c):
    y0, y1 = int(r * cs), int((r + 1) * cs)
    x0, x1 = int(c * cs), int((c + 1) * cs)
    return img[y0:y1, x0:x1]

path = np.zeros((G, G), bool)
for r in range(G):
    for c in range(G):
        blk = cell_pixels(r, c)
        path[r, c] = (blk.sum(2) > 0).mean() > 0.5

# ---- locate objects by colour ----
def find(color):
    m = (img == np.array(color, np.uint8)).all(2)
    ys, xs = np.where(m)
    return m, (int(ys.mean() / cs), int(xs.mean() / cs))

GREEN = (0, 255, 0)
agent_mask, start = find(GREEN)
colors = {tuple(c) for c in np.unique(img.reshape(-1, 3), axis=0)}
colors -= {(0, 0, 0), (255, 255, 255), GREEN}
keys, door = {}, None
for col in colors:
    m, cell = find(col)
    # hollow square (door) has a white interior; diamond is solid
    ys, xs = np.where(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    if m[cy, cx]:
        keys[cell] = m
    else:
        door = cell
assert door is not None and keys

# ---- BFS shortest paths ----
def bfs(src):
    dist = {src: 0}; prev = {src: None}; q = deque([src])
    while q:
        u = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + d[0], u[1] + d[1])
            if 0 <= v[0] < G and 0 <= v[1] < G and path[v] and v not in dist:
                dist[v] = dist[u] + 1; prev[v] = u; q.append(v)
    return dist, prev

def route(a, b):
    dist, prev = bfs(a)
    seq = [b]
    while seq[-1] != a:
        seq.append(prev[seq[-1]])
    return seq[::-1]

nodes = [start] + list(keys) + [door]
D = {a: bfs(a)[0] for a in nodes}
best = None
for perm in itertools.permutations(list(keys)):
    order = [start] + list(perm) + [door]
    tot = sum(D[order[i]][order[i + 1]] for i in range(len(order) - 1))
    if best is None or tot < best[0]:
        best = (tot, perm)
order = [start] + list(best[1]) + [door]
cells = [start]
for a, b in zip(order, order[1:]):
    cells += route(a, b)[1:]
print('order', order, 'total steps', best[0])

# ---- rendering ----
base = img.copy()
base[agent_mask] = 255                      # remove agent from background
for m in keys.values():
    base[m] = 255                           # remove keys (re-added per frame)
ys, xs = np.where(agent_mask)
ay0, ax0 = ys.min(), xs.min()
sprite = agent_mask[ay0:ys.max() + 1, ax0:xs.max() + 1]
sc = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)  # sprite centre in source

def cell_px(cell):
    return np.array(cell_center(*cell), float)

def frame_at(t):
    """t in [0, 1] along the whole path."""
    s = t * (len(cells) - 1)
    i = min(int(np.floor(s)), len(cells) - 2)
    f = s - i
    p = cell_px(cells[i]) * (1 - f) + cell_px(cells[i + 1]) * f
    reached = set(cells[:i + 1]) | ({cells[i + 1]} if f >= 1 - 1e-9 else set())
    fr = base.copy()
    for cell, m in keys.items():
        if cell not in reached:
            fr[m] = np.array(list(colors_of[cell]), np.uint8)
    dx, dy = int(round(p[0] - sc[0])), int(round(p[1] - sc[1]))
    sub = fr[ay0 + dy:ay0 + dy + sprite.shape[0], ax0 + dx:ax0 + dx + sprite.shape[1]]
    sub[sprite] = GREEN
    return fr

colors_of = {cell: tuple(img[m][0]) for cell, m in keys.items()}

frames = []
for k in range(N_FRAMES):
    t = k / (N_FRAMES - 1)
    frames.append(frame_at(t))
assert (frames[0] == img).all()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                      '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                      '-pix_fmt', 'yuv420p', '-crf', '12', OUT], stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
print('wrote', OUT)
