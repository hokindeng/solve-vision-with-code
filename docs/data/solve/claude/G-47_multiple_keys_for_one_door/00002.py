import itertools, os, subprocess
from collections import deque
import numpy as np
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 63, 16

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape

# ---- grid detection ----
blk = np.all(img == 0, axis=2)
def edges(a):
    return np.nonzero(np.abs(np.diff(a)) > 0.05)[0] + 1
re_, ce_ = edges(blk.mean(axis=1)), edges(blk.mean(axis=0))
y0, x0 = re_[0], ce_[0]
cell = int(round(np.median(np.diff(re_))))
rows = int(round((re_[-1] - y0) / cell)); cols = int(round((ce_[-1] - x0) / cell))

def center(c, r):
    return (x0 + c * cell + cell / 2, y0 + r * cell + cell / 2)

def cell_of(mask):
    ys, xs = np.nonzero(mask)
    return (int((xs.mean() - x0) // cell), int((ys.mean() - y0) // cell))

# passable if the cell centre area is not black
passable = np.zeros((rows, cols), bool)
for r in range(rows):
    for c in range(cols):
        cx, cy = center(c, r)
        patch = blk[int(cy) - 3:int(cy) + 4, int(cx) - 3:int(cx) + 4]
        passable[r, c] = not patch.all()

# ---- objects ----
AGENT_COL = (0, 255, 0)
agent_mask = np.all(img == AGENT_COL, axis=2)
agent_cell = cell_of(agent_mask)

# door: hollow square (low fill ratio inside bbox); keys: filled diamonds
objs = {}
for col in {tuple(c) for c in np.unique(img.reshape(-1, 3), axis=0)} - {(0, 0, 0), (255, 255, 255), AGENT_COL}:
    m = np.all(img == np.array(col), axis=2)
    ys, xs = np.nonzero(m)
    fill = m.sum() / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    objs[col] = (cell_of(m), m, fill)
door_col = min(objs, key=lambda k: objs[k][2])
door_cell = objs[door_col][0]
keys = {k: v for k, v in objs.items() if k != door_col}

# ---- BFS ----
def bfs(src):
    dist = {src: 0}; prev = {}
    q = deque([src])
    while q:
        c, r = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and passable[nr, nc] and (nc, nr) not in dist:
                dist[(nc, nr)] = dist[(c, r)] + 1; prev[(nc, nr)] = (c, r); q.append((nc, nr))
    return dist, prev

def path(a, b):
    dist, prev = bfs(a)
    p = [b]
    while p[-1] != a:
        p.append(prev[p[-1]])
    return p[::-1]

key_cells = [v[0] for v in keys.values()]
key_cols = list(keys.keys())
best = None
for perm in itertools.permutations(range(len(key_cells))):
    pts = [agent_cell] + [key_cells[i] for i in perm] + [door_cell]
    total = sum(bfs(pts[i])[0][pts[i + 1]] for i in range(len(pts) - 1))
    if best is None or total < best[0]:
        best = (total, perm)
total, perm = best
order = [key_cells[i] for i in perm]
print('grid', rows, cols, 'agent', agent_cell, 'keys', key_cells, 'door', door_cell)
print('optimal order', order, 'total steps', total)

# full path of cells
full = [agent_cell]
for tgt in order + [door_cell]:
    full += path(full[-1], tgt)[1:]
pickup_step = {}  # step index at which each key is collected
for col, (kc, m, _) in keys.items():
    pickup_step[col] = full.index(kc)

# ---- rendering ----
# agent sprite (offsets relative to its cell centre)
ays, axs = np.nonzero(agent_mask)
acx, acy = center(*agent_cell)
sprite_dy = ays - int(round(acy)); sprite_dx = axs - int(round(acx))

base = img.copy()
base[agent_mask] = (255, 255, 255)  # background under agent is white path

def render(t):
    """t in [0, total] steps along the path."""
    frame = base.copy()
    for col, (kc, m, _) in keys.items():
        if t >= pickup_step[col]:
            frame[m] = (255, 255, 255)
    i = min(int(np.floor(t)), len(full) - 1)
    f = t - i
    if i + 1 < len(full):
        (c0, r0), (c1, r1) = full[i], full[i + 1]
        cx = (1 - f) * center(c0, r0)[0] + f * center(c1, r1)[0]
        cy = (1 - f) * center(c0, r0)[1] + f * center(c1, r1)[1]
    else:
        cx, cy = center(*full[-1])
    ys = sprite_dy + int(round(cy)); xs = sprite_dx + int(round(cx))
    frame[ys, xs] = AGENT_COL
    return frame

frames = []
for k in range(N_FRAMES):
    if k == 0:
        frames.append(img.copy())
        continue
    t = total * k / (N_FRAMES - 1)
    frames.append(render(t))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = '/app/output/frames'
os.makedirs(tmp, exist_ok=True)
for k, fr in enumerate(frames):
    Image.fromarray(fr).save(f'{tmp}/{k:04d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%04d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', OUT], check=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print('wrote', OUT, len(frames), 'frames')
