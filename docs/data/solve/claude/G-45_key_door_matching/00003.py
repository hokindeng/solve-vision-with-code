import numpy as np, subprocess, os
from collections import deque
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 94, 16
N = 13
b = [round(68 + i * 887 / N) for i in range(N + 1)]  # cell boundaries

base = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = base.shape

def center(r, c):
    return ((b[c] + b[c + 1]) // 2, (b[r] + b[r + 1]) // 2)  # x, y

def color_at(r, c):
    x, y = center(r, c)
    return tuple(base[y, x])

# --- classify cells ---
open_cells = np.zeros((N, N), bool)
for r in range(N):
    for c in range(N):
        cell = base[b[r]:b[r + 1], b[c]:b[c + 1]]
        open_cells[r, c] = not (cell == 0).all()

def find_color(col):
    m = (base == np.array(col, np.uint8)).all(2)
    ys, xs = np.where(m)
    return m, ys, xs

GREEN, YELLOW = (0, 255, 0), (255, 255, 0)
gm, gy, gx = find_color(GREEN)
start = (int(np.searchsorted(b, gy.mean(), 'right') - 1), int(np.searchsorted(b, gx.mean(), 'right') - 1))

# yellow: key is filled diamond, door is hollow square. Split by cell.
ym, yy, yx = find_color(YELLOW)
cells = {}
for y, x in zip(yy, yx):
    rc = (int(np.searchsorted(b, y, 'right') - 1), int(np.searchsorted(b, x, 'right') - 1))
    cells[rc] = cells.get(rc, 0) + 1
# key (filled) has many more pixels than hollow door
key_cell = max(cells, key=cells.get)
door_cell = min(cells, key=cells.get)

def bfs(s, t):
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < N and 0 <= v[1] < N and open_cells[v] and v not in prev:
                prev[v] = u
                q.append(v)
    path = []
    u = t
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]

p1 = bfs(start, key_cell)
p2 = bfs(key_cell, door_cell)
path = p1 + p2[1:]
key_idx = len(p1) - 1  # index in path where key is collected

# --- agent sprite: copy green pixels relative to start cell ---
sy0, sx0 = b[start[0]], b[start[1]]
sprite_mask = gm[sy0:b[start[0] + 1], sx0:b[start[1] + 1]]
ys, xs = np.where(sprite_mask)
sprite_off = np.stack([ys, xs], 1)  # offsets relative to cell top-left

# background without agent and (later) without key
bg = base.copy()
bg[gm] = 255  # agent sits on white path
bg_nokey = bg.copy()
ky0, kx0 = b[key_cell[0]], b[key_cell[1]]
kcell = bg_nokey[ky0:b[key_cell[0] + 1], kx0:b[key_cell[1] + 1]]
kcell[(kcell == np.array(YELLOW, np.uint8)).all(2)] = 255

# --- timing: move over frames 0..MOVE_END, then hold ---
HOLD = 6
move_frames = N_FRAMES - 1 - HOLD
L = len(path) - 1
frames = []
for f in range(N_FRAMES):
    t = min(f / move_frames, 1.0) * L
    i = min(int(np.floor(t)), L - 1)
    a = t - i
    r0, c0 = path[i]
    r1, c1 = path[i + 1]
    y = b[r0] + (b[r1] - b[r0]) * a
    x = b[c0] + (b[c1] - b[c0]) * a
    collected = t >= key_idx
    img = (bg_nokey if collected else bg).copy()
    yy_ = (sprite_off[:, 0] + int(round(y))).clip(0, H - 1)
    xx_ = (sprite_off[:, 1] + int(round(x))).clip(0, W - 1)
    img[yy_, xx_] = GREEN
    frames.append(img)

frames[0] = base.copy()  # exact first frame

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
       '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
print('start', start, 'key', key_cell, 'door', door_cell, 'path len', L, 'frames', len(frames), '->', OUT)
