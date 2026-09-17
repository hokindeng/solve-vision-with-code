#!/usr/bin/env python3
"""Mirror the left half of the grid onto the right half, animated over 35 frames."""
import numpy as np
from PIL import Image
import subprocess, os, shutil

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, NFRAMES = 16, 35
FILL = np.array([124, 58, 237], np.uint8)
LINE = np.array([203, 213, 225], np.uint8)
WHITE = np.array([255, 255, 255], np.uint8)

base = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = base.shape

# --- detect grid geometry from grid-line pixels ---
g = np.all(base == LINE, axis=2)
def groups(idx):
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1: cur.append(i)
        else: out.append(cur); cur = [i]
    out.append(cur); return out
rg = groups(np.where(g.sum(1) > W // 4)[0])
cg = groups(np.where(g.sum(0) > H // 4)[0])
row_bounds = [(rg[i][-1] + 1, rg[i + 1][0]) for i in range(len(rg) - 1)]  # [y0,y1)
col_bounds = [(cg[i][-1] + 1, cg[i + 1][0]) for i in range(len(cg) - 1)]
R, C = len(row_bounds), len(col_bounds)

def cell_state(r, c):
    y0, y1 = row_bounds[r]; x0, x1 = col_bounds[c]
    patch = base[y0:y1, x0:x1].reshape(-1, 3)
    return np.all(patch == FILL, axis=1).mean() > 0.5

state = np.array([[cell_state(r, c) for c in range(C)] for r in range(R)])
half = C // 2
# cells to fill: right side, mirror of a filled left cell, currently empty
todo = [(r, C - 1 - c) for r in range(R) for c in range(half)
        if state[r, c] and not state[r, C - 1 - c]]
# order: top-to-bottom, inside-out
todo.sort(key=lambda rc: (rc[0], rc[1]))
print(f'grid {R}x{C}, filling {len(todo)} cells: {todo}')

def ease(t):
    return t * t * (3 - 2 * t)

def draw(frame_idx):
    img = base.copy()
    n = len(todo)
    # global progress; each cell grows over a window, staggered
    T = (NFRAMES - 1)
    span = 10  # frames per cell animation
    total = T - span  # last cell starts here so it finishes at final frame
    for k, (r, c) in enumerate(todo):
        start = total * k / max(n - 1, 1) if n > 1 else 0
        t = np.clip((frame_idx - start) / span, 0, 1)
        if frame_idx >= T: t = 1.0
        if t <= 0: continue
        s = ease(t)
        y0, y1 = row_bounds[r]; x0, x1 = col_bounds[c]
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        hh, hw = (y1 - y0) / 2 * s, (x1 - x0) / 2 * s
        ya, yb = int(round(cy - hh)), int(round(cy + hh))
        xa, xb = int(round(cx - hw)), int(round(cx + hw))
        if t >= 1: ya, yb, xa, xb = y0, y1, x0, x1
        sub = img[ya:yb, xa:xb]
        mask = np.all(sub == WHITE, axis=2)
        sub[mask] = FILL
    return img

tmp = '/app/output/frames'
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
for i in range(NFRAMES):
    Image.fromarray(draw(i)).save(f'{tmp}/{i:04d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-preset', 'slow', OUT], check=True)
shutil.rmtree(tmp)
print('wrote', OUT)
