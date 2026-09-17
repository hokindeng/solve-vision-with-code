#!/usr/bin/env python3
"""Generate the solution video: circle every point that lies fully inside the
overlap of the two semi-transparent shapes in first_frame.png."""
import numpy as np, cv2, subprocess, os, tempfile
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, 'first_frame.png')
OUT = os.path.join(BASE, 'output', 'video.mp4')
W = H = 1024
FPS = 16
N_FRAMES = 37

img = np.array(Image.open(FIRST).convert('RGB'))

# ---- 1. colours: white bg, shape A, shape B, overlap, black dots -------------
cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
order = np.argsort(-counts)
cols, counts = cols[order], counts[order]
black = np.all(img == 0, axis=2)
white = np.all(img == 255, axis=2)
shape_cols = [c for c in cols if not (np.all(c == 0) or np.all(c == 255))][:3]
# The overlap colour is the one that is a blend of the other two (closest to
# the region bounded by both shapes). Determine it geometrically: the overlap
# region is adjacent to both other colours; pick the colour whose region has
# the smallest area among the three (overlap is partial).
masks = {tuple(c): np.all(img == c, axis=2) for c in shape_cols}
overlap_col = min(masks, key=lambda c: masks[c].sum())
overlap = masks[overlap_col].astype(np.uint8)

# fill holes (dots sitting inside the overlap)
ff = overlap.copy()
h, w = ff.shape
flood = np.zeros((h + 2, w + 2), np.uint8)
inv = (1 - ff).copy()
cv2.floodFill(inv, flood, (0, 0), 0)          # inv now = holes only
filled = (ff | inv).astype(bool)

# ---- 2. locate the dots -----------------------------------------------------
n, lab, stats, cent = cv2.connectedComponentsWithStats(black.astype(np.uint8), 8)
areas = stats[1:, 4]
base_area = int(np.median(areas))
base_size = int(np.median(stats[1:, 2]))            # typical dot diameter
R = base_size / 2.0
k = np.zeros((base_size, base_size), np.uint8)
cv2.circle(k, (base_size // 2, base_size // 2), base_size // 2, 1, -1)
# match full disk template to separate touching dots
resp = cv2.filter2D(black.astype(np.float32), -1, k.astype(np.float32),
                    borderType=cv2.BORDER_CONSTANT)
full = resp >= k.sum() - 0.5
pts = []
for i in range(1, n):
    x, y, bw, bh, area = stats[i]
    if area <= base_area * 1.3:
        pts.append(tuple(cent[i]))
    else:
        sub = full[y:y + bh, x:x + bw].astype(np.uint8)
        m, sl, ss, sc = cv2.connectedComponentsWithStats(sub, 8)
        for j in range(1, m):
            pts.append((sc[j][0] + x, sc[j][1] + y))

# ---- 3. which dots are fully inside the overlap ----------------------------
inside = []
for (cx, cy) in pts:
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (R + 1.5) ** 2   # dot + 1px margin
    if filled[disk].all():
        inside.append((cx, cy))
inside.sort(key=lambda p: (p[0], p[1]))
print(f'{len(pts)} points, {len(inside)} inside overlap:', [(round(x), round(y)) for x, y in inside])

# ---- 4. animate -------------------------------------------------------------
RED = (220, 30, 30)
FINAL_R = int(round(R * 2.2))
THICK = 3
GROW = 4                       # frames for one circle to grow in
first_anim = 3                 # frames of "thinking" before first circle
last_full = N_FRAMES - 2       # index at which all circles are complete
n_c = len(inside)
if n_c:
    span = last_full - first_anim - GROW
    starts = [first_anim + int(round((i + 1) * span / (n_c + 1))) for i in range(n_c)]
else:
    starts = []

def draw_frame(t):
    fr = img.copy()
    for (cx, cy), s in zip(inside, starts):
        if t < s:
            continue
        prog = min(1.0, (t - s + 1) / GROW)
        r = int(round(R + 2 + (FINAL_R - R - 2) * prog))
        # supersampled anti-aliased circle
        cv2.circle(fr, (int(round(cx)), int(round(cy))), r, RED, THICK, lineType=cv2.LINE_AA)
    return fr

tmp = tempfile.mkdtemp()
for t in range(N_FRAMES):
    Image.fromarray(draw_frame(t)).save(os.path.join(tmp, f'{t:04d}.png'))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', os.path.join(tmp, '%04d.png'), '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT], check=True)
print('wrote', OUT)
