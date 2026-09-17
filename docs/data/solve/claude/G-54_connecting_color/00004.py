#!/usr/bin/env python3
"""Draw one smooth curve per color connecting its leftmost and rightmost shapes."""
import os, subprocess
import numpy as np, cv2
from PIL import Image

W = H = 1024
FPS, N_FRAMES = 16, 48
SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
OUT = os.path.join(OUT_DIR, 'video.mp4')

base = np.array(Image.open(SRC).convert('RGB'))

# --- detect shapes grouped by exact color -----------------------------------
flat = base.reshape(-1, 3)
cols, counts = np.unique(flat, axis=0, return_counts=True)
bg_col = cols[counts.argmax()]
shape_cols = [tuple(int(v) for v in c) for c, n in zip(cols, counts)
              if n > 500 and not np.array_equal(c, bg_col)]

shapes_mask = np.zeros((H, W), np.uint8)
groups = {}
for col in shape_cols:
    m = np.all(base == col, axis=2).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    comps = []
    for i in range(1, n):
        if st[i][4] < 500:
            continue
        comps.append({'centroid': cen[i], 'mask': (lab == i).astype(np.uint8)})
        shapes_mask |= (lab == i).astype(np.uint8)
    groups[col] = sorted(comps, key=lambda c: c['centroid'][0])

# a small dilation keeps curve endpoints a hair away from shape edges
keepout = cv2.dilate(shapes_mask, np.ones((7, 7), np.uint8))

def bezier(p0, p1, p2, n=600):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def clip_to_gap(pts, mA, mB):
    """Keep only the portion of the polyline between the two shapes."""
    dA = cv2.dilate(mA, np.ones((9, 9), np.uint8))
    dB = cv2.dilate(mB, np.ones((9, 9), np.uint8))
    xi = np.clip(pts[:, 0].round().astype(int), 0, W - 1)
    yi = np.clip(pts[:, 1].round().astype(int), 0, H - 1)
    inA = dA[yi, xi] > 0
    inB = dB[yi, xi] > 0
    start = np.argmax(~inA)                      # first point outside A
    end = len(pts) - np.argmax(~inB[::-1])       # last point outside B
    return pts[start:end]

# --- build curves (leftmost -> rightmost shape of each color) ----------------
curves = []  # (color, points)
# bulge direction: alternate so the two curves stay apart from each other
bulges = {}
order = sorted(groups.items(), key=lambda kv: kv[1][0]['centroid'][1])  # top to bottom
for k, (col, comps) in enumerate(order):
    a, b = comps[0], comps[-1]
    p0, p2 = a['centroid'], b['centroid']
    mid = (p0 + p2) / 2
    d = p2 - p0
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    sign = -1 if k == 0 else 1        # upper curve bulges up, lower bulges down
    ctrl = mid + sign * 0.18 * np.linalg.norm(d) * nrm
    pts = bezier(p0, ctrl, p2)
    pts = clip_to_gap(pts, a['mask'], b['mask'])
    curves.append((col, pts))

# sanity: curves must not touch shapes or each other
def raster(pts, thick=6):
    c = np.zeros((H, W), np.uint8)
    cv2.polylines(c, [pts.round().astype(np.int32)], False, 1, thick, cv2.LINE_8)
    return c
rs = [raster(p) for _, p in curves]
for r in rs:
    assert not (r & shapes_mask).any(), 'curve touches a shape'
for i in range(len(rs)):
    for j in range(i + 1, len(rs)):
        assert not (rs[i] & rs[j]).any(), 'curves cross'

# --- render frames ----------------------------------------------------------
THICK = 6
def render(progress):
    """progress in [0, len(curves)]: how much of the curve sequence is drawn."""
    img = base.copy()
    for idx, (col, pts) in enumerate(curves):
        frac = np.clip(progress - idx, 0, 1)
        if frac <= 0:
            continue
        n = max(2, int(round(frac * len(pts))))
        seg = pts[:n].round().astype(np.int32)
        layer = img.copy()
        cv2.polylines(layer, [seg], False, col, THICK, cv2.LINE_AA)
        # never alter the shapes themselves
        img = np.where(shapes_mask[..., None] > 0, img, layer)
    return img

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for f in range(N_FRAMES):
    # frame 0 is untouched; the last frame is complete; curves drawn one after another
    if f == 0:
        frames.append(base.copy())
        continue
    t = f / (N_FRAMES - 1)
    # smooth ease per curve
    p = t * len(curves)
    frames.append(render(p))
frames[-1] = render(len(curves))

tmp = os.path.join(OUT_DIR, 'frames_tmp')
os.makedirs(tmp, exist_ok=True)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(os.path.join(tmp, f'{i:04d}.png'))
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', os.path.join(tmp, '%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12',
                '-preset', 'slow', OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print('wrote', OUT)
