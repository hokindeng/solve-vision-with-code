#!/usr/bin/env python3
"""Draw a smooth curve between the leftmost and rightmost shape of each color."""
import numpy as np, cv2, subprocess, os, shutil
from PIL import Image

W = H = 1024
FPS, NFRAMES = 16, 48
SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
TMP = '/app/output/_frames'
BG = (255, 255, 255)
THICK = 6

img = np.array(Image.open(SRC).convert('RGB'))

# ---- detect shapes ------------------------------------------------------
cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
colors = [tuple(int(v) for v in c) for c, n in zip(cols, counts)
          if tuple(c) != BG and n > 2000]
shapes = {}  # color -> list of (mask, centroid)
for c in colors:
    m = np.all(img == c, axis=2).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    comps = [(lab == i, cen[i]) for i in range(1, n) if st[i][4] > 2000]
    shapes[c] = comps
assert len(shapes) == 3 and all(len(v) == 2 for v in shapes.values()), shapes

all_shapes = np.zeros((H, W), np.uint8)
for v in shapes.values():
    for m, _ in v:
        all_shapes |= m.astype(np.uint8)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THICK + 3, THICK + 3))
shapes_dil = cv2.dilate(all_shapes, k)  # keep-out zone for curves


def bezier(p0, p1, p2, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def trim(pts, ma, mb):
    """Keep only the part of the polyline outside the two endpoint shapes."""
    ins_a = [ma[int(round(y)), int(round(x))] for x, y in pts]
    ins_b = [mb[int(round(y)), int(round(x))] for x, y in pts]
    i0 = max(i for i, v in enumerate(ins_a) if v) + 1
    i1 = min(i for i, v in enumerate(ins_b) if v) - 1
    return pts[i0:i1 + 1]


def path_mask(pts, thick):
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, thick)
    return m


curves = []  # (color, pts) in drawing order: top to bottom
occupied = np.zeros((H, W), np.uint8)
order = sorted(shapes.items(), key=lambda kv: min(c[1] for _, c in kv[1]))
for color, comps in order:
    comps = sorted(comps, key=lambda mc: mc[1][0])  # leftmost, rightmost
    (ma, ca), (mb, cb) = comps
    p0, p2 = np.array(ca), np.array(cb)
    d = p2 - p0
    nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)  # "up" for L->R
    chosen = None
    for off in [60, -60, 90, -90, 120, -120, 150, -150, 30, -30, 0]:
        ctrl = (p0 + p2) / 2 + nrm * off
        pts = trim(bezier(p0, ctrl, p2), ma, mb)
        pm = path_mask(pts, THICK)
        # keep-out: other shapes (dilated) and previously drawn curves (dilated)
        keep_out = shapes_dil & ~cv2.dilate(ma.astype(np.uint8) | mb.astype(np.uint8), k)
        if not (pm & keep_out).any() and not (pm & occupied).any():
            chosen = pts
            break
    assert chosen is not None, f'no free path for {color}'
    curves.append((color, chosen))
    occupied |= cv2.dilate(path_mask(chosen, THICK), k)

# ---- animation ----------------------------------------------------------
def draw_partial(base, color, pts, frac):
    n = max(2, int(round(len(pts) * frac)))
    if frac <= 0:
        return base
    cv2.polylines(base, [np.round(pts[:n]).astype(np.int32)], False,
                  color, THICK, lineType=cv2.LINE_AA)
    return base


# schedule: frame 0 = original; 3 curves drawn sequentially; last frames hold
hold_start, hold_end = 2, 4
active = NFRAMES - hold_start - hold_end  # frames spent drawing
per = active / len(curves)

if os.path.isdir(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)
for f in range(NFRAMES):
    frame = img.copy()
    t = f - hold_start
    for i, (color, pts) in enumerate(curves):
        frac = np.clip((t - i * per) / (per - 1), 0, 1) if t >= 0 else 0
        # ease in/out for a natural stroke
        frac = 0.5 - 0.5 * np.cos(np.pi * frac)
        if frac > 0:
            draw_partial(frame, color, pts, frac)
    Image.fromarray(frame).save(f'{TMP}/{f:04d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{TMP}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '15', '-r', str(FPS), OUT], check=True)
shutil.rmtree(TMP)
print('wrote', OUT)
