#!/usr/bin/env python3
"""Move each colored animal face (left) onto its matching dark outline (right)."""
import subprocess, os
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

W = H = 1024
FPS, N = 16, 64
SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'

base = np.array(Image.open(SRC).convert('RGB'))
bg = base[5, 5].astype(int)
diff = np.abs(base.astype(int) - bg).max(axis=2)

# --- segment left faces (any deviation from bg, incl. near-white muzzles) ---
mask = diff >= 3
mask[:, 505:521] = False               # ignore the divider
lab, n = ndimage.label(mask)
comps = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(ys) < 400:
        continue
    comps.append(dict(idx=i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max()))
faces = [c for c in comps if c['x1'] < 512]
outlines = [c for c in comps if c['x0'] > 512]
for c in faces + outlines:
    c['w'] = c['x1'] - c['x0'] + 1; c['h'] = c['y1'] - c['y0'] + 1
    c['cx'] = (c['x0'] + c['x1']) / 2; c['cy'] = (c['y0'] + c['y1']) / 2

# --- match faces to outlines by bounding-box size ---
cost = np.array([[abs(f['w'] - o['w']) + abs(f['h'] - o['h']) for o in outlines] for f in faces])
ri, ci = linear_sum_assignment(cost)

# --- sprites ---
sprites = []
for fi, oi in zip(ri, ci):
    f, o = faces[fi], outlines[oi]
    m = np.zeros((H, W), bool)
    m[f['y0']:f['y1'] + 1, f['x0']:f['x1'] + 1] = mask[f['y0']:f['y1'] + 1, f['x0']:f['x1'] + 1]
    m = ndimage.binary_fill_holes(m)
    sub = m[f['y0']:f['y1'] + 1, f['x0']:f['x1'] + 1]
    rgb = base[f['y0']:f['y1'] + 1, f['x0']:f['x1'] + 1].copy()
    sprites.append(dict(rgb=rgb, alpha=sub, start=(f['x0'], f['y0']),
                        delta=(o['cx'] - f['cx'], o['cy'] - f['cy']), full=m))

sprites.sort(key=lambda s: -s['alpha'].sum())  # draw big sprites first, small on top

# background with faces removed
clean = base.copy()
for s in sprites:
    clean[s['full']] = bg

def ease(t):  # smooth ease-in-out
    return t * t * (3 - 2 * t)

def blit(frame, s, dx, dy):
    x0 = int(round(s['start'][0] + dx)); y0 = int(round(s['start'][1] + dy))
    h, w = s['alpha'].shape
    region = frame[y0:y0 + h, x0:x0 + w]
    region[s['alpha']] = s['rgb'][s['alpha']]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p', '-crf', '15', OUT], stdin=subprocess.PIPE)
for k in range(N):
    t = ease(k / (N - 1))
    if k == 0:
        frame = base.copy()
    else:
        frame = clean.copy()
        for s in sprites:
            blit(frame, s, s['delta'][0] * t, s['delta'][1] * t)
    ff.stdin.write(frame.tobytes())
ff.stdin.close(); ff.wait()
print('wrote', OUT, 'sprites:', [(s['start'], tuple(round(d) for d in s['delta'])) for s in sprites])
