#!/usr/bin/env python3
"""Rearrange shapes: group by type, sort by size (small->large, left->right), one horizontal row."""
import os, subprocess, tempfile, shutil
import numpy as np, cv2
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N_FRAMES = 16, 96
HOLD_START, HOLD_END = 8, 8

img = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = img.shape
bg_color = np.array([235, 235, 235], np.uint8)

# --- detect shapes ---------------------------------------------------------
mask = (np.abs(img.astype(int) - bg_color).sum(2) > 0).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
shapes = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    m = (lab[y:y+h, x:x+w] == i)
    sprite = img[y:y+h, x:x+w].copy()
    # type via fill colour (most common non-outline colour), size via bbox
    px = sprite[m]
    px = px[(px != [80, 80, 80]).any(1)]
    vals, cnt = np.unique(px, axis=0, return_counts=True)
    fill = tuple(vals[cnt.argmax()])
    # solidity separates triangle (~0.5) from circle (~0.785)
    solidity = area / float(w * h)
    kind = 'circle' if solidity > 0.65 else 'triangle'
    shapes.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), size=max(w, h),
                       kind=kind, fill=fill, sprite=sprite, mask=m, cx=cent[i][0]))

# background with shapes removed
background = np.empty_like(img); background[:] = bg_color

# --- target layout -------------------------------------------------------
kinds = sorted({s['kind'] for s in shapes},
               key=lambda k: np.mean([s['cx'] for s in shapes if s['kind'] == k]))
ordered = []
for k in kinds:
    ordered += sorted([s for s in shapes if s['kind'] == k], key=lambda s: s['size'])

total_w = sum(s['w'] for s in ordered)
margin = 40
gap = (W - 2 * margin - total_w) / (len(ordered) - 1)
row_cy = H / 2.0
x = margin
for s in ordered:
    s['tx'] = int(round(x))
    s['ty'] = int(round(row_cy - s['h'] / 2.0))
    x += s['w'] + gap

# --- animation -----------------------------------------------------------
def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep

move_frames = N_FRAMES - HOLD_START - HOLD_END

def render(t):
    frame = background.copy()
    for s in shapes:
        px = int(round(s['x'] + (s['tx'] - s['x']) * t))
        py = int(round(s['y'] + (s['ty'] - s['y']) * t))
        region = frame[py:py+s['h'], px:px+s['w']]
        region[s['mask']] = s['sprite'][s['mask']]
    return frame

tmp = tempfile.mkdtemp()
for f in range(N_FRAMES):
    if f < HOLD_START:
        t = 0.0
    elif f >= N_FRAMES - HOLD_END:
        t = 1.0
    else:
        t = ease((f - HOLD_START) / float(move_frames - 1))
    Image.fromarray(render(t)).save(os.path.join(tmp, f'{f:04d}.png'))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', os.path.join(tmp, '%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-preset', 'slow', OUT], check=True)
shutil.rmtree(tmp)
print('wrote', OUT)
