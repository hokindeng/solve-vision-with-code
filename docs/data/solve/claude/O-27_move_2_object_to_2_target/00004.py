#!/usr/bin/env python3
"""Move each filled object to its matching dashed outline along a straight line."""
import os, subprocess, colorsys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
OUT = os.path.join(OUT_DIR, 'video.mp4')
N_FRAMES, FPS = 35, 16

img = np.array(Image.open(SRC).convert('RGB'))
bg = img[0, 0].astype(int)
diff = np.abs(img.astype(int) - bg).sum(2)
fg = diff > 30
# Join dashed outlines into single components.
lab, n = ndi.label(ndi.binary_dilation(fg, iterations=12))

comps = []
for i in range(1, n + 1):
    m = (lab == i) & fg
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    fill = m.sum() / ((x1 - x0 + 1) * (y1 - y0 + 1))
    med = np.median(img[m], 0) / 255.0
    hue = colorsys.rgb_to_hsv(*med)[0]
    comps.append(dict(mask=m, center=((x0 + x1) / 2.0, (y0 + y1) / 2.0), fill=fill, hue=hue))

objects = [c for c in comps if c['fill'] > 0.15]
outlines = [c for c in comps if c['fill'] <= 0.15]
assert len(objects) == 2 and len(outlines) == 2, (len(objects), len(outlines))

def hue_dist(a, b):
    d = abs(a - b); return min(d, 1 - d)

# Match objects to outlines by colour (hue); pick the assignment with minimal total hue distance.
best = min(([0, 1], [1, 0]), key=lambda p: sum(hue_dist(objects[i]['hue'], outlines[p[i]]['hue']) for i in range(2)))

# Static background: scene with the moving objects erased (background is flat).
base = img.copy()
for o in objects:
    m = ndi.binary_dilation(o['mask'], iterations=1)
    base[m] = bg

# Extract object sprites (RGBA) with soft alpha at anti-aliased edges.
sprites = []
for i, o in enumerate(objects):
    m = ndi.binary_dilation(o['mask'], iterations=1)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = img[y0:y1, x0:x1]
    alpha = np.clip(diff[y0:y1, x0:x1] / 60.0, 0, 1) * m[y0:y1, x0:x1]
    alpha[o['mask'][y0:y1, x0:x1]] = 1.0
    start = np.array(o['center']); end = np.array(outlines[best[i]]['center'])
    sprites.append(dict(rgb=rgb.astype(float), alpha=alpha[..., None], origin=np.array([x0, y0], float),
                        start=start, end=end))

def composite(frame, spr, shift):
    ox, oy = np.round(spr['origin'] + shift).astype(int)
    h, w = spr['alpha'].shape[:2]
    H, W = frame.shape[:2]
    fx0, fy0 = max(ox, 0), max(oy, 0); fx1, fy1 = min(ox + w, W), min(oy + h, H)
    if fx1 <= fx0 or fy1 <= fy0: return
    sx0, sy0 = fx0 - ox, fy0 - oy
    a = spr['alpha'][sy0:sy0 + fy1 - fy0, sx0:sx0 + fx1 - fx0]
    c = spr['rgb'][sy0:sy0 + fy1 - fy0, sx0:sx0 + fx1 - fx0]
    region = frame[fy0:fy1, fx0:fx1].astype(float)
    frame[fy0:fy1, fx0:fx1] = np.clip(region * (1 - a) + c * a, 0, 255).astype(np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
frames = []
for k in range(N_FRAMES):
    t = k / (N_FRAMES - 1)
    s = t * t * (3 - 2 * t)  # smoothstep easing; both objects share the same timing
    if k == 0:
        frames.append(img.copy()); continue
    f = base.copy()
    for spr in sprites:
        composite(f, spr, (spr['end'] - spr['start']) * s)
    frames.append(f)

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
       '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in frames: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
print('wrote', OUT, 'frames', len(frames))
