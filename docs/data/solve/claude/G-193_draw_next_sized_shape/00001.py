#!/usr/bin/env python3
"""Draw the next shape in the size cycle inside the empty box, step by step.

Row: medium, small, large, medium, small  ->  next = large pentagon.
Only pixels inside the empty dashed box are ever changed.
"""
import subprocess, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N = 16, 60
W = H = 1024

base = np.array(Image.open(SRC).convert('RGB'))
img = base.astype(int)

# ---- analyse the scene -------------------------------------------------
shape_mask = (img != 255).any(2) & ~((img < 60).all(2))          # coloured pixels
color = tuple(int(v) for v in np.median(img[shape_mask], axis=0))
lab, n = ndimage.label(shape_mask)
shapes = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    shapes.append(dict(id=i, x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                       cx=(xs.min() + xs.max()) / 2, h=ys.max() - ys.min() + 1))
shapes.sort(key=lambda s: s['cx'])

# empty dashed box (black pixels)
dark = (img < 60).all(2)
ys, xs = np.where(dark)
bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
box_cx = (bx0 + bx1) / 2

# size classes by height, cycle detection
heights = [s['h'] for s in shapes]
levels = sorted(set(heights))
def cls(h):
    return min(range(len(levels)), key=lambda k: abs(levels[k] - h))
seq = [cls(h) for h in heights]
period = next(p for p in range(1, len(seq) + 1)
              if all(seq[i] == seq[i - p] for i in range(p, len(seq))))
next_cls = seq[len(seq) % period]
template = next(s for s in shapes if cls(s['h']) == next_cls)   # a shape of the wanted size
dx = int(round(box_cx - template['cx']))                        # shift into the empty box

# exact pixel copy of the template shape (keeps anti-aliasing identical)
tm = (lab == template['id'])
final = base.copy()
tys, txs = np.where(tm)
final[tys, txs + dx] = base[tys, txs]

# geometry of the new pentagon (pointing up) for the animated drawing
tx0, tx1, ty0, ty1 = template['x0'] + dx, template['x1'] + dx, template['y0'], template['y1']
R = (ty1 - ty0 + 1) / (1 + np.cos(np.pi / 5))
cx, cy = (tx0 + tx1 + 1) / 2, ty0 + R
verts = [(cx + R * np.sin(2 * np.pi * k / 5), cy - R * np.cos(2 * np.pi * k / 5)) for k in range(5)]

# region we're allowed to touch: strictly inside the dashed box
inner = np.zeros((H, W), bool)
inner[by0 + 4:by1 - 3, bx0 + 4:bx1 - 3] = True

def compose(layer_rgba):
    """Alpha-composite an RGBA overlay onto the base, clipped to the box interior."""
    ov = np.array(layer_rgba).astype(float)
    a = (ov[..., 3:] / 255.0) * inner[..., None]
    out = base * (1 - a) + ov[..., :3] * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

def smooth(t):
    return t * t * (3 - 2 * t)

frames = []
HOLD0, TRACE, FILL = 10, 24, 20          # 10 + 24 + 20 = 54, then hold 6
for f in range(N):
    if f < HOLD0:
        frames.append(base.copy()); continue
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    t = f - HOLD0
    if t < TRACE:                                     # trace outline edge by edge
        prog = smooth((t + 1) / TRACE) * 5
        for k in range(5):
            if prog <= k: break
            p, q = verts[k], verts[(k + 1) % 5]
            frac = min(1.0, prog - k)
            end = (p[0] + (q[0] - p[0]) * frac, p[1] + (q[1] - p[1]) * frac)
            d.line([p, end], fill=color + (255,), width=3)
        for k in range(5):
            if prog >= k:
                v = verts[k]; d.ellipse([v[0]-1.5, v[1]-1.5, v[0]+1.5, v[1]+1.5], fill=color + (255,))
        frames.append(compose(layer)); continue
    t -= TRACE
    if t < FILL:                                      # fill sweeps top -> bottom
        frac = smooth((t + 1) / FILL)
        d.polygon(verts, outline=color + (255,), width=3)
        fill_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(fill_layer).polygon(verts, fill=color + (255,))
        fl = np.array(fill_layer)
        ylim = int(round(ty0 + (ty1 + 1 - ty0) * frac))
        fl[ylim:, :, 3] = 0
        layer = Image.alpha_composite(layer, Image.fromarray(fl))
        if frac >= 1.0:
            frames.append(final.copy())
        else:
            frames.append(compose(layer))
        continue
    frames.append(final.copy())

frames[-1] = final.copy()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = '/app/output/_frames'
os.makedirs(tmp, exist_ok=True)
for i, fr in enumerate(frames):
    Image.fromarray(fr).save(f'{tmp}/{i:04d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-preset', 'slow', OUT], check=True)
for fn in os.listdir(tmp):
    os.remove(f'{tmp}/{fn}')
os.rmdir(tmp)
print('sizes:', seq, 'period', period, '-> next class', next_cls, '| shift', dx, '| wrote', OUT)
