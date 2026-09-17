#!/usr/bin/env python3
"""Animate drawing a black circle around the tangent point of the two touching circles."""
import numpy as np, cv2, subprocess, os, math
from PIL import Image

BASE = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, N = 16, 60
base = np.array(Image.open(BASE).convert('RGB'))
H, W = base.shape[:2]

# ---- detect the circles by colour -------------------------------------------
def circles():
    px = base.reshape(-1, 3)
    cols, cnt = np.unique(px, axis=0, return_counts=True)
    bg = cols[cnt.argmax()]
    out = []
    a = base.astype(int)
    for c, n in zip(cols, cnt):
        if n < 2000 or np.array_equal(c, bg):
            continue
        m = np.abs(a - c).sum(-1) < 60
        ys, xs = np.nonzero(m)
        out.append((xs.mean(), ys.mean(), math.sqrt(m.sum() / math.pi)))
    return out

cs = circles()
best, pair = 1e9, None
for i in range(len(cs)):
    for j in range(i + 1, len(cs)):
        (x1, y1, r1), (x2, y2, r2) = cs[i], cs[j]
        d = math.hypot(x2 - x1, y2 - y1)
        gap = abs(d - (r1 + r2))
        if gap < best:
            best, pair = gap, (cs[i], cs[j])
(x1, y1, r1), (x2, y2, r2) = pair
t = r1 / (r1 + r2)
tx, ty = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t

# ---- animation ---------------------------------------------------------------
R, THICK, SS = 38, 5, 4          # marker radius, stroke, supersample factor
DRAW_FRAMES = 50                 # frames spent sweeping the arc, rest holds

def frame(k):
    img = base.copy()
    prog = min(1.0, k / (DRAW_FRAMES - 1)) if k > 0 else 0.0
    if prog <= 0:
        return img
    # ease-in-out sweep
    p = 0.5 - 0.5 * math.cos(math.pi * prog)
    sweep = 360.0 * p
    big = cv2.resize(img, (W * SS, H * SS), interpolation=cv2.INTER_NEAREST)
    c = (int(round(tx * SS)), int(round(ty * SS)))
    if sweep >= 359.9:
        cv2.circle(big, c, R * SS, (0, 0, 0), THICK * SS, cv2.LINE_AA)
    else:
        cv2.ellipse(big, c, (R * SS, R * SS), 0, -90, -90 + sweep, (0, 0, 0), THICK * SS, cv2.LINE_AA)
        # round caps
        for ang in (-90, -90 + sweep):
            a = math.radians(ang)
            pt = (int(round(c[0] + R * SS * math.cos(a))), int(round(c[1] + R * SS * math.sin(a))))
            cv2.circle(big, pt, THICK * SS // 2, (0, 0, 0), -1, cv2.LINE_AA)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
ff = subprocess.Popen(
    ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
     '-crf', '15', '-preset', 'slow', OUT], stdin=subprocess.PIPE)
for k in range(N):
    ff.stdin.write(frame(k).tobytes())
ff.stdin.close(); ff.wait()
print(f'tangent point ({tx:.1f}, {ty:.1f}); wrote {OUT}')
