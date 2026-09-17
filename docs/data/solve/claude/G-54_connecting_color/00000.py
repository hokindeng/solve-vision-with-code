#!/usr/bin/env python3
"""Draw a smooth curve between the leftmost and rightmost shape of each color,
step by step, without crossing any shape or other curve."""
import numpy as np, cv2, subprocess, os, shutil, tempfile
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
FPS, NFRAMES = 16, 48
THICK = 6

img0 = np.array(Image.open(SRC).convert('RGB'))
H, W = img0.shape[:2]
bg = img0[0, 0].copy()

# --- find colors and shapes -----------------------------------------------
cols, counts = np.unique(img0.reshape(-1, 3), axis=0, return_counts=True)
colors = [tuple(int(v) for v in c) for c, n in zip(cols, counts)
          if n > 2000 and tuple(c) != tuple(bg)]

shapes_all = np.zeros((H, W), np.uint8)
groups = {}
for col in colors:
    m = np.all(img0 == np.array(col, np.uint8), axis=2).astype(np.uint8)
    shapes_all |= m
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    comps = [(cen[i], lab == i) for i in range(1, n) if st[i][4] > 500]
    comps.sort(key=lambda c: c[0][0])
    groups[col] = (comps[0], comps[-1])

# --- curve construction ----------------------------------------------------
def bezier(p0, p1, p2, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def clip_to_boundaries(pts, mA, mB):
    """Trim curve so it starts/ends at the shape boundaries."""
    inside = lambda p, m: m[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]
    i = 0
    while i < len(pts) - 1 and inside(pts[i], mA): i += 1
    j = len(pts) - 1
    while j > 0 and inside(pts[j], mB): j -= 1
    return pts[max(i - 1, 0):j + 2]

def collides(pts, obstacle):
    canvas = np.zeros((H, W), np.uint8)
    cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 1, THICK + 4)
    return bool((canvas & obstacle).any())

# order colors by the x of their leftmost shape
order = sorted(colors, key=lambda c: groups[c][0][0][0])
curves = []
drawn = np.zeros((H, W), np.uint8)
for col in order:
    (cA, mA), (cB, mB) = groups[col]
    p0, p2 = np.array(cA, float), np.array(cB, float)
    chord = p2 - p0
    L = np.linalg.norm(chord)
    nrm = np.array([-chord[1], chord[0]]) / L
    other_shapes = (shapes_all & ~(mA | mB)).astype(np.uint8)
    obstacle = other_shapes | drawn
    ok = None
    for amp in [0.0] + [s * a for a in np.arange(0.1, 1.6, 0.1) for s in (1, -1)]:
        ctrl = (p0 + p2) / 2 + nrm * amp * L
        pts = clip_to_boundaries(bezier(p0, ctrl, p2), mA, mB)
        if not collides(pts, obstacle):
            ok = pts; break
    if ok is None:
        ok = clip_to_boundaries(bezier(p0, (p0 + p2) / 2, p2), mA, mB)
    curves.append((col, ok))
    cv2.polylines(drawn, [np.round(ok).astype(np.int32)], False, 1, THICK)

# --- render frames ----------------------------------------------------------
def draw_partial(img, col, pts, frac):
    n = max(2, int(round(len(pts) * frac)))
    if frac <= 0: return
    cv2.polylines(img, [np.round(pts[:n]).astype(np.int32)], False, col,
                  THICK, lineType=cv2.LINE_AA)

tmp = tempfile.mkdtemp()
K = len(curves)
for f in range(NFRAMES):
    frame = img0.copy()
    t = f / (NFRAMES - 1)          # 0 at first frame, 1 at last
    for k, (col, pts) in enumerate(curves):
        frac = np.clip((t * K - k), 0, 1)
        # ease for smoother stroke
        frac = frac * frac * (3 - 2 * frac)
        draw_partial(frame, col, pts, frac)
    Image.fromarray(frame).save(f'{tmp}/{f:04d}.png')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '15', '-r', str(FPS), OUT], check=True)
shutil.rmtree(tmp)
print('wrote', OUT)
