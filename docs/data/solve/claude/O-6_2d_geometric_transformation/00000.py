import numpy as np, cv2, subprocess, os
from PIL import Image, ImageDraw

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 70, 16
FILL, EDGE, DASH, BG = (50, 129, 77), (50, 50, 50), (100, 100, 100), (240, 240, 240)
CX, CY = 434.0, 434.0          # rotation center (marker center)

im = np.array(Image.open(SRC).convert('RGB'))
H, W = im.shape[:2]

# ---- recover polygon vertices --------------------------------------------
poly_mask = ((im == FILL).all(-1) | (im == EDGE).all(-1)).astype(np.uint8)
cnts, _ = cv2.findContours(poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
approx = cv2.approxPolyDP(cnt.astype(np.int32), 2, True).reshape(-1, 2).astype(float)
far = np.hypot(cnt[:, 0] - CX, cnt[:, 1] - CY) > 16      # drop points hidden by marker

def seg_dist(p, a, b):
    ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

lines = []   # (point, direction) for each real edge, in contour order
n = len(approx)
for i in range(n):
    a, b = approx[i], approx[(i + 1) % n]
    if np.linalg.norm(b - a) < 20:          # marker-induced short edge
        continue
    sel = far & (seg_dist(cnt, a, b) < 2.0)
    pts = cnt[sel]
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    lines.append((c, d))

def intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([d1, -d2]).T
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1

V = np.array([intersect(lines[i], lines[(i + 1) % len(lines)]) for i in range(len(lines))])

# ---- fit CCW rotation angle to dashed outline ----------------------------
dash = (im == DASH).all(-1)
ys, xs = np.nonzero(dash)
D = np.stack([xs, ys], 1).astype(float)

def rot(P, th):     # visually counterclockwise rotation by th (image coords, y down)
    dx, dy = P[:, 0] - CX, P[:, 1] - CY
    c, s = np.cos(th), np.sin(th)
    return np.stack([CX + dx * c + dy * s, CY - dx * s + dy * c], 1)

def cost(th):
    R = rot(V, th); m = len(R)
    best = np.full(len(D), 1e9)
    for i in range(m):
        best = np.minimum(best, seg_dist(D, R[i], R[(i + 1) % m]))
    return best.mean()

grid = np.deg2rad(np.arange(0, 360, 0.5))
th = grid[np.argmin([cost(t) for t in grid])]
for step in (0.05, 0.005):
    g = th + np.deg2rad(np.arange(-1, 1, step))
    th = g[np.argmin([cost(t) for t in g])]
THETA = th
print('vertices:\n', np.round(V, 2))
print('CCW angle: %.3f deg, residual %.3f px' % (np.degrees(THETA), cost(THETA)))

# ---- rendering ------------------------------------------------------------
marker = ((im == (0, 0, 0)).all(-1) | (im == (255, 255, 255)).all(-1))
base = im.copy()
base[poly_mask.astype(bool)] = BG          # scene without the polygon

def render(t):
    img = Image.fromarray(base.copy())
    dr = ImageDraw.Draw(img)
    P = rot(V, t)
    dr.polygon([tuple(p) for p in P], fill=FILL, outline=EDGE, width=1)
    arr = np.array(img)
    arr[marker] = im[marker]               # marker stays on top
    return arr

def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = '/app/output/_frames'
os.makedirs(tmp, exist_ok=True)
for k in range(N_FRAMES):
    if k == 0:
        fr = im
    else:
        fr = render(THETA * ease(k / (N_FRAMES - 1)))
    Image.fromarray(fr).save(f'{tmp}/{k:04d}.png')

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', f'{tmp}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '12', '-preset', 'slow', OUT], check=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print('wrote', OUT)
