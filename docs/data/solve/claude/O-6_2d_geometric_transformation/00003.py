import numpy as np, cv2, subprocess, os
from PIL import Image

SRC = '/app/first_frame.png'
OUT = '/app/output/video.mp4'
N_FRAMES, FPS = 70, 16
BG = (240, 240, 240); POLY = (192, 152, 55); EDGE = (50, 50, 50); DASH = (100, 100, 100)

im = np.array(Image.open(SRC).convert('RGB'))
H, W, _ = im.shape

def mask(c): return np.all(im == np.array(c), axis=2)

# rotation center = centroid of the black marker pixels
ys, xs = np.nonzero(mask((0, 0, 0)))
C = np.array([xs.mean() + 0.5, ys.mean() + 0.5])

# ---- polygon vertices: fit lines to the dark outline, intersect neighbours ----
shape = (mask(POLY) | mask(EDGE)).astype(np.uint8)
cs, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
ap = cv2.approxPolyDP(cnt.astype(np.int32), 3, True).reshape(-1, 2).astype(float)
# drop approx vertices hidden under the marker (within 14 px of center); real vertex is at C
keep = [v for v in ap if np.linalg.norm(v - C) > 14]
# order vertices by angle around polygon centroid
cen = np.mean(keep + [C], axis=0)
verts = sorted(keep + [C], key=lambda v: np.arctan2(v[1] - cen[1], v[0] - cen[0]))
verts = np.array(verts)
# refine each edge by a least-squares line through contour points near it
edge_pts = mask(EDGE)
ey, ex = np.nonzero(edge_pts); E = np.stack([ex + 0.5, ey + 0.5], 1)
lines = []
n = len(verts)
for i in range(n):
    a, b = verts[i], verts[(i + 1) % n]
    ab = b - a; L = np.linalg.norm(ab); d = ab / L; nrm = np.array([-d[1], d[0]])
    t = (E - a) @ d; off = np.abs((E - a) @ nrm)
    sel = (t > 8) & (t < L - 8) & (off < 2.5)
    if sel.sum() < 6:
        lines.append((a, d)); continue
    P = E[sel]; m = P.mean(0); u, s, vt = np.linalg.svd(P - m); lines.append((m, vt[0]))
def intersect(l1, l2):
    p, d = l1; q, e = l2
    A = np.array([d, -e]).T; t = np.linalg.solve(A, q - p); return p + t[0] * d
V = []
for i in range(n):
    v = intersect(lines[(i - 1) % n], lines[i])
    V.append(v)
V = np.array(V)
# vertex nearest to the center is the center itself
k = int(np.argmin(np.linalg.norm(V - C, axis=1))); V[k] = C

# ---- target angle: rotate polygon CW about C to best match dashed pixels ----
dy, dx = np.nonzero(mask(DASH)); D = np.stack([dx + 0.5, dy + 0.5], 1)
def rot(P, th):
    r = np.deg2rad(th); R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return (P - C) @ R.T + C
def segdist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
def cost(th):
    Wv = rot(V, th)
    return np.min([segdist(D, Wv[i], Wv[(i + 1) % n]) for i in range(n)], axis=0).mean()
ths = np.arange(1, 360, 0.5); best = ths[int(np.argmin([cost(t) for t in ths]))]
fine = np.arange(best - 1.5, best + 1.5, 0.01); THETA = float(fine[int(np.argmin([cost(t) for t in fine]))])
print('vertices', V.round(2).tolist(), 'theta', THETA, 'cost', cost(THETA))

# ---- static layers ----
base = im.copy()
base[shape.astype(bool)] = BG                       # erase polygon; dashes & marker untouched
yy, xx = np.mgrid[0:H, 0:W]
marker_mask = (xx + 0.5 - C[0]) ** 2 + (yy + 0.5 - C[1]) ** 2 <= 12.5 ** 2  # marker disk, pasted on top
marker_px = im.copy()

def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)

def render(th):
    f = base.copy()
    P = rot(V, th)
    pts = np.round((P - 0.5)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(f, [pts], POLY, lineType=cv2.LINE_8)
    cv2.polylines(f, [pts], True, EDGE, 1, lineType=cv2.LINE_8)
    f[marker_mask] = marker_px[marker_mask]
    return f

os.makedirs('/app/output', exist_ok=True)
frames = []
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    if i == 0: frames.append(im.copy())
    else: frames.append(render(THETA * ease(t)))

tmp = '/app/output/frames'; os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames): Image.fromarray(f).save(f'{tmp}/{i:04d}.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%04d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT], check=True)
for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print('wrote', OUT)
