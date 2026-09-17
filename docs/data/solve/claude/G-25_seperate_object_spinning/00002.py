#!/usr/bin/env python3
"""Rotate each object in place to match its dashed target, then slide it right into the target."""
import math, subprocess, os
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'first_frame.png')
OUT = os.path.join(HERE, 'output', 'video.mp4')
W = H = 1024; FPS = 16; NF = 48

OUTLINE = (120, 120, 120)   # object outline color
DASH = (90, 90, 90)         # dashed target color
BG = (255, 255, 255)

img = np.array(Image.open(SRC).convert('RGB'))

def mask(c):
    return np.all(img == np.array(c, np.uint8), axis=2)

# ---------- objects ----------
allmask = ~mask(BG) & ~mask(DASH) & ~mask(OUTLINE)
fill_colors = [tuple(int(v) for v in c) for c in np.unique(img[allmask].reshape(-1, 3), axis=0)]
ol = mask(OUTLINE)

objects = []
for c in fill_colors:
    fm = mask(c)
    d = cv2.dilate(fm.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    om = fm | (d & ol)                       # fill + its outline
    ys, xs = np.nonzero(om)
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5  # pixel centers
    cnt = max(cv2.findContours(om.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    area = om.sum()
    # circularity test
    per = cv2.arcLength(cnt, True)
    circ = 4 * math.pi * area / (per * per)
    ow = int(round(len(np.nonzero(d & ol)[0]) / per))  # outline width estimate
    if len(cv2.approxPolyDP(cnt, 4, True)) >= 10:
        R = math.sqrt(area / math.pi)
        objects.append(dict(color=c, kind='circle', N=0, cx=cx, cy=cy, R=R, theta=0.0, ow=max(1, ow)))
        continue
    ap = cv2.approxPolyDP(cnt, 4, True).reshape(-1, 2).astype(float) + 0.5
    # merge near-duplicate vertices
    verts = []
    for p in ap:
        if not verts or np.linalg.norm(p - verts[-1]) > 12:
            verts.append(p)
    if len(verts) > 1 and np.linalg.norm(verts[0] - verts[-1]) <= 12:
        verts.pop()
    verts = np.array(verts)
    N = len(verts)
    rel = verts - [cx, cy]
    R = np.linalg.norm(rel, axis=1).mean()
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    # circular mean of vertex angles modulo 2pi/N
    z = np.exp(1j * ang * N).mean()
    theta = np.angle(z) / N
    objects.append(dict(color=c, kind='poly', N=N, cx=cx, cy=cy, R=R, theta=theta, ow=max(1, ow)))


def render_mask(o, cx, cy, R, theta):
    im = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(im)
    oo = dict(o); oo['R'] = R
    if o['kind'] == 'circle':
        dr.ellipse([cx - 0.5 - R, cy - 0.5 - R, cx - 0.5 + R, cy - 0.5 + R], fill=255)
    else:
        V = poly_pts(oo, cx, cy, theta)
        dr.polygon([(float(x) - 0.5, float(y) - 0.5) for x, y in V], fill=255, outline=255, width=o['ow'])
    return np.array(im) > 0

def refine_object(o, target):
    cx, cy, R, th = o['cx'], o['cy'], o['R'], o['theta']
    def cost(cx_, cy_, R_, th_):
        return int((render_mask(o, cx_, cy_, R_, th_) ^ target).sum())
    cur = cost(cx, cy, R, th)
    for step in [1.0, 0.5, 0.25, 0.125]:
        improved = True
        while improved:
            improved = False
            moves = [(step, 0, 0, 0), (-step, 0, 0, 0), (0, step, 0, 0), (0, -step, 0, 0), (0, 0, step, 0), (0, 0, -step, 0)]
            if o['kind'] != 'circle':
                a = step / o['R']
                moves += [(0, 0, 0, a), (0, 0, 0, -a)]
            for dcx, dcy, dR, dth in moves:
                s = cost(cx + dcx, cy + dcy, R + dR, th + dth)
                if s < cur:
                    cx, cy, R, th, cur = cx + dcx, cy + dcy, R + dR, th + dth, s
                    improved = True
    o['cx'], o['cy'], o['R'], o['theta'] = cx, cy, R, th
    return cur

def poly_pts(o, cx, cy, theta):
    N = o['N']
    a = theta + 2 * np.pi * np.arange(N) / N
    return np.stack([cx + o['R'] * np.cos(a), cy + o['R'] * np.sin(a)], 1)

def dist_to_boundary(P, o, cx, cy, theta):
    """distance of points P (M,2) to shape boundary."""
    if o['kind'] == 'circle':
        return np.abs(np.linalg.norm(P - [cx, cy], axis=1) - o['R'])
    V = poly_pts(o, cx, cy, theta)
    best = np.full(len(P), np.inf)
    for i in range(len(V)):
        A, B = V[i], V[(i + 1) % len(V)]
        AB = B - A; L2 = AB @ AB
        t = np.clip(((P - A) @ AB) / L2, 0, 1)
        proj = A + t[:, None] * AB
        best = np.minimum(best, np.linalg.norm(P - proj, axis=1))
    return best

for o in objects:
    fm = mask(o['color'])
    d = cv2.dilate(fm.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    resid = refine_object(o, fm | (d & ol))
    print('refined', o['color'], o['kind'], f"cx={o['cx']:.2f} cy={o['cy']:.2f} R={o['R']:.2f} th={math.degrees(o['theta']):.2f} resid={resid}")

# ---------- dashed targets: fit each object's shape to dash pixels ----------
dm = mask(DASH).astype(np.uint8)
ncomp, lab = cv2.connectedComponents(dm, connectivity=8)
dash_pts_all = np.column_stack(np.nonzero(dm))[:, ::-1].astype(float) + 0.5
dash_lab = lab[dm > 0]
free = np.ones(len(dash_pts_all), bool)

for o in sorted(objects, key=lambda o: -o['R']):
    P = dash_pts_all[free]; labs = dash_lab[free]
    band = np.abs(P[:, 1] - o['cy']) <= o['R'] + 6
    Pb = P[band]
    best = None
    nth = 1 if o['kind'] == 'circle' else int(round(360 / o['N']))
    thetas = np.deg2rad(np.arange(0, nth, 1.0))
    for cx in np.arange(W / 2, W - o['R'] + 1, 2.0):
        near = np.abs(Pb[:, 0] - cx) <= o['R'] + 6
        Q = Pb[near]
        if len(Q) < 50:
            continue
        for th in thetas:
            n_in = int((dist_to_boundary(Q, o, cx, o['cy'], o['theta'] + th) <= 3.0).sum())
            if best is None or n_in > best[0]:
                best = (n_in, cx, th)
    _, cx0, th0 = best
    # refine cx, cy, theta by mean-distance minimisation on inliers
    cx, cy, th = cx0, o['cy'], o['theta'] + th0
    for step in [1.0, 0.5, 0.25, 0.1]:
        improved = True
        while improved:
            improved = False
            d0 = dist_to_boundary(Pb, o, cx, cy, th)
            inl = d0 <= 4.0
            def score(cx_, cy_, th_):
                return dist_to_boundary(Pb[inl], o, cx_, cy_, th_).mean()
            cur = score(cx, cy, th)
            for dcx, dcy, dth in [(step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0),
                                  (0, 0, np.deg2rad(step)), (0, 0, -np.deg2rad(step))]:
                s = score(cx + dcx, cy + dcy, th + dth)
                if s < cur - 1e-6:
                    cx, cy, th, cur = cx + dcx, cy + dcy, th + dth, s
                    improved = True
    # assign dash components whose majority of pixels are inliers
    d = dist_to_boundary(P, o, cx, cy, th)
    inl = d <= 4.0
    taken = np.zeros(len(P), bool)
    for l in np.unique(labs[inl]):
        sel = labs == l
        if inl[sel].mean() > 0.5:
            taken |= sel
    idx = np.nonzero(free)[0]
    free[idx[taken]] = False
    # rotation: minimal signed rotation to an equivalent orientation
    if o['kind'] == 'circle':
        o['dtheta'] = 0.0
    else:
        per = 2 * np.pi / o['N']
        dth = (th - o['theta'] + per / 2) % per - per / 2
        o['dtheta'] = dth
    o['tx'], o['ty'] = cx, cy
    print(f"{o['color']} {o['kind']} N={o['N']} R={o['R']:.1f} from ({o['cx']:.1f},{o['cy']:.1f}) "
          f"-> ({cx:.1f},{cy:.1f}) rot={math.degrees(o['dtheta']):.1f} deg  inliers={taken.sum()}")

# ---------- rendering ----------
obj_any = np.zeros((H, W), bool)
for o in objects:
    fm = mask(o['color'])
    d = cv2.dilate(fm.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    obj_any |= fm | (d & ol)
base = img.copy()
base[obj_any] = BG                 # background with objects erased
dash_mask = mask(DASH)
dash_rgb = img[dash_mask]

def draw_object(draw, o, cx, cy, theta):
    if o['kind'] == 'circle':
        R = o['R']
        bbox = [cx - 0.5 - R, cy - 0.5 - R, cx - 0.5 + R, cy - 0.5 + R]
        draw.ellipse(bbox, fill=o['color'], outline=OUTLINE, width=o['ow'])
    else:
        V = poly_pts(o, cx, cy, theta)
        pts = [(float(x) - 0.5, float(y) - 0.5) for x, y in V]
        draw.polygon(pts, fill=o['color'], outline=OUTLINE, width=o['ow'])

def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

def render(i):
    if i == 0:
        return img.copy()
    t = i / (NF - 1)
    split = 0.5
    r = ease(t / split)                      # rotation progress
    m = ease((t - split) / (1 - split))      # translation progress
    im = Image.fromarray(base.copy())
    dr = ImageDraw.Draw(im)
    for o in sorted(objects, key=lambda o: -o['R']):
        th = o['theta'] + o['dtheta'] * r
        cx = o['cx'] + (o['tx'] - o['cx']) * m
        cy = o['cy'] + (o['ty'] - o['cy']) * m
        draw_object(dr, o, cx, cy, th)
    fr = np.array(im)
    fr[dash_mask] = dash_rgb                 # dashed outlines stay exactly as in the source
    return fr

# sanity: how close is the redraw at t=0 to the original?
im0 = Image.fromarray(base.copy()); dr0 = ImageDraw.Draw(im0)
for o in objects:
    draw_object(dr0, o, o['cx'], o['cy'], o['theta'])
diff = np.any(np.array(im0) != img, axis=2).sum()
print('redraw-vs-original differing pixels at t=0:', int(diff))
for o in objects:
    fm = mask(o['color']); fm2 = np.all(np.array(im0) == np.array(o['color'], np.uint8), axis=2)
    print('  ', o['color'], o['kind'], 'ow', o['ow'], 'fill mismatch', int((fm ^ fm2).sum()), 'orig', int(fm.sum()), 'new', int(fm2.sum()))

frames = [render(i) for i in range(NF)]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = os.path.join(HERE, 'output', 'frames')
os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(tmp, f'{i:03d}.png'))
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(tmp, '%03d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'slow', OUT], check=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print('wrote', OUT)
