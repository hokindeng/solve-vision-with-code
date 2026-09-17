#!/usr/bin/env python3
"""Rotate each object in place to its target orientation, then slide it right into its dashed target."""
import os, subprocess, shutil
import numpy as np, cv2
from PIL import Image
from scipy.optimize import minimize

SRC = '/app/first_frame.png'
OUT_DIR = '/app/output'
FRAMES_DIR = os.path.join(OUT_DIR, 'frames')
N_FRAMES, FPS = 48, 16
ROT_FRAC = 0.4  # fraction of time spent rotating, remainder translating

img = np.array(Image.open(SRC).convert('RGB'))
H, W = img.shape[:2]

def mask_of(c):
    return np.all(img == np.array(c, dtype=np.uint8), axis=2)

DASH, OUTLINE = (90, 90, 90), (120, 120, 120)
m_dash = mask_of(DASH).astype(np.uint8)
m_out = mask_of(OUTLINE).astype(np.uint8)
dt = np.minimum(cv2.distanceTransform(1 - m_dash, cv2.DIST_L2, 5), 10.0)
_, out_lab = cv2.connectedComponents(m_out)

# object fill colour, rotational symmetry (deg), rough target centre guess
OBJECTS = [
    ((173, 216, 230), 90,  (660, 630)),   # light-blue square
    ((255, 165, 0),   90,  (662, 478)),   # orange square
    ((255, 150, 50),  360, (822, 485)),   # orange circle
    ((230, 170, 95),  120, (815, 615)),   # tan triangle
    ((255, 255, 224), 60,  (825, 365)),   # pale hexagon
]

def fit(colour, sym, guess):
    fill = mask_of(colour)
    ys, xs = np.nonzero(fill)
    cx, cy = xs.mean(), ys.mean()
    dil = cv2.dilate(fill.astype(np.uint8), np.ones((3, 3), np.uint8))
    lab = np.bincount(out_lab[(dil > 0) & (m_out > 0)]).argmax()
    full = fill | (out_lab == lab)
    oys, oxs = np.nonzero(out_lab == lab)
    px, py = oxs - cx, oys - cy

    def cost(p):
        tx, ty, th = p
        t = np.deg2rad(th)
        X = tx + px * np.cos(t) - py * np.sin(t)
        Y = ty + px * np.sin(t) + py * np.cos(t)
        return dt[np.clip(np.round(Y).astype(int), 0, H - 1),
                  np.clip(np.round(X).astype(int), 0, W - 1)].mean()

    best = None
    for th in np.arange(0, sym, 2.0):
        for dx in range(-12, 13, 4):
            for dy in range(-12, 13, 4):
                v = cost((guess[0] + dx, guess[1] + dy, th))
                if best is None or v < best[0]:
                    best = (v, (guess[0] + dx, guess[1] + dy, th))
    r = minimize(cost, best[1], method='Nelder-Mead',
                 options={'xatol': 0.01, 'fatol': 1e-4})
    th = r.x[2] % sym
    if th > sym / 2:
        th -= sym
    if sym == 360:
        th = 0.0  # a circle has no orientation to correct
    # move is purely horizontal: keep the object's own y
    return dict(cx=cx, cy=cy, tx=float(r.x[0]), ty=float(cy), theta=float(th), mask=full)

objs = [fit(*o) for o in OBJECTS]
for o in objs:
    print(f"obj at ({o['cx']:.1f},{o['cy']:.1f}) -> x={o['tx']:.1f}, rot={o['theta']:.2f} deg")

# Background: the frame with every moving object erased to white
background = img.copy()
for o in objs:
    background[o['mask']] = 255

# Sprites: RGBA cut-outs of each object
for o in objs:
    ys, xs = np.nonzero(o['mask'])
    x0, x1, y0, y1 = xs.min() - 2, xs.max() + 3, ys.min() - 2, ys.max() + 3
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    rgba[..., :3] = img[y0:y1, x0:x1]
    rgba[..., 3] = o['mask'][y0:y1, x0:x1] * 255.0
    o['sprite'] = rgba
    o['sx0'], o['sy0'] = x0, y0

def ease(u):
    return u * u * (3 - 2 * u)  # smoothstep

def render(t):
    """t in [0,1]. Rotation during [0,ROT_FRAC], translation during [ROT_FRAC,1]."""
    u_rot = ease(min(1.0, max(0.0, t / ROT_FRAC)))
    u_mov = ease(min(1.0, max(0.0, (t - ROT_FRAC) / (1 - ROT_FRAC))))
    canvas = background.astype(np.float32)
    for o in objs:
        ang = o['theta'] * u_rot
        cx = o['cx'] + (o['tx'] - o['cx']) * u_mov
        cy = o['cy'] + (o['ty'] - o['cy']) * u_mov
        # affine mapping sprite coords -> canvas: rotate about object centre, then translate
        t_ = np.deg2rad(ang)
        c, s = np.cos(t_), np.sin(t_)
        ocx, ocy = o['cx'] - o['sx0'], o['cy'] - o['sy0']  # centre in sprite coords
        M = np.array([[c, -s, cx - (c * ocx - s * ocy)],
                      [s,  c, cy - (s * ocx + c * ocy)]], np.float32)
        warped = cv2.warpAffine(o['sprite'], M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4] / 255.0
        canvas = canvas * (1 - a) + warped[..., :3] * a
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)

if os.path.isdir(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR)
for i in range(N_FRAMES):
    t = i / (N_FRAMES - 1)
    frame = img if i == 0 else render(t)
    Image.fromarray(frame).save(os.path.join(FRAMES_DIR, f'{i:03d}.png'))

out = os.path.join(OUT_DIR, 'video.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', os.path.join(FRAMES_DIR, '%03d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12',
                '-r', str(FPS), out], check=True)
print('wrote', out)
