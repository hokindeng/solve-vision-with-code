#!/usr/bin/env python3
"""Animate the orange ball rolling along the dashed platform path.

Reads /app/first_frame.png, detects the ball and the platform dashes, and
renders 64 frames (16 fps, 4 s) to /app/output/video.mp4.  Only the ball
moves; every other pixel is kept from the first frame.
"""
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 64

BALL_FILL = (255, 140, 0)
BALL_EDGE = (175, 60, 0)
DASH_FILL = (70, 130, 200)
DASH_EDGE = (10, 70, 140)

# --------------------------------------------------------------------------
# Scene analysis
# --------------------------------------------------------------------------
frame0 = np.array(Image.open(SRC).convert("RGB"))
a = frame0.astype(int)
r, g, b = a[..., 0], a[..., 1], a[..., 2]

ball_mask = (r > 150) & (b < 80) & (g < 200)           # fill + outline
ys, xs = np.nonzero(ball_mask)
cx0 = (xs.min() + xs.max()) / 2.0
cy0 = (ys.min() + ys.max()) / 2.0
R = int(round((xs.max() - xs.min() + 1) / 2.0)) - 1    # outer radius 40 for 81 px extent
start = np.array([cx0, cy0])

blue_mask = (b > 100) & (r < 150)
n, lab, stats, cent = cv2.connectedComponentsWithStats(blue_mask.astype(np.uint8))
comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 30]
comps.sort(key=lambda i: cent[i][0])                   # left -> right along the path

# Typical dash geometry from un-occluded dashes (those not touching the ball)
ball_dil = cv2.dilate(ball_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
sizes, angles = [], []
for i in comps:
    m = lab == i
    if (m & ball_dil).any():
        continue
    pts = np.column_stack(np.nonzero(m))[:, ::-1].astype(np.float32)
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    sizes.append((w + h) / 2.0)
dash_size = float(np.median(sizes)) + 1.0              # initial guess (refined below)


def dash_poly(c, ang_deg, size):
    """Corner points of a rotated square dash."""
    t = np.deg2rad(ang_deg)
    ux = np.array([np.cos(t), np.sin(t)]) * size / 2.0
    uy = np.array([-np.sin(t), np.cos(t)]) * size / 2.0
    c = np.asarray(c, float)
    return np.array([c - ux - uy, c + ux - uy, c + ux + uy, c - ux + uy])


def draw_dash(img, c, ang_deg, size):
    p = np.round(dash_poly(c, ang_deg, size)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [p], DASH_FILL)
    cv2.polylines(img, [p], True, DASH_EDGE, 2)


# Calibrate the rendered square size against the un-occluded dashes.
def _render_mismatch(m, c, ang, size):
    tmp = np.zeros((H, W), np.uint8)
    p = np.round(dash_poly(c, ang, size)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(tmp, [p], 1)
    cv2.polylines(tmp, [p], True, 1, 2)
    return tmp > 0

best_size, best_err = dash_size, None
for size in np.arange(dash_size - 4, dash_size + 2.01, 0.25):
    err = 0
    for i in comps:
        m = lab == i
        if (m & ball_dil).any():
            continue
        pts = np.column_stack(np.nonzero(m))[:, ::-1].astype(np.float32)
        (mx, my), _, ang = cv2.minAreaRect(pts)
        loc = None
        for dx in (-0.5, 0, 0.5):
            for dy in (-0.5, 0, 0.5):
                e = np.count_nonzero(_render_mismatch(m, (mx + dx, my + dy), ang, size) ^ m)
                loc = e if loc is None else min(loc, e)
        err += loc
    if best_err is None or err < best_err:
        best_size, best_err = size, err
dash_size = float(best_size)

# Platform centres (and local angle) in path order.  Dashes partially hidden
# by the ball are reconstructed by fitting a rotated square to the visible part.
centers, dash_params = [], []
for i in comps:
    m = lab == i
    occluded = (m & ball_dil).any()
    pts = np.column_stack(np.nonzero(m))[:, ::-1].astype(np.float32)
    (mx, my), (w, h), ang = cv2.minAreaRect(pts)
    if not occluded:
        centers.append((mx, my))
        dash_params.append(((mx, my), ang))
        continue
    # Grid search a square that matches the visible pixels outside the ball.
    visible_ok = ~ball_mask
    # neighbouring un-occluded dash gives the expected orientation
    j = comps[min(comps.index(i) + 1, len(comps) - 1)]
    npts = np.column_stack(np.nonzero(lab == j))[:, ::-1].astype(np.float32)
    _, _, nang = cv2.minAreaRect(npts)
    best = None
    for dx in np.arange(-5, 5.01, 0.5):
        for dy in np.arange(-5, 5.01, 0.5):
            for da in np.arange(-14, 14.1, 1.0):
                c = (mx + dx, my + dy)
                rend = _render_mismatch(m, c, nang + da, dash_size)
                mism = np.count_nonzero((rend ^ m) & visible_ok)
                if best is None or mism < best[0]:
                    best = (mism, c, nang + da)
    _, c, an = best
    centers.append(c)
    dash_params.append((c, an))
centers = np.array(centers, float)

# --------------------------------------------------------------------------
# Background: first frame with the ball removed and hidden dash pixels restored
# --------------------------------------------------------------------------
background = frame0.copy()
background[ball_mask] = (255, 255, 255)
restore = np.zeros_like(frame0)
restore[:] = 255
for (c, an) in dash_params:
    draw_dash(restore, c, an, dash_size)
background[ball_mask] = restore[ball_mask]

# --------------------------------------------------------------------------
# Path: Catmull-Rom spline through platform centres, offset to the top side
# --------------------------------------------------------------------------
P = centers
M = len(P)


def spline(u):
    """Catmull-Rom position and tangent at parameter u in [0, M-1]."""
    u = float(np.clip(u, 0, M - 1))
    i = min(int(np.floor(u)), M - 2)
    t = u - i
    p0 = P[max(i - 1, 0)]
    p1 = P[i]
    p2 = P[i + 1]
    p3 = P[min(i + 2, M - 1)]
    t2, t3 = t * t, t * t * t
    pos = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    tan = 0.5 * ((-p0 + p2) + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t
                 + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * t2)
    return pos, tan


OFFSET = R + dash_size / 2.0 + 1.0                     # ball resting on top of a dash


def rest_pos(u):
    pos, tan = spline(u)
    tan = tan / (np.linalg.norm(tan) + 1e-9)
    nrm = np.array([tan[1], -tan[0]])                  # rotate to get a normal
    if nrm[1] > 0:                                     # want the upward (screen -y) side
        nrm = -nrm
    return pos + nrm * OFFSET


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_in_out(x):
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


# Timeline (frames): approach first platform, hop platform to platform, settle.
F_APPROACH = 8
F_SETTLE = 6
F_TRAVEL = N_FRAMES - 1 - F_APPROACH - F_SETTLE
UP = np.array([0.0, -1.0])


def ball_center(f):
    if f <= 0:
        return start
    if f <= F_APPROACH:
        s = ease_in_out(f / F_APPROACH)
        target = rest_pos(0)
        c = start + (target - start) * s
        c = c + UP * (18.0 * np.sin(np.pi * s))      # small arc up onto the first platform
        return c
    if f <= F_APPROACH + F_TRAVEL:
        prog = (f - F_APPROACH) / F_TRAVEL * (M - 1)  # 0 .. M-1 platforms
        seg = min(int(np.floor(prog)), M - 2)
        frac = prog - seg
        # ease within each segment so the ball momentarily settles on every platform
        u = seg + smoothstep(frac)
        hop = 7.0 * np.sin(np.pi * frac)               # short hop between platforms
        return rest_pos(u) + UP * hop
    # settle on the last platform with a tiny damped bounce
    s = (f - F_APPROACH - F_TRAVEL) / F_SETTLE
    bounce = 4.0 * np.exp(-3.0 * s) * abs(np.sin(2 * np.pi * s))
    if f >= N_FRAMES - 1:
        bounce = 0.0
    return rest_pos(M - 1) + UP * bounce


# Ball sprite cut from the first frame (exact pixels), moved by integer offsets.
_by0, _by1 = ys.min(), ys.max() + 1
_bx0, _bx1 = xs.min(), xs.max() + 1
SPRITE = frame0[_by0:_by1, _bx0:_bx1].copy()
SPRITE_MASK = ball_mask[_by0:_by1, _bx0:_bx1].copy()


def draw_ball(img, c):
    dx = int(round(c[0] - cx0))
    dy = int(round(c[1] - cy0))
    y0, x0 = _by0 + dy, _bx0 + dx
    h, w = SPRITE.shape[:2]
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1, sx1 = min(h, H - y0), min(w, W - x0)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    dst = img[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
    msk = SPRITE_MASK[sy0:sy1, sx0:sx1]
    dst[msk] = SPRITE[sy0:sy1, sx0:sx1][msk]


def render(f):
    if f == 0:
        return frame0.copy()
    img = background.copy()
    draw_ball(img, ball_center(f))
    return img


# --------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "16",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        proc.stdin.write(np.ascontiguousarray(render(f)).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT, "frames:", N_FRAMES, "platforms:", M, "ball start:", start.round(1))
