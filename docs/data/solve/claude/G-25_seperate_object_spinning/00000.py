#!/usr/bin/env python3
"""Rotate each object in place to match its dashed target, then slide it right into the target."""
import os, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 48, 16

img = cv2.imread(SRC)[:, :, ::-1].copy()  # RGB
H, W = img.shape[:2]
imi = img.astype(int)
sat = imi.max(2) - imi.min(2)
colored = sat > 30
gray = (sat <= 30) & (imi.max(2) < 200)
xs_grid = np.arange(W)[None, :]
dash = gray & (xs_grid > W // 2)          # dashed targets (right half)
dys, dxs = np.nonzero(dash)
P = np.stack([dxs, dys], 1).astype(np.float64) + 0.5


def fit_square(Q, cxs, cys, ths, ss, tol):
    """Best (score, cx, cy, th, s) so that square edges pass through points Q."""
    best = None
    for th in ths:
        c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
        for cx in cxs:
            for cy in cys:
                dx, dy = Q[:, 0] - cx, Q[:, 1] - cy
                u = dx * c + dy * sn
                v = -dx * sn + dy * c
                d = np.maximum(np.abs(u), np.abs(v))
                for s in ss:
                    sc = np.sum(np.abs(d - s / 2) < tol)
                    if best is None or sc > best[0]:
                        best = (sc, cx, cy, th, s)
    return best


def fit_circle(Q, cxs, cy, rs, tol):
    best = None
    for cx in cxs:
        d = np.hypot(Q[:, 0] - cx, Q[:, 1] - cy)
        for r in rs:
            sc = np.sum(np.abs(d - r) < tol)
            if best is None or sc > best[0]:
                best = (sc, cx, r)
    return best


# ---- segment objects (left side) ----
n, lab, st, cen = cv2.connectedComponentsWithStats(colored.astype(np.uint8))
objects = []
for i in range(1, n):
    if st[i, 4] < 200:
        continue
    m = (lab == i)
    # include the thin gray outline hugging the shape
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & gray & (xs_grid <= W // 2)
    full = m | ring
    cx, cy = cen[i][0] + 0.5, cen[i][1] + 0.5
    area = st[i, 4]
    # circle test: filled area vs bbox
    bw, bh = st[i, 2], st[i, 3]
    mys, mxs = np.nonzero(m)
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([mxs, mys], 1).astype(np.float32))
    is_circle = area / (rw * rh) < 0.9   # square fills its min-area rect; circle fills ~78%
    if is_circle:
        r_obj = bw / 2.0
        b = fit_circle(P, np.arange(W // 2 + 60, W - 60, 0.5), cy, np.arange(r_obj - 6, r_obj + 12, 0.5), 1.5)
        tcx = b[1]
        dtheta = 0.0
    else:
        edge = m.astype(np.uint8) - cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
        eys, exs = np.nonzero(edge)
        Q = np.stack([exs, eys], 1) + 0.5
        s0 = np.sqrt(area)
        bo = fit_square(Q, [cx], [cy], np.arange(0, 90, 0.25), np.arange(s0 - 3, s0 + 3, 0.5), 1.0)
        th_obj, s_obj = bo[3], bo[4]
        # target: same row, same size; coarse then fine search over cx, theta
        bt = fit_square(P, np.arange(W // 2 + 60, W - 60, 2), [cy], np.arange(0, 90, 2), [s_obj], 2.0)
        bt = fit_square(P, np.arange(bt[1] - 4, bt[1] + 4, 0.25), [cy], np.arange(bt[3] - 4, bt[3] + 4, 0.25), [s_obj], 1.5)
        tcx, th_tgt = bt[1], bt[3]
        dtheta = (th_tgt - th_obj + 45) % 90 - 45   # smallest equivalent rotation
    objects.append(dict(mask=full, cx=cx, cy=cy, tcx=tcx, dtheta=dtheta))
    print(f"object at ({cx:.1f},{cy:.1f}) circle={is_circle} -> target cx={tcx:.2f}, rotate {dtheta:+.2f} deg")

# ---- background with objects erased ----
bg = img.copy()
for o in objects:
    bg[o["mask"]] = 255
    o["rgba"] = np.dstack([img, (o["mask"] * 255).astype(np.uint8)]).astype(np.float32)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def render(t):
    if t <= 0:
        return img.copy()
    rot = ease(t / 0.5)               # first half: rotate
    mov = ease((t - 0.5) / 0.5)       # second half: translate
    out = bg.astype(np.float32)
    for o in objects:
        ang = o["dtheta"] * rot
        cx, cy = o["cx"], o["cy"]
        shift = (o["tcx"] - cx) * mov
        M = cv2.getRotationMatrix2D((cx - 0.5, cy - 0.5), -ang, 1.0)  # image-coords clockwise positive
        M[0, 2] += shift
        warped = cv2.warpAffine(o["rgba"], M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))
        a = warped[:, :, 3:4] / 255.0
        out = out * (1 - a) + warped[:, :, :3] * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for k in range(N_FRAMES):
    t = k / (N_FRAMES - 1)
    fr = render(t)
    cv2.imwrite(os.path.join(frames_dir, f"{k:03d}.png"), fr[:, :, ::-1])

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(frames_dir, "%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
                "-r", str(FPS), OUT], check=True)
import shutil
shutil.rmtree(frames_dir, ignore_errors=True)
print("wrote", OUT)
