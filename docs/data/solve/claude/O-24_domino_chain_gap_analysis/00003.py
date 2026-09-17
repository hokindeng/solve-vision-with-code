#!/usr/bin/env python3
"""Domino chain gap analysis: push domino 1, dominos topple in sequence until the wide gap.

The first frame is used as the background; the dominos that fall are cut out as sprites,
rotated about their bottom-right corner, and composited back. Rotation angles are constrained
by collision with the next domino (separating-axis test) so the chain leans realistically and
domino 5 falls flat into the gap without reaching domino 6.
"""
import os, subprocess, tempfile
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 54

TOP, BOTTOM = 583, 723          # domino outer rows: 583..722 inclusive
GROUND_Y = 721                  # ground line top row (brown, rows 721..725)
BROWN = (160, 82, 45)

# --- detect dominos (blue fill columns) -------------------------------------
img = np.array(Image.open(FIRST).convert("RGB"))
blue = (np.abs(img[..., 0].astype(int) - 30) < 6) & (np.abs(img[..., 1].astype(int) - 144) < 6) & (img[..., 2] > 240)
cols = np.where(blue.any(axis=0))[0]
runs, s, p = [], cols[0], cols[0]
for x in cols[1:]:
    if x != p + 1:
        runs.append((s, p)); s = x
    p = x
runs.append((s, p))
# outer border extends 2 px beyond blue fill on each side
dominos = [(int(a) - 2, int(b) + 3) for a, b in runs]   # [x_left, x_right) exclusive right
gaps = [dominos[i + 1][0] - dominos[i][1] for i in range(len(dominos) - 1)]
# chain stops at the first gap wider than the domino height (can't reach the next domino)
height = BOTTOM - TOP
typical = float(np.median(gaps))
stop_idx = next(i for i, g in enumerate(gaps) if g > 1.8 * typical)
falling = list(range(0, stop_idx + 1))   # indices of dominos that fall (0-based)
print("dominos:", len(dominos), "gaps:", gaps, "-> last to fall: #%d" % (stop_idx + 1))

# --- background with falling dominos removed --------------------------------
bg = img.copy()
for i in falling:
    xl, xr = dominos[i]
    bg[TOP:GROUND_Y, xl:xr] = 255
    bg[GROUND_Y:BOTTOM, xl:xr] = BROWN

# --- sprites (RGBA) ----------------------------------------------------------
sprites = {}
for i in falling:
    xl, xr = dominos[i]
    rgb = img[TOP:BOTTOM, xl:xr]
    a = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    sprites[i] = np.concatenate([rgb, a], axis=2)

# --- geometry helpers --------------------------------------------------------
def corners(i, theta_deg):
    """Corners of domino i rotated clockwise by theta about its bottom-right corner."""
    xl, xr = dominos[i]
    px, py = float(xr), float(BOTTOM)
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    pts = np.array([[xl, TOP], [xr, TOP], [xr, BOTTOM], [xl, BOTTOM]], float) - (px, py)
    # clockwise rotation in y-down screen coords
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + (px, py)

def overlap(P, Q, tol=0.25):
    """Separating axis test for convex polygons; True if they overlap by more than tol."""
    for poly in (P, Q):
        n = len(poly)
        for k in range(n):
            e = poly[(k + 1) % n] - poly[k]
            ax = np.array([-e[1], e[0]])
            ax /= np.linalg.norm(ax)
            p0, p1 = (P @ ax).min(), (P @ ax).max()
            q0, q1 = (Q @ ax).min(), (Q @ ax).max()
            if p1 <= q0 + tol or q1 <= p0 + tol:
                return False
    return True

def max_angle(i, desired, blockers):
    """Largest angle <= desired at which domino i doesn't intersect any blocker polygon."""
    desired = min(desired, 90.0)
    def ok(th):
        P = corners(i, th)
        return not any(overlap(P, B) for B in blockers)
    if ok(desired):
        return desired
    lo, hi = 0.0, desired
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if ok(mid): lo = mid
        else: hi = mid
    return lo

# --- animation ---------------------------------------------------------------
DUR = 18.0      # frames for an unobstructed fall 0 -> 90 deg
POW = 1.7       # accelerating (gravity-like) ease-in
start = {falling[0]: 2}          # domino 1 is pushed at frame 2
angles_per_frame = []
for f in range(N_FRAMES):
    angles = {i: 0.0 for i in falling}
    # resolve from the far end so each domino is limited by the (already updated) next one
    triggered = []
    for i in reversed(falling):
        if i not in start or f < start[i]:
            continue
        prog = min(1.0, (f - start[i]) / DUR)
        desired = 90.0 * prog ** POW
        blockers = []
        if i + 1 < len(dominos):
            blockers.append(corners(i + 1, angles.get(i + 1, 0.0)))
        allowed = max_angle(i, desired, blockers)
        angles[i] = allowed
        if allowed < desired - 1e-3 and i + 1 in sprites and (i + 1) not in start:
            triggered.append(i + 1)   # pushing on the next domino -> it starts to fall
    for j in triggered:
        start[j] = f + 1
    angles_per_frame.append(angles)

def render(angles):
    frame = bg.copy()
    canvas = frame.astype(np.float32)
    for i in falling:
        th = angles[i]
        xl, xr = dominos[i]
        if th <= 1e-6:
            canvas[TOP:BOTTOM, xl:xr] = img[TOP:BOTTOM, xl:xr]
            continue
        spr = sprites[i]
        # place sprite on a full-size layer and rotate about pivot
        layer = np.zeros((H, W, 4), np.uint8)
        layer[TOP:BOTTOM, xl:xr] = spr
        M = cv2.getRotationMatrix2D((float(xr), float(BOTTOM)), -th, 1.0)
        rot = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = rot[..., 3:4].astype(np.float32) / 255.0
        canvas = canvas * (1 - a) + rot[..., :3].astype(np.float32) * a
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for f, angles in enumerate(angles_per_frame):
        fr = img if f == 0 else render(angles)
        Image.fromarray(fr).save(os.path.join(td, "f%04d.png" % f))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(td, "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-movflags", "+faststart", OUT], check=True)
print("wrote", OUT)
print("final angles:", {k + 1: round(v, 1) for k, v in angles_per_frame[-1].items()})
print("start frames:", {k + 1: v for k, v in start.items()})
