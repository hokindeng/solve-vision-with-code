#!/usr/bin/env python3
"""Domino chain gap analysis: push domino 1, dominoes 1-3 fall, chain stops at the wide gap before 4."""
import math, os, subprocess, shutil
import numpy as np
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 50, 16

base = Image.open(SRC).convert("RGB")
arr = np.array(base)

# ---- scene measurements (from first_frame.png) ----
GROUND_Y = 720            # top row of the brown ground line
TOP_Y = 587               # first row of domino outline (incl. anti-alias)
BOT_Y = 721               # last row of domino outline
H = GROUND_Y - TOP_Y      # height above ground (133)
# x extents (incl. outline) of the falling dominoes 1..3; domino 4 starts at x=503
FALLING = [(145, 190), (204, 249), (265, 310)]
NEXT_LEFT = 503
W = FALLING[0][1] - FALLING[0][0] + 1  # 46

# ---- clean background: dominoes 1-3 erased, everything else untouched ----
bg = arr.copy()
for x0, x1 in FALLING:
    bg[TOP_Y:GROUND_Y, x0:x1 + 1] = 255
    for y in range(GROUND_Y, BOT_Y + 1):
        bg[y, x0:x1 + 1] = arr[y, 400]  # ground line colour from an uncovered column
bg_img = Image.fromarray(bg).convert("RGBA")

# ---- sprites (exact crops of the original dominoes, incl. labels) ----
sprites, pivots = [], []
for x0, x1 in FALLING:
    sprites.append(base.crop((x0, TOP_Y, x1 + 1, BOT_Y + 1)).convert("RGBA"))
    pivots.append((x1 + 1.0, float(GROUND_Y)))  # bottom-right corner = rotation pivot

# ---- geometry helpers ----
def corners(pivot, theta_deg):
    """Corners of a domino rotated clockwise (falling right) by theta about its bottom-right corner."""
    px, py = pivot
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    pts = []
    for dx, dy in [(-W, 0), (0, 0), (0, -H), (-W, -H)]:
        pts.append((px + dx * c - dy * s, py + dx * s + dy * c))
    return pts

def polys_overlap(p, q, tol=0.5):
    """Separating-axis test for two convex quads (True if they overlap by more than tol)."""
    for poly in (p, q):
        for i in range(len(poly)):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
            nx, ny = y1 - y2, x2 - x1
            ln = math.hypot(nx, ny); nx, ny = nx / ln, ny / ln
            pa = [x * nx + y * ny for x, y in p]
            qa = [x * nx + y * ny for x, y in q]
            if max(pa) <= min(qa) + tol or max(qa) <= min(pa) + tol:
                return False
    return True

def contact_limit(i, theta_next):
    """Largest angle of domino i that does not penetrate domino i+1 at angle theta_next."""
    nxt = corners(pivots[i + 1], theta_next)
    lo, hi = 0.0, 90.0
    if not polys_overlap(corners(pivots[i], hi), nxt):
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if polys_overlap(corners(pivots[i], mid), nxt):
            hi = mid
        else:
            lo = mid
    return lo

# ---- timing: gravity-like angular fall, next domino starts when hit ----
A = 0.15                        # deg / frame^2
START = [2.0]                   # push happens right after the first frame
for i in range(len(FALLING) - 1):
    theta_c = contact_limit(i, 0.0)
    START.append(START[i] + math.sqrt(theta_c / A))

def angles_at(t):
    th = [0.0] * len(FALLING)
    for i in reversed(range(len(FALLING))):
        tau = max(0.0, t - START[i])
        free = min(90.0, A * tau * tau)
        if i < len(FALLING) - 1:
            free = min(free, contact_limit(i, th[i + 1]))
        th[i] = free
    return th

def render(thetas):
    frame = bg_img.copy()
    for i in reversed(range(len(FALLING))):  # draw 3, then 2 on top, then 1 on top
        th = thetas[i]
        px, py = pivots[i]
        x0 = FALLING[i][0]
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        # output (u,v) -> sprite coords: inverse rotation about pivot, then shift to crop origin
        a, b = c, s
        cc = -c * px - s * py + px - x0
        d, e = -s, c
        ff = s * px - c * py + py - TOP_Y
        layer = sprites[i].transform(frame.size, Image.AFFINE, (a, b, cc, d, e, ff),
                                     resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
        frame = Image.alpha_composite(frame, layer)
    return frame.convert("RGB")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for f in range(N_FRAMES):
        img = base if f == 0 else render(angles_at(float(f)))
        img.save(os.path.join(tmp, f"{f:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "16", "-r", str(FPS), OUT], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("start frames:", [round(s, 1) for s in START])
    print("final angles:", [round(a, 1) for a in angles_at(N_FRAMES - 1)])
    print("wrote", OUT)

if __name__ == "__main__":
    main()
