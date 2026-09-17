#!/usr/bin/env python3
"""Domino chain gap analysis: dominos 1-3 fall; the chain stops at the wide gap before 4."""
import os
import subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 53

# Domino boxes (inclusive pixel bounds) measured from first_frame.png
TOP, BOT = 598, 738            # domino rows (bottom 2 rows overlap the ground band)
GROUND_ROWS = (737, 741)       # brown ground band
GROUND_RGB = (139, 90, 43)
DOMINOS = [(143, 185), (219, 261), (278, 320), (464, 506), (539, 581),
           (599, 641), (664, 706), (739, 781), (814, 856), (878, 920)]
H = BOT - TOP + 1              # 141
W = DOMINOS[0][1] - DOMINOS[0][0] + 1  # 43

# --- Chain analysis: find first gap wider than the typical spacing -------------
gaps = [DOMINOS[i + 1][0] - DOMINOS[i][1] for i in range(len(DOMINOS) - 1)]
typical = float(np.median(gaps))
falling = len(DOMINOS)
for i, g in enumerate(gaps):
    if g > 1.5 * typical and g > H * 0.9:   # too wide for a falling domino to bridge
        falling = i + 1
        break
print(f"gaps={gaps}; dominos that fall: 1..{falling} (last to fall: #{falling})")


def poly(idx, theta):
    """Corners of domino idx rotated clockwise by theta about its bottom-right pivot."""
    x0, x1 = DOMINOS[idx]
    px, py = x1 + 1.0, BOT + 1.0
    c, s = np.cos(theta), np.sin(theta)
    pts = []
    for (x, y) in [(x0, TOP), (x1 + 1, TOP), (x1 + 1, BOT + 1), (x0, BOT + 1)]:
        dx, dy = x - px, y - py
        pts.append((px + dx * c - dy * s, py + dx * s + dy * c))
    return np.array(pts)


def points_in_poly(pts, P):
    """Vectorised even-odd test: pts (N,2) against convex/concave polygon P (M,2)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(P)
    for i in range(n):
        xa, ya = P[i]
        xb, yb = P[(i + 1) % n]
        cond = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= cond & (x < xi)
    return inside


def collides(idx, theta, other_poly):
    """Does the leading (right) face of domino idx at angle theta touch other_poly?"""
    P = poly(idx, theta)
    a, b = P[1], P[2]  # top-right -> bottom-right
    t = np.linspace(0, 1, 300)[:, None]
    return bool(points_in_poly(a + (b - a) * t, other_poly).any())


def first_angle(idx, other_poly, steps=90):
    """Smallest clockwise angle at which domino idx touches other_poly:
    coarse scan to bracket the first contact, then bisection."""
    grid = np.linspace(0.0, np.pi / 2, steps + 1)
    hit = next((k for k, th in enumerate(grid) if collides(idx, th, other_poly)), None)
    if hit is None:
        return np.pi / 2
    if hit == 0:
        return 0.0
    lo, hi = grid[hit - 1], grid[hit]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if collides(idx, mid, other_poly):
            hi = mid
        else:
            lo = mid
    return lo


# Contact angles (when domino i first hits upright i+1) and rest angles (final pose).
contact = []
for i in range(falling - 1):
    contact.append(first_angle(i, poly(i + 1, 0.0)))
rest = [0.0] * falling
rest[falling - 1] = np.pi / 2  # last one lies flat on the ground
for i in range(falling - 2, -1, -1):
    rest[i] = first_angle(i, poly(i + 1, rest[i + 1]))
print("contact angles (deg):", np.degrees(contact).round(1))
print("rest angles (deg):", np.degrees(rest).round(1))

# --- Timeline ---------------------------------------------------------------------
# Domino i starts at frame start[i]; its angle grows like a gravity-driven topple
# (slow start, accelerating), and the next domino begins when contact is reached.
START0 = 5            # a short hold before the push lands
POWER = 1.7
DUR = 17.0            # frames for a free domino to go from upright to flat


def free_angle(i, f):
    tau = max(0.0, f - start[i]) / DUR
    return min(rest[i], (np.pi / 2) * min(1.0, tau) ** POWER)


def angles_at(f):
    """All falling-domino angles at frame f; each is capped so it never
    penetrates the (already computed) pose of the domino in front of it."""
    th = [0.0] * falling
    for i in range(falling - 1, -1, -1):
        a = free_angle(i, f)
        if i < falling - 1:
            a = min(a, first_angle(i, poly(i + 1, th[i + 1])))
        th[i] = a
    return th


start = [START0]
for i in range(falling - 1):
    f = start[i]
    while free_angle(i, f) < contact[i]:
        f += 0.25
    start.append(f)
print("start frames:", start)
end_frame = max(start[i] + DUR * (rest[i] / (np.pi / 2)) ** (1 / POWER) for i in range(falling))
assert end_frame < N_FRAMES - 4, "animation must settle before the video ends"

# --- Rendering ----------------------------------------------------------------------
base = Image.open(FIRST).convert("RGB")
base_np = np.array(base)

# Sprites of the falling dominos, and a background with them erased.
sprites = []
bg = base_np.copy()
for i in range(falling):
    x0, x1 = DOMINOS[i]
    spr = np.zeros((H, W, 4), np.uint8)
    spr[..., :3] = base_np[TOP:BOT + 1, x0:x1 + 1]
    spr[..., 3] = 255
    sprites.append(Image.fromarray(spr, "RGBA"))
    bg[TOP:BOT + 1, x0:x1 + 1] = 255
    bg[GROUND_ROWS[0]:GROUND_ROWS[1] + 1, x0:x1 + 1] = GROUND_RGB
bg_img = Image.fromarray(bg, "RGB")


def render(f):
    if f == 0:
        return base.copy()
    frame = bg_img.copy()
    # Draw from the last falling domino backwards so leaning ones overlap correctly.
    th_all = angles_at(f)
    for i in range(falling - 1, -1, -1):
        th = th_all[i]
        x0, x1 = DOMINOS[i]
        px, py = x1 + 1.0, BOT + 1.0
        c, s = np.cos(th), np.sin(th)
        # Inverse map: output (X,Y) -> sprite (u,v); sprite pixel (u,v) sits at
        # canvas (x0+u, TOP+v) before rotation.
        # forward: X = px + (x-px)c - (y-py)s ; Y = py + (x-px)s + (y-py)c
        # inverse: x = px + (X-px)c + (Y-py)s ; y = py - (X-px)s + (Y-py)c
        a_, b_, c_ = c, s, px - px * c - py * s - x0
        d_, e_, f_ = -s, c, py + px * s - py * c - TOP
        layer = sprites[i].transform((1024, 1024), Image.AFFINE,
                                     (a_, b_, c_, d_, e_, f_), resample=Image.BILINEAR)
        frame.paste(layer, (0, 0), layer)
    return frame


os.makedirs(OUT_DIR, exist_ok=True)
frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for f in range(N_FRAMES):
    render(f).save(os.path.join(frames_dir, f"{f:03d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
    "-i", os.path.join(frames_dir, "%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT,
], check=True)
print("wrote", OUT)
