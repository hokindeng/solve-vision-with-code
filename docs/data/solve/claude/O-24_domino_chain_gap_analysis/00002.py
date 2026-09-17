#!/usr/bin/env python3
"""Domino chain gap analysis: push domino 1, dominos fall until the wide gap.

The first frame has 10 dominos. The gap between #2 and #3 is much wider than
the rest, so only dominos 1 and 2 fall. Domino 1 tips onto domino 2, domino 2
falls flat and stops short of domino 3. Everything else stays untouched.
"""
import math
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 50

GROUND_RGB = (139, 115, 85)
BORDER_RGB = (20, 20, 20)
TOP_Y, BOT_Y = 617, 747          # inclusive pixel rows of a domino (incl. border)
GROUND_TOP = 746                 # ground line starts here (overlaps bottom border)
DOM_H = BOT_Y - TOP_Y + 1        # 131
DOM_W = 39


def detect_dominos(a):
    """Return list of (x0, x1) inclusive column spans of dominos at row 700."""
    row = a[700]
    nz = np.where(np.any(row < 240, axis=1))[0]
    segs, s, p = [], nz[0], nz[0]
    for x in nz[1:]:
        if x != p + 1:
            segs.append((int(s), int(p)))
            s = x
        p = x
    segs.append((int(s), int(p)))
    return segs


def find_gap(segs):
    """Index i such that the gap between domino i and i+1 is the outlier."""
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    med = float(np.median(gaps))
    return max(range(len(gaps)), key=lambda i: gaps[i] - med)


def rect_pts(x0, x1, ang):
    """Corners of domino spanning cols x0..x1 rotated clockwise by ang (rad)
    about its bottom-right corner (falling to the right)."""
    px, py = x1 + 1.0, BOT_Y + 1.0
    w, h = (x1 - x0 + 1), DOM_H
    local = [(-w, -h), (0, -h), (0, 0), (-w, 0)]
    c, s = math.cos(ang), math.sin(ang)
    # clockwise on screen (y down): (dx,dy) -> (dx*c - dy*s, dx*s + dy*c)
    return [(px + dx * c - dy * s, py + dx * s + dy * c) for dx, dy in local]


def convex_overlap(P, Q, eps=1e-6):
    """Separating-axis test for two convex polygons."""
    def axes(poly):
        for i in range(len(poly)):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % len(poly)]
            yield (-(y1 - y0), x1 - x0)
    for ax, ay in list(axes(P)) + list(axes(Q)):
        p = [x * ax + y * ay for x, y in P]
        q = [x * ax + y * ay for x, y in Q]
        if max(p) < min(q) + eps or max(q) < min(p) + eps:
            return False
    return True


def contact_angle(d1, d2, phi, lo=0.0):
    """Largest angle of domino d1 (>= lo) that does not penetrate domino d2 at
    angle phi. Binary search on the first intersecting angle."""
    Q = rect_pts(*d2, phi)
    hi = math.pi / 2
    if not convex_overlap(rect_pts(*d1, hi), Q):
        return hi
    a, b = lo, hi
    for _ in range(40):
        m = 0.5 * (a + b)
        if convex_overlap(rect_pts(*d1, m), Q):
            b = m
        else:
            a = m
    return a


def sprite_layer(a, x0, x1):
    """Full-canvas RGBA layer holding just this domino; transparent RGB set to
    border colour so bilinear edges blend darkly (into the border) not black."""
    layer = np.zeros((H, W, 4), np.uint8)
    layer[..., :3] = BORDER_RGB
    layer[TOP_Y:BOT_Y + 1, x0:x1 + 1, :3] = a[TOP_Y:BOT_Y + 1, x0:x1 + 1]
    layer[TOP_Y:BOT_Y + 1, x0:x1 + 1, 3] = 255
    return Image.fromarray(layer, "RGBA")


def rotate_layer(layer, x1, ang):
    """Rotate layer clockwise by ang about pivot (x1+1, BOT_Y+1)."""
    px, py = x1 + 1.0, BOT_Y + 1.0
    c, s = math.cos(ang), math.sin(ang)
    # output (X,Y) -> input: inverse rotation (counter-clockwise by ang)
    #  dx =  (X-px)*c + (Y-py)*s ; dy = -(X-px)*s + (Y-py)*c
    coeffs = (c, s, px - px * c - py * s,
              -s, c, py + px * s - py * c)
    return layer.transform((W, H), Image.AFFINE, coeffs, resample=Image.BILINEAR)


def ease_in(u, p=1.7):
    u = min(max(u, 0.0), 1.0)
    return u ** p


def main():
    src = Image.open(SRC).convert("RGB")
    a = np.array(src)
    segs = detect_dominos(a)
    gi = find_gap(segs)                 # dominos 0..gi fall
    falling = segs[:gi + 1]

    # background with falling dominos erased
    bg = a.copy()
    for x0, x1 in falling:
        bg[TOP_Y:GROUND_TOP, x0:x1 + 1] = 255
        bg[GROUND_TOP:BOT_Y + 1, x0:x1 + 1] = GROUND_RGB
    bg_img = Image.fromarray(bg, "RGB")
    layers = [sprite_layer(a, x0, x1) for x0, x1 in falling]

    n = len(falling)
    # Timeline: last falling domino goes 0 -> 90deg between F_START and F_END,
    # every earlier domino is driven by contact with the next one.
    F_LAST_START, F_END = 9, 42
    # free-fall angle of domino 0 before it touches domino 1 (if n > 1)
    theta_c = contact_angle(falling[0], falling[1], 0.0) if n > 1 else math.pi / 2

    frames = []
    prev = [0.0] * n
    for f in range(N_FRAMES):
        angs = [0.0] * n
        if n == 1:
            angs[0] = (math.pi / 2) * ease_in((f - 1) / (F_END - 1))
        else:
            if f < F_LAST_START:
                angs[0] = theta_c * ease_in((f - 1) / (F_LAST_START - 1), 2.0)
            else:
                angs[-1] = (math.pi / 2) * ease_in((f - F_LAST_START) / (F_END - F_LAST_START))
                for i in range(n - 2, -1, -1):
                    angs[i] = contact_angle(falling[i], falling[i + 1], angs[i + 1], lo=prev[i])
        angs = [max(x, y) for x, y in zip(angs, prev)]   # never un-fall
        prev = angs

        if f == 0:
            frames.append(src.copy())
            continue
        frame = bg_img.copy()
        for i in range(n - 1, -1, -1):        # later dominos underneath earlier ones
            rl = rotate_layer(layers[i], falling[i][1], angs[i])
            frame.paste(rl, (0, 0), rl)
        frames.append(frame)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.asarray(fr.convert("RGB"), np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    frames[-1].convert("RGB").save("/app/output/last_frame.png")
    print(f"gap after domino {gi + 1}; wrote {len(frames)} frames to {OUT}")


if __name__ == "__main__":
    main()
