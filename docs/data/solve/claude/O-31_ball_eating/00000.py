#!/usr/bin/env python3
"""Black ball eats the colored balls smallest-first, growing after each meal.

Frame 0 is exactly first_frame.png.  Background is pure white; the balls are
crisp (non-antialiased) filled circles, redrawn each frame with the same
rasterisation as the source frame.
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 108
GROWTH = 0.6          # r_black += GROWTH * r_eaten
EAT_FRAMES = 7
HOLD_END = 5

BG = (255, 255, 255)
BLACK = {"c": (150.0, 191.0), "r": 24.0, "col": (0, 0, 0)}
BALLS = [  # measured from first_frame.png
    {"name": "pink",   "c": (562.0, 78.0),  "r": 20.0, "col": (255, 105, 180)},
    {"name": "green",  "c": (403.0, 849.0), "r": 28.0, "col": (60, 179, 113)},
    {"name": "orange", "c": (650.0, 92.0),  "r": 50.0, "col": (255, 140, 0)},
    {"name": "red",    "c": (469.0, 360.0), "r": 75.0, "col": (220, 20, 60)},
]

def draw_circle(draw, c, r, col):
    """Same rasterisation as the source frame (PIL ellipse, no antialiasing)."""
    if r <= 0:
        return
    draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col)


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def seg_hits(p, q, c, R):
    """Does segment p-q pass within R of c?"""
    p, q, c = map(np.asarray, (p, q, c))
    d = q - p
    L2 = d @ d
    t = 0.0 if L2 == 0 else float(np.clip(((c - p) @ d) / L2, 0, 1))
    return np.linalg.norm(p + t * d - c) < R


def plan_path(p, q, obstacles, rb, depth=0):
    """Polyline from p to q keeping the black ball (radius rb) clear of the
    obstacle circles (center, radius)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if depth > 12:
        return [p, q]
    for c, R in obstacles:
        c = np.asarray(c, float)
        Rf = R + rb + 12.0
        if not seg_hits(p, q, c, Rf):
            continue
        v = p - c
        if np.linalg.norm(v) < Rf:
            # already adjacent to this obstacle: back straight out first
            wp = c + v / (np.linalg.norm(v) + 1e-9) * (Rf + 1.0)
        else:
            d = q - p
            t = float(np.clip(((c - p) @ d) / (d @ d), 0, 1))
            foot = p + t * d
            u = foot - c
            if np.linalg.norm(u) < 1e-6:
                u = np.array([-d[1], d[0]])
            wp = c + u / np.linalg.norm(u) * (Rf + 1.0)
        wp = np.clip(wp, rb + 2, W - rb - 2)
        if np.linalg.norm(wp - p) < 1e-6:
            continue
        return plan_path(p, wp, obstacles, rb, depth + 1)[:-1] + \
            plan_path(wp, q, obstacles, rb, depth + 1)
    return [p, q]


def polyline_point(pts, s):
    """Point at arc length s along polyline."""
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        if s <= L:
            return a + (b - a) * (s / L if L else 0.0)
        s -= L
    return pts[-1]


def build_schedule():
    """Return list of per-frame states: (black_c, black_r, [(c, r) per ball])."""
    order = sorted(range(len(BALLS)), key=lambda i: BALLS[i]["r"])
    bc, br = np.array(BLACK["c"]), BLACK["r"]
    eaten = set()
    segments = []  # (path, dist, target index, r_before, r_after)
    for i in order:
        tgt = BALLS[i]
        assert br > tgt["r"], f"cannot eat {tgt['name']}"
        tc = np.array(tgt["c"])
        obstacles = [(BALLS[j]["c"], BALLS[j]["r"]) for j in range(len(BALLS))
                     if j != i and j not in eaten]
        # approach until circles touch
        path = plan_path(bc, tc, obstacles, br)
        last_dir = path[-1] - path[-2]
        last_dir /= np.linalg.norm(last_dir) + 1e-9
        path[-1] = tc - last_dir * (br + tgt["r"])
        dist = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
        r_after = br + GROWTH * tgt["r"]
        segments.append((path, dist, i, br, r_after))
        # after eating, black sits centered over where the target was
        bc, br = tc.copy(), r_after
        eaten.add(i)

    travel_budget = N_FRAMES - 1 - HOLD_END - EAT_FRAMES * len(segments)
    total = sum(s[1] for s in segments)
    travel_frames = [max(3, round(travel_budget * s[1] / total)) for s in segments]
    # fix rounding so the sum matches exactly
    travel_frames[-1] += travel_budget - sum(travel_frames)

    states = []
    balls = [(np.array(b["c"], float), b["r"]) for b in BALLS]
    bc, br = np.array(BLACK["c"]), BLACK["r"]
    states.append((bc.copy(), br, [(c.copy(), r) for c, r in balls]))
    for (path, dist, i, r0, r1), nf in zip(segments, travel_frames):
        tc, tr = balls[i]
        for k in range(1, nf + 1):
            s = dist * ease(k / nf)
            bc = polyline_point(path, s)
            states.append((bc.copy(), br, [(c.copy(), r) for c, r in balls]))
        contact = bc.copy()
        for k in range(1, EAT_FRAMES + 1):
            t = ease(k / EAT_FRAMES)
            bc = contact + (tc - contact) * t
            br = r0 + (r1 - r0) * t
            cur = [(c.copy(), r) for c, r in balls]
            cur[i] = (tc + (bc - tc) * t * 0.5, tr * (1 - t))
            states.append((bc.copy(), br, cur))
        balls[i] = (tc, 0.0)
    while len(states) < N_FRAMES:
        states.append(states[-1])
    return states[:N_FRAMES]


def render(state):
    bc, br, balls = state
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    # colored balls first (largest first), black on top
    for (c, r), b in sorted(zip(balls, BALLS), key=lambda x: -x[0][1]):
        draw_circle(d, c, r, b["col"])
    draw_circle(d, bc, br, BLACK["col"])
    return np.array(im)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = np.array(Image.open(FIRST).convert("RGB"))
    states = build_schedule()
    frames = [first] + [render(s) for s in states[1:]]
    assert np.array_equal(render(states[0]), first), "frame-0 reproduction mismatch"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print(f"wrote {OUT}: {len(frames)} frames, final black r={states[-1][1]:.1f}")


if __name__ == "__main__":
    main()
