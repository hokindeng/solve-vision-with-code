#!/usr/bin/env python3
"""Simulate a ball bouncing twice off the boundary walls (elastic reflection).

Frame 0 is first_frame.png unchanged. From frame 1 on, the direction arrow (an
indicator of the initial velocity) is removed and the ball moves at constant
speed, reflecting off the inner wall faces, stopping when it touches the wall
of the 2nd collision.
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

FPS = 16
N_FRAMES = 80
HOLD_FRAMES = 3           # frames the ball rests at the final wall
N_BOUNCES = 2

BALL_COLOR = (0, 191, 255)
ARROW_COLOR = (255, 140, 0)
BG_COLOR = (255, 255, 255)


def measure(first):
    """Recover ball centre/radius, arrow direction and inner wall faces."""
    a = np.array(first)
    ball = (a == BALL_COLOR).all(2)
    arrow = (a == ARROW_COLOR).all(2)
    ys, xs = np.where(ball)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = (x1 - x0) / 2.0
    # arrow direction: from ball centre to the arrow pixel farthest away (tip)
    ay, ax = np.where(arrow)
    d = np.hypot(ax - cx, ay - cy)
    i = int(d.argmax())
    ang = math.atan2(ay[i] - cy, ax[i] - cx)
    # walls: non-white, non-ball, non-arrow pixels -> the frame border
    wall = ~((a == BG_COLOR).all(2) | ball | arrow)
    wy, wx = np.where(wall)
    row = wall[int(cy)]
    cols = np.where(row)[0]
    left_face = cols[cols < cx].max() + 1      # first free column
    right_face = cols[cols > cx].min() - 1     # last free column
    col = wall[:, int(cx)]
    rows = np.where(col)[0]
    top_face = rows[rows < cy].max() + 1
    bot_face = rows[rows > cy].min() - 1
    return (cx, cy, r, ang, left_face, right_face, top_face, bot_face,
            ball | arrow)


def simulate(cx, cy, r, ang, L, R, T, B, n_bounces):
    """Return list of (point, cumulative distance) waypoints for the path."""
    # ball centre must stay within [L+r, R-r] x [T+r, B-r]
    xmin, xmax = L + r, R - r
    ymin, ymax = T + r, B - r
    x, y = cx, cy
    dx, dy = math.cos(ang), math.sin(ang)
    pts = [(x, y)]
    for _ in range(n_bounces):
        tx = ((xmax if dx > 0 else xmin) - x) / dx if dx != 0 else math.inf
        ty = ((ymax if dy > 0 else ymin) - y) / dy if dy != 0 else math.inf
        t = min(tx, ty)
        x, y = x + dx * t, y + dy * t
        pts.append((x, y))
        if tx <= ty:
            dx = -dx
        if ty <= tx:
            dy = -dy
    return pts


def draw_ball(bg, x, y, r):
    im = bg.copy()
    cx, cy = int(round(x)), int(round(y))
    ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], fill=BALL_COLOR)
    return im


def main():
    first = Image.open(FIRST).convert("RGB")
    cx, cy, r, ang, L, R, T, B, mask = measure(first)
    ri = int(round(r))

    # background = first frame with ball and arrow erased
    bg_arr = np.array(first).copy()
    bg_arr[mask] = BG_COLOR
    bg = Image.fromarray(bg_arr)

    pts = simulate(cx, cy, r, ang, L, R, T, B, N_BOUNCES)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    move_frames = N_FRAMES - 1 - HOLD_FRAMES   # frames 1..move_frames travel
    # Allocate frames to each segment proportionally to its length, rounded so
    # every collision lands exactly on a frame (ball visibly touching the wall).
    seg_len = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    total = sum(seg_len)
    bounds = [int(round(move_frames * sum(seg_len[:i]) / total))
              for i in range(len(seg_len) + 1)]
    frames = [first]
    for k in range(1, N_FRAMES):
        kk = min(k, move_frames)
        seg = min(max(i for i in range(len(seg_len)) if bounds[i] <= kk),
                  len(seg_len) - 1)
        f = (kk - bounds[seg]) / (bounds[seg + 1] - bounds[seg])
        (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        frames.append(draw_ball(bg, x, y, ri))

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(os.path.join(frames_dir, f"{i:04d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-preset", "slow", "-r", str(FPS), OUT,
    ], check=True)
    print("bounce points:", [(round(x, 1), round(y, 1)) for x, y in pts])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
