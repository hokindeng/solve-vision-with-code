#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: a ball bouncing 5 times elastically inside the box."""
import math
import os
import subprocess

import numpy as np
from PIL import Image
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 80
N_BOUNCES = 5

BALL_COLOR = (210, 105, 30)
ARROW_COLOR = (255, 140, 0)


def measure(img):
    """Extract ball centre/radius, arrow direction and inner wall bounds from the first frame."""
    a = np.asarray(img).astype(int)
    ball = (a == BALL_COLOR).all(-1)
    arrow = (a == ARROW_COLOR).all(-1)
    wall = (a == (100, 100, 100)).all(-1)

    ys, xs = np.where(ball)
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    r = (xs.max() - xs.min()) // 2

    # Arrow tip = arrow pixel farthest from the ball centre.
    ys, xs = np.where(arrow)
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    i = d.argmax()
    ang = math.atan2(ys[i] - cy, xs[i] - cx)

    # Inner playing area: the white region enclosed by the wall.
    mid_row = np.where(wall[H // 2])[0]
    mid_col = np.where(wall[:, W // 2])[0]
    left = mid_row[mid_row < W // 2].max() + 1
    right = mid_row[mid_row > W // 2].min() - 1
    top = mid_col[mid_col < H // 2].max() + 1
    bottom = mid_col[mid_col > H // 2].min() - 1

    return (cx, cy), r, ang, (left, top, right, bottom), ball | arrow


def simulate(pos, r, ang, bounds, n_bounces):
    """Return list of polyline vertices: start, each collision point, ending at the last collision."""
    left, top, right, bottom = bounds
    xmin, xmax = left + r, right - r
    ymin, ymax = top + r, bottom - r
    x, y = pos
    dx, dy = math.cos(ang), math.sin(ang)
    pts = [(x, y)]
    for _ in range(n_bounces):
        tx = ((xmax - x) / dx if dx > 0 else (xmin - x) / dx) if dx != 0 else math.inf
        ty = ((ymax - y) / dy if dy > 0 else (ymin - y) / dy) if dy != 0 else math.inf
        t = min(tx, ty)
        x, y = x + dx * t, y + dy * t
        if tx <= ty:
            x = xmax if dx > 0 else xmin
            dx = -dx
        if ty <= tx:
            y = ymax if dy > 0 else ymin
            dy = -dy
        pts.append((x, y))
    return pts


def point_along(pts, s):
    """Point at arc length s along polyline pts."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if s <= seg or (x1, y1) == pts[-1]:
            f = min(max(s / seg, 0.0), 1.0)
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        s -= seg
    return pts[-1]


def main():
    first = Image.open(FIRST).convert("RGB")
    pos, r, ang, bounds, mask = measure(first)
    pts = simulate(pos, r, ang, bounds, N_BOUNCES)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    # Background: first frame with the ball and its direction arrow removed.
    bg = np.asarray(first).copy()
    bg[mask] = (255, 255, 255)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        if i == 0:
            frame = np.asarray(first)
        else:
            s = total * i / (N_FRAMES - 1)
            x, y = point_along(pts, s)
            frame = bg.copy()
            cv2.circle(frame, (int(round(x)), int(round(y))), r, BALL_COLOR, -1, cv2.LINE_8)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("bounce points:", [(round(x, 1), round(y, 1)) for x, y in pts])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
