#!/usr/bin/env python3
"""Locate the intersection of the two lines in first_frame.png and animate
drawing a single red circle around it. Everything else stays untouched."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 30
RED = (255, 0, 0)
RADIUS, THICK = 42, 5


def fit_lines(img):
    """Find the two non-white colours and fit a line to each."""
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    lines = []
    for i in order:
        c = cols[i]
        if np.all(c > 240):          # background
            continue
        mask = np.all(img == c, axis=2)
        ys, xs = np.nonzero(mask)
        if len(xs) < 200:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
        if len(lines) == 2:
            break
    assert len(lines) == 2, "expected exactly two lines"
    return lines


def intersect(l1, l2):
    p, d = l1
    q, e = l2
    A = np.array([d, -e]).T
    t, _ = np.linalg.solve(A, q - p)
    return p + t * d


def draw_arc(base, center, sweep_deg):
    """Return a copy of base with a red arc (anti-aliased) drawn on it."""
    frame = base.copy()
    if sweep_deg <= 0:
        return frame
    shift = 4
    c = tuple(int(round(v * (1 << shift))) for v in center)
    r = RADIUS << shift
    # start at top (-90 deg) and sweep clockwise
    cv2.ellipse(frame, c, (r, r), 0, -90, -90 + sweep_deg, RED, THICK,
                lineType=cv2.LINE_AA, shift=shift)
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    l1, l2 = fit_lines(base)
    center = intersect(l1, l2)
    print("intersection at", center)

    # Frame 0: untouched. Frames 1..N-2: arc grows with ease-in-out.
    # Last frame: complete circle.
    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(base.copy())
            continue
        u = i / (N_FRAMES - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * u)          # ease in/out
        sweep = 360.0 * s
        if i >= N_FRAMES - 2:
            sweep = 360.0
        frames.append(draw_arc(base, center, sweep))

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
        "-x264-params", "keyint=1", "-r", str(FPS), OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
