#!/usr/bin/env python3
"""Locate the intersection of two line segments in first_frame.png and animate
drawing a single red circle around it, producing output/video.mp4."""
import os, subprocess, tempfile, math
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 30, 16
RED = (255, 0, 0)


def find_intersection(img):
    """Return (x, y) of the crossing point of the two drawn lines."""
    white = np.all(img > 235, axis=2)
    mask = (~white).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)

    # Two-line fit via iterative reassignment (k=2 lines), seeded from Hough.
    lines = cv2.HoughLines(mask, 1, np.pi / 180, threshold=60)
    thetas = lines[:, 0, 1]
    # split hough angles into two clusters
    t0 = thetas[0]
    d = np.abs(((thetas - t0 + np.pi / 2) % np.pi) - np.pi / 2)
    others = lines[d > np.deg2rad(10)]
    seeds = [lines[0, 0], others[0, 0]]

    def line_from_rt(rho, th):
        return np.cos(th), np.sin(th), -rho  # a x + b y + c = 0

    L = [line_from_rt(*s) for s in seeds]
    for _ in range(10):
        dist = np.stack([np.abs(a * pts[:, 0] + b * pts[:, 1] + c) for a, b, c in L], 1)
        lab = dist.argmin(1)
        newL = []
        for k in range(2):
            p = pts[lab == k]
            mean = p.mean(0)
            _, _, vt = np.linalg.svd(p - mean, full_matrices=False)
            dx, dy = vt[0]
            a, b = -dy, dx
            c = -(a * mean[0] + b * mean[1])
            newL.append((a, b, c))
        L = newL
    (a1, b1, c1), (a2, b2, c2) = L
    det = a1 * b2 - a2 * b1
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return x, y


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    cx, cy = find_intersection(base)
    print(f"intersection at ({cx:.1f}, {cy:.1f})")

    radius, thick = 40, 6
    start, end = 3, 27  # frames over which the arc is swept
    with tempfile.TemporaryDirectory() as td:
        for i in range(N_FRAMES):
            fr = base.copy()
            if i >= start:
                t = min(1.0, (i - start) / (end - start))
                sweep = 360.0 * ease(t)
                if sweep > 0:
                    if sweep >= 359.5:
                        cv2.circle(fr, (int(round(cx)), int(round(cy))), radius, RED,
                                   thick, lineType=cv2.LINE_AA)
                    else:
                        cv2.ellipse(fr, (int(round(cx)), int(round(cy))), (radius, radius),
                                    0, -90, -90 + sweep, RED, thick, lineType=cv2.LINE_AA)
            Image.fromarray(fr).save(os.path.join(td, f"f{i:03d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS), OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
