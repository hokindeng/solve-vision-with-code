#!/usr/bin/env python3
"""Regenerate /app/output/video.mp4: animate a black circle drawn around the
tangent point of the two touching circles in first_frame.png."""
import os, subprocess, tempfile
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 60


def fit_circles(img):
    """Segment the non-white flat-colour discs and fit a circle to each."""
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    circles = []
    for c, n in zip(cols, counts):
        if n < 2000 or (c > 245).all():
            continue
        m = (np.abs(img.astype(int) - c.astype(int)).sum(2) < 30)
        ys, xs = np.nonzero(m)
        # bbox is robust even when another disc overlaps the edge slightly
        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
        circles.append((cx, cy, r))
    return circles


def tangent_point(circles):
    best = None
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = np.hypot(x2 - x1, y2 - y1)
            gap = abs(d - (r1 + r2))
            if best is None or gap < best[0]:
                best = (gap, i, j, d)
    _, i, j, d = best
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    # point on the centre line, r1 from circle i (== r2 from circle j when tangent)
    t = r1 / d
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    circles = fit_circles(base)
    px, py = tangent_point(circles)
    R, THICK = 32, 5
    SS = 4  # supersampling for a smooth stroke

    os.makedirs(OUT_DIR, exist_ok=True)
    draw_start, draw_end = 4, 52  # frames over which the circle is traced
    frames = []
    for k in range(N_FRAMES):
        frame = base.copy()
        if k >= draw_start:
            u = min(1.0, (k - draw_start) / (draw_end - draw_start))
            sweep = 360.0 * ease(u)
            if sweep > 0:
                # render the arc on a supersampled transparent layer, composite
                h, w = base.shape[:2]
                layer = np.zeros((h * SS, w * SS), np.uint8)
                center = (int(round(px * SS)), int(round(py * SS)))
                cv2.ellipse(layer, center, (R * SS, R * SS), 0, -90, -90 + sweep,
                            255, THICK * SS, cv2.LINE_AA)
                alpha = cv2.resize(layer, (w, h), interpolation=cv2.INTER_AREA)
                a = alpha[..., None].astype(np.float32) / 255.0
                frame = (frame.astype(np.float32) * (1 - a)).round().astype(np.uint8)
        frames.append(frame)

    with tempfile.TemporaryDirectory() as td:
        for k, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(td, f"{k:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow", "-tune", "stillimage",
            OUT], check=True)
    print(f"tangent point ({px:.1f}, {py:.1f}); wrote {OUT}")


if __name__ == "__main__":
    main()
