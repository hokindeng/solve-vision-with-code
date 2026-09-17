#!/usr/bin/env python3
"""Identify hollow circles in first_frame.png and circle each with a red ring,
animated step by step into output/video.mp4 (1024x1024, 16 fps, 80 frames)."""
import os, math, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, SIZE = 16, 80, 1024


def detect_circles(img):
    """Return list of (cx, cy, r, hollow) for each dark blob."""
    a = np.array(img.convert("L"))
    dark = a < 128
    lab, n = ndimage.label(dark)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 50:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        r = ((x1 - x0) + (y1 - y0)) / 4.0
        fill_ratio = len(xs) / (math.pi * r * r)
        hollow = fill_ratio < 0.6
        out.append((cx, cy, r, hollow))
    # left-to-right, top-to-bottom reading order
    out.sort(key=lambda c: (c[1] // 200, c[0]))
    return out


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    circles = detect_circles(base)
    hollow = [c for c in circles if c[3]]
    print(f"found {len(circles)} circles, {len(hollow)} hollow")

    RED = (255, 0, 0)
    W = 6          # ring stroke width
    GAP = 14       # ring radius offset beyond circle outline
    hold_start, hold_end = 8, 10
    active = N_FRAMES - hold_start - hold_end
    per = active / len(hollow)
    overlap = 0.35  # fraction of a slot the next ring starts early

    def ring_progress(k, f):
        start = hold_start + k * per * (1 - overlap * 0.4)
        dur = per * 1.15
        return ease((f - start) / dur)

    frames = []
    for f in range(N_FRAMES):
        im = base.copy()
        d = ImageDraw.Draw(im)
        for k, (cx, cy, r, _) in enumerate(hollow):
            p = ring_progress(k, f)
            if p <= 0:
                continue
            R = r + GAP
            box = [cx - R, cy - R, cx + R, cy + R]
            if p >= 1:
                d.ellipse(box, outline=RED, width=W)
            else:
                d.arc(box, start=-90, end=-90 + 360 * p, fill=RED, width=W)
        frames.append(im)

    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            fr.save(os.path.join(td, f"{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS), OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
