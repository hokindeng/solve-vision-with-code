#!/usr/bin/env python3
"""Mark the peaks of the wave in first_frame.png, one by one, left to right."""
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 10
RED = (255, 0, 0)
R_CIRCLE = 22
R_DOT = 5
LINE_W = 3


def wave_profile(img):
    """Return (xs, ys): centerline y for every column that has ink."""
    g = np.array(img.convert("L"))
    ink = g < 128
    xs, ys = [], []
    for x in range(g.shape[1]):
        rows = np.nonzero(ink[:, x])[0]
        if len(rows):
            xs.append(x)
            ys.append(rows.mean())
    return np.array(xs), np.array(ys)


def find_peaks(xs, ys):
    """Local maxima of wave value (= minimal image y). Flat tops -> centre."""
    h = -ys  # wave value: up is larger
    # Smooth lightly to suppress single-pixel rasterisation jitter.
    k = 5
    hs = np.convolve(np.pad(h, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    peaks = []
    n = len(hs)
    i = 1
    while i < n - 1:
        if hs[i] > hs[i - 1]:
            j = i
            while j + 1 < n and hs[j + 1] == hs[j]:
                j += 1
            if j < n - 1 and hs[j] > hs[j + 1]:
                # refine: within a window take the raw top columns
                lo, hi = max(0, i - 6), min(n, j + 7)
                seg = h[lo:hi]
                top = np.nonzero(seg == seg.max())[0] + lo
                c = int(round(top.mean()))
                peaks.append((int(xs[c]), float(ys[c])))
            i = j + 1
        else:
            i += 1
    return peaks


def draw_marker(draw, x, y, circle_r, dot_r):
    if dot_r > 0:
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=RED)
    if circle_r > 0:
        draw.ellipse([x - circle_r, y - circle_r, x + circle_r, y + circle_r],
                     outline=RED, width=LINE_W)


def main():
    base = Image.open(SRC).convert("RGB")
    xs, ys = wave_profile(base)
    peaks = find_peaks(xs, ys)
    print("peaks:", peaks)

    # Schedule: frame 0 untouched; remaining frames split evenly among peaks,
    # each peak grows in over its slot (dot first, then circle expands).
    steps = N_FRAMES - 1
    slot = steps / max(1, len(peaks))
    frames = []
    for f in range(N_FRAMES):
        im = base.copy()
        d = ImageDraw.Draw(im)
        for k, (px, py) in enumerate(peaks):
            start = 1 + k * slot
            end = 1 + (k + 1) * slot - 1  # frame at which this peak is complete
            if f < start - 1e-9:
                continue
            t = 1.0 if end <= start else min(1.0, (f - start) / (end - start))
            t = max(0.0, t)
            circle_r = int(round(R_CIRCLE * t)) if t > 0 else 0
            draw_marker(d, px, py, circle_r, R_DOT)
        frames.append(im)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
