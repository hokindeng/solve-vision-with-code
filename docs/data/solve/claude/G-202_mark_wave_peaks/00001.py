#!/usr/bin/env python3
"""Mark the local maxima of the wave in first_frame.png, one by one, left to right."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS = 10, 16
RED = (220, 30, 30)
R_CIRCLE, W_CIRCLE, R_DOT = 22, 4, 6


def find_peaks(img):
    """Return (x, y) of every local maximum of the wave (min image-y => max value)."""
    g = np.array(img.convert("L"))
    mask = g < 128
    W = g.shape[1]
    prof = np.full(W, np.nan)
    for x in range(W):
        ys = np.nonzero(mask[:, x])[0]
        if len(ys):
            prof[x] = ys.mean()
    valid = ~np.isnan(prof)
    # smooth lightly to suppress rasterisation jitter of the line centre
    k = 9
    filled = np.where(valid, prof, 1e9)
    sm = np.convolve(filled, np.ones(k) / k, mode="same")
    peaks = []
    h = k // 2
    x = h + 1
    while x < W - h - 2:
        if valid[x - h - 1] and valid[x + h + 1] and sm[x] < sm[x - 1] and sm[x] <= sm[x + 1]:
            # plateau handling: take centre of equal run
            x2 = x
            while x2 + 1 < W - h and sm[x2 + 1] == sm[x]:
                x2 += 1
            xc = (x + x2) / 2.0
            # refine: centroid of the topmost wave pixels near xc
            lo, hi = int(xc) - 12, int(xc) + 13
            sub = mask[:, lo:hi]
            ys, xs = np.nonzero(sub)
            top = ys.min()
            sel = ys <= top + 1
            xc = lo + xs[sel].mean()
            yc = prof[int(round(xc))]
            peaks.append((float(xc), float(yc)))
            x = x2 + 1
        x += 1
    return peaks


def draw_marker(draw, x, y, progress):
    """progress in (0,1]: dot appears first, then the circle sweeps closed."""
    if progress <= 0:
        return
    draw.ellipse([x - R_DOT, y - R_DOT, x + R_DOT, y + R_DOT], fill=RED)
    sweep = min(1.0, progress) * 360.0
    box = [x - R_CIRCLE, y - R_CIRCLE, x + R_CIRCLE, y + R_CIRCLE]
    if sweep >= 359.9:
        draw.ellipse(box, outline=RED, width=W_CIRCLE)
    else:
        draw.arc(box, start=-90, end=-90 + sweep, fill=RED, width=W_CIRCLE)


def main():
    base = Image.open(SRC).convert("RGB")
    peaks = find_peaks(base)
    print("peaks:", [(round(x, 1), round(y, 1)) for x, y in peaks])
    n = len(peaks)
    # frame 0 untouched; remaining frames split evenly across peaks, left to right
    steps = N_FRAMES - 1
    frames = []
    for f in range(N_FRAMES):
        im = base.copy()
        d = ImageDraw.Draw(im)
        if f > 0:
            t = f / steps  # 0..1 over the animation
            for i, (x, y) in enumerate(peaks):
                p = t * n - i  # local progress of peak i
                if p > 0:
                    # first sub-step shows the dot with a partial arc, finishes on next
                    draw_marker(d, x, y, min(1.0, p if p >= 1 else 0.5 + 0.5 * p))
        frames.append(im)
    # ensure last frame is fully complete
    d = ImageDraw.Draw(frames[-1])
    for x, y in peaks:
        draw_marker(d, x, y, 1.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(frames):
            im.save(os.path.join(td, f"f{i:03d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow",
            "-r", str(FPS), OUT,
        ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
