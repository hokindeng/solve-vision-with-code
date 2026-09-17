#!/usr/bin/env python3
"""Draw a red perpendicular line through the dot at the vertical midpoint of the
two horizontal parallel lines, growing from the upper line to the lower line."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS, N_FRAMES = 16, 50
RED = np.array([255, 0, 0], dtype=np.uint8)


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = base.shape

    # Locate the two black horizontal lines.
    black_rows = np.where((base == 0).all(axis=2).all(axis=1))[0]
    gaps = np.where(np.diff(black_rows) > 1)[0]
    upper = black_rows[: gaps[0] + 1]
    lower = black_rows[gaps[0] + 1:]
    y_top, y_bot = int(upper[0]), int(lower[-1])
    thickness = len(upper)
    mid_y = (upper.mean() + lower.mean()) / 2.0

    # Find the point (dot) whose centre sits at the vertical midpoint.
    between = np.zeros(h, bool)
    between[upper[-1] + 1: lower[0]] = True
    mask = (~(base == 255).all(axis=2)) & between[:, None]
    colors = np.unique(base[mask].reshape(-1, 3), axis=0)
    best, best_d = None, 1e9
    for c in colors:
        cm = (base == c).all(axis=2) & between[:, None]
        ys, xs = np.where(cm)
        # A colour may belong to several dots: cluster by connected rows.
        order = np.argsort(ys)
        ys, xs = ys[order], xs[order]
        splits = np.where(np.diff(ys) > 1)[0] + 1
        for gy, gx in zip(np.split(ys, splits), np.split(xs, splits)):
            d = abs(gy.mean() - mid_y)
            if d < best_d:
                best_d, best = d, (gx.mean(), gy.mean())
    x_mid = int(round(best[0]))
    half = thickness // 2
    x0, x1 = x_mid - half, x_mid - half + thickness  # match line thickness

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    total = y_bot - y_top + 1
    for i in range(N_FRAMES):
        frame = base.copy()
        t = i / (N_FRAMES - 1)
        t = t * t * (3 - 2 * t)  # ease in/out
        length = int(round(t * total))
        if length > 0:
            frame[y_top: y_top + length, x0:x1] = RED
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {OUT}: line x={x0}..{x1 - 1}, y={y_top}..{y_bot}")


if __name__ == "__main__":
    main()
