#!/usr/bin/env python3
"""Outline the innermost concentric square in blue, traced step by step."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 85
BLUE = np.array([0, 0, 255], dtype=np.uint8)
STROKE = 6  # stroke width in px, centered on the square boundary


def find_innermost_square(img):
    """Concentric squares share a center; the innermost is the smallest
    solid-colour square containing the image's shape center."""
    h, w, _ = img.shape
    bg = tuple(img[0, 0])
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    best = None
    for c, n in zip(map(tuple, cols), counts):
        if c == bg or n < 50:
            continue
        m = np.all(img == c, axis=2)
        ys, xs = np.where(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        # must be a filled square
        if abs((x1 - x0) - (y1 - y0)) > 2:
            continue
        if m[y0:y1 + 1, x0:x1 + 1].mean() < 0.98:
            continue
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if best is None or area < best[0]:
            best = (area, x0, y0, x1, y1)
    return best[1:]


def edge_points(x0, y0, x1, y1):
    """Ordered points along the square boundary (clockwise from top-left)."""
    pts = []
    pts += [(x, y0) for x in range(x0, x1 + 1)]          # top
    pts += [(x1, y) for y in range(y0 + 1, y1 + 1)]      # right
    pts += [(x, y1) for x in range(x1 - 1, x0 - 1, -1)]  # bottom
    pts += [(x0, y) for y in range(y1 - 1, y0, -1)]      # left
    return pts


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = base.shape
    x0, y0, x1, y1 = find_innermost_square(base)
    pts = edge_points(x0, y0, x1, y1)
    total = len(pts)
    half = STROKE // 2

    # Pace: frame 0 untouched; tracing runs frames 1..N-6; hold finished result.
    trace_frames = N_FRAMES - 6
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        t = min(1.0, max(0.0, i / trace_frames))
        n = int(round(t * total))
        if n > 0:
            mask = np.zeros((h, w), dtype=bool)
            for (px, py) in pts[:n]:
                mask[max(0, py - half):py + half + 1 - (STROKE % 2 == 0),
                     max(0, px - half):px + half + 1 - (STROKE % 2 == 0)] = True
            f[mask] = BLUE
        frames.append(f)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "12",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print(f"square=({x0},{y0})-({x1},{y1}) frames={len(frames)} -> {OUT}")


if __name__ == "__main__":
    main()
