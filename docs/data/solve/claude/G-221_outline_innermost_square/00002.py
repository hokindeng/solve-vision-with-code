#!/usr/bin/env python3
"""Outline the innermost of several concentric squares with a blue outline,
drawn step by step (edge by edge) over the duration of the video."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
FPS = 16
N_FRAMES = 85
BLUE = np.array([0, 0, 255], dtype=np.uint8)
THICK = 8  # outline thickness in px (centred on the square's edge)


def find_squares(img):
    """Return list of (x0, y0, x1, y1, area) for each solid-colour square,
    largest first. The background is the colour touching the image border."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    colors, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(h, w)
    border = np.concatenate([inv[0], inv[-1], inv[:, 0], inv[:, -1]])
    bg = np.bincount(border).argmax()
    squares = []
    for ci in range(len(colors)):
        if ci == bg or counts[ci] < 100:
            continue
        ys, xs = np.where(inv == ci)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        squares.append((int(x0), int(y0), int(x1), int(y1), int((x1 - x0 + 1) * (y1 - y0 + 1))))
    squares.sort(key=lambda s: -s[4])
    return squares


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    squares = find_squares(base)
    # innermost = smallest bounding box (all share the same centre)
    x0, y0, x1, y1, _ = squares[-1]
    print("squares (outer->inner):", [(s[0], s[1], s[2], s[3]) for s in squares])
    print("innermost:", (x0, y0, x1, y1))

    # Perimeter path (clockwise from the top-left corner), one segment per edge.
    edges = [
        ((x0, y0), (x1, y0)),  # top
        ((x1, y0), (x1, y1)),  # right
        ((x1, y1), (x0, y1)),  # bottom
        ((x0, y1), (x0, y0)),  # left
    ]
    lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in edges]
    total = float(sum(lengths))

    half = THICK // 2
    h, w, _ = base.shape

    def draw_partial(frame, progress):
        """Draw the outline up to `progress` (0..1) of the perimeter length."""
        remaining = progress * total
        for (ax, ay), (bx, by), L in zip([e[0] for e in edges], [e[1] for e in edges], lengths):
            if remaining <= 0:
                break
            frac = min(1.0, remaining / L)
            ex = ax + (bx - ax) * frac
            ey = ay + (by - ay) * frac
            xa, xb = sorted((ax, int(round(ex))))
            ya, yb = sorted((ay, int(round(ey))))
            xs0, xs1 = max(0, xa - half), min(w, xb + half + 1)
            ys0, ys1 = max(0, ya - half), min(h, yb + half + 1)
            frame[ys0:ys1, xs0:xs1] = BLUE
            remaining -= L

    # Timing: hold the original, then draw the four edges one after another
    # with a brief pause after each, then hold the completed result.
    hold_start = 6
    hold_end = 8
    draw_frames = N_FRAMES - hold_start - hold_end  # 71 frames of drawing
    per_edge = draw_frames / 4.0
    pause = 3  # frames of pause at the end of each edge

    def progress_at(i):
        t = i - hold_start
        if t <= 0:
            return 0.0
        if t >= draw_frames:
            return 1.0
        e = int(t // per_edge)
        local = (t - e * per_edge) / (per_edge - pause)
        local = min(1.0, max(0.0, local))
        return (e + local) / 4.0

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "12",
        "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        frame = base.copy()
        p = progress_at(i)
        if p > 0:
            draw_partial(frame, p)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
