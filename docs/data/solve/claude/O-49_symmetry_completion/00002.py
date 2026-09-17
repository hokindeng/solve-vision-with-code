"""Complete the striped pattern by mirroring the left half onto the right half.

Renders 35 frames at 16 fps (1024x1024, H.264 yuv420p) to /app/output/video.mp4.
Frame 0 is exactly first_frame.png; the missing right-side cells fade in one
after another, and the final frame shows the completed symmetric stripes.
"""
import os
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 35
FPS = 16

BLUE = np.array([37, 99, 235], dtype=np.uint8)
GRID = np.array([203, 213, 225], dtype=np.uint8)


def segments(mask_1d, offset):
    """Return contiguous runs of True in a 1-D mask as (start, end) inclusive."""
    xs = np.where(mask_1d)[0] + offset
    segs, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x != p + 1:
            segs.append((s, p))
            s = x
        p = x
    segs.append((s, p))
    return segs


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    grid = (base == GRID).all(axis=2)
    blue = (base == BLUE).all(axis=2)

    # Grid bounding box and cell interiors (runs of non-grid pixels between lines).
    gy, gx = np.where(grid)
    x0, x1, y0, y1 = gx.min(), gx.max(), gy.min(), gy.max()
    probe_y = y0 + (y1 - y0) // 16          # inside the first row of cells
    probe_x = x0 + (x1 - x0) // 16
    col_segs = segments(~grid[probe_y, x0:x1 + 1], x0)
    row_segs = segments(~grid[y0:y1 + 1, probe_x], y0)
    n_rows, n_cols = len(row_segs), len(col_segs)

    def filled(r, c):
        ys, xs = row_segs[r], col_segs[c]
        return blue[(ys[0] + ys[1]) // 2, (xs[0] + xs[1]) // 2]

    # Target: mirror the left half across the vertical centre line.
    targets = []
    for r in range(n_rows):
        for c in range(n_cols // 2, n_cols):
            if filled(r, n_cols - 1 - c) and not filled(r, c):
                targets.append((r, c))
    targets.sort()  # top-to-bottom, left-to-right

    # Schedule: each cell fades in over `fade` frames, staggered across frames 1..34.
    fade = 6
    last_start = N_FRAMES - 1 - fade
    starts = np.linspace(1, last_start, len(targets)) if targets else []

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        img = base.copy()
        for (r, c), st in zip(targets, starts):
            a = np.clip((f - st + 1) / fade, 0.0, 1.0)
            if f == N_FRAMES - 1:
                a = 1.0
            if a <= 0:
                continue
            # smoothstep easing
            a = a * a * (3 - 2 * a)
            ys, xs = row_segs[r], col_segs[c]
            cell = img[ys[0]:ys[1] + 1, xs[0]:xs[1] + 1].astype(np.float32)
            img[ys[0]:ys[1] + 1, xs[0]:xs[1] + 1] = np.round(
                cell * (1 - a) + BLUE.astype(np.float32) * a).astype(np.uint8)
        frames.append(img)

    # Encode with ffmpeg via raw RGB pipe.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x1024", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "12",
        "-pix_fmt", "yuv420p", "-r", str(FPS), OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: {len(frames)} frames, filled {len(targets)} cells: {targets}")


if __name__ == "__main__":
    main()
