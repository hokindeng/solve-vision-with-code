#!/usr/bin/env python3
"""Outline the innermost of several concentric squares with a blue outline,
drawn progressively side by side."""
import os, subprocess, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 85
BLUE = np.array([0, 0, 255], dtype=np.uint8)
STROKE = 7  # outline thickness, centered on the square's edge


def find_innermost_square(img):
    """Squares are flat-colored; the innermost is the one whose bbox is smallest
    (flood from the exact center colour)."""
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    col = img[cy, cx]
    mask = np.all(img == col, axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_partial_outline(base, box, progress):
    """progress in [0,1]: fraction of the perimeter traced clockwise from top-left."""
    x0, y0, x1, y1 = box
    img = base.copy()
    half = STROKE // 2
    w = x1 - x0
    hgt = y1 - y0
    perim = 2 * (w + hgt)
    length = progress * perim
    # side segments as (start_point, end_point) clockwise
    sides = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
             ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    remaining = length
    for (sx, sy), (ex, ey) in sides:
        seg_len = abs(ex - sx) + abs(ey - sy)
        if remaining <= 0:
            break
        t = min(1.0, remaining / seg_len)
        px = int(round(sx + (ex - sx) * t))
        py = int(round(sy + (ey - sy) * t))
        xa, xb = sorted((sx, px)); ya, yb = sorted((sy, py))
        img[max(0, ya - half):yb + half + 1, max(0, xa - half):xb + half + 1] = BLUE
        remaining -= seg_len
    return img


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    box = find_innermost_square(base)
    os.makedirs(OUT_DIR, exist_ok=True)

    hold_start, hold_end = 10, 12
    draw_frames = N_FRAMES - hold_start - hold_end
    frames = []
    for i in range(N_FRAMES):
        if i < hold_start:
            frames.append(base)
        elif i >= N_FRAMES - hold_end:
            frames.append(draw_partial_outline(base, box, 1.0))
        else:
            p = (i - hold_start + 1) / draw_frames
            frames.append(draw_partial_outline(base, box, p))

    h, w, _ = base.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {OUT}: innermost square bbox={box}, {len(frames)} frames")


if __name__ == "__main__":
    main()
