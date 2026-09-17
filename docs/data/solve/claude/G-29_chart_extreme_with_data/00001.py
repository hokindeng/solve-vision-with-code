"""Highlight the maximum-value point of the scatter chart with a red rectangular border.

The chart's max value is 97 (light-blue dot). The red border is traced progressively
around that dot over 48 frames; every other pixel stays identical to first_frame.png.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

FPS, N_FRAMES = 16, 48
RED = (255, 0, 0)
THICK = 3


def find_max_point(img):
    """Locate the centre of the max-value (97) point: the light-blue (135,206,235) dot."""
    a = np.asarray(img).astype(int)
    m = (abs(a[..., 0] - 135) < 30) & (abs(a[..., 1] - 206) < 30) & (abs(a[..., 2] - 235) < 30)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
    return cx, cy, r


def rect_path(x0, y0, x1, y1):
    """Corners of the rectangle in drawing order (clockwise from top-left)."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def draw_partial_rect(img, corners, frac):
    """Draw the first `frac` of the rectangle perimeter as a red stroke."""
    if frac <= 0:
        return img
    d = ImageDraw.Draw(img)
    segs = list(zip(corners[:-1], corners[1:]))
    total = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in segs)
    remain = frac * total
    for a, b in segs:
        L = abs(b[0] - a[0]) + abs(b[1] - a[1])
        if remain <= 0:
            break
        t = min(1.0, remain / L)
        end = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        # draw as a filled box so joins/corners are square and crisp
        h = THICK // 2
        xa, xb = sorted([a[0], end[0]])
        ya, yb = sorted([a[1], end[1]])
        d.rectangle([xa - h, ya - h, xb + h - 1 + (THICK % 2), yb + h - 1 + (THICK % 2)], fill=RED)
        remain -= L
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(SRC).convert("RGB")
    cx, cy, r = find_max_point(base)
    pad = 4
    half = int(round(r + pad))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = int(round(cx)) + half, int(round(cy)) + half
    corners = rect_path(x0, y0, x1, y1)

    frames = []
    for i in range(N_FRAMES):
        # frame 0 is the untouched first frame; last frame is the completed border
        frac = i / (N_FRAMES - 1)
        f = base.copy()
        draw_partial_rect(f, corners, frac)
        frames.append(np.asarray(f))

    writer = imageio.get_writer(
        OUT, fps=FPS, codec="libx264", pixelformat="yuv420p",
        ffmpeg_params=["-crf", "12"], macro_block_size=1,
    )
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"wrote {OUT}: {len(frames)} frames, max point at ({cx:.1f},{cy:.1f}) r={r:.1f}")


if __name__ == "__main__":
    main()
