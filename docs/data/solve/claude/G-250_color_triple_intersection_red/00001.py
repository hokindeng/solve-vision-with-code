#!/usr/bin/env python3
"""Color the triple intersection of the three Venn circles red, animated over 60 frames."""
import subprocess
import numpy as np
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
N_FRAMES, FPS = 60, 16
RED = np.array([255, 0, 0], dtype=np.uint8)


def fit_circle(mask):
    ys, xs = np.nonzero(mask)
    A = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]
    b = xs ** 2 + ys ** 2
    cx, cy, k = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(k + cx ** 2 + cy ** 2)
    d = np.hypot(xs - cx, ys - cy)
    return cx, cy, d.min()  # inner edge of the stroke


def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    h, w, _ = img.shape
    bg = np.array([255, 255, 255])
    # Stroke colors = every non-background color present (first three circles).
    cols, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    strokes = [c for c in cols[np.argsort(-counts)] if not np.array_equal(c, bg)][:3]

    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.ones((h, w), bool)
    for c in strokes:
        cx, cy, r = fit_circle(np.all(img == c, axis=2))
        inside &= (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    region = inside & np.all(img == bg, axis=2)  # fill only background pixels

    ry, rx = np.nonzero(region)
    cyc, cxc = ry.mean(), rx.mean()
    dist = np.hypot(xx - cxc, yy - cyc)
    rmax = dist[region].max() + 1.0

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        t = t * t * (3 - 2 * t)  # smoothstep easing
        frame = img.copy()
        if i == N_FRAMES - 1:
            m = region
        else:
            m = region & (dist <= t * rmax)
        frame[m] = RED
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(frame).save("/app/output/last_frame.png")


if __name__ == "__main__":
    main()
