#!/usr/bin/env python3
"""Find the rectangle closest to a square and circle it in red, animated."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48


def find_rectangles(img):
    """Return list of (x0, y0, x1, y1) for each filled colored rectangle
    (bbox extended by one pixel to include the black outline)."""
    h, w, _ = img.shape
    white = np.all(img == 255, axis=2)
    black = np.all(img == 0, axis=2)
    fill = ~(white | black)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), 8)
    rects = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 200:
            continue
        # outline is 1px around fill
        rects.append((x - 1, y - 1, x + bw, y + bh))
    return rects


def squareness(r):
    x0, y0, x1, y1 = r
    w, h = x1 - x0 + 1, y1 - y0 + 1
    ratio = w / h
    return abs(np.log(ratio))  # 0 == perfect square


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    rects = find_rectangles(base)
    scores = [squareness(r) for r in rects]
    for r, s in zip(rects, scores):
        print(f"rect {r}: w={r[2]-r[0]+1} h={r[3]-r[1]+1} |log(w/h)|={s:.4f}")
    best = rects[int(np.argmin(scores))]
    print("closest to square:", best)

    x0, y0, x1, y1 = best
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_diag = np.hypot((x1 - x0 + 1) / 2.0, (y1 - y0 + 1) / 2.0)
    radius = int(round(half_diag + 12))
    thickness = 5
    red = (255, 0, 0)

    # Timeline: hold (comparison), sweep the circle, hold finished result.
    sweep_start, sweep_end = 6, 43
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        if i >= sweep_start:
            t = min(1.0, (i - sweep_start) / float(sweep_end - sweep_start))
            t = t * t * (3 - 2 * t)  # ease in/out
            ang = 360.0 * t
            if ang > 0:
                cv2.ellipse(f, (int(round(cx)), int(round(cy))), (radius, radius),
                            -90, 0, ang, red, thickness, lineType=cv2.LINE_AA)
        frames.append(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    h, w, _ = base.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
           "-r", str(FPS), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
