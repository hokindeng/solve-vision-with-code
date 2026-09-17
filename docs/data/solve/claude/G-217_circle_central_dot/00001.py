#!/usr/bin/env python3
"""Animate a red circle being drawn around the middle-by-count dot."""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS = 16
N_FRAMES = 22


def find_dots(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < 128).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    dots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        dots.append((float(cent[i][0]), float(cent[i][1]), max(w, h) / 2.0))
    dots.sort(key=lambda d: d[0])  # left to right
    return dots


def draw_arc(base, cx, cy, r, frac, color=(255, 0, 0), thickness=5):
    """Draw an arc covering `frac` of a full circle, starting at the top, clockwise."""
    if frac <= 0:
        return base.copy()
    img = base.copy()
    scale = 4  # supersample for smooth anti-aliased edges
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    start = -90
    end = -90 + 360 * min(frac, 1.0)
    cv2.ellipse(big, (int(round(cx * scale)), int(round(cy * scale))),
                (int(round(r * scale)), int(round(r * scale))),
                0, start, end, color, thickness * scale, lineType=cv2.LINE_AA)
    if frac >= 1.0:
        cv2.circle(big, (int(round(cx * scale)), int(round(cy * scale))),
                   int(round(r * scale)), color, thickness * scale, lineType=cv2.LINE_AA)
    return cv2.resize(big, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    dots = find_dots(base)
    mid = dots[len(dots) // 2]
    cx, cy, dot_r = mid
    circle_r = dot_r * 2.2

    frames = [base.copy()]  # frame 0 is exactly the first frame
    for i in range(1, N_FRAMES):
        t = i / (N_FRAMES - 1)
        t = t * t * (3 - 2 * t)  # ease in/out
        frames.append(draw_arc(base, cx, cy, circle_r, t))
    frames[-1] = draw_arc(base, cx, cy, circle_r, 1.0)

    for k, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(OUT_DIR, f"frame_{k:03d}.png"))

    cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
           "-i", os.path.join(OUT_DIR, "frame_%03d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for k in range(len(frames)):
        os.remove(os.path.join(OUT_DIR, f"frame_{k:03d}.png"))
    print(f"wrote {OUT} ({len(frames)} frames, dot at ({cx:.0f},{cy:.0f}))")


if __name__ == "__main__":
    main()
