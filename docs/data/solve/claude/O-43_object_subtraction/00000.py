#!/usr/bin/env python3
"""Remove all cube (square) objects from first_frame.png, animated over 96 frames.

Squares are detected as connected non-background components whose pixel area
fills their bounding box (fill ratio ~1). Each square shrinks toward its centre
with an eased schedule; the three squares are staggered so the action covers
the whole clip. Every pixel outside the squares' bounding boxes is untouched.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 96


def find_squares(img):
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(2) > 30).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    squares = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area / float(w * h) > 0.97 and abs(w - h) <= 2 and area > 100:
            squares.append((int(x), int(y), int(w), int(h)))
    squares.sort(key=lambda s: (s[1], s[0]))  # top-to-bottom order
    return squares, tuple(int(v) for v in bg)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    base = np.array(Image.open(SRC).convert("RGB"))
    squares, bg = find_squares(base)
    assert squares, "no squares found"

    # Staggered schedules: (start_frame, end_frame) for each square.
    n = len(squares)
    dur = 44
    span = (N_FRAMES - 6 - dur) / max(n - 1, 1)
    sched = [(int(round(4 + i * span)), int(round(4 + i * span)) + dur) for i in range(n)]

    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{base.shape[1]}x{base.shape[0]}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow", "-tune", "stillimage", "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        frame = base.copy()
        for (x, y, w, h), (s0, s1) in zip(squares, sched):
            p = ease((f - s0) / float(s1 - s0))
            if p <= 0:
                continue
            color = base[y + h // 2, x + w // 2].copy()
            frame[y:y + h, x:x + w] = bg
            scale = 1.0 - p
            nw, nh = int(round(w * scale)), int(round(h * scale))
            if nw > 0 and nh > 0:
                cx, cy = x + w / 2.0, y + h / 2.0
                x0, y0 = int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))
                frame[y0:y0 + nh, x0:x0 + nw] = color
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, "ffmpeg failed"
    print(f"wrote {OUT}: {len(squares)} squares removed, schedules {sched}")


if __name__ == "__main__":
    main()
