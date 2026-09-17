#!/usr/bin/env python3
"""Find the shape with the unique color and progressively outline it in black."""
import os, subprocess
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, THICK = 21, 16, 6


def find_unique_shape(img):
    """Return the mask of the single shape whose color appears exactly once."""
    flat = img.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    bg = tuple(cols[np.argmax(counts)])
    best = None
    for c, n in zip(cols, counts):
        if tuple(c) == bg or n < 200:
            continue
        m = np.all(img == c, axis=2).astype(np.uint8)
        k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        comps = [i for i in range(1, k) if stats[i, cv2.CC_STAT_AREA] > 200]
        if len(comps) == 1:
            best = (lab == comps[0]).astype(np.uint8)
    assert best is not None, "no unique-colored shape found"
    return best


def main():
    img = np.array(Image.open(FIRST).convert("RGB"))
    mask = find_unique_shape(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    cnt = np.vstack([cnt, cnt[:1]])  # close the loop
    total = len(cnt) - 1

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        fr = img.copy()
        if f > 0:
            k = int(round(total * f / (N_FRAMES - 1)))
            pts = cnt[: k + 1]
            cv2.polylines(fr, [pts.reshape(-1, 1, 2)], False, (0, 0, 0), THICK, cv2.LINE_8)
        frames.append(fr)

    h, w = img.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    print("wrote", OUT)


if __name__ == "__main__":
    main()
