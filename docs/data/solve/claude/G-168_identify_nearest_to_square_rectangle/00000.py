#!/usr/bin/env python3
"""Find the rectangle closest to a square and circle it in red, step by step."""
import math
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 48
RED = (220, 30, 30)


def detect_rects(img):
    """Return list of (x, y, w, h) for non-background connected components."""
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(2) > 30).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    rects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > 200:
            rects.append((int(x), int(y), int(w), int(h)))
    return rects


def squareness(r):
    """Distance from a 1:1 ratio; 0 means a perfect square."""
    _, _, w, h = r
    ratio = w / h
    return abs(math.log(ratio))  # symmetric in w/h vs h/w


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def draw_arc(frame, center, radius, frac, color, thickness):
    if frac <= 0:
        return
    if frac >= 1:
        cv2.circle(frame, center, radius, color, thickness, lineType=cv2.LINE_AA)
        return
    cv2.ellipse(frame, center, (radius, radius), -90, 0, 360 * frac,
                color, thickness, lineType=cv2.LINE_AA)


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    rects = detect_rects(base)
    order = sorted(rects, key=lambda r: (r[1], r[0]))  # scan order for comparison step
    best = min(rects, key=squareness)
    bx, by, bw, bh = best
    cx, cy = bx + bw // 2, by + bh // 2
    radius = int(math.hypot(bw, bh) / 2) + 14

    # Timeline (frames): 0-5 hold, 6-27 compare each rectangle, 28-43 draw circle, 44-47 hold.
    hold_end = 6
    cmp_start, cmp_end = 6, 28
    circ_start, circ_end = 28, 44
    per = (cmp_end - cmp_start) / len(order)

    frames = []
    for f in range(N_FRAMES):
        frame = base.copy()
        if cmp_start <= f < cmp_end:
            # Comparison step: outline the rectangle currently being examined.
            k = min(int((f - cmp_start) / per), len(order) - 1)
            x, y, w, h = order[k]
            local = ((f - cmp_start) - k * per) / per
            gray = int(90 + 120 * abs(2 * local - 1))  # pulse in and out
            cv2.rectangle(frame, (x - 6, y - 6), (x + w + 5, y + h + 5),
                          (gray, gray, gray), 2, lineType=cv2.LINE_AA)
        elif f >= circ_start:
            frac = ease((f - circ_start + 1) / (circ_end - circ_start))
            draw_arc(frame, (cx, cy), radius, frac, RED, 5)
        frames.append(frame)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames_raw.rgb")
    with open(tmp, "wb") as fh:
        for fr in frames:
            fh.write(np.ascontiguousarray(fr).tobytes())
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1024x1024", "-r", str(FPS), "-i", tmp,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "medium",
        "-r", str(FPS), OUT], check=True)
    os.remove(tmp)
    print(f"rects={rects}\nbest={best} ratio={bw/bh:.3f}\nwrote {OUT}")


if __name__ == "__main__":
    main()
