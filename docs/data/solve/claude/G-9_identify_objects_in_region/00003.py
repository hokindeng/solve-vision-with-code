#!/usr/bin/env python3
"""Outline all circles inside the circular region with a green border.

Detects the circular region and the circle-shaped objects inside it from
first_frame.png, then animates a green outline sweeping around each circle
over 40 frames (16 fps). All other pixels are left untouched.
"""
import os
import subprocess
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES = 40
FPS = 16
GREEN = (0, 200, 0)  # BGR
THICK = 4


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    cnts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    info = []
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if a < 50:
            continue
        p = cv2.arcLength(c, True)
        (x, y), r = cv2.minEnclosingCircle(c)
        info.append(dict(i=i, parent=hier[i][3], area=a, circ=4 * np.pi * a / (p * p),
                         cx=x, cy=y, r=r, cnt=c))
    # Region containers: top-level contours (large). Circular region = high circularity.
    regions = [d for d in info if d["parent"] == -1 and d["area"] > 20000]
    circ_region = max(regions, key=lambda d: d["circ"])
    # Inner boundary of the region's border ring is its direct child; objects are its children.
    inner = [d for d in info if d["parent"] == circ_region["i"] and d["area"] > 20000]
    inner_idx = inner[0]["i"] if inner else circ_region["i"]
    objs = [d for d in info if d["parent"] == inner_idx]
    circles = [d for d in objs if d["circ"] > 0.85]
    return circles


def main():
    base = cv2.imread(SRC)
    circles = detect(base)
    print("circles to outline:", [(round(c["cx"]), round(c["cy"]), round(c["r"])) for c in circles])

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    for k in range(N_FRAMES):
        frame = base.copy()
        t = k / (N_FRAMES - 1)  # 0 .. 1
        sweep = 360.0 * t
        if sweep > 0:
            for c in circles:
                center = (int(round(c["cx"])), int(round(c["cy"])))
                rad = int(round(c["r"])) + THICK // 2 + 1
                if sweep >= 360:
                    cv2.circle(frame, center, rad, GREEN, THICK, lineType=cv2.LINE_AA)
                else:
                    cv2.ellipse(frame, center, (rad, rad), 0, -90, -90 + sweep, GREEN, THICK,
                                lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(frames_dir, f"f{k:03d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
