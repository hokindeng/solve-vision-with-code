#!/usr/bin/env python3
"""Rotate the 4 objects in first_frame.png counterclockwise by 163 degrees
around their own centroids, leaving everything else untouched."""
import os, subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, TOTAL_DEG = 17, 16, 163.0
MIN_AREA = 1000  # objects are large; title glyphs are small

def main():
    img = np.array(Image.open(SRC).convert("RGB"))
    H, W = img.shape[:2]
    nonwhite = (np.abs(img.astype(int) - 255).sum(2) > 30).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nonwhite, connectivity=8)

    objects = []
    background = img.copy()
    kernel = np.ones((5, 5), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        m = (lab == i).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)   # fill interior gaps
        # fill holes so the interior is part of the object
        ff = m.copy(); pad = np.zeros((H + 2, W + 2), np.uint8)
        cv2.floodFill(ff, pad, (0, 0), 1)
        m = (m | (1 - ff)).astype(np.uint8)
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()          # centroid of the filled shape
        md = cv2.dilate(m, np.ones((3, 3), np.uint8))  # include anti-aliased rim
        layer = np.dstack([img, md * 255]).astype(np.uint8)  # RGBA
        objects.append((layer, (cx, cy)))
        background[md > 0] = 255               # clear object from background

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for f in range(N_FRAMES):
        ang = TOTAL_DEG * f / (N_FRAMES - 1)
        frame = background.astype(np.float32)
        for layer, (cx, cy) in objects:
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)  # +angle = CCW
            rot = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            a = rot[..., 3:4].astype(np.float32) / 255.0
            frame = frame * (1 - a) + rot[..., :3].astype(np.float32) * a
        fr = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
        frames.append(fr if f > 0 else img.copy())  # frame 0 is exactly the source

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print("wrote", OUT, len(frames), "frames")

if __name__ == "__main__":
    main()
