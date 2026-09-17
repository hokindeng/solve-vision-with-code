#!/usr/bin/env python3
"""Identify hollow (outlined, unfilled) circles and circle each with a red ring."""
import os, subprocess, math
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 80

def detect_hollow(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 128).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30:
            continue
        r = (w + h) / 4.0
        fill_ratio = area / (math.pi * r * r)
        cx, cy = cents[i]
        shapes.append(dict(cx=float(cx), cy=float(cy), r=r, hollow=fill_ratio < 0.5))
    hollow = [s for s in shapes if s["hollow"]]
    hollow.sort(key=lambda s: (s["cy"], s["cx"]))  # top-to-bottom, left-to-right
    return shapes, hollow

def draw_ring(img, cx, cy, r, frac, color=(0, 0, 255), thick=6):
    if frac <= 0:
        return
    if frac >= 1:
        cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)), color, thick, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (int(round(cx)), int(round(cy))), (int(round(r)), int(round(r))),
                    0, -90, -90 + 360 * frac, color, thick, cv2.LINE_AA)

def main():
    base = cv2.imread(SRC)
    shapes, hollow = detect_hollow(base)
    print(f"found {len(shapes)} shapes, {len(hollow)} hollow:",
          [(round(s['cx']), round(s['cy']), round(s['r'])) for s in hollow])

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    # Timeline: hold first frame, then each hollow point gets its ring drawn in turn,
    # then hold the final result.
    hold_start, hold_end = 8, 12
    active = N_FRAMES - hold_start - hold_end
    per = active / max(1, len(hollow))
    frames = []
    for k in range(N_FRAMES):
        img = base.copy()
        t = k - hold_start
        for i, s in enumerate(hollow):
            frac = (t - i * per) / (per * 0.85)  # draw over 85% of its slot, brief pause after
            frac = max(0.0, min(1.0, frac))
            draw_ring(img, s["cx"], s["cy"], s["r"] + 14, frac)
        frames.append(img)
    frames[0] = base.copy()  # first frame identical to source
    for k, img in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f"f{k:04d}.png"), img)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-preset", "slow", OUT], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
