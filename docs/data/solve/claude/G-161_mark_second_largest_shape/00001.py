#!/usr/bin/env python3
"""Detect the 4 circles in first_frame.png, find the second largest, and
animate a red ring being drawn around it over 40 frames at 16 fps."""
import os, subprocess, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(ROOT, "first_frame.png")
OUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N = 16, 40
RED = (255, 0, 0)

def detect_circles(img):
    mask = ~np.all(img == 255, axis=2)
    lab, n = ndimage.label(mask)
    circles = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        cx, cy = xs.mean(), ys.mean()
        r = (xs.max() - xs.min() + 1 + ys.max() - ys.min() + 1) / 4.0
        circles.append((cx, cy, r, len(ys)))
    return circles

def main():
    base = Image.open(FIRST).convert("RGB")
    arr = np.array(base)
    circles = detect_circles(arr)
    assert len(circles) == 4, circles
    circles.sort(key=lambda c: -c[3])            # largest -> smallest
    cx, cy, r, _ = circles[1]                    # second largest
    ring_r = r + 22                              # gap around the shape
    width = 6

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    S = 4  # supersample for smooth arc
    for i in range(N):
        # frame 0 unchanged; ring sweeps in from frame 1, complete by ~frame 34
        t = 0.0 if i == 0 else min(1.0, i / 34.0)
        frame = base.copy()
        if t > 0:
            big = Image.new("RGBA", (base.width * S, base.height * S), (0, 0, 0, 0))
            d = ImageDraw.Draw(big)
            box = [(cx - ring_r) * S, (cy - ring_r) * S, (cx + ring_r) * S, (cy + ring_r) * S]
            start = -90
            end = start + 360 * t
            if t >= 1.0:
                d.ellipse(box, outline=RED + (255,), width=width * S)
            else:
                d.arc(box, start=start, end=end, fill=RED + (255,), width=width * S)
                # round caps
                for ang in (start, end):
                    a = math.radians(ang)
                    px, py = (cx + ring_r * math.cos(a)) * S, (cy + ring_r * math.sin(a)) * S
                    hw = width * S / 2
                    d.ellipse([px - hw, py - hw, px + hw, py + hw], fill=RED + (255,))
            overlay = big.resize(base.size, Image.LANCZOS)
            frame.paste(overlay, (0, 0), overlay)
        frame.save(os.path.join(frames_dir, f"{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-r", str(FPS), OUT,
    ], check=True)
    print("wrote", OUT, "target circle:", (round(cx), round(cy), round(r)))

if __name__ == "__main__":
    main()
