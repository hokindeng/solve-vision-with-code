#!/usr/bin/env python3
"""Animate a red circle drawn around the middle dot (by count) of a row of dots."""
import os, subprocess, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES = 16, 22
RED = (220, 30, 30)


def find_dots(img):
    gray = np.array(img.convert("L"))
    mask = (gray < 128).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    dots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        dots.append((float(cents[i][0]), float(cents[i][1]), max(w, h) / 2.0))
    dots.sort(key=lambda d: d[0])  # left to right
    return dots


def draw_arc_frame(base, cx, cy, r, width, frac):
    """Return base image with a red arc from -90deg sweeping frac*360deg, anti-aliased."""
    if frac <= 0:
        return base.copy()
    S = 4  # supersampling
    W, H = base.size
    overlay = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(overlay)
    box = [(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S]
    start = -90.0
    end = start + 360.0 * min(frac, 1.0)
    if frac >= 1.0:
        d.ellipse(box, outline=255, width=int(round(width * S)))
    else:
        d.arc(box, start=start, end=end, fill=255, width=int(round(width * S)))
        # round caps at both ends of the arc
        cap_r = width * S / 2.0
        for ang in (start, end):
            a = math.radians(ang)
            px = cx * S + (r - width / 2.0) * S * math.cos(a)
            py = cy * S + (r - width / 2.0) * S * math.sin(a)
            d.ellipse([px - cap_r, py - cap_r, px + cap_r, py + cap_r], fill=255)
    alpha = overlay.resize((W, H), Image.LANCZOS)
    red = Image.new("RGB", (W, H), RED)
    out = base.copy()
    out.paste(red, (0, 0), alpha)
    return out


def main():
    base = Image.open(FIRST).convert("RGB")
    dots = find_dots(base)
    mid = dots[len(dots) // 2]  # equal number of dots on each side
    cx, cy, dot_r = mid
    circle_r = dot_r * 1.45  # sits in the gap between the dot and its neighbors
    width = 3.5

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        # first frame untouched; circle completes at the final frame with ease-in-out
        t = i / (N_FRAMES - 1)
        frac = 0.0 if i == 0 else (1.0 if i == N_FRAMES - 1 else 0.5 - 0.5 * math.cos(math.pi * t))
        frames.append(draw_arc_frame(base, cx, cy, circle_r, width, frac))

    tmp = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp, f"f_{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "f_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "1", "-preset", "slow", "-tune", "stillimage",
        "-r", str(FPS), OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print(f"wrote {OUT}: {N_FRAMES} frames, middle dot at ({cx:.0f},{cy:.0f}) of {len(dots)}")


if __name__ == "__main__":
    main()
