#!/usr/bin/env python3
"""Identify hollow (outlined, unfilled) circles in first_frame.png and circle each
with a red ring, animated step by step. Output: /app/output/video.mp4"""
import os, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
FPS, N_FRAMES, SS = 16, 80, 4  # supersample factor for smooth rings


def detect_circles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = (gray < 128).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, connectivity=8)
    circles = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        cx, cy = cent[i]
        r = (w + h) / 4.0
        fill_ratio = area / (math.pi * r * r)
        circles.append(dict(cx=float(cx), cy=float(cy), r=float(r),
                            hollow=fill_ratio < 0.5))
    return circles


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(SRC).convert("RGB"))
    H, W = base.shape[:2]
    circles = detect_circles(base)
    hollow = sorted([c for c in circles if c["hollow"]], key=lambda c: (c["cy"], c["cx"]))
    print(f"found {len(circles)} circles, {len(hollow)} hollow:")
    for c in hollow:
        print(f"  center=({c['cx']:.1f},{c['cy']:.1f}) r={c['r']:.1f}")

    # Schedule: short hold, then each hollow point gets a sweep (ring drawn as growing arc),
    # then hold at end.
    hold_start, hold_end = 6, 10
    active = N_FRAMES - hold_start - hold_end
    per = active / max(len(hollow), 1)
    RED = (220, 30, 30)
    ring_w = 6
    margin = 10

    frames = []
    for f in range(N_FRAMES):
        overlay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for k, c in enumerate(hollow):
            t0 = hold_start + k * per
            p = (f - t0) / (per * 0.85)  # sweep finishes a bit before next starts
            if p <= 0:
                continue
            p = ease(min(p, 1.0))
            R = (c["r"] + margin) * SS
            box = [(c["cx"] * SS - R), (c["cy"] * SS - R), (c["cx"] * SS + R), (c["cy"] * SS + R)]
            if p >= 1.0:
                d.ellipse(box, outline=RED + (255,), width=ring_w * SS)
            else:
                d.arc(box, start=-90, end=-90 + 360 * p, fill=RED + (255,), width=ring_w * SS)
        ov = overlay.resize((W, H), Image.LANCZOS)
        frame = Image.fromarray(base.copy())
        frame.paste(ov, (0, 0), ov)  # only pixels under the ring change
        frames.append(np.array(frame))

    # Frame 0 must equal the source exactly.
    frames[0] = base.copy()

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "medium", OUT], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
