#!/usr/bin/env python3
"""Locate the intersection of two line segments and animate a red circle around it."""
import os, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(BASE, "first_frame.png")
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")
N_FRAMES, FPS, SS = 30, 16, 4  # SS = supersampling factor for the overlay


def fit_line(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], float), np.array([vx, vy], float)


def intersect(p1, d1, p2, d2):
    # p1 + t*d1 = p2 + s*d2
    A = np.array([d1, -d2]).T
    t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + t * d1


def main():
    img = Image.open(FIRST).convert("RGB")
    arr = np.array(img)
    white = np.all(arr > 240, axis=2)
    colored = ~white
    # split the non-background pixels into the two dominant colours
    cols, inv, cnt = np.unique(arr[colored].reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
    top = np.argsort(-cnt)[:2]
    masks = [np.all(arr == cols[i], axis=2) for i in top]
    (p1, d1), (p2, d2) = fit_line(masks[0]), fit_line(masks[1])
    cx, cy = intersect(p1, d1, p2, d2)
    print(f"intersection at ({cx:.1f}, {cy:.1f})")

    radius, width = 34, 6
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        frame = img.copy()
        if i > 0:
            # sweep the arc from 0 to 360 degrees across frames 1..N-1
            frac = i / (N_FRAMES - 1)
            sweep = 360.0 * frac
            W, H = img.size
            ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            r, w = radius * SS, width * SS
            box = [cx * SS - r, cy * SS - r, cx * SS + r, cy * SS + r]
            start = -90.0
            if sweep >= 359.9:
                d.ellipse(box, outline=(220, 30, 30, 255), width=w)
            else:
                d.arc(box, start, start + sweep, fill=(220, 30, 30, 255), width=w)
                # round caps
                for ang in (start, start + sweep):
                    a = np.deg2rad(ang)
                    ex, ey = cx * SS + (r - w / 2) * np.cos(a), cy * SS + (r - w / 2) * np.sin(a)
                    d.ellipse([ex - w / 2, ey - w / 2, ex + w / 2, ey + w / 2], fill=(220, 30, 30, 255))
            ov = ov.resize((W, H), Image.LANCZOS)
            frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
        frames.append(frame)

    tmp = os.path.join(OUT_DIR, "frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp, f"{i:03d}.png"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp, "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", OUT,
    ], check=True)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
