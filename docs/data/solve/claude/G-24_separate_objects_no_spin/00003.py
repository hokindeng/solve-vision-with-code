#!/usr/bin/env python3
"""Move the two squares horizontally into their dashed target outlines."""
import os, subprocess, tempfile
import numpy as np, cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
N_FRAMES, FPS = 30, 16


def rect_center(mask):
    """Center of a rotated square from its pixels (robust to dashed gaps)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (_, _), (_, _), th = cv2.minAreaRect(pts)
    t = np.deg2rad(th)
    n1 = np.array([np.cos(t), np.sin(t)]); n2 = np.array([-np.sin(t), np.cos(t)])
    p1, p2 = pts @ n1, pts @ n2
    mid = np.array([(p1.min() + p1.max()) / 2, (p2.min() + p2.max()) / 2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ mid


def main():
    base = np.array(Image.open(FIRST).convert("RGB"))
    a = base.astype(int)
    nonwhite = np.abs(a - 255).sum(2) > 30
    X = np.arange(1024)[None, :]; Y = np.arange(1024)[:, None]

    # Object masks (left side): red square (x<345) and yellow square (345<=x<600)
    red = nonwhite & (X < 345)
    yel = nonwhite & (X >= 345) & (X < 600)
    # Dashed outlines (right side), split along a line between the two outlines
    gray = nonwhite & (X > 640)
    split = Y > 470 + (X - 696) * 38 / 121
    big, small = gray & split, gray & ~split

    moves = []
    for obj, tgt in [(red, big), (yel, small)]:
        c_obj, c_tgt = rect_center(obj), rect_center(tgt)
        dx = int(round(c_tgt[0] - c_obj[0]))
        # dilate the object mask slightly so anti-aliased edge pixels travel too
        m = cv2.dilate(obj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        moves.append((m, dx))
        print(f"object center {c_obj.round(2)} -> target {c_tgt.round(2)}  dx={dx}")

    # Background with objects removed (background is pure white)
    bg = base.copy()
    for m, _ in moves:
        bg[m] = 255

    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        s = i / (N_FRAMES - 1)
        e = s * s * (3 - 2 * s)  # smoothstep ease in/out
        frame = bg.copy()
        for m, dx in moves:
            d = int(round(dx * e))
            ys, xs = np.nonzero(m)
            frame[ys, xs + d] = base[ys, xs]
        if i == 0:
            frame = base.copy()
        Image.fromarray(frame).save(os.path.join(tmp, f"f{i:03d}.png"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
