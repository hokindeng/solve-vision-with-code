#!/usr/bin/env python3
"""Rotate the polygon counterclockwise about the marked center onto the dashed target outline."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

APP = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(APP, "first_frame.png")
OUT = os.path.join(APP, "output", "video.mp4")
W = H = 1024
FPS = 16
N_FRAMES = 70

BG = np.array([240, 240, 240], dtype=np.uint8)
POLY_COLOR = (132, 152, 174)
EDGE_COLOR = (50, 50, 50)
CENTER = np.array([779.0, 359.0])            # rotation center (marker dot)
VERTS = np.array([[780.04, 358.93], [887.89, 459.94], [948.99, 395.10],
                  [905.81, 354.83], [885.19, 377.36], [819.15, 316.00]])
TOTAL_DEG = 200.72                           # counterclockwise rotation to reach target outline


def rotate(verts, deg):
    """Visually counterclockwise rotation (image y axis points down)."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s], [-s, c]])
    return (verts - CENTER) @ R.T + CENTER


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))
    fi = first.astype(int)
    poly_mask = (np.abs(fi - np.array(POLY_COLOR)).sum(2) < 40) | (np.abs(fi - np.array(EDGE_COLOR)).sum(2) < 10)
    # Background with the polygon erased.
    base = first.copy()
    base[poly_mask] = BG
    # Static overlay (dashed outline, marker) drawn on top of the moving polygon.
    static_mask = (~poly_mask) & (np.abs(fi - BG).sum(2) > 0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = tempfile.mkdtemp()
    for i in range(N_FRAMES):
        if i == 0:
            frame = first
        else:
            deg = TOTAL_DEG * ease(i / (N_FRAMES - 1))
            pts = [(int(round(x)), int(round(y))) for x, y in rotate(VERTS, deg)]
            layer = Image.fromarray(base.copy())
            ImageDraw.Draw(layer).polygon(pts, fill=POLY_COLOR, outline=EDGE_COLOR)
            frame = np.array(layer)
            frame[static_mask] = first[static_mask]
        Image.fromarray(frame).save(os.path.join(tmp, f"f{i:04d}.png"))

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
                    "-r", str(FPS), OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
