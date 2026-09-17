#!/usr/bin/env python3
"""Rotate the L-shaped polygon counterclockwise about the marked center until it
matches the dashed target outline. Everything except the polygon is kept
pixel-identical to first_frame.png."""
import math, os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 70

BG = (240, 240, 240)
FILL = (60, 139, 119)
EDGE = (50, 50, 50)
DASH = (100, 100, 100)

# Geometry measured from first_frame.png
CX, CY = 793.0, 214.0                       # rotation center (marker)
LOCAL = [(0, 0), (150, 0), (150, -90), (90, -90), (90, -60), (0, -60)]  # L-shape, vertex at center
ANGLE0 = 18.9                               # initial orientation (deg, image coords)
ANGLE1 = -57.1                              # target orientation matching dashed outline
# ANGLE1 < ANGLE0: decreasing angle in y-down image coords == counterclockwise on screen


def verts(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [(CX + x * c - y * s, CY + x * s + y * c) for x, y in LOCAL]


def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * u)


def main():
    first = np.array(Image.open(FIRST).convert("RGB"))

    poly_mask = np.all(first == FILL, axis=2) | np.all(first == EDGE, axis=2)
    base = first.copy()
    base[poly_mask] = BG                      # scene without the polygon

    # elements that must stay on top / unchanged: dashed outline + center marker
    overlay_mask = (np.all(first == DASH, axis=2) | np.all(first == (0, 0, 0), axis=2)
                    | np.all(first == (255, 255, 255), axis=2))

    frames = []
    for i in range(N_FRAMES):
        if i == 0:
            frames.append(first.copy())
            continue
        u = ease(i / (N_FRAMES - 1))
        ang = ANGLE0 + (ANGLE1 - ANGLE0) * u
        img = Image.fromarray(base.copy())
        d = ImageDraw.Draw(img)
        d.polygon(verts(ang), fill=FILL, outline=EDGE, width=1)
        arr = np.array(img)
        arr[overlay_mask] = first[overlay_mask]
        frames.append(arr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(td, f"f{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
            "-r", str(FPS), OUT,
        ], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
