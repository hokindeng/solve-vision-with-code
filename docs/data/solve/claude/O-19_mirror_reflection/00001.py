#!/usr/bin/env python3
"""Generate the mirror-reflection video from first_frame.png.

Action: fade out the angle annotation (arc + "θ = 30°" label), then grow the
reflected ray from the hit point toward the top edge of the image. The
reflected ray is drawn at 46% intensity (mirror reflectivity 0.46).
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES = 35
FPS = 16
REFLECTIVITY = 0.46

# Scene geometry measured from first_frame.png
HIT = (685, 462)          # incidence point (x, y) on mirror top surface
NORMAL_X = 685            # gray normal line column
NORMAL_Y0 = 434           # top of normal line
THETA = np.deg2rad(30.0)  # angle of incidence w.r.t. normal
LINE_W = 4
RAY_COLOR = np.array([0, 0, 255], dtype=np.float32)      # RGB
WHITE = np.array([255, 255, 255], dtype=np.float32)
ANNOT_BOX = (650, 418, 830, 458)  # x0, y0, x1, y1 region containing arc + text


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def build_clean(base):
    """Return base image with the angle annotation removed."""
    img = base.copy()
    x0, y0, x1, y1 = ANNOT_BOX
    reg = img[y0:y1, x0:x1].astype(int)
    is_blue = (reg[:, :, 0] == 0) & (reg[:, :, 1] == 0) & (reg[:, :, 2] == 255)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    is_normal = (xs == NORMAL_X) & (ys >= NORMAL_Y0) & (reg[:, :, 0] == 150)
    keep = is_blue | is_normal
    reg[~keep] = 255
    img[y0:y1, x0:x1] = reg.astype(np.uint8)
    return img


def reflected_endpoint():
    """Point where the reflected ray leaves the image (top edge)."""
    d = np.array([np.sin(THETA), -np.cos(THETA)])
    t = HIT[1] / -d[1]  # reach y = 0
    return np.array(HIT, dtype=float) + d * t, d


def draw_reflected(img, frac):
    """Draw reflected ray of fractional length `frac` with arrowhead at tip."""
    if frac <= 0:
        return img
    end, d = reflected_endpoint()
    start = np.array(HIT, dtype=float)
    tip = start + (end - start) * frac
    color = REFLECTIVITY * RAY_COLOR + (1 - REFLECTIVITY) * WHITE
    col = tuple(int(round(c)) for c in color)

    layer = img.copy()
    p0 = tuple(int(round(v)) for v in start)
    p1 = tuple(int(round(v)) for v in tip)
    cv2.line(layer, p0, p1, col, LINE_W, lineType=cv2.LINE_8)
    # arrowhead: two wings at +-45 deg from the reversed direction
    wing = 30.0
    back = -d
    for s in (+1, -1):
        ang = s * np.pi / 4
        w = np.array([back[0] * np.cos(ang) - back[1] * np.sin(ang),
                      back[0] * np.sin(ang) + back[1] * np.cos(ang)])
        q = tip + w * wing
        cv2.line(layer, p1, tuple(int(round(v)) for v in q), col, LINE_W,
                 lineType=cv2.LINE_8)

    # Keep existing scene elements (incident ray, normal, mirror) on top.
    existing = np.any(img != 255, axis=2)
    layer[existing] = img[existing]
    return layer


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    clean = build_clean(base)

    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        # annotation fade: frames 0..12
        fade = ease(i / 12.0)
        img = (base.astype(np.float32) * (1 - fade)
               + clean.astype(np.float32) * fade)
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
        # ray growth: frames 4..34
        grow = ease((i - 4) / (N_FRAMES - 1 - 4))
        if i == N_FRAMES - 1:
            grow = 1.0
        if i == 0:
            img = base.copy()
        else:
            img = draw_reflected(img, grow)
        frames.append(img)

    tmp = os.path.join(OUT_DIR, "frames_%03d.png")
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp % i)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "14", "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    for i in range(N_FRAMES):
        os.remove(tmp % i)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
