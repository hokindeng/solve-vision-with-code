#!/usr/bin/env python3
"""Generate the mirror-reflection video from first_frame.png.

Scene: a blue ray comes down onto a horizontal mirror at 6 deg from the normal.
Task: remove the angle annotation (arc + "theta = 6deg" label), then draw the
reflected ray (angle of reflection = angle of incidence) from the hit point out to
the image boundary. Reflectivity 0.71 -> the reflected ray is drawn as blue at
71% opacity over the white background.
"""
import math
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT_DIR = os.path.join(HERE, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

W = H = 1024
FPS = 16
N_FRAMES = 35

REFLECTIVITY = 0.71
BLUE = np.array([0, 0, 255], dtype=np.float64)          # incident ray colour (RGB)
WHITE = np.array([255, 255, 255], dtype=np.float64)
REFL_COLOR = tuple(int(round(v)) for v in (REFLECTIVITY * BLUE + (1 - REFLECTIVITY) * WHITE))

# Geometry measured from first_frame.png
HIT = (597.0, 594.0)          # incidence point (centre of the normal line / mirror surface)
THETA_DEG = 6.0               # angle of incidence w.r.t. the normal (ray comes from upper-left)
RAY_W = 3                     # stroke width of the rays in the source image
ARROW_LEN = 26                # arrowhead leg length (as in the source image)
ARROW_HALF_ANGLE = math.radians(30)


def base_frame_without_annotation(first):
    """Return a copy of the first frame with the angle annotation removed."""
    img = first.copy()
    # Label "theta = 6deg": bounding box x 647..714, y 554..568 -> background white.
    img[554:569, 645:717] = 255
    # Small angle arc between ray and normal: a black bar rows 554-555, x 592..597.
    img[554:556, 592:598] = 255
    # Restore the incident ray pixels that the arc covered (ray occupies x 592..594 there).
    img[554:556, 592:595] = (0, 0, 255)
    return img


def reflected_endpoint():
    """Extend the reflected ray from HIT to the top image boundary."""
    x0, y0 = HIT
    dx, dy = math.sin(math.radians(THETA_DEG)), -math.cos(math.radians(THETA_DEG))
    t = -y0 / dy                       # reach y = 0
    return (x0 + t * dx, y0 + t * dy), (dx, dy)


def draw_ray(img, p0, p1, direction, color, with_arrow):
    """Draw a ray segment p0->p1 (float coords) plus an open-V arrowhead at p1."""
    c = tuple(int(v) for v in color)
    q0 = (int(round(p0[0])), int(round(p0[1])))
    q1 = (int(round(p1[0])), int(round(p1[1])))
    cv2.line(img, q0, q1, c, RAY_W, lineType=cv2.LINE_8)
    if with_arrow:
        dx, dy = direction
        ang = math.atan2(dy, dx)
        for s in (-1, 1):
            a = ang + math.pi + s * ARROW_HALF_ANGLE
            e = (int(round(p1[0] + ARROW_LEN * math.cos(a))),
                 int(round(p1[1] + ARROW_LEN * math.sin(a))))
            cv2.line(img, q1, e, c, RAY_W, lineType=cv2.LINE_8)


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    first = np.array(Image.open(FIRST).convert("RGB"))
    assert first.shape == (H, W, 3)
    base = base_frame_without_annotation(first)
    end, direction = reflected_endpoint()

    frames = [first.copy()]  # frame 0 must be the original first frame
    for i in range(1, N_FRAMES):
        t = ease(i / (N_FRAMES - 1))
        img = base.copy()
        tip = (HIT[0] + t * (end[0] - HIT[0]), HIT[1] + t * (end[1] - HIT[1]))
        if t > 0:
            draw_ray(img, HIT, tip, direction, REFL_COLOR, with_arrow=True)
        frames.append(img)

    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for k, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp_dir, f"{k:04d}.png"))
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, "last_frame.png"))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", os.path.join(tmp_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
        "-x264-params", "keyint=1", OUT,
    ], check=True)
    for k in range(len(frames)):
        os.remove(os.path.join(tmp_dir, f"{k:04d}.png"))
    os.rmdir(tmp_dir)
    print("wrote", OUT, "frames:", len(frames), "reflected colour:", REFL_COLOR, "endpoint:", end)


if __name__ == "__main__":
    main()
