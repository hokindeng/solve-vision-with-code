#!/usr/bin/env python3
"""Generate the refraction video: remove the angle annotation, then draw the
red refracted ray inside the glass according to Snell's law."""
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

N_FRAMES = 70
FPS = 16
W = H = 1024

# Scene geometry (measured from first_frame.png)
P0 = (512, 512)            # incidence point on the interface (x, y)
THETA_I = 56.3             # incident angle from normal, degrees
N_AIR, N_GLASS = 1.00, 1.857
RAY_THICKNESS = 3
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Angle annotation (arc + "θ = 56°" text) bounding box, inclusive
ANN_X0, ANN_X1, ANN_Y0, ANN_Y1 = 478, 642, 472, 490
NORMAL_X = 512             # gray normal line column (must be preserved)

FADE_FRAMES = 18           # frames 1..FADE_FRAMES: annotation fades out
                           # remaining frames: red ray grows to the boundary


def annotation_mask(base):
    """Pixels belonging to the angle annotation (arc + text)."""
    white = (base == 255).all(axis=2)
    blue = (base[:, :, 2] == 255) & (base[:, :, 0] == 0) & (base[:, :, 1] == 0)
    gray = (base == 150).all(axis=2)
    m = np.zeros(base.shape[:2], bool)
    m[ANN_Y0:ANN_Y1 + 1, ANN_X0:ANN_X1 + 1] = True
    m &= ~white & ~blue
    m[:, NORMAL_X] &= ~gray[:, NORMAL_X]
    return m, blue


def clean_frame(base):
    """Background with the annotation removed (arc pixels that covered the
    blue ray are restored to blue)."""
    mask, blue = annotation_mask(base)
    target = base.copy()
    target[mask] = 255
    # The arc was drawn on top of the incident ray; the ray footprint is
    # 6 px wide per row, so restore blue where the arc sat inside it.
    for y in range(ANN_Y0, ANN_Y1 + 1):
        xs = np.nonzero(blue[y])[0]
        if len(xs) == 0:
            continue
        lo = xs.min()
        for x in np.nonzero(mask[y])[0]:
            if lo <= x <= lo + 5:
                target[y, x] = BLUE
    return target, mask


def refracted_direction():
    s = N_AIR * math.sin(math.radians(THETA_I)) / N_GLASS
    theta_t = math.asin(s)
    return math.sin(theta_t), math.cos(theta_t)   # (dx, dy), going down-right


def ray_end(dx, dy):
    """Point where the refracted ray leaves the image."""
    x0, y0 = P0
    t_candidates = []
    if dy > 0:
        t_candidates.append((H - 1 - y0) / dy)
    if dx > 0:
        t_candidates.append((W - 1 - x0) / dx)
    elif dx < 0:
        t_candidates.append((0 - x0) / dx)
    t = min(t_candidates)
    return t, (x0 + t * dx, y0 + t * dy)


def draw_ray(img, frac, dx, dy, t_full):
    """Draw the refracted ray from the interface out to fraction `frac`."""
    if frac <= 0:
        return img
    x0, y0 = P0
    # start just below the 3px interface line so the interface stays intact
    y_start = y0 + 2
    t_start = (y_start - y0) / dy
    t_end = t_full * frac
    if t_end <= t_start:
        return img
    p_a = (int(round(x0 + t_start * dx)), int(round(y0 + t_start * dy)))
    p_b = (int(round(x0 + t_end * dx)), int(round(y0 + t_end * dy)))
    cv2.line(img, p_a, p_b, RED, RAY_THICKNESS, cv2.LINE_8)
    return img


def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * u)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = np.array(Image.open(FIRST).convert("RGB"))
    clean, mask = clean_frame(base)
    dx, dy = refracted_direction()
    t_full, _ = ray_end(dx, dy)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    grow_frames = N_FRAMES - 1 - FADE_FRAMES
    for i in range(N_FRAMES):
        if i == 0:
            frame = base.copy()
        elif i <= FADE_FRAMES:
            a = i / FADE_FRAMES
            frame = base.copy()
            blend = (base.astype(np.float32) * (1 - a) + clean.astype(np.float32) * a)
            frame[mask] = np.clip(blend[mask] + 0.5, 0, 255).astype(np.uint8)
        else:
            frac = ease((i - FADE_FRAMES) / grow_frames)
            frame = draw_ray(clean.copy(), frac, dx, dy, t_full)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    # also save the final frame for inspection
    Image.fromarray(frame).save(os.path.join(OUT_DIR, "last_frame.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
