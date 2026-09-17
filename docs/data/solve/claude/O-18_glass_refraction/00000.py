#!/usr/bin/env python3
"""Generate the refraction video: remove the angle annotation, then draw the
red refracted ray (Snell's law) growing from the incidence point to the image
boundary. Every other pixel stays identical to first_frame.png."""
import math
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST = os.path.join(HERE, "first_frame.png")
OUT = os.path.join(HERE, "output", "video.mp4")

N_FRAMES = 70
FPS = 16
FADE_FRAMES = 14           # frames 1..14 fade the annotation out
THETA_I_DEG = 69.8
N_AIR, N_GLASS = 1.00, 1.850

base = np.array(Image.open(FIRST).convert("RGB"))
H, W, _ = base.shape

# --- locate scene elements from the image itself ---------------------------
black = base.sum(2) < 100
rows_full = [r for r in range(H) if black[r].sum() > W // 2]
y_if = int(round(np.mean(rows_full)))                     # interface row (512)

blue = (base[:, :, 2] > 150) & (base[:, :, 0] < 100) & (base[:, :, 1] < 100)
ys, xs = np.nonzero(blue)
x_if = int(xs.max()) - 1                                  # incidence x (512)
# refine using the normal line column (gray 150) if present
g150 = (base[:, :, 0] == 150) & (base[:, :, 1] == 150) & (base[:, :, 2] == 150)
cols, cnt = np.unique(np.nonzero(g150)[1], return_counts=True)
if len(cols):
    x_if = int(cols[np.argmax(cnt)])

# --- annotation mask: non-white, non-blue pixels above the interface, except
#     the gray normal line -----------------------------------------------------
nonwhite = base.sum(2) < 765
annot = nonwhite & ~blue
annot[y_if - 6:, :] = False
annot[:, x_if] &= ~g150[:, x_if]

clean = base.copy()
clean[annot] = 255

# --- refracted ray geometry --------------------------------------------------
sin_t = N_AIR * math.sin(math.radians(THETA_I_DEG)) / N_GLASS
theta_t = math.asin(sin_t)
dx, dy = math.sin(theta_t), math.cos(theta_t)              # rightward, downward
p0 = np.array([x_if, y_if], dtype=float)
# extend to image boundary
t_candidates = []
if dy > 0:
    t_candidates.append((H - 1 - p0[1]) / dy)
if dx > 0:
    t_candidates.append((W - 1 - p0[0]) / dx)
t_max = min(t_candidates)
p1 = p0 + t_max * np.array([dx, dy])
RED = (255, 0, 0)
THICK = 2


def draw_ray(img, frac):
    if frac <= 0:
        return img
    pe = p0 + frac * (p1 - p0)
    layer = img.copy()
    cv2.line(layer, (int(round(p0[0])), int(round(p0[1]))),
             (int(round(pe[0])), int(round(pe[1]))), RED, THICK, cv2.LINE_8)
    # the refracted ray exists only inside the glass (at/below the interface)
    img[y_if:] = layer[y_if:]
    return img


def make_frame(i):
    if i == 0:
        return base.copy()
    if i <= FADE_FRAMES:
        a = i / FADE_FRAMES
        f = base.astype(float) * (1 - a) + clean.astype(float) * a
        return np.clip(f + 0.5, 0, 255).astype(np.uint8)
    frac = (i - FADE_FRAMES) / (N_FRAMES - 1 - FADE_FRAMES)
    frac = min(1.0, max(0.0, frac))
    return draw_ray(clean.copy(), frac)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i in range(N_FRAMES):
            Image.fromarray(make_frame(i)).save(os.path.join(td, f"f{i:04d}.png"))
        Image.fromarray(make_frame(N_FRAMES - 1)).save(os.path.join(HERE, "output", "last_frame.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(td, "f%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p", "-r", str(FPS), OUT,
        ], check=True)
    print(f"incidence ({x_if},{y_if}); theta_t = {math.degrees(theta_t):.2f} deg; "
          f"ray end ({p1[0]:.1f},{p1[1]:.1f}); wrote {OUT}")


if __name__ == "__main__":
    main()
