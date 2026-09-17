#!/usr/bin/env python3
"""Generate the refraction video: remove the angle annotation and grow the
red refracted ray (Snell's law) inside the glass down to the image boundary."""
import math
import subprocess
import numpy as np
import cv2
from PIL import Image

SRC = "/app/first_frame.png"
OUT = "/app/output/video.mp4"
W = H = 1024
FPS = 16
N_FRAMES = 70

THETA_I = 30.0          # incidence angle from the normal, degrees
N_AIR, N_GLASS = 1.00, 1.630
INCIDENCE = (512, 512)  # where the blue ray meets the interface (x, y)
RED = (255, 0, 0)
THICKNESS = 3

base = np.array(Image.open(SRC).convert("RGB"))

# ---- annotation mask: arc (pure black, above the interface) + text "θ = 30°" ----
ann = np.zeros((H, W), bool)
# arc region: between normal and blue ray, above interface line (rows 511-513)
arc_reg = base[440:511, 440:560]
arc_black = (arc_reg == 0).all(axis=2)
ann[440:511, 440:560] = arc_black
# text region: everything non-white here belongs to the label
txt_reg = base[460:500, 540:700]
ann[460:500, 540:700] = (txt_reg != 255).any(axis=2)

# ---- blue ray pixels hidden under the arc: restore them instead of whitening ----
blue = (base[:, :, 2] > 150) & (base[:, :, 0] < 100) & (base[:, :, 1] < 100)
bys, bxs = np.nonzero(blue)
sel = bys < 450
pfit = np.polyfit(bys[sel], bxs[sel], 1)          # x = p0*y + p1 (ray centre line)
under_ray = np.zeros_like(ann)
ays, axs = np.nonzero(ann)
for y, x in zip(ays, axs):
    cx = pfit[0] * y + pfit[1]
    if abs(x - cx) <= 1.55:
        under_ray[y, x] = True
ann_white = ann & ~under_ray          # pixels that fade to white
ann_blue = ann & under_ray            # pixels that fade back to blue
BLUE = np.array([0, 0, 255], float)

# ---- refracted ray geometry ----
sin_r = N_AIR * math.sin(math.radians(THETA_I)) / N_GLASS
theta_r = math.asin(sin_r)
dx, dy = math.sin(theta_r), math.cos(theta_r)   # down-right, like the incident ray
x0, y0 = INCIDENCE
# extend past bottom boundary
L_full = (H + 10 - y0) / dy
x_end = x0 + dx * L_full
if x_end > W + 10:  # would exit the right edge first
    L_full = (W + 10 - x0) / dx

FADE_END = 14          # annotation fully gone by this frame
RAY_START = 14         # ray begins growing here
RAY_END = N_FRAMES - 1


def make_frame(i):
    img = base.copy()
    # annotation fade-out (blend toward white)
    if i > 0:
        a = 1.0 - min(1.0, i / FADE_END)   # remaining annotation opacity
        px = img[ann_white].astype(float)
        img[ann_white] = np.clip(255 - (255 - px) * a, 0, 255).astype(np.uint8)
        px = img[ann_blue].astype(float)
        img[ann_blue] = np.clip(BLUE + (px - BLUE) * a, 0, 255).astype(np.uint8)
    # red refracted ray growing
    if i >= RAY_START:
        t = (i - RAY_START) / (RAY_END - RAY_START)
        t = min(1.0, max(0.0, t))
        L = L_full * t
        if L > 0:
            x1 = int(round(x0 + dx * L))
            y1 = int(round(y0 + dy * L))
            cv2.line(img, (x0, y0), (x1, y1), RED, THICKNESS, lineType=cv2.LINE_8)
    return img


def main():
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-preset", "slow",
           "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        fr = make_frame(i)
        if i == 0:
            assert np.array_equal(fr, base)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(make_frame(N_FRAMES - 1)).save("/app/output/last_frame.png")
    print("theta_r = %.2f deg, ray end x=%.1f" % (math.degrees(theta_r), x0 + dx * L_full))


if __name__ == "__main__":
    main()
