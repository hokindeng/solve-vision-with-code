#!/usr/bin/env python3
"""Animate the reflection of a light ray off a mirror (reflectivity 0.60).

Frame 0 is first_frame.png. Over the clip the angle annotation (arc + label)
fades away and the reflected ray grows from the hit point to the image edge.
Everything else is left untouched.
"""
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(APP, "first_frame.png")
OUT_DIR = os.path.join(APP, "output")
OUT = os.path.join(OUT_DIR, "video.mp4")

N_FRAMES, FPS = 35, 16
REFLECTIVITY = 0.60
RAY_COLOR = np.array([0, 0, 255], dtype=np.float32)      # incident ray colour (RGB)
THICKNESS = 4

base = np.array(Image.open(SRC).convert("RGB"))
H, W = base.shape[:2]

# ---- geometry, measured from first_frame.png -----------------------------
hit = np.array([629.0, 600.0])            # where the incident ray meets the mirror
start = np.array([144.0, 49.0])           # start of incident ray
d_in = hit - start
d_in /= np.linalg.norm(d_in)
d_ref = np.array([d_in[0], -d_in[1]])     # mirror is horizontal: flip vertical component

# distance to image edge along the reflected direction
ts = []
if d_ref[0] > 0: ts.append((W - 1 - hit[0]) / d_ref[0])
if d_ref[0] < 0: ts.append(-hit[0] / d_ref[0])
if d_ref[1] > 0: ts.append((H - 1 - hit[1]) / d_ref[1])
if d_ref[1] < 0: ts.append(-hit[1] / d_ref[1])
T_EDGE = min(ts)
end = hit + d_ref * T_EDGE

# ---- clean background: annotation (arc + label) removed ------------------
clean = base.copy()
y0, y1, x0, x1 = 550, 600, 590, 780
roi = base[y0:y1, x0:x1].astype(int)
white = (roi == 255).all(2)
blue = (roi[:, :, 0] == 0) & (roi[:, :, 1] == 0) & (roi[:, :, 2] == 255)
ann = ~(white | blue)
xs = np.arange(x0, x1)[None, :].repeat(y1 - y0, 0)
ys = np.arange(y0, y1)[:, None].repeat(x1 - x0, 1)
normal_px = (xs == 629) & (ys >= 572)     # keep the normal line
ann &= ~normal_px
ann_mask = np.zeros((H, W), bool)
ann_mask[y0:y1, x0:x1] = ann
clean[ann_mask] = 255

# normal line pixels (drawn over the rays in the source) to restore afterwards
normal_mask = np.zeros((H, W), bool)
normal_mask[572:633, 629] = True


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def render(i):
    # annotation fade: frames 1..8
    fade = ease((i - 0) / 8.0) if i > 0 else 0.0
    frame = base.astype(np.float32)
    frame[ann_mask] = (1 - fade) * base[ann_mask] + fade * clean[ann_mask]

    # reflected ray growth: frames 3..34
    grow = ease((i - 3) / (N_FRAMES - 1 - 3)) if i >= 3 else 0.0
    if grow > 0:
        length = T_EDGE * grow
        p1 = hit + d_ref * length
        layer = np.zeros((H, W), np.uint8)
        cv2.line(layer, tuple(np.round(hit).astype(int)), tuple(np.round(p1).astype(int)),
                 255, THICKNESS, cv2.LINE_8)
        if grow >= 0.999:
            # arrow head at the far end (tip on the image edge)
            L, wdt = 26.0, 10.0
            perp = np.array([-d_ref[1], d_ref[0]])
            tip = end
            b1 = tip - d_ref * L + perp * wdt
            b2 = tip - d_ref * L - perp * wdt
            pts = np.round(np.array([tip, b1, b2])).astype(np.int32)
            cv2.fillPoly(layer, [pts], 255)
        m = layer > 0
        m[599:, :] = False                     # never paint on/below the mirror surface
        m &= ~normal_mask                      # normal line stays on top
        # reflectivity 0.60 -> ray drawn at 60% intensity over the background
        frame[m] = (1 - REFLECTIVITY) * frame[m] + REFLECTIVITY * RAY_COLOR
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(N_FRAMES):
        f = render(i)
        if i == 0:
            assert np.array_equal(f, base), "frame 0 must equal first_frame.png"
        Image.fromarray(f).save(os.path.join(frames_dir, f"{i:04d}.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(frames_dir, "%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow",
           "-r", str(FPS), OUT]
    subprocess.run(cmd, check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
